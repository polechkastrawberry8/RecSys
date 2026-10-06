"""Offline evaluation: ranking quality, business proxies, monitoring, failure cases.

The split is per user by time. A test purchase is relevant. Metrics are averaged
only over users who have at least one relevant purchase in that slice, which is
the usual ranking protocol. Revenue@K is the sum of prices of those hits. It is
not incremental revenue: the user already bought the item in the log.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from recsys.collaborative import ItemBasedCF, UserBasedCF, rating_error_report
from recsys.content import ContentBasedRecommender
from recsys.data import load_dataset, season_of, time_split
from recsys.heuristics import (
    AlsoBoughtRecommender,
    PopularityRecommender,
    RandomRecommender,
    SameCategoryRecommender,
    SeasonalRecommender,
)
from recsys.hybrid import COLD_WEIGHTS, HybridRecommender
from recsys.metrics import average_precision_at_k, ndcg_at_k, precision_at_k, recall_at_k
from recsys.monitor import RecommendationMonitor
from recsys.vulnerabilities import keyword_stuffing, sensitive_co_purchase, shilling_attack

K = 10
ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"


def _models() -> list:
    return [
        RandomRecommender(),
        PopularityRecommender(),
        PopularityRecommender(half_life_days=90),
        SeasonalRecommender(),
        SameCategoryRecommender(),
        AlsoBoughtRecommender(),
        ContentBasedRecommender(),
        UserBasedCF(),
        ItemBasedCF(),
        HybridRecommender(),
    ]


def _rename(model) -> str:
    if isinstance(model, PopularityRecommender) and model.half_life_days:
        return "popularity_recent"
    return model.name


def _purchases(frame: pd.DataFrame) -> dict[int, set[int]]:
    bought = frame[frame["event_type"].eq("purchase")]
    return {
        int(user): {int(item) for item in items}
        for user, items in bought.groupby("user_id")["item_id"]
    }


def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(np.mean(values))


def evaluate_slice(
    model,
    users: list[int],
    relevant: dict[int, set[int]],
    request_season: dict[int, str],
    prices: dict[int, float],
    categories: dict[int, str],
    favorites: dict[int, set[str]],
    top_popular: set[int],
    popularity_prob: dict[int, float],
    similarity: np.ndarray,
    item_index: dict[int, int],
    k: int = K,
) -> dict[str, float]:
    precision, recall, ndcg, average_precision = [], [], [], []
    hit, revenue, avg_price, spread, novelty, bias, recovery, diversity = [], [], [], [], [], [], [], []
    union: set[int] = set()
    empty = 0
    for user_id in users:
        recs = model.recommend(user_id, k=k, context={"season": request_season.get(user_id, "all")})
        recommended = [int(item) for item in recs["item_id"]]
        if not recommended:
            empty += 1
        rel = relevant[user_id]
        precision.append(precision_at_k(recommended, rel, k))
        recall.append(recall_at_k(recommended, rel, k))
        ndcg.append(ndcg_at_k(recommended, rel, k))
        average_precision.append(average_precision_at_k(recommended, rel, k))
        hits = [item for item in recommended if item in rel]
        hit.append(1.0 if hits else 0.0)
        revenue.append(sum(prices.get(item, 0.0) for item in hits))
        avg_price.append(_mean([prices.get(item, 0.0) for item in recommended]) if recommended else 0.0)
        spread.append(float(len({categories.get(item, "") for item in recommended})))
        novelty.append(
            _mean([-np.log2(popularity_prob.get(item, 1.0)) for item in recommended]) if recommended else 0.0
        )
        bias.append(_mean([1.0 if item in top_popular else 0.0 for item in recommended]) if recommended else 0.0)
        fav = favorites.get(user_id, set())
        recovery.append(
            _mean([1.0 if categories.get(item) in fav else 0.0 for item in recommended]) if recommended else 0.0
        )
        diversity.append(_diversity(recommended, similarity, item_index))
        union.update(recommended)

    diversity_values = [value for value in diversity if np.isfinite(value)]
    return {
        "users": float(len(users)),
        "precision": _mean(precision),
        "recall": _mean(recall),
        "ndcg": _mean(ndcg),
        "map": _mean(average_precision),
        "hit_rate": _mean(hit),
        "revenue_at_k": _mean(revenue),
        "avg_price": _mean(avg_price),
        "category_spread": _mean(spread),
        "novelty": _mean(novelty),
        "pop_bias": _mean(bias),
        "pref_recovery": _mean(recovery),
        "diversity": _mean(diversity_values),
        "coverage": len(union) / max(len(item_index), 1),
        "empty_rate": empty / max(len(users), 1),
    }


def _diversity(recommended: list[int], similarity: np.ndarray, item_index: dict[int, int]) -> float:
    positions = [item_index[item] for item in recommended if item in item_index]
    if len(positions) < 2:
        return float("nan")
    block = similarity[np.ix_(positions, positions)]
    triangle = block[np.triu_indices(len(positions), k=1)]
    return float(np.mean(1.0 - triangle))




def _table(frame: pd.DataFrame) -> str:
    show = frame.copy()
    for column in show.columns:
        if column != "model":
            show[column] = show[column].map(lambda value: f"{value:.3f}")
    header = "| " + " | ".join(show.columns) + " |"
    rule = "| " + " | ".join("---" for _ in show.columns) + " |"
    body = ["| " + " | ".join(str(value) for value in row) + " |" for row in show.itertuples(index=False)]
    return "\n".join([header, rule, *body])


def main() -> None:
    products, users, interactions = load_dataset()
    train, test = time_split(interactions)
    models = _models()
    fitted = []
    for model in models:
        model.fit(train, products)
        fitted.append((_rename(model), model))

    content = next(model for name, model in fitted if name == "content")
    similarity = content.pairwise_similarity()
    item_index = content.item_index
    prices = {int(row.item_id): float(row.price) for row in products.itertuples(index=False)}
    categories = {int(row.item_id): str(row.category) for row in products.itertuples(index=False)}
    seasons = {int(row.item_id): str(row.season) for row in products.itertuples(index=False)}
    favorites = {
        int(row.user_id): set(str(row.fav_categories).split("|"))
        for row in users.itertuples(index=False)
    }
    train_size = {int(user): int(size) for user, size in train.groupby("user_id").size().items()}
    relevant = _purchases(test)
    request_season = {
        int(user): season_of(timestamp)
        for user, timestamp in test.groupby("user_id")["timestamp"].min().items()
    }
    purchase_counts = train[train["event_type"].eq("purchase")].groupby("item_id").size()
    n_top = max(1, int(round(0.1 * len(products))))
    top_popular = {int(item) for item in purchase_counts.nlargest(n_top).index}
    n_users = int(train["user_id"].nunique())
    popularity_prob = {int(item): (float(purchase_counts.get(item, 0.0)) + 1.0) / (n_users + 1.0) for item in item_index}

    warm = [user for user, items in relevant.items() if train_size.get(user, 0) >= 8 and items]
    cold = [user for user, items in relevant.items() if train_size.get(user, 0) <= 2 and items]
    seasonal_relevant = {
        user: {item for item in items if seasons.get(item, "all") != "all"}
        for user, items in relevant.items()
        if train_size.get(user, 0) >= 8
    }
    seasonal_users = [user for user, items in seasonal_relevant.items() if items]

    slices = {
        "warm": (warm, relevant),
        "cold": (cold, relevant),
        "seasonal_items": (seasonal_users, seasonal_relevant),
    }
    tables: dict[str, dict] = {}
    frames: list[pd.DataFrame] = []
    for slice_name, (user_ids, rel) in slices.items():
        rows = []
        for name, model in fitted:
            metrics = evaluate_slice(
                model,
                user_ids,
                rel,
                request_season,
                prices,
                categories,
                favorites,
                top_popular,
                popularity_prob,
                similarity,
                item_index,
            )
            tables.setdefault(slice_name, {})[name] = metrics
            rows.append({"model": name, **metrics})
        frame = pd.DataFrame(rows)
        frames.append(frame.assign(slice=slice_name))
        print(f"\n## {slice_name} (users={len(user_ids)})\n")
        print(_table(frame))

    rmse = rating_error_report(train, test, products)
    print("\n## rating error\n")
    print(json.dumps(rmse, indent=2))

    hybrid = next(model for name, model in fitted if name == "hybrid")
    print("\n## hybrid weights\n")
    print(json.dumps({"warm": hybrid.warm_weights, "cold": COLD_WEIGHTS}, indent=2))
    monitor = RecommendationMonitor(
        [int(item) for item in products["item_id"]],
        sensitive_ids=set(products.loc[products["sensitive"].astype(int) == 1, "item_id"].astype(int)),
    )
    import time

    for user_id in warm:
        started = time.perf_counter()
        recs = hybrid.recommend(user_id, k=K, context={"season": request_season.get(user_id, "all")})
        monitor.observe(recs, (time.perf_counter() - started) * 1000)
    monitoring = monitor.report()
    print("\n## monitoring (hybrid, warm users)\n")
    print(json.dumps(monitoring, indent=2))

    print("\n## vulnerabilities\n")
    attack = shilling_attack(train, products)
    stuffing = keyword_stuffing(train, products)
    private = sensitive_co_purchase(train, products)
    print(json.dumps({"shilling": attack, "keyword_stuffing": stuffing, "sensitive": private}, ensure_ascii=False, indent=2))

    purchases = interactions[interactions["event_type"].eq("purchase")]
    nnz = purchases.groupby(["user_id", "item_id"]).ngroups
    dataset = {
        "products": int(len(products)),
        "users": int(len(users)),
        "cold_users": int((users["segment"] == "cold").sum()),
        "interactions": int(len(interactions)),
        "purchases": int(len(purchases)),
        "purchase_density": nnz / (len(users) * len(products)),
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "warm_eval_users": len(warm),
        "cold_eval_users": len(cold),
        "seasonal_eval_users": len(seasonal_users),
    }
    ARTIFACTS.mkdir(exist_ok=True)
    pd.concat(frames, ignore_index=True).to_csv(ARTIFACTS / "metrics.csv", index=False)
    summary = {
        "dataset": dataset,
        "slices": tables,
        "rmse": rmse,
        "hybrid_weights": {"warm": hybrid.warm_weights, "cold": COLD_WEIGHTS},
        "monitoring": monitoring,
        "shilling": attack,
        "keyword_stuffing": stuffing,
        "sensitive": private,
    }
    (ARTIFACTS / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nWrote {ARTIFACTS / 'summary.json'}")


if __name__ == "__main__":
    main()
