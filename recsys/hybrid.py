"""Weighted hybrid of item-item CF, content, and popularity.

User-user CF is implemented and compared on its own. It is not inside the hybrid:
every new purchase changes the neighbor graph, which is a poor fit for a page
that has to answer quickly. Item-item similarities can be computed once.

Warm-user weights are chosen on a later slice of the training log, then the
components are refit on the whole training log. The test window is not used.
Cold users, with fewer than three training events, keep a fixed mix that does
not trust collaborative filtering.
"""

from __future__ import annotations

import pandas as pd

from recsys.collaborative import ItemBasedCF
from recsys.content import ContentBasedRecommender
from recsys.data import as_recs, empty_recs, time_split
from recsys.heuristics import PopularityRecommender
from recsys.metrics import ndcg_at_k

WARM_WEIGHTS = {"item_cf": 0.55, "content": 0.30, "popularity": 0.15}
COLD_WEIGHTS = {"item_cf": 0.10, "content": 0.45, "popularity": 0.45}
COLD_EVENTS = 3
WEIGHT_GRID = (
    {"item_cf": 0.25, "content": 0.45, "popularity": 0.30},
    {"item_cf": 0.25, "content": 0.30, "popularity": 0.45},
    {"item_cf": 0.40, "content": 0.45, "popularity": 0.15},
    {"item_cf": 0.40, "content": 0.30, "popularity": 0.30},
    {"item_cf": 0.40, "content": 0.15, "popularity": 0.45},
    {"item_cf": 0.55, "content": 0.30, "popularity": 0.15},
    {"item_cf": 0.55, "content": 0.15, "popularity": 0.30},
    {"item_cf": 0.70, "content": 0.15, "popularity": 0.15},
)


def _minmax(scores: dict[int, float]) -> dict[int, float]:
    if not scores:
        return {}
    values = list(scores.values())
    low, high = min(values), max(values)
    if high - low < 1e-9:
        return {item: 1.0 for item in scores}
    return {item: (value - low) / (high - low) for item, value in scores.items()}


def _shelf_scores(frame: pd.DataFrame) -> dict[int, float]:
    if frame.empty:
        return {}
    return {int(row.item_id): float(row.score) for row in frame.itertuples(index=False)}


class HybridRecommender:
    name = "hybrid"

    def __init__(self, block_sensitive: bool = True, tune: bool = True) -> None:
        self.block_sensitive = block_sensitive
        self.tune = tune
        self.item_cf = ItemBasedCF()
        self.content = ContentBasedRecommender()
        self.popularity = PopularityRecommender()
        self.activity: dict[int, int] = {}
        self.sensitive: set[int] = set()
        self.warm_weights = dict(WARM_WEIGHTS)

    def fit(self, interactions: pd.DataFrame, products: pd.DataFrame) -> HybridRecommender:
        self.sensitive = set(products.loc[products["sensitive"].astype(int) == 1, "item_id"].astype(int))
        if self.tune:
            self.warm_weights = self._tune(interactions, products)
        else:
            self.warm_weights = dict(WARM_WEIGHTS)
        self._fit_children(interactions, products)
        self.activity = {int(user): int(count) for user, count in interactions.groupby("user_id").size().items()}
        return self

    def _fit_children(self, interactions: pd.DataFrame, products: pd.DataFrame) -> None:
        self.item_cf.fit(interactions, products)
        self.content.fit(interactions, products)
        self.popularity.fit(interactions, products)

    def _parts(self, user_id: int) -> dict[str, dict[int, float]]:
        return {
            "item_cf": _minmax(_shelf_scores(self.item_cf.recommend(user_id, k=80))),
            "content": _minmax(_shelf_scores(self.content.recommend(user_id, k=80))),
            "popularity": _minmax(_shelf_scores(self.popularity.recommend(user_id, k=40))),
        }

    def _blend(self, parts: dict[str, dict[int, float]], weights: dict[str, float], k: int) -> pd.DataFrame:
        active = {name: weight for name, weight in weights.items() if parts.get(name)}
        if not active:
            return empty_recs()
        total = sum(active.values())
        active = {name: weight / total for name, weight in active.items()}
        candidates: set[int] = set()
        for name in active:
            candidates.update(parts[name])
        if self.block_sensitive:
            candidates -= self.sensitive
        ranked: list[tuple[int, float, str]] = []
        for item in candidates:
            contribution = {name: active[name] * parts[name].get(item, 0.0) for name in active}
            score = float(sum(contribution.values()))
            dominant = max(contribution, key=contribution.get)
            ranked.append((item, score, f"hybrid:{dominant}"))
        ranked.sort(key=lambda row: row[1], reverse=True)
        chosen = ranked[:k]
        if not chosen:
            return empty_recs()
        frame = as_recs([item for item, _, _ in chosen], [score for _, score, _ in chosen], "hybrid")
        frame["reason"] = [reason for _, _, reason in chosen]
        return frame

    def _tune(self, interactions: pd.DataFrame, products: pd.DataFrame) -> dict[str, float]:
        inner_train, validation = time_split(interactions)
        if validation.empty:
            return dict(WARM_WEIGHTS)
        self._fit_children(inner_train, products)
        inner_size = {int(user): int(count) for user, count in inner_train.groupby("user_id").size().items()}
        relevant = {
            int(user): {int(item) for item in items}
            for user, items in validation[validation["event_type"].eq("purchase")].groupby("user_id")["item_id"]
        }
        val_users = [user for user, items in relevant.items() if items and inner_size.get(user, 0) >= 8]
        val_users = sorted(val_users)[:200]
        if not val_users:
            return dict(WARM_WEIGHTS)
        cached = {user: self._parts(user) for user in val_users}
        best_weights = dict(WARM_WEIGHTS)
        best_score = -1.0
        for weights in WEIGHT_GRID:
            scores = []
            for user in val_users:
                recommended = [int(item) for item in self._blend(cached[user], weights, k=10)["item_id"]]
                scores.append(ndcg_at_k(recommended, relevant[user], 10))
            quality = float(sum(scores) / len(scores))
            if quality > best_score:
                best_score = quality
                best_weights = dict(weights)
        return best_weights

    def recommend(self, user_id: int | None = None, k: int = 10, context: dict | None = None) -> pd.DataFrame:
        del context
        if user_id is None:
            return self.popularity.recommend(k=k)
        user_id = int(user_id)
        weights = COLD_WEIGHTS if self.activity.get(user_id, 0) < COLD_EVENTS else self.warm_weights
        return self._blend(self._parts(user_id), weights, k)
