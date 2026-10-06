"""Controlled checks of how the shelves fail when the input is manipulated.

These are simulations on our own generated data, not attacks on a live service.
They answer the assignment question about vulnerabilities: which shelves move
when someone injects fake purchases or rewrites a product card, and which do not.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from recsys.collaborative import ItemBasedCF, UserBasedCF
from recsys.content import ContentBasedRecommender
from recsys.heuristics import AlsoBoughtRecommender


def _top_hit_rate(model, users: list[int], target: int, k: int = 10) -> float:
    hits = 0
    scored = 0
    for user_id in users:
        scores = model.score_user(int(user_id))
        if scores is None:
            continue
        scored += 1
        finite = np.isfinite(scores)
        if not finite.any():
            continue
        order = np.argsort(np.where(finite, -scores, np.inf))
        top: list[int] = []
        for pos in order:
            if not finite[pos]:
                break
            top.append(int(model.item_ids[pos]))
            if len(top) >= k:
                break
        if target in top:
            hits += 1
    if scored == 0:
        return 0.0
    return hits / scored


def shilling_attack(train: pd.DataFrame, products: pd.DataFrame, n_fakes: int = 40) -> dict[str, float | int | str]:
    """Fake users who praise a niche item together with the current bestsellers.

    User-based CF treats them as good neighbors of real buyers. Item-based CF
    starts to see the niche item as similar to those bestsellers. Content-based
    recommendations ignore purchases, so the text of the card is unchanged and
    the hit rate should stay put.
    """
    purchases = train[train["event_type"].eq("purchase")]
    counts = purchases.groupby("item_id").size().sort_values(ascending=False)
    popular = [int(item) for item in counts.head(8).index]
    band = counts[(counts >= 3) & (counts <= 15)]
    if band.empty:
        band = counts.tail(10)
    target = int(band.index[-1])
    target_title = str(products.loc[products["item_id"].eq(target), "title"].iloc[0])

    buyers = set(purchases.loc[purchases["item_id"].eq(target), "user_id"].astype(int))
    warm = [int(user) for user, size in train.groupby("user_id").size().items() if size >= 8 and int(user) not in buyers]
    sample = warm[:120]

    user_cf = UserBasedCF().fit(train, products)
    item_cf = ItemBasedCF().fit(train, products)
    content = ContentBasedRecommender().fit(train, products)
    before = {
        "user_cf": _top_hit_rate(user_cf, sample, target),
        "item_cf": _top_hit_rate(item_cf, sample, target),
        "content": _top_hit_rate(content, sample, target),
    }

    rows = []
    stamp = train["timestamp"].max()
    for offset in range(n_fakes):
        user_id = 1_000_000 + offset
        for item_id in popular + [target]:
            rows.append(
                {
                    "user_id": user_id,
                    "item_id": item_id,
                    "event_type": "purchase",
                    "rating": 5.0,
                    "timestamp": stamp,
                }
            )
    poisoned = pd.concat([train, pd.DataFrame(rows)], ignore_index=True)
    user_cf.fit(poisoned, products)
    item_cf.fit(poisoned, products)
    content.fit(poisoned, products)
    after = {
        "user_cf": _top_hit_rate(user_cf, sample, target),
        "item_cf": _top_hit_rate(item_cf, sample, target),
        "content": _top_hit_rate(content, sample, target),
    }
    return {
        "target_item_id": target,
        "target_title": target_title,
        "train_purchases_of_target": int(counts.get(target, 0)),
        "fake_users": n_fakes,
        "users_scored": len(sample),
        "user_cf_top10_before": before["user_cf"],
        "user_cf_top10_after": after["user_cf"],
        "item_cf_top10_before": before["item_cf"],
        "item_cf_top10_after": after["item_cf"],
        "content_top10_before": before["content"],
        "content_top10_after": after["content"],
    }


def keyword_stuffing(train: pd.DataFrame, products: pd.DataFrame) -> dict[str, float | int | str]:
    """Rewrite a low-demand card so its text copies a popular smartphone.

    Content similarity follows the text. Collaborative scores do not read it.
    """
    purchases = train[train["event_type"].eq("purchase")]
    phones = products[products["tags"].str.contains("smartphone", na=False)]
    phone_counts = purchases[purchases["item_id"].isin(phones["item_id"])].groupby("item_id").size()
    if phone_counts.empty:
        source_id = int(phones.iloc[0]["item_id"])
    else:
        source_id = int(phone_counts.idxmax())
    victim = products[products["title"].str.contains("Пластырь", na=False)].iloc[0]
    victim_id = int(victim["item_id"])
    source = products.loc[products["item_id"].eq(source_id)].iloc[0]

    clean = ContentBasedRecommender().fit(train, products)
    before = clean.similar_items(source_id, k=10)
    before_hit = int(victim_id in set(before["item_id"].astype(int)))

    stuffed = products.copy()
    mask = stuffed["item_id"].eq(victim_id)
    stuffed.loc[mask, "tags"] = source["tags"]
    stuffed.loc[mask, "description"] = source["description"] + " " + str(source["title"])
    stuffed.loc[mask, "category"] = source["category"]
    attacked = ContentBasedRecommender().fit(train, stuffed)
    after = attacked.similar_items(source_id, k=10)
    after_ids = [int(item) for item in after["item_id"]]
    rank = after_ids.index(victim_id) + 1 if victim_id in after_ids else None
    return {
        "source_item_id": source_id,
        "source_title": str(source["title"]),
        "victim_item_id": victim_id,
        "victim_title": str(victim["title"]),
        "in_similar_top10_before": before_hit,
        "in_similar_top10_after": int(victim_id in after_ids),
        "rank_after": -1 if rank is None else rank,
    }


def sensitive_co_purchase(train: pd.DataFrame, products: pd.DataFrame, k: int = 5) -> dict[str, object]:
    """Show a private product on an unfiltered 'also bought' shelf, then hide it."""
    vitamin = products[products["title"].str.contains("Фолиевая", na=False)].iloc[0]
    private = products[products["sensitive"].astype(int).eq(1)].iloc[0]
    vitamin_id = int(vitamin["item_id"])
    private_id = int(private["item_id"])
    open_shelf = AlsoBoughtRecommender(block_sensitive=False).fit(train, products)
    closed_shelf = AlsoBoughtRecommender(block_sensitive=True).fit(train, products)
    context = {"item_id": vitamin_id}
    open_ids = [int(item) for item in open_shelf.recommend(k=k, context=context)["item_id"]]
    closed_ids = [int(item) for item in closed_shelf.recommend(k=k, context=context)["item_id"]]
    titles = {int(row.item_id): str(row.title) for row in products.itertuples(index=False)}
    return {
        "seed_title": str(vitamin["title"]),
        "private_title": str(private["title"]),
        "open_shelf": [titles[item] for item in open_ids],
        "closed_shelf": [titles[item] for item in closed_ids],
        "private_shown_without_filter": private_id in open_ids,
        "private_shown_with_filter": private_id in closed_ids,
    }
