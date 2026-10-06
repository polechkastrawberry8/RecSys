"""Small deterministic checks. They do not need the generated store dataset."""

from __future__ import annotations

import pandas as pd

from recsys.collaborative import ItemBasedCF, UserBasedCF
from recsys.content import ContentBasedRecommender
from recsys.data import time_split
from recsys.heuristics import AlsoBoughtRecommender, PopularityRecommender, RecentlyViewedRecommender
from recsys.hybrid import HybridRecommender


def _catalog() -> tuple[pd.DataFrame, pd.DataFrame]:
    products = pd.DataFrame(
        [
            {
                "item_id": 1,
                "title": "Phone Nova",
                "category": "electronics",
                "brand": "Nova",
                "price": 100,
                "season": "all",
                "tags": "smartphone phone",
                "description": "smartphone phone nova electronics",
                "bundle": "phone",
                "sensitive": 0,
            },
            {
                "item_id": 2,
                "title": "Case Nova",
                "category": "electronics",
                "brand": "Nova",
                "price": 10,
                "season": "all",
                "tags": "case phone",
                "description": "case phone nova electronics",
                "bundle": "phone",
                "sensitive": 0,
            },
            {
                "item_id": 3,
                "title": "Coat",
                "category": "fashion",
                "brand": "Forma",
                "price": 80,
                "season": "winter",
                "tags": "coat winter",
                "description": "coat winter fashion",
                "bundle": "",
                "sensitive": 0,
            },
            {
                "item_id": 4,
                "title": "Private test",
                "category": "health",
                "brand": "Flora",
                "price": 5,
                "season": "all",
                "tags": "private test",
                "description": "private health test",
                "bundle": "private",
                "sensitive": 1,
            },
            {
                "item_id": 5,
                "title": "Vitamin",
                "category": "health",
                "brand": "Purelab",
                "price": 6,
                "season": "all",
                "tags": "vitamin health",
                "description": "vitamin health",
                "bundle": "private",
                "sensitive": 0,
            },
        ]
    )
    interactions = pd.DataFrame(
        [
            ("u", 1, 1, "purchase", 5, "2025-01-01"),
            ("u", 1, 2, "purchase", 5, "2025-01-02"),
            ("u", 2, 1, "purchase", 4, "2025-01-03"),
            ("u", 2, 2, "purchase", 4, "2025-01-04"),
            ("u", 3, 1, "purchase", 5, "2025-02-01"),
            ("u", 3, 4, "purchase", 5, "2025-02-02"),
            ("u", 4, 3, "purchase", 5, "2025-03-01"),
            ("u", 4, 3, "view", None, "2025-03-02"),
            ("u", 5, 5, "purchase", 4, "2025-04-01"),
            ("u", 5, 4, "purchase", 5, "2025-04-02"),
            ("u", 6, 1, "view", None, "2025-05-01"),
        ],
        columns=["kind", "user_id", "item_id", "event_type", "rating", "timestamp"],
    ).drop(columns=["kind"])
    interactions["timestamp"] = pd.to_datetime(interactions["timestamp"])
    return products, interactions


def test_time_split_does_not_leak_the_future() -> None:
    _, interactions = _catalog()
    train, test = time_split(interactions)
    for user_id, group in test.groupby("user_id"):
        past = train.loc[train["user_id"] == user_id, "timestamp"]
        if past.empty:
            continue
        assert past.max() <= group["timestamp"].min()


def test_popularity_ranks_the_most_purchased_item_first() -> None:
    products, interactions = _catalog()
    model = PopularityRecommender().fit(interactions, products)
    recs = model.recommend(user_id=999, k=3)
    assert int(recs.iloc[0]["item_id"]) == 1


def test_recommendations_skip_items_the_user_already_touched() -> None:
    products, interactions = _catalog()
    model = PopularityRecommender().fit(interactions, products)
    recs = model.recommend(user_id=1, k=5)
    assert 1 not in set(recs["item_id"].astype(int))
    assert 2 not in set(recs["item_id"].astype(int))


def test_also_bought_can_hide_a_sensitive_item() -> None:
    products, interactions = _catalog()
    open_shelf = AlsoBoughtRecommender(block_sensitive=False).fit(interactions, products)
    closed_shelf = AlsoBoughtRecommender(block_sensitive=True).fit(interactions, products)
    context = {"item_id": 5}
    open_ids = set(open_shelf.recommend(k=5, context=context)["item_id"].astype(int))
    closed_ids = set(closed_shelf.recommend(k=5, context=context)["item_id"].astype(int))
    assert 4 in open_ids
    assert 4 not in closed_ids


def test_cold_user_still_gets_a_hybrid_list() -> None:
    products, interactions = _catalog()
    hybrid = HybridRecommender().fit(interactions, products)
    user_cf = UserBasedCF().fit(interactions, products)
    # User 6 has a single view, which collaborative filtering does not use.
    assert user_cf.recommend(6, k=3).empty
    recs = hybrid.recommend(6, k=3)
    assert len(recs) == 3
    assert recs["item_id"].is_unique


def test_content_and_item_cf_fit() -> None:
    products, interactions = _catalog()
    content = ContentBasedRecommender().fit(interactions, products)
    similar = content.similar_items(1, k=2)
    assert 1 not in set(similar["item_id"].astype(int))
    assert len(similar) == 2
    item_cf = ItemBasedCF().fit(interactions, products)
    recs = item_cf.recommend(1, k=3)
    assert recs["item_id"].is_unique
    assert 1 not in set(recs["item_id"].astype(int))


def test_recently_viewed_returns_history() -> None:
    products, interactions = _catalog()
    model = RecentlyViewedRecommender().fit(interactions, products)
    recs = model.recommend(4, k=5)
    assert list(recs["item_id"].astype(int)) == [3]


if __name__ == "__main__":
    test_time_split_does_not_leak_the_future()
    test_popularity_ranks_the_most_purchased_item_first()
    test_recommendations_skip_items_the_user_already_touched()
    test_also_bought_can_hide_a_sensitive_item()
    test_cold_user_still_gets_a_hybrid_list()
    test_content_and_item_cf_fit()
    test_recently_viewed_returns_history()
    print("ok")
