"""Shelves that do not need a trained model.

These are the baselines a store can ship before any ML model is ready, and the
fallbacks the ML shelves use when a user or a product has no history.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd

from recsys.data import as_recs, empty_recs, season_of, seen_items, top_k


class PopularityRecommender:
    """Most purchased products. The default baseline and the cold-start shelf."""

    name = "popularity"

    def __init__(self, half_life_days: float | None = None) -> None:
        self.half_life_days = half_life_days
        self.item_ids = np.array([], dtype=int)
        self.scores = np.array([])
        self._seen: dict[int, set[int]] = {}

    def fit(self, interactions: pd.DataFrame, products: pd.DataFrame) -> PopularityRecommender:
        self.item_ids = products["item_id"].astype(int).to_numpy()
        purchases = interactions[interactions["event_type"].eq("purchase")].copy()
        scores = {int(item): 0.0 for item in self.item_ids}
        if not purchases.empty:
            if self.half_life_days:
                horizon = purchases["timestamp"].max()
                age_days = (horizon - purchases["timestamp"]).dt.total_seconds() / 86400
                weight = np.exp(-np.log(2) * age_days / self.half_life_days)
                purchases = purchases.assign(weight=weight)
                grouped = purchases.groupby("item_id")["weight"].sum()
            else:
                grouped = purchases.groupby("item_id").size()
            for item_id, score in grouped.items():
                scores[int(item_id)] = float(score)
        self.scores = np.array([scores[int(item)] for item in self.item_ids], dtype=float)
        self._seen = seen_items(interactions)
        return self

    def recommend(self, user_id: int | None = None, k: int = 10, context: dict | None = None) -> pd.DataFrame:
        del context
        exclude = self._seen.get(int(user_id), set()) if user_id is not None else set()
        return top_k(self.item_ids, self.scores, k, exclude, self.name)


class SeasonalRecommender:
    """Popularity with a boost for the season of the user's last event.

    The season is taken from training history only, so the test window does not
    leak into the shelf.
    """

    name = "seasonal"

    def __init__(self) -> None:
        self.item_ids = np.array([], dtype=int)
        self.popularity = np.array([])
        self.seasons: list[str] = []
        self.user_season: dict[int, str] = {}
        self._seen: dict[int, set[int]] = {}

    def fit(self, interactions: pd.DataFrame, products: pd.DataFrame) -> SeasonalRecommender:
        self.item_ids = products["item_id"].astype(int).to_numpy()
        self.seasons = products["season"].fillna("all").astype(str).tolist()
        purchases = interactions[interactions["event_type"].eq("purchase")]
        counts = purchases.groupby("item_id").size() if not purchases.empty else pd.Series(dtype=float)
        self.popularity = np.array([float(counts.get(int(item), 0.0)) for item in self.item_ids])
        last = interactions.sort_values("timestamp").groupby("user_id")["timestamp"].max()
        self.user_season = {int(user): season_of(ts) for user, ts in last.items()}
        self._seen = seen_items(interactions)
        return self

    def _scores_for(self, season: str) -> np.ndarray:
        scores = self.popularity.copy()
        for pos, item_season in enumerate(self.seasons):
            if item_season == season:
                scores[pos] *= 3.0
            elif item_season != "all":
                scores[pos] *= 0.35
        return scores

    def recommend(self, user_id: int | None = None, k: int = 10, context: dict | None = None) -> pd.DataFrame:
        season = (context or {}).get("season")
        if season is None and user_id is not None:
            season = self.user_season.get(int(user_id), "all")
        season = season or "all"
        exclude = self._seen.get(int(user_id), set()) if user_id is not None else set()
        return top_k(self.item_ids, self._scores_for(str(season)), k, exclude, "seasonal")


class AlsoBoughtRecommender:
    """People who bought this also bought that.

    The raw pair count is divided by sqrt(popularity of the other item). Without
    that discount the shelf collapses into the global top sellers. The discount
    is a heuristic, not a guarantee: a rare accidental pair can still float up.
    """

    name = "also_bought"

    def __init__(self, block_sensitive: bool = True) -> None:
        self.block_sensitive = block_sensitive
        self.pairs: dict[int, dict[int, float]] = {}
        self.item_count: dict[int, int] = {}
        self.user_purchases: dict[int, set[int]] = {}
        self.sensitive: set[int] = set()
        self._seen: dict[int, set[int]] = {}

    def fit(self, interactions: pd.DataFrame, products: pd.DataFrame) -> AlsoBoughtRecommender:
        self.sensitive = set(products.loc[products["sensitive"].astype(int) == 1, "item_id"].astype(int))
        purchases = interactions[interactions["event_type"].eq("purchase")]
        baskets: dict[int, set[int]] = {}
        counts: dict[int, int] = defaultdict(int)
        for row in purchases.itertuples(index=False):
            baskets.setdefault(int(row.user_id), set()).add(int(row.item_id))
            counts[int(row.item_id)] += 1
        self.item_count = dict(counts)
        self.user_purchases = baskets
        pairs: dict[int, dict[int, float]] = defaultdict(lambda: defaultdict(float))
        for items in baskets.values():
            unique = list(items)
            for left in unique:
                for right in unique:
                    if left != right:
                        pairs[left][right] += 1.0
        self.pairs = {item: dict(others) for item, others in pairs.items()}
        self._seen = seen_items(interactions)
        return self

    def _rank(self, seed_items: list[int], exclude: set[int], k: int) -> pd.DataFrame:
        totals: dict[int, float] = defaultdict(float)
        for seed in seed_items:
            for other, count in self.pairs.get(seed, {}).items():
                if other in exclude or other in seed_items:
                    continue
                if self.block_sensitive and other in self.sensitive:
                    continue
                popularity = max(self.item_count.get(other, 1), 1)
                totals[other] += count / np.sqrt(popularity)
        if not totals:
            return empty_recs()
        ranked = sorted(totals.items(), key=lambda pair: pair[1], reverse=True)[:k]
        return as_recs([item for item, _ in ranked], [score for _, score in ranked], "also_bought")

    def recommend(self, user_id: int | None = None, k: int = 10, context: dict | None = None) -> pd.DataFrame:
        context = context or {}
        seed = context.get("item_id")
        if seed is not None:
            return self._rank([int(seed)], set(), k)
        if user_id is None:
            return empty_recs()
        seeds = list(self.user_purchases.get(int(user_id), set()))
        exclude = self._seen.get(int(user_id), set())
        return self._rank(seeds, exclude, k)


class SameBrandRecommender:
    name = "same_brand"

    def fit(self, interactions: pd.DataFrame, products: pd.DataFrame) -> SameBrandRecommender:
        self.products = products
        self.by_brand: dict[str, list[int]] = {}
        for row in products.itertuples(index=False):
            self.by_brand.setdefault(str(row.brand), []).append(int(row.item_id))
        self.item_brand = {int(row.item_id): str(row.brand) for row in products.itertuples(index=False)}
        purchases = interactions[interactions["event_type"].eq("purchase")]
        self.user_brand = {}
        if not purchases.empty:
            merged = purchases.merge(products[["item_id", "brand"]], on="item_id", how="left")
            for user_id, group in merged.groupby("user_id"):
                self.user_brand[int(user_id)] = str(group["brand"].value_counts().index[0])
        self.popularity = PopularityRecommender().fit(interactions, products)
        self._seen = seen_items(interactions)
        return self

    def recommend(self, user_id: int | None = None, k: int = 10, context: dict | None = None) -> pd.DataFrame:
        context = context or {}
        seed = context.get("item_id")
        if seed is not None:
            brand = self.item_brand.get(int(seed))
            exclude = {int(seed)}
        elif user_id is not None:
            brand = self.user_brand.get(int(user_id))
            exclude = self._seen.get(int(user_id), set())
        else:
            return empty_recs()
        if not brand:
            return empty_recs()
        allowed = set(self.by_brand.get(brand, []))
        scores = self.popularity.scores.copy()
        for pos, item_id in enumerate(self.popularity.item_ids):
            if int(item_id) not in allowed:
                scores[pos] = -np.inf
        return top_k(self.popularity.item_ids, scores, k, exclude, "same_brand")


class SameCategoryRecommender:
    """You may also be interested: other products of the same category, by popularity."""

    name = "same_category"

    def fit(self, interactions: pd.DataFrame, products: pd.DataFrame) -> SameCategoryRecommender:
        self.by_category: dict[str, list[int]] = {}
        for row in products.itertuples(index=False):
            self.by_category.setdefault(str(row.category), []).append(int(row.item_id))
        self.item_category = {int(row.item_id): str(row.category) for row in products.itertuples(index=False)}
        purchases = interactions[interactions["event_type"].eq("purchase")]
        self.user_category: dict[int, str] = {}
        if not purchases.empty:
            merged = purchases.merge(products[["item_id", "category"]], on="item_id", how="left")
            for user_id, group in merged.groupby("user_id"):
                self.user_category[int(user_id)] = str(group["category"].value_counts().index[0])
        self.popularity = PopularityRecommender().fit(interactions, products)
        self._seen = seen_items(interactions)
        return self

    def recommend(self, user_id: int | None = None, k: int = 10, context: dict | None = None) -> pd.DataFrame:
        context = context or {}
        seed = context.get("item_id")
        if seed is not None:
            category = self.item_category.get(int(seed))
            exclude = {int(seed)}
        elif user_id is not None:
            category = self.user_category.get(int(user_id))
            exclude = self._seen.get(int(user_id), set())
        else:
            return empty_recs()
        if not category:
            return self.popularity.recommend(user_id=user_id, k=k).assign(reason="same_category_fallback_popular")
        allowed = set(self.by_category.get(category, []))
        scores = self.popularity.scores.copy()
        for pos, item_id in enumerate(self.popularity.item_ids):
            if int(item_id) not in allowed:
                scores[pos] = -np.inf
        return top_k(self.popularity.item_ids, scores, k, exclude, "same_category")


class RecentlyViewedRecommender:
    """A history shelf, not a predictor of the next purchase."""

    name = "recently_viewed"

    def fit(self, interactions: pd.DataFrame, products: pd.DataFrame) -> RecentlyViewedRecommender:
        del products
        views = interactions[interactions["event_type"].eq("view")].sort_values("timestamp")
        self.history: dict[int, list[int]] = {}
        for user_id, group in views.groupby("user_id"):
            ordered = [int(item) for item in group["item_id"]]
            unique_latest: list[int] = []
            for item in reversed(ordered):
                if item not in unique_latest:
                    unique_latest.append(item)
            self.history[int(user_id)] = unique_latest
        return self

    def recommend(self, user_id: int | None = None, k: int = 10, context: dict | None = None) -> pd.DataFrame:
        del context
        if user_id is None:
            return empty_recs()
        items = self.history.get(int(user_id), [])[:k]
        scores = [float(len(items) - pos) for pos in range(len(items))]
        return as_recs(items, scores, "recently_viewed")


class RandomRecommender:
    """Sanity-check floor. Every real shelf should beat this."""

    name = "random"

    def __init__(self, seed: int = 42) -> None:
        self.seed = seed
        self.item_ids = np.array([], dtype=int)
        self._seen: dict[int, set[int]] = {}

    def fit(self, interactions: pd.DataFrame, products: pd.DataFrame) -> RandomRecommender:
        self.item_ids = products["item_id"].astype(int).to_numpy()
        self._seen = seen_items(interactions)
        return self

    def recommend(self, user_id: int | None = None, k: int = 10, context: dict | None = None) -> pd.DataFrame:
        del context
        exclude = self._seen.get(int(user_id), set()) if user_id is not None else set()
        pool = [int(item) for item in self.item_ids if int(item) not in exclude]
        if not pool:
            return empty_recs()
        rng = np.random.default_rng(self.seed + (0 if user_id is None else int(user_id)))
        take = min(k, len(pool))
        chosen = rng.choice(pool, size=take, replace=False)
        return as_recs([int(item) for item in chosen], [1.0] * take, "random")
