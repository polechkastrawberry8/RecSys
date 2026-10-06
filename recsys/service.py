"""Storefront shelves.

Home and the product page are different tasks. The home page ranks what this
user may buy next. The product page explains the item they are already looking
at: bundles, brand, category, similar content, and the browsing history.
"""

from __future__ import annotations

import pandas as pd

from recsys.collaborative import UserBasedCF
from recsys.content import ContentBasedRecommender
from recsys.heuristics import (
    AlsoBoughtRecommender,
    PopularityRecommender,
    RecentlyViewedRecommender,
    SameBrandRecommender,
    SameCategoryRecommender,
    SeasonalRecommender,
)
from recsys.hybrid import HybridRecommender


class Storefront:
    def __init__(self) -> None:
        self.hybrid = HybridRecommender()
        self.user_cf = UserBasedCF()
        self.content = ContentBasedRecommender()
        self.popularity = PopularityRecommender()
        self.seasonal = SeasonalRecommender()
        self.also_bought = AlsoBoughtRecommender(block_sensitive=True)
        self.also_bought_unfiltered = AlsoBoughtRecommender(block_sensitive=False)
        self.same_brand = SameBrandRecommender()
        self.same_category = SameCategoryRecommender()
        self.recently_viewed = RecentlyViewedRecommender()
        self.products: pd.DataFrame | None = None

    def fit(self, interactions: pd.DataFrame, products: pd.DataFrame) -> Storefront:
        self.products = products
        self.hybrid.fit(interactions, products)
        self.user_cf.fit(interactions, products)
        self.content.fit(interactions, products)
        self.popularity.fit(interactions, products)
        self.seasonal.fit(interactions, products)
        self.also_bought.fit(interactions, products)
        self.also_bought_unfiltered.fit(interactions, products)
        self.same_brand.fit(interactions, products)
        self.same_category.fit(interactions, products)
        self.recently_viewed.fit(interactions, products)
        return self

    def home(self, user_id: int, k: int = 5, season: str | None = None) -> dict[str, pd.DataFrame]:
        context = {"season": season} if season else None
        return {
            "for_you": self.hybrid.recommend(user_id, k=k),
            "user_cf": self.user_cf.recommend(user_id, k=k),
            "content": self.content.recommend(user_id, k=k),
            "popular": self.popularity.recommend(user_id, k=k),
            "seasonal": self.seasonal.recommend(user_id, k=k, context=context),
            "also_bought": self.also_bought.recommend(user_id, k=k),
            "recently_viewed": self.recently_viewed.recommend(user_id, k=k),
        }

    def product_page(self, item_id: int, k: int = 5, user_id: int | None = None) -> dict[str, pd.DataFrame]:
        context = {"item_id": int(item_id)}
        return {
            "also_bought": self.also_bought.recommend(k=k, context=context),
            "same_brand": self.same_brand.recommend(k=k, context=context),
            "you_may_also_like": self.same_category.recommend(k=k, context=context),
            "similar_content": self.content.similar_items(int(item_id), k=k),
            "recently_viewed": self.recently_viewed.recommend(user_id, k=k) if user_id else self.recently_viewed.recommend(k=0),
        }
