"""Content-based recommender.

Each product is a TF-IDF vector of its title, category, brand, season, and tags.
A user profile is the strength-weighted average of products they already touched.
The score of a candidate is the cosine between that profile and the product.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

from recsys.data import add_strength, empty_recs, seen_items, top_k


class ContentBasedRecommender:
    name = "content"

    def __init__(self) -> None:
        self.vectorizer = TfidfVectorizer(lowercase=True, min_df=1, ngram_range=(1, 2))
        self.item_ids: np.ndarray | None = None
        self.item_index: dict[int, int] = {}
        self.item_vectors: np.ndarray | None = None
        self.user_items: dict[int, list[tuple[int, float]]] = {}
        self._seen: dict[int, set[int]] = {}
        self.products: pd.DataFrame | None = None

    def fit(self, interactions: pd.DataFrame, products: pd.DataFrame) -> ContentBasedRecommender:
        self.products = products.reset_index(drop=True)
        documents = (
            products["title"].fillna("")
            + " "
            + products["category"].fillna("")
            + " "
            + products["brand"].fillna("")
            + " "
            + products["season"].fillna("")
            + " "
            + products["tags"].fillna("")
            + " "
            + products["description"].fillna("")
        )
        raw = self.vectorizer.fit_transform(documents)
        self.item_vectors = normalize(raw).toarray()
        self.item_ids = products["item_id"].astype(int).to_numpy()
        self.item_index = {int(item): pos for pos, item in enumerate(self.item_ids)}

        weighted = add_strength(interactions)
        grouped = weighted.groupby(["user_id", "item_id"], as_index=False)["strength"].max()
        self.user_items = {}
        for row in grouped.itertuples(index=False):
            self.user_items.setdefault(int(row.user_id), []).append((int(row.item_id), float(row.strength)))
        self._seen = seen_items(interactions)
        return self

    def profile(self, user_id: int) -> np.ndarray | None:
        history = self.user_items.get(int(user_id))
        if not history or self.item_vectors is None:
            return None
        acc = np.zeros(self.item_vectors.shape[1], dtype=float)
        for item_id, strength in history:
            pos = self.item_index.get(item_id)
            if pos is None:
                continue
            acc += self.item_vectors[pos] * strength
        norm = np.linalg.norm(acc)
        if norm <= 0:
            return None
        return acc / norm

    def score_items(self, user_id: int) -> np.ndarray | None:
        profile = self.profile(user_id)
        if profile is None or self.item_vectors is None:
            return None
        return self.item_vectors @ profile

    def score_user(self, user_id: int) -> np.ndarray | None:
        scores = self.score_items(user_id)
        if scores is None:
            return None
        masked = scores.copy()
        for item_id in self._seen.get(int(user_id), set()):
            pos = self.item_index.get(item_id)
            if pos is not None:
                masked[pos] = -np.inf
        return masked

    def recommend(self, user_id: int | None = None, k: int = 10, context: dict | None = None) -> pd.DataFrame:
        if self.item_ids is None:
            return empty_recs()
        seed = (context or {}).get("item_id")
        if seed is not None:
            return self.similar_items(int(seed), k=k)
        if user_id is None:
            return empty_recs()
        scores = self.score_items(int(user_id))
        if scores is None:
            return empty_recs()
        return top_k(self.item_ids, scores, k, self._seen.get(int(user_id), set()), "content")

    def similar_items(self, item_id: int, k: int = 10) -> pd.DataFrame:
        if self.item_vectors is None or self.item_ids is None:
            return empty_recs()
        pos = self.item_index.get(int(item_id))
        if pos is None:
            return empty_recs()
        scores = self.item_vectors @ self.item_vectors[pos]
        return top_k(self.item_ids, scores, k, {int(item_id)}, "content_similar")

    def pairwise_similarity(self) -> np.ndarray:
        if self.item_vectors is None:
            return np.zeros((0, 0))
        return self.item_vectors @ self.item_vectors.T
