"""Neighborhood collaborative filtering on purchases.

Two similarity formulas are implemented.

Ranking uses cosine of the raw purchase-strength vectors (implicit feedback).
On this log, mean-centered "adjusted cosine" was checked separately: items from
the same category were no more similar than items from different categories
(about -0.01 either way), and top-N from that matrix matched a random shelf.
Cosine on the raw purchases keeps a category signal, so that is what
``recommend`` uses. A neighbor is kept only if it is among the closest ones and
clears a minimum similarity, otherwise weak neighbors wash the score out.

Rating error is a different question and still uses adjusted cosine. Those
numbers are compared with a user-mean baseline in ``rating_error_report``.
Views and carts are not in the matrix: a view is exposure, a cart is intent,
and both made neighbors collapse toward popular items.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from recsys.data import empty_recs, top_k, user_item_strength

CF_EVENTS = {"purchase"}


def _positive_cosine(vectors: np.ndarray) -> np.ndarray:
    n_rows = vectors.shape[0]
    if n_rows == 0 or vectors.shape[1] == 0:
        return np.zeros((n_rows, n_rows))
    similarity = cosine_similarity(vectors)
    similarity[~np.isfinite(similarity)] = 0.0
    np.fill_diagonal(similarity, 0.0)
    similarity[similarity < 0] = 0.0
    return similarity


def _keep_neighbors(similarity: np.ndarray, n_neighbors: int, min_similarity: float) -> np.ndarray:
    masked = np.zeros_like(similarity)
    n_rows = similarity.shape[0]
    if n_rows == 0:
        return masked
    width = min(n_neighbors, n_rows)
    chosen = np.argpartition(similarity, -width, axis=1)[:, -width:]
    rows = np.arange(n_rows)[:, None]
    picked = similarity[rows, chosen]
    picked = np.where(picked >= min_similarity, picked, 0.0)
    masked[rows, chosen] = picked
    np.fill_diagonal(masked, 0.0)
    return masked


class _NeighborhoodCF:
    def __init__(self, n_neighbors: int, min_similarity: float) -> None:
        self.n_neighbors = n_neighbors
        self.min_similarity = min_similarity
        self.user_ids = np.array([], dtype=int)
        self.item_ids = np.array([], dtype=int)
        self.user_index: dict[int, int] = {}
        self.matrix = np.zeros((0, 0))
        self.seen = np.zeros((0, 0), dtype=bool)
        self.neighbor_sim = np.zeros((0, 0))

    def _prepare(self, interactions: pd.DataFrame, products: pd.DataFrame) -> None:
        self.item_ids = products["item_id"].astype(int).to_numpy()
        self.user_ids, self.matrix, self.seen = user_item_strength(
            interactions, self.item_ids, events=CF_EVENTS
        )
        self.user_index = {int(user): pos for pos, user in enumerate(self.user_ids)}

    def recommend(self, user_id: int | None = None, k: int = 10, context: dict | None = None) -> pd.DataFrame:
        del context
        if user_id is None:
            return empty_recs()
        scores = self.score_user(int(user_id))
        if scores is None:
            return empty_recs()
        return top_k(self.item_ids, scores, k, set(), self.name)

    def score_user(self, user_id: int) -> np.ndarray | None:
        raise NotImplementedError


class UserBasedCF(_NeighborhoodCF):
    """Score an item by how strongly similar buyers purchased it."""

    name = "user_cf"

    def __init__(self, n_neighbors: int = 10, min_similarity: float = 0.3) -> None:
        super().__init__(n_neighbors=n_neighbors, min_similarity=min_similarity)

    def fit(self, interactions: pd.DataFrame, products: pd.DataFrame) -> UserBasedCF:
        self._prepare(interactions, products)
        similarity = _positive_cosine(self.matrix)
        self.neighbor_sim = _keep_neighbors(similarity, self.n_neighbors, self.min_similarity)
        return self

    def score_user(self, user_id: int) -> np.ndarray | None:
        pos = self.user_index.get(int(user_id))
        if pos is None or self.neighbor_sim.size == 0:
            return None
        weights = self.neighbor_sim[pos]
        numer = weights @ self.matrix
        denom = weights @ self.seen.astype(float)
        predicted = np.full(len(self.item_ids), np.nan)
        valid = denom > 1e-9
        predicted[valid] = numer[valid] / denom[valid]
        predicted[self.seen[pos]] = -np.inf
        return predicted


class ItemBasedCF(_NeighborhoodCF):
    """Score an item from the user's purchases of similar items.

    Similarities are computed once. A request only mixes the rows of items that
    this user already bought, which is why item-item CF is the usual production
    choice between the two neighborhood methods.
    """

    name = "item_cf"

    def __init__(self, n_neighbors: int = 5, min_similarity: float = 0.2) -> None:
        super().__init__(n_neighbors=n_neighbors, min_similarity=min_similarity)

    def fit(self, interactions: pd.DataFrame, products: pd.DataFrame) -> ItemBasedCF:
        self._prepare(interactions, products)
        n_items = len(self.item_ids)
        if len(self.user_ids) == 0 or n_items == 0:
            self.neighbor_sim = np.zeros((n_items, n_items))
            return self
        similarity = _positive_cosine(self.matrix.T)
        self.neighbor_sim = _keep_neighbors(similarity, self.n_neighbors, self.min_similarity)
        return self

    def score_user(self, user_id: int) -> np.ndarray | None:
        pos = self.user_index.get(int(user_id))
        if pos is None or self.neighbor_sim.size == 0:
            return None
        ratings = self.matrix[pos]
        if not np.any(ratings > 0):
            return np.full(len(self.item_ids), np.nan)
        numer = self.neighbor_sim @ ratings
        denom = self.neighbor_sim @ (ratings > 0).astype(float)
        predicted = np.full(len(self.item_ids), np.nan)
        valid = denom > 1e-9
        predicted[valid] = numer[valid] / denom[valid]
        predicted[ratings > 0] = -np.inf
        return predicted


def _errors(pairs: list[tuple[float, float]]) -> dict[str, float]:
    if not pairs:
        return {"rmse": float("nan"), "mae": float("nan"), "n": 0.0}
    error = np.array([pred - actual for pred, actual in pairs], dtype=float)
    return {
        "rmse": float(np.sqrt(np.mean(error**2))),
        "mae": float(np.mean(np.abs(error))),
        "n": float(len(pairs)),
    }


def rating_error_report(train: pd.DataFrame, test: pd.DataFrame, products: pd.DataFrame) -> dict[str, dict[str, float]]:
    """Adjusted-cosine rating error against a user-mean baseline, on the same pairs.

    This is not the score used for ranking. It answers whether the classic
    rating formula beats "predict this user's average" on held-out purchases.
    """
    item_ids = products["item_id"].astype(int).to_numpy()
    user_ids, matrix, seen = user_item_strength(train, item_ids, events={"purchase"})
    user_index = {int(user): pos for pos, user in enumerate(user_ids)}
    item_index = {int(item): pos for pos, item in enumerate(item_ids)}
    means = np.zeros(len(user_ids))
    centered = np.zeros_like(matrix)
    for row in range(len(user_ids)):
        mask = matrix[row] > 0
        if not mask.any():
            continue
        means[row] = matrix[row, mask].mean()
        centered[row, mask] = matrix[row, mask] - means[row]
    user_sim = _positive_cosine(centered) if len(user_ids) else np.zeros((0, 0))
    item_sim = _positive_cosine(centered.T) if len(user_ids) else np.zeros((len(item_ids), len(item_ids)))
    user_neighbors = _keep_neighbors(user_sim, n_neighbors=40, min_similarity=1e-6)
    item_neighbors = _keep_neighbors(item_sim, n_neighbors=30, min_similarity=1e-6)

    bought = test[test["event_type"].eq("purchase") & test["rating"].notna()]
    user_pairs_model: list[tuple[float, float]] = []
    user_pairs_base: list[tuple[float, float]] = []
    item_pairs_model: list[tuple[float, float]] = []
    item_pairs_base: list[tuple[float, float]] = []
    for user_id, group in bought.groupby("user_id"):
        pos = user_index.get(int(user_id))
        if pos is None:
            continue
        rated = np.where(seen[pos])[0]
        weights = user_neighbors[pos]
        neighbor_rows = np.where(weights > 0)[0]
        for row in group.itertuples(index=False):
            item_pos = item_index.get(int(row.item_id))
            if item_pos is None or seen[pos, item_pos]:
                continue
            actual = float(row.rating)
            baseline = float(means[pos])
            if len(neighbor_rows):
                neigh_ratings = matrix[neighbor_rows, item_pos]
                observed = neigh_ratings > 0
                if observed.any():
                    w = weights[neighbor_rows][observed]
                    deviation = neigh_ratings[observed] - means[neighbor_rows][observed]
                    pred = float(np.clip(baseline + np.sum(w * deviation) / np.sum(np.abs(w)), 1, 5))
                    user_pairs_model.append((pred, actual))
                    user_pairs_base.append((baseline, actual))
            if len(rated) == 0:
                continue
            sims = item_neighbors[item_pos, rated]
            positive = sims > 0
            if not positive.any():
                continue
            w = sims[positive]
            pred = float(np.clip(np.sum(w * matrix[pos, rated][positive]) / np.sum(w), 1, 5))
            item_pairs_model.append((pred, actual))
            item_pairs_base.append((float(means[pos]), actual))

    return {
        "user_mean_on_user_cf_pairs": _errors(user_pairs_base),
        "user_cf_adjusted_cosine": _errors(user_pairs_model),
        "user_mean_on_item_cf_pairs": _errors(item_pairs_base),
        "item_cf_adjusted_cosine": _errors(item_pairs_model),
    }
