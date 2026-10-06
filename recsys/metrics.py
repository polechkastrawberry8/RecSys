"""Offline ranking metrics.

Relevance is binary: a product the user later purchased. Graded ratings are
reported separately as RMSE for the two collaborative models, whose scores
still live on the rating scale.
"""

from __future__ import annotations

import math

import numpy as np


def precision_at_k(recommended: list[int], relevant: set[int], k: int) -> float:
    if k <= 0:
        return 0.0
    hits = sum(1 for item in recommended[:k] if item in relevant)
    return hits / k


def recall_at_k(recommended: list[int], relevant: set[int], k: int) -> float:
    if not relevant:
        return 0.0
    hits = sum(1 for item in recommended[:k] if item in relevant)
    return hits / len(relevant)


def ndcg_at_k(recommended: list[int], relevant: set[int], k: int) -> float:
    hits = [1.0 if item in relevant else 0.0 for item in recommended[:k]]
    dcg = sum(value / math.log2(index + 2) for index, value in enumerate(hits))
    ideal = sum(1.0 / math.log2(index + 2) for index in range(min(k, len(relevant))))
    if ideal == 0:
        return 0.0
    return dcg / ideal


def average_precision_at_k(recommended: list[int], relevant: set[int], k: int) -> float:
    if not relevant:
        return 0.0
    hits = 0
    total = 0.0
    for index, item in enumerate(recommended[:k], start=1):
        if item in relevant:
            hits += 1
            total += hits / index
    return total / min(len(relevant), k)


def gini(counts: np.ndarray) -> float:
    values = np.sort(np.asarray(counts, dtype=float))
    n = len(values)
    total = float(values.sum())
    if n == 0 or total <= 0:
        return 0.0
    index = np.arange(1, n + 1)
    return float((2 * np.sum(index * values) / (n * total)) - (n + 1) / n)
