"""Dataset loading and the time-based train/test split."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"

EVENT_WEIGHT = {"view": 1.0, "cart": 2.0}


def load_dataset(data_dir: Path | None = None) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    folder = Path(data_dir) if data_dir else DATA_DIR
    products = pd.read_csv(folder / "products.csv", encoding="utf-8")
    users = pd.read_csv(folder / "users.csv", encoding="utf-8")
    interactions = pd.read_csv(folder / "interactions.csv", encoding="utf-8")
    interactions["timestamp"] = pd.to_datetime(interactions["timestamp"])
    products["sensitive"] = products["sensitive"].fillna(0).astype(int)
    return products, users, interactions


def add_strength(interactions: pd.DataFrame) -> pd.DataFrame:
    """Map events to a preference strength. Views stay weak on purpose."""
    frame = interactions.copy()
    frame["strength"] = frame["event_type"].map(EVENT_WEIGHT).astype(float)
    purchased = frame["event_type"].eq("purchase")
    frame.loc[purchased, "strength"] = frame.loc[purchased, "rating"].fillna(4.0).astype(float)
    return frame


def user_item_strength(
    interactions: pd.DataFrame,
    item_ids: list[int] | np.ndarray,
    events: set[str] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Max strength per user-item pair.

    Returns user ids, a dense matrix aligned with ``item_ids``, and a boolean
    mask of observed pairs. Missing entries stay 0 and must not be treated as
    a real score of zero outside the similarity math.
    """
    frame = add_strength(interactions)
    if events is not None:
        frame = frame[frame["event_type"].isin(events)]
    if frame.empty:
        return np.array([], dtype=int), np.zeros((0, len(item_ids))), np.zeros((0, len(item_ids)), dtype=bool)

    grouped = frame.groupby(["user_id", "item_id"], as_index=False)["strength"].max()
    users = np.array(sorted(grouped["user_id"].unique()), dtype=int)
    user_index = {int(user): pos for pos, user in enumerate(users)}
    item_index = {int(item): pos for pos, item in enumerate(item_ids)}
    matrix = np.zeros((len(users), len(item_ids)), dtype=float)
    for row in grouped.itertuples(index=False):
        item_pos = item_index.get(int(row.item_id))
        if item_pos is None:
            continue
        matrix[user_index[int(row.user_id)], item_pos] = float(row.strength)
    return users, matrix, matrix > 0


def time_split(
    interactions: pd.DataFrame,
    test_frac: float = 0.2,
    min_events_for_fraction: int = 8,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Hold out each user's latest interactions.

    Users with a short history keep a single last event as test, so cold-start
    users can be scored. A global random split would leak future behaviour into
    training and overstate quality.
    """
    train_parts: list[pd.DataFrame] = []
    test_parts: list[pd.DataFrame] = []
    for _, group in interactions.groupby("user_id", sort=False):
        ordered = group.sort_values("timestamp")
        if len(ordered) < 2:
            train_parts.append(ordered)
            continue
        if len(ordered) < min_events_for_fraction:
            n_test = 1
        else:
            n_test = max(1, int(round(len(ordered) * test_frac)))
        train_parts.append(ordered.iloc[:-n_test])
        test_parts.append(ordered.iloc[-n_test:])
    train = pd.concat(train_parts, ignore_index=True)
    test = pd.concat(test_parts, ignore_index=True) if test_parts else interactions.iloc[0:0].copy()
    return train, test


def season_of(timestamp: pd.Timestamp) -> str:
    month = int(pd.Timestamp(timestamp).month)
    if month in (12, 1, 2):
        return "winter"
    if month in (3, 4, 5):
        return "spring"
    if month in (6, 7, 8):
        return "summer"
    return "autumn"


def empty_recs() -> pd.DataFrame:
    return pd.DataFrame(columns=["item_id", "score", "reason"])


def as_recs(item_ids: list[int], scores: list[float], reason: str) -> pd.DataFrame:
    if not item_ids:
        return empty_recs()
    frame = pd.DataFrame({"item_id": item_ids, "score": scores, "reason": reason})
    frame["item_id"] = frame["item_id"].astype(int)
    return frame.reset_index(drop=True)


def seen_items(interactions: pd.DataFrame) -> dict[int, set[int]]:
    return {
        int(user): {int(item) for item in items}
        for user, items in interactions.groupby("user_id")["item_id"]
    }


def top_k(item_ids: np.ndarray, scores: np.ndarray, k: int, exclude: set[int], reason: str) -> pd.DataFrame:
    order = np.argsort(-scores)
    chosen_ids: list[int] = []
    chosen_scores: list[float] = []
    for pos in order:
        score = float(scores[pos])
        if not np.isfinite(score):
            continue
        item_id = int(item_ids[pos])
        if item_id in exclude:
            continue
        chosen_ids.append(item_id)
        chosen_scores.append(score)
        if len(chosen_ids) >= k:
            break
    return as_recs(chosen_ids, chosen_scores, reason)
