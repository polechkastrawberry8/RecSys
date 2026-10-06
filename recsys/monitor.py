"""In-process stand-in for the production dashboard.

A real service would export the same numbers (latency, empty rate, coverage,
concentration, sensitive-item rate) to its metrics backend. Here they are
collected in memory so the evaluation run can show what would be watched.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from recsys.metrics import gini


class RecommendationMonitor:
    def __init__(self, catalog_ids: list[int], sensitive_ids: set[int] | None = None) -> None:
        self.catalog_ids = [int(item) for item in catalog_ids]
        self.sensitive_ids = sensitive_ids or set()
        self.latencies_ms: list[float] = []
        self.lengths: list[int] = []
        self.served: list[int] = []
        self.sensitive_impressions = 0

    def observe(self, recommendations: pd.DataFrame, latency_ms: float) -> None:
        item_ids = [int(item) for item in recommendations["item_id"]] if not recommendations.empty else []
        self.latencies_ms.append(float(latency_ms))
        self.lengths.append(len(item_ids))
        self.served.extend(item_ids)
        self.sensitive_impressions += sum(1 for item in item_ids if item in self.sensitive_ids)

    def report(self) -> dict[str, float]:
        latencies = np.array(self.latencies_ms, dtype=float) if self.latencies_ms else np.array([0.0])
        counts = {item: 0 for item in self.catalog_ids}
        for item in self.served:
            if item in counts:
                counts[item] += 1
        impressions = max(len(self.served), 1)
        return {
            "requests": float(len(self.lengths)),
            "latency_p50_ms": float(np.percentile(latencies, 50)),
            "latency_p95_ms": float(np.percentile(latencies, 95)),
            "empty_rate": float(np.mean([length == 0 for length in self.lengths])) if self.lengths else 0.0,
            "mean_list_size": float(np.mean(self.lengths)) if self.lengths else 0.0,
            "catalog_coverage": float(sum(1 for value in counts.values() if value > 0) / max(len(counts), 1)),
            "gini_served": gini(np.array(list(counts.values()), dtype=float)),
            "sensitive_impression_share": self.sensitive_impressions / impressions,
        }
