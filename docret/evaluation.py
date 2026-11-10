"""Retrieval metrics computed against exact labels."""
from __future__ import annotations

import math

import numpy as np


def recall_at_k(ranked: list[int], gold: int, k: int) -> float:
    return 1.0 if gold in ranked[:k] else 0.0


def reciprocal_rank(ranked: list[int], gold: int) -> float:
    for pos, idx in enumerate(ranked, start=1):
        if idx == gold:
            return 1.0 / pos
    return 0.0


def ndcg_at_k(ranked: list[int], gold: int, k: int) -> float:
    """Binary-relevance nDCG@k with a single relevant document."""
    for pos, idx in enumerate(ranked[:k], start=1):
        if idx == gold:
            return 1.0 / math.log2(pos + 1)
    return 0.0


def evaluate(orderings: dict[str, list[int]], golds: dict[str, int], k: int = 5) -> dict[str, float]:
    ids = list(orderings)
    if not ids:
        return {"recall@k": 0.0, "mrr": 0.0, "ndcg@k": 0.0, "n": 0.0}
    r = sum(recall_at_k(orderings[i], golds[i], k) for i in ids) / len(ids)
    m = sum(reciprocal_rank(orderings[i], golds[i]) for i in ids) / len(ids)
    d = sum(ndcg_at_k(orderings[i], golds[i], k) for i in ids) / len(ids)
    return {"recall@k": r, "mrr": m, "ndcg@k": d, "n": float(len(ids))}
