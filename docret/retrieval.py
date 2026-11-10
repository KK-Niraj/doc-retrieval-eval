"""Two retrieval routes and a rank-fusion combiner.

BM25 and TF-IDF cosine are implemented directly so the repo has no retrieval
dependency. Both return full score vectors; fusion works on ranks, because the
raw scores are not comparable across routes.
"""
from __future__ import annotations

import math
import re
from collections import Counter

import numpy as np

_TOKEN = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


class BM25:
    """Okapi BM25 over a fixed list of tokenized documents."""

    def __init__(self, docs: list[list[str]], k1: float = 1.5, b: float = 0.75) -> None:
        self.k1, self.b = k1, b
        self.n = len(docs)
        self.tfs = [Counter(d) for d in docs]
        self.lens = [len(d) for d in docs]
        self.avgdl = (sum(self.lens) / self.n) if self.n else 0.0
        df: Counter = Counter()
        for d in docs:
            df.update(set(d))
        self.idf = {
            t: math.log(1 + (self.n - c + 0.5) / (c + 0.5)) for t, c in df.items()
        }

    def scores(self, query: str) -> np.ndarray:
        out = np.zeros(self.n, dtype=float)
        for term in tokenize(query):
            idf = self.idf.get(term)
            if idf is None:
                continue
            for i in range(self.n):
                f = self.tfs[i].get(term, 0)
                if not f:
                    continue
                denom = f + self.k1 * (1 - self.b + self.b * self.lens[i] / self.avgdl)
                out[i] += idf * (f * (self.k1 + 1)) / denom
        return out


class TfidfCosine:
    """TF-IDF vectors with L2 normalisation, scored by cosine similarity."""

    def __init__(self, docs: list[list[str]]) -> None:
        self.vocab = sorted({t for d in docs for t in d})
        self.index = {t: i for i, t in enumerate(self.vocab)}
        df = Counter(t for d in docs for t in set(d))
        n = len(docs) or 1
        self.idf = np.array(
            [math.log((1 + n) / (1 + df[t])) + 1.0 for t in self.vocab], dtype=float
        )
        self.matrix = np.vstack([self._vec(d) for d in docs]) if docs else np.zeros((0, len(self.vocab)))

    def _vec(self, tokens: list[str]) -> np.ndarray:
        v = np.zeros(len(self.vocab), dtype=float)
        for t in tokens:
            i = self.index.get(t)
            if i is not None:
                v[i] += 1.0
        v *= self.idf
        norm = np.linalg.norm(v)
        return v / norm if norm else v

    def scores(self, query: str) -> np.ndarray:
        q = self._vec(tokenize(query))
        if not self.vocab:
            return np.zeros(0)
        return self.matrix @ q


def ranks_from_scores(scores: np.ndarray) -> list[int]:
    """Indices ordered best-first; ties broken by index for determinism."""
    return sorted(range(len(scores)), key=lambda i: (-scores[i], i))


def reciprocal_rank_fusion(
    ranked_lists: list[list[int]], weights: list[float] | None = None, k: int = 60
) -> np.ndarray:
    """Weighted RRF over pre-ranked index lists. Higher is better."""
    if not ranked_lists:
        return np.zeros(0)
    n = max(max(r) for r in ranked_lists if r) + 1 if any(ranked_lists) else 0
    weights = weights or [1.0] * len(ranked_lists)
    fused = np.zeros(n, dtype=float)
    for ranked, w in zip(ranked_lists, weights):
        for pos, idx in enumerate(ranked, start=1):
            if idx < n:
                fused[idx] += w / (k + pos)
    return fused
