"""Agreement / stability metrics between attribution vectors.

All metrics compare features ranked by ABSOLUTE attribution, i.e. "which
features drive this prediction", ignoring direction.
"""
from __future__ import annotations

from itertools import combinations

import numpy as np
from scipy.stats import spearmanr


def top_k(a: np.ndarray, k: int) -> set[int]:
    a = np.abs(np.asarray(a, dtype=float))
    # stable tie-break on index so results are deterministic
    order = np.lexsort((np.arange(len(a)), -a))
    return set(order[:k].tolist())


def top1(a: np.ndarray) -> int:
    return min(top_k(a, 1))


def jaccard_topk(a: np.ndarray, b: np.ndarray, k: int) -> float:
    A, B = top_k(a, k), top_k(b, k)
    return len(A & B) / len(A | B)


def rank_corr(a: np.ndarray, b: np.ndarray) -> float:
    a, b = np.abs(a), np.abs(b)
    if np.allclose(a, a[0]) or np.allclose(b, b[0]):
        return np.nan
    return float(spearmanr(a, b).statistic)


def pair_metrics(a: np.ndarray, b: np.ndarray, k: int) -> dict:
    return {"jaccard": jaccard_topk(a, b, k), "spearman": rank_corr(a, b),
            "top1_same": float(top1(a) == top1(b))}


def set_stability(vectors: list[np.ndarray], k: int) -> dict:
    """Mean pairwise agreement within a set of explanations of the SAME point."""
    pairs = [pair_metrics(a, b, k) for a, b in combinations(vectors, 2)]
    tops = [top1(v) for v in vectors]
    modal = max(set(tops), key=tops.count)
    return {
        "jaccard": float(np.mean([p["jaccard"] for p in pairs])),
        "spearman": float(np.nanmean([p["spearman"] for p in pairs])) if pairs else np.nan,
        "top1_same": float(np.mean([p["top1_same"] for p in pairs])),
        "n_distinct_top1": len(set(tops)),
        "modal_top1_share": tops.count(modal) / len(tops),
    }


def against_reference(ref: np.ndarray, others: list[np.ndarray], k: int) -> dict:
    """Agreement of perturbed explanations with the unperturbed reference."""
    ms = [pair_metrics(ref, o, k) for o in others]
    return {"jaccard": float(np.mean([m["jaccard"] for m in ms])),
            "spearman": float(np.nanmean([m["spearman"] for m in ms])),
            "top1_changed": 1 - float(np.mean([m["top1_same"] for m in ms]))}
