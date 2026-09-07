"""Predicate-aware routing semantics shared by calibration and online search."""

import numpy as np

from utils_cpp import log_scale


def predicate_log_scores(selectivity, beta, *, out=None, threads=0):
    """Compute beta*ln(p), excluding p=0 even when beta is zero."""
    if out is None:
        out = np.empty_like(selectivity, dtype=np.float32)
    if beta == 0:
        out.fill(0)
    else:
        log_scale(selectivity, beta, out=out, eps=0.0, threads=threads)
    out[selectivity <= 0] = -np.inf
    return out


def exclude_empty_probes(rankings, selectivity):
    """FAISS skips -1 list IDs, including padding beyond eligible lists."""
    result = np.ascontiguousarray(rankings, dtype=np.int64).copy()
    eligible = np.take_along_axis(selectivity, result, axis=1) > 0
    result[~eligible] = -1
    return result
