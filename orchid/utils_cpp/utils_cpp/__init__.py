"""Python bindings for cluster legals kernels."""

from __future__ import annotations

import numpy as np

from . import _cluster_legals as _impl

prepare_cluster_order = _impl.prepare_cluster_order
reorder_mask = _impl.reorder_mask


def topk_small(
    scores: np.ndarray,
    k: int,
    out: np.ndarray | None = None,
    threads: int = 0,
) -> np.ndarray:
    """Return descending top-k column IDs for each contiguous float32 row."""
    scores = np.asarray(scores)
    if scores.dtype != np.float32 or not scores.flags.c_contiguous:
        raise ValueError("scores must be a C-contiguous float32 array")
    if scores.ndim != 2:
        raise ValueError("scores must be two-dimensional")
    k = int(k)
    if k <= 0 or k > scores.shape[1]:
        raise ValueError("k must be in [1, scores.shape[1]]")
    if out is None:
        out = np.empty((scores.shape[0], k), dtype=np.int64)
    elif (
        out.dtype != np.int64
        or not out.flags.c_contiguous
        or out.shape != (scores.shape[0], k)
    ):
        raise ValueError("out must be a C-contiguous int64 array of shape (rows, k)")
    return _impl.topk_small_out(scores, k, out, int(threads))


def ids_to_bitmap(
    offsets: np.ndarray,
    ids: np.ndarray,
    n_docs: int,
    out: np.ndarray | None = None,
    clear: bool = True,
    threads: int = 0,
    validate: bool = True,
) -> np.ndarray:
    """Materialize batched CSR document-ID lists as packed bitmaps.

    Each query owns one output row, so the C++ kernel can parallelize across
    queries without atomics. Bit order matches ``np.packbits(...,
    bitorder="little")``. Inputs are range-checked here rather than in the
    per-ID C++ loop. Trusted hot paths can pass ``validate=False`` after
    validating the result set during preparation.
    """
    offsets = np.ascontiguousarray(offsets, dtype=np.int64)
    ids = np.ascontiguousarray(ids, dtype=np.uint32)
    n_docs = int(n_docs)
    if n_docs <= 0:
        raise ValueError("n_docs must be positive")
    if validate and ids.size and int(ids.max()) >= n_docs:
        raise ValueError("ids contains a document ID outside [0, n_docs)")
    if out is None:
        out = np.empty((offsets.size - 1, (n_docs + 7) // 8), dtype=np.uint8)
    return _impl.ids_to_bitmap_out(offsets, ids, n_docs, out, clear, threads)


predicate_results_to_bitmap = _impl.predicate_results_to_bitmap_out


def predicate_results_to_mask(
    posting_lists: list[np.ndarray | None],
    dense_masks: list[np.ndarray | None],
    n_docs: int,
    out: np.ndarray | None = None,
    threads: int = 0,
) -> np.ndarray:
    """Materialize predicate results as one byte per document.

    Dense inputs are already byte masks and are copied into ``out`` by the C++
    kernel. Sparse inputs are scattered into the same output layout. The
    returned array has shape ``(n_queries, n_docs)`` and contains only zero and
    one.
    """
    n_queries = len(posting_lists)
    if len(dense_masks) != n_queries:
        raise ValueError("dense_masks must have one entry per query")
    n_docs = int(n_docs)
    if n_docs <= 0:
        raise ValueError("n_docs must be positive")
    if out is None:
        out = np.empty((n_queries, n_docs), dtype=np.uint8)
    elif out.dtype != np.uint8 or not out.flags.c_contiguous:
        raise ValueError("out must be a C-contiguous uint8 array")
    return _impl.predicate_results_to_mask_out(
        posting_lists, dense_masks, n_docs, out, int(threads)
    )


def cluster_legals(
    mask_packed: np.ndarray,
    cluster_offsets: np.ndarray,
    cluster_counts: np.ndarray,
    out: np.ndarray | None = None,
    threads: int = 0,
) -> np.ndarray:
    if cluster_offsets is None or cluster_counts is None:
        raise ValueError("cluster_offsets and cluster_counts are required")
    if out is None:
        n_queries = int(mask_packed.shape[0])
        n_list = int(cluster_counts.shape[0])
        out = np.empty((n_queries, n_list), dtype=np.float32)
    return _impl.cluster_legals_preordered_packed_out(
        mask_packed, cluster_offsets, cluster_counts, out, threads
    )


def log_scale(
    in_arr: np.ndarray,
    beta: float,
    out: np.ndarray | None = None,
    eps: float = 1e-9,
    threads: int = 0,
) -> np.ndarray:
    if out is None:
        out = np.empty_like(in_arr, dtype=np.float32)
    return _impl.log_scale_out(in_arr, float(beta), out, float(eps), threads)


def evaluate_predicate_conjunctions(
    keys: np.ndarray,
    counts: np.ndarray,
    sparse_offsets: np.ndarray,
    postings: np.ndarray,
    dense_rows: np.ndarray,
    dense_masks: np.ndarray,
    n_docs: int,
    query_offsets: np.ndarray,
    query_terms: np.ndarray,
    packed: bool = True,
    threads: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate a batch of same-field categorical conjunctions.

    Returns per-query kinds, sparse CSR results, dense-row mappings, and the
    representation-native dense result matrix. Sparse results remain
    document-ID lists; this kernel does not perform final materialization.
    """
    return _impl.evaluate_predicate_conjunctions(
        np.ascontiguousarray(keys, dtype=np.int64),
        np.ascontiguousarray(counts, dtype=np.int64),
        np.ascontiguousarray(sparse_offsets, dtype=np.int64),
        np.ascontiguousarray(postings, dtype=np.uint32),
        np.ascontiguousarray(dense_rows, dtype=np.int32),
        np.ascontiguousarray(dense_masks, dtype=np.uint8),
        int(n_docs),
        np.ascontiguousarray(query_offsets, dtype=np.int64),
        np.ascontiguousarray(query_terms, dtype=np.int64),
        bool(packed),
        int(threads),
    )


__all__ = [
    "topk_small",
    "cluster_legals",
    "evaluate_predicate_conjunctions",
    "ids_to_bitmap",
    "log_scale",
    "predicate_results_to_bitmap",
    "predicate_results_to_mask",
    "prepare_cluster_order",
    "reorder_mask",
]
