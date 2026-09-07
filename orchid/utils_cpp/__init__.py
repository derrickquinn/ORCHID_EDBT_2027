"""Convenience imports for the utils_cpp package.

This makes `import utils_cpp` work when running from the repo root, where the
project directory itself is on `sys.path`.
"""

from .utils_cpp import (
    cluster_legals,
    evaluate_predicate_conjunctions,
    ids_to_bitmap,
    log_scale,
    predicate_results_to_bitmap,
    predicate_results_to_mask,
    prepare_cluster_order,
    reorder_mask,
    topk_small,
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
