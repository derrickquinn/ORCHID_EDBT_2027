"""Manuscript query partitions and bounded array reads (no dataset I/O here)."""

from __future__ import annotations

FIT_QUERIES = 512
VALIDATION_QUERIES = 512
TEST_QUERIES = 10_000
TEST_START = FIT_QUERIES + VALIDATION_QUERIES


def query_slice(total: int, start: int, count: int, *, name: str = "queries") -> slice:
    if start < 0 or count <= 0:
        raise ValueError("query start must be nonnegative and count must be positive")
    stop = start + count
    if stop > total:
        remaining = max(0, total - start)
        raise ValueError(
            f"{name} has {total} rows; requested [{start}, {stop}). "
            f"Only {remaining} remain after the held-out prefix. "
            "Prepare additional aligned queries, or explicitly reduce num_queries "
            "for a bounded check; held-out queries are never reused."
        )
    return slice(start, stop)


def validate_split(
    start: int, count: int, fit_queries: int, *, calibration: bool
) -> None:
    if fit_queries <= 0 or count <= 0:
        raise ValueError("fit and evaluation query counts must be positive")
    minimum = fit_queries if calibration else fit_queries + VALIDATION_QUERIES
    if start < minimum:
        raise ValueError(
            f"query_start must be at least {minimum} to keep query sets disjoint"
        )


def batch_size_for(gpu: bool, requested: int | None = None) -> int:
    expected = 1 if gpu else 100
    if requested is not None and requested != expected:
        raise ValueError(
            f"the manuscript requires batch_size={expected} for {'GPU' if gpu else 'CPU'}"
        )
    return expected


def batch_slices(count: int, batch_size: int):
    if count <= 0 or batch_size <= 0:
        raise ValueError("query count and batch size must be positive")
    for start in range(0, count, batch_size):
        yield slice(start, min(start + batch_size, count))
