"""Published predicate workloads over query-independent document indexes."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from itertools import chain
from pathlib import Path

import numpy as np

from predicate_index import (
    HybridCategoricalIndex,
    MaskFormat,
    PredicateResult,
    SortedRangeIndex,
)
from src.dataset_config import DatasetConfig


ROOT = Path(
    os.environ.get("ORCHID_ASSET_ROOT", Path(__file__).resolve().parent.parent)
)
LAION_KEYWORDS = ROOT / "setup/laion/keywords.jsonl"
LAION_QUERIES = ROOT / "setup/laion/predicate_queries.txt"
YFCC_DOCUMENTS = ROOT / "data/raw/yfcc100M/base.metadata.10M.spmat"
YFCC_QUERIES = ROOT / "data/raw/yfcc100M/query.metadata.private.2727415019.100K.spmat"

SIFT_EQUALITY = {"sift12": 12, "sift24": 24, "sift48": 48}
SIFT_RANGES = {"sift_r10": 10, "sift_r20": 20, "sift_r40": 40}
LAION_MODES = {
    "laion_pos": "positive",
    "laion_neg": "negative",
    "laion_all": "alternating",
}
LAION_TERM_COUNT = 30


@dataclass(slots=True)
class PredicateWorkload:
    index: HybridCategoricalIndex | SortedRangeIndex
    operation: str
    first: np.ndarray
    second: np.ndarray | None = None

    @property
    def n_queries(self) -> int:
        return self.first.size - 1 if self.operation == "conjunction" else self.first.size

    def slice_queries(self, start: int, count: int) -> "PredicateWorkload":
        from query_settings import query_slice

        rows = query_slice(self.n_queries, start, count, name="predicate metadata")
        if self.operation == "conjunction":
            offsets = self.first[start : rows.stop + 1].copy()
            terms = self.second[int(offsets[0]) : int(offsets[-1])]
            offsets -= offsets[0]
            return PredicateWorkload(self.index, self.operation, offsets, terms)
        second = None if self.second is None else self.second[rows]
        return PredicateWorkload(self.index, self.operation, self.first[rows], second)

    def evaluate_batch(
        self, start: int, stop: int, *, threads: int = 0
    ) -> list[PredicateResult]:
        if not 0 <= start < stop <= self.n_queries:
            raise ValueError("predicate batch lies outside the query workload")
        if self.operation == "equality":
            assert isinstance(self.index, HybridCategoricalIndex)
            return self.index.equal_batch(self.first[start:stop])
        if self.operation == "range":
            assert isinstance(self.index, SortedRangeIndex) and self.second is not None
            return self.index.range_batch(self.first[start:stop], self.second[start:stop])

        assert self.operation == "conjunction"
        assert isinstance(self.index, HybridCategoricalIndex) and self.second is not None
        offsets = self.first[start : stop + 1].copy()
        term_start, term_stop = int(offsets[0]), int(offsets[-1])
        offsets -= term_start
        terms = self.second[term_start:term_stop]
        lengths = np.diff(offsets)
        if np.all(lengths == 1):
            return self.index.equal_batch(terms)
        if np.all(lengths > 1):
            return self.index.conjunction_batch(offsets, terms, threads=threads)

        results: list[PredicateResult | None] = [None] * lengths.size
        singles = np.flatnonzero(lengths == 1)
        for position, result in zip(
            singles, self.index.equal_batch(terms[offsets[singles]])
        ):
            results[int(position)] = result

        multiples = np.flatnonzero(lengths > 1)
        multiple_lengths = lengths[multiples]
        multiple_offsets = np.r_[0, np.cumsum(multiple_lengths)]
        multiple_terms = np.concatenate(
            [terms[offsets[q] : offsets[q + 1]] for q in multiples]
        )
        for position, result in zip(
            multiples,
            self.index.conjunction_batch(
                multiple_offsets, multiple_terms, threads=threads
            ),
        ):
            results[int(position)] = result
        return [result for result in results if result is not None]


def _load_index(
    path: Path,
    index_type: type[HybridCategoricalIndex] | type[SortedRangeIndex],
    n_docs: int,
) -> HybridCategoricalIndex | SortedRangeIndex | None:
    if not (path / "index.json").exists():
        return None
    # Loading is offline. Keep the online path on ordinary resident arrays;
    # NumPy memmap slices add substantial Python overhead to batched lookups.
    index = index_type.load(path, mmap_mode=None)
    if index.n_docs != n_docs:
        raise ValueError("cached predicate index has the wrong document count")
    return index


def _save_index(
    index: HybridCategoricalIndex | SortedRangeIndex, path: Path
) -> HybridCategoricalIndex | SortedRangeIndex:
    index.save(path)
    return index


def _sift_workload(
    name: str, path: Path, document_ids: np.ndarray, n_docs: int, query_count: int
) -> PredicateWorkload:
    cardinality = SIFT_EQUALITY.get(name, 1000)
    values, rng = _sift_document_values(cardinality, n_docs)
    if name in SIFT_EQUALITY:
        queries = rng.integers(0, cardinality, size=query_count)
        index = _load_index(path, HybridCategoricalIndex, n_docs)
        if index is None:
            index = _save_index(
                HybridCategoricalIndex.build_scalar(values, document_ids=document_ids),
                path,
            )
        return PredicateWorkload(index, "equality", queries)

    width = SIFT_RANGES[name]
    lower = rng.integers(0, cardinality - width + 1, size=query_count)
    index = _load_index(path, SortedRangeIndex, n_docs)
    if index is None:
        index = _save_index(
            SortedRangeIndex.build(values, document_ids=document_ids), path
        )
    return PredicateWorkload(index, "range", lower, lower + width)


def _sift_document_values(
    cardinality: int, n_docs: int
) -> tuple[np.ndarray, np.random.Generator]:
    rng = np.random.default_rng(42)
    return rng.integers(0, cardinality, size=n_docs), rng


def acorn_construction_metadata(
    dataset: DatasetConfig, n_docs: int, mode: str
) -> np.ndarray:
    """Return the scalar metadata used only while constructing an ACORN graph."""
    if n_docs <= 0:
        raise ValueError("n_docs must be positive")
    if mode == "dummy":
        return np.zeros(n_docs, dtype=np.int32)
    if mode != "workload_scalar":
        raise ValueError(f"unsupported ACORN metadata mode: {mode!r}")
    if dataset.name not in SIFT_EQUALITY:
        raise ValueError(
            "workload_scalar metadata is only defined for SIFT equality workloads"
        )

    values, _ = _sift_document_values(SIFT_EQUALITY[dataset.name], n_docs)
    return np.ascontiguousarray(
        values, dtype=np.int32
    )


def _laion_workload(
    name: str, path: Path, document_ids: np.ndarray, n_docs: int, query_count: int
) -> PredicateWorkload:
    mode = LAION_MODES[name]
    query_terms = np.atleast_1d(np.loadtxt(LAION_QUERIES, dtype=np.int64))[
        :query_count
    ]
    if query_terms.size != query_count:
        raise ValueError("LAION query-term list is too short")

    index = _load_index(path, HybridCategoricalIndex, n_docs)
    if index is None:
        terms_per_document = 6 if mode == "alternating" else 3
        document_terms = np.empty((n_docs, terms_per_document), dtype=np.int64)
        with LAION_KEYWORDS.open() as source:
            first = json.loads(next(source))
            vocabulary = first["top_30_keywords"]
            if len(vocabulary) != LAION_TERM_COUNT:
                raise ValueError("LAION keyword vocabulary must contain 30 terms")
            term_ids = {term: i for i, term in enumerate(vocabulary)}
            records = chain([first], (json.loads(line) for line in source))
            seen = 0
            for document, record in enumerate(records):
                if document >= n_docs:
                    break
                ranking = record["top_30_keywords"]
                top = [term_ids[term] for term in ranking[:3]]
                bottom = [term_ids[term] for term in ranking[-3:]]
                if mode == "positive":
                    document_terms[document] = top
                elif mode == "negative":
                    document_terms[document] = bottom
                else:
                    document_terms[document, :3] = top
                    document_terms[document, 3:] = np.add(bottom, len(vocabulary))
                seen += 1
        if seen != n_docs:
            raise ValueError("LAION keyword source is too short")
        offsets = np.arange(
            0, document_terms.size + 1, terms_per_document, dtype=np.int64
        )
        index = _save_index(
            HybridCategoricalIndex.build(
                offsets, document_terms.ravel(), document_ids=document_ids
            ),
            path,
        )

    if mode == "alternating":
        query_terms = query_terms.copy()
        query_terms[1::2] += LAION_TERM_COUNT
    return PredicateWorkload(index, "equality", query_terms)


def _read_sparse_matrix(path: Path, rows: int) -> tuple[np.ndarray, np.ndarray]:
    n_rows, _, nnz = map(int, np.fromfile(path, dtype=np.int64, count=3))
    if not 0 < rows <= n_rows:
        raise ValueError(f"requested {rows} rows from {n_rows}-row sparse matrix")
    indptr = np.memmap(path, mode="r", dtype=np.int64, offset=24, shape=(n_rows + 1,))
    indices = np.memmap(
        path,
        mode="r",
        dtype=np.int32,
        offset=24 + (n_rows + 1) * 8,
        shape=(nnz,),
    )
    return np.asarray(indptr[: rows + 1]), np.asarray(indices[: int(indptr[rows])])


def _yfcc_workload(
    name: str, path: Path, document_ids: np.ndarray, n_docs: int, query_count: int
) -> PredicateWorkload:
    index = _load_index(path, HybridCategoricalIndex, n_docs)
    if index is None:
        document_offsets, document_terms = _read_sparse_matrix(YFCC_DOCUMENTS, n_docs)
        index = _save_index(
            HybridCategoricalIndex.build(
                document_offsets, document_terms, document_ids=document_ids
            ),
            path,
        )

    metadata_rows = int(np.fromfile(YFCC_QUERIES, dtype=np.int64, count=1)[0])
    query_offsets, query_terms = _read_sparse_matrix(YFCC_QUERIES, metadata_rows)
    if name == "yfcc_single":
        rows = np.flatnonzero(np.diff(query_offsets) == 1)
        if rows.size < query_count:
            raise ValueError("YFCC query pool has too few single-tag queries")
        return PredicateWorkload(
            index, "equality", query_terms[query_offsets[rows[:query_count]]]
        )
    if query_count >= query_offsets.size:
        raise ValueError("YFCC metadata has too few query rows")
    stop = int(query_offsets[query_count])
    return PredicateWorkload(
        index,
        "conjunction",
        np.asarray(query_offsets[: query_count + 1]),
        np.asarray(query_terms[:stop]),
    )


def load_or_build_predicate_workload(
    dataset: DatasetConfig,
    directory: str | Path,
    document_ids: np.ndarray,
    n_docs: int,
    query_count: int,
    *,
    mask_format: MaskFormat = "packed",
    query_start: int = 0,
) -> PredicateWorkload:
    if query_count <= 0 or query_start < 0:
        raise ValueError("query_count must be positive and query_start nonnegative")
    requested_count = query_count
    query_count += query_start
    if mask_format not in {"packed", "byte"}:
        raise ValueError(f"unsupported mask format: {mask_format!r}")
    path = Path(directory) / "index"
    if dataset.name in SIFT_EQUALITY or dataset.name in SIFT_RANGES:
        workload = _sift_workload(
            dataset.name, path, document_ids, n_docs, query_count
        )
    elif dataset.name in LAION_MODES:
        workload = _laion_workload(
            dataset.name, path, document_ids, n_docs, query_count
        )
    elif dataset.name in {"yfcc", "yfcc_single"}:
        workload = _yfcc_workload(
            dataset.name, path, document_ids, n_docs, query_count
        )
    else:
        raise ValueError(f"no published predicate workload for {dataset.name!r}")
    if query_start:
        workload = workload.slice_queries(query_start, requested_count)
    if isinstance(workload.index, HybridCategoricalIndex):
        workload.index.set_mask_format(mask_format)
    elif mask_format != "packed":
        # Range indexes return postings in either mode; their materializer owns
        # the final representation.
        assert isinstance(workload.index, SortedRangeIndex)
    return workload


__all__ = [
    "PredicateWorkload",
    "acorn_construction_metadata",
    "load_or_build_predicate_workload",
]
