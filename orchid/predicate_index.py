"""Compact, query-independent predicate indexes.

Indexes are built offline over cluster-major document IDs. Online lookups
return either uint32 document IDs or an existing dense mask in the configured
representation; materializing those results is deliberately left to the next
pipeline stage.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
from utils_cpp import (
    evaluate_predicate_conjunctions,
    predicate_results_to_bitmap,
    predicate_results_to_mask,
)


_VERSION = 1
_EMPTY_IDS = np.empty(0, dtype=np.uint32)


@dataclass(frozen=True, slots=True)
class PostingList:
    ids: np.ndarray


@dataclass(frozen=True, slots=True)
class PackedBitmap:
    bits: np.ndarray


@dataclass(frozen=True, slots=True)
class ByteMask:
    bytes: np.ndarray


MaskFormat = Literal["packed", "byte"]
PredicateResult = PostingList | PackedBitmap | ByteMask


def _readonly(array: np.ndarray) -> np.ndarray:
    array.flags.writeable = False
    return array


class HybridCategoricalIndex:
    """Integer-coded terms stored as postings or packed bitmaps."""

    _ARRAYS = (
        "keys",
        "counts",
        "sparse_offsets",
        "postings",
        "dense_rows",
        "bitmaps",
    )

    def __init__(
        self,
        n_docs: int,
        keys: np.ndarray,
        counts: np.ndarray,
        sparse_offsets: np.ndarray,
        postings: np.ndarray,
        dense_rows: np.ndarray,
        bitmaps: np.ndarray,
        dense_threshold: int,
    ) -> None:
        self.n_docs = int(n_docs)
        self.keys = keys
        self.counts = counts
        self.sparse_offsets = sparse_offsets
        self.postings = postings
        self.dense_rows = dense_rows
        self.bitmaps = bitmaps
        self.dense_threshold = int(dense_threshold)
        self.mask_format: MaskFormat = "packed"
        self.byte_masks: np.ndarray | None = None

    def set_mask_format(self, mask_format: MaskFormat) -> None:
        if mask_format not in {"packed", "byte"}:
            raise ValueError(f"unsupported mask format: {mask_format!r}")
        self.mask_format = mask_format
        self.byte_masks = (
            _readonly(
                np.ascontiguousarray(
                    np.unpackbits(
                        self.bitmaps,
                        axis=1,
                        count=self.n_docs,
                        bitorder="little",
                    )
                )
            )
            if mask_format == "byte"
            else None
        )

    @classmethod
    def build(
        cls,
        document_offsets: np.ndarray,
        terms: np.ndarray,
        *,
        document_ids: np.ndarray | None = None,
        dense_threshold: int | None = None,
    ) -> HybridCategoricalIndex:
        """Build from CSR document-to-term data.

        ``document_ids`` maps input rows to offline cluster-major IDs. Repeated
        terms within one document have set semantics.
        """

        offsets = np.asarray(document_offsets, dtype=np.int64)
        terms = np.asarray(terms, dtype=np.int64)
        if (
            offsets.ndim != 1
            or offsets.size < 2
            or terms.ndim != 1
            or offsets[0] != 0
            or offsets[-1] != terms.size
            or np.any(offsets[1:] < offsets[:-1])
        ):
            raise ValueError("invalid CSR term data")

        n_docs = offsets.size - 1
        if n_docs > np.iinfo(np.uint32).max:
            raise ValueError("uint32 postings cannot represent n_docs")
        if document_ids is None:
            row_ids = np.arange(n_docs, dtype=np.uint32)
        else:
            row_ids = np.asarray(document_ids)
            if (
                row_ids.shape != (n_docs,)
                or row_ids.dtype.kind not in "iu"
                or (row_ids.size and (row_ids.min() < 0 or row_ids.max() >= n_docs))
            ):
                raise ValueError("document_ids must contain one valid ID per row")
            row_ids = row_ids.astype(np.uint32, copy=False)

        threshold = (
            max(1, (((n_docs + 7) // 8) + 3) // 4)
            if dense_threshold is None
            else int(dense_threshold)
        )
        if threshold <= 0:
            raise ValueError("dense_threshold must be positive")

        docs = np.repeat(row_ids, np.diff(offsets))
        order = np.argsort(terms, kind="stable")
        sorted_terms, docs = terms[order], docs[order]
        if sorted_terms.size > 1:
            unique = np.r_[
                True,
                (sorted_terms[1:] != sorted_terms[:-1])
                | (docs[1:] != docs[:-1]),
            ]
            sorted_terms, docs = sorted_terms[unique], docs[unique]

        keys, starts, counts = np.unique(
            sorted_terms, return_index=True, return_counts=True
        )
        counts = counts.astype(np.int64, copy=False)
        dense = counts >= threshold
        dense_rows = np.full(keys.size, -1, dtype=np.int32)
        dense_rows[dense] = np.arange(dense.sum(), dtype=np.int32)
        sparse_offsets = np.r_[0, np.cumsum(np.where(dense, 0, counts))].astype(
            np.int64, copy=False
        )
        postings = np.empty(sparse_offsets[-1], dtype=np.uint32)
        bitmaps = np.zeros((dense.sum(), (n_docs + 7) // 8), dtype=np.uint8)

        for key, (start, count) in enumerate(zip(starts, counts)):
            ids = docs[start : start + count]
            row = dense_rows[key]
            if row >= 0:
                bitmap = np.zeros(n_docs, dtype=np.bool_)
                bitmap[ids] = True
                bitmaps[row] = np.packbits(bitmap, bitorder="little")
            else:
                begin, end = sparse_offsets[key : key + 2]
                postings[begin:end] = np.sort(ids)

        return cls(
            n_docs,
            _readonly(np.ascontiguousarray(keys, dtype=np.int64)),
            _readonly(np.ascontiguousarray(counts)),
            _readonly(sparse_offsets),
            _readonly(postings),
            _readonly(dense_rows),
            _readonly(bitmaps),
            threshold,
        )

    @classmethod
    def build_scalar(
        cls,
        values: np.ndarray,
        *,
        document_ids: np.ndarray | None = None,
        dense_threshold: int | None = None,
    ) -> HybridCategoricalIndex:
        values = np.asarray(values)
        if values.ndim != 1:
            raise ValueError("values must be one-dimensional")
        return cls.build(
            np.arange(values.size + 1, dtype=np.int64),
            values,
            document_ids=document_ids,
            dense_threshold=dense_threshold,
        )

    def equal_batch(self, values: np.ndarray) -> list[PredicateResult]:
        values = np.asarray(values, dtype=np.int64)
        if values.ndim != 1:
            raise ValueError("values must be one-dimensional")
        indices = np.searchsorted(self.keys, values)
        found = indices < self.keys.size
        found[found] &= self.keys[indices[found]] == values[found]

        results: list[PredicateResult] = []
        for index, present in zip(indices, found):
            if not present:
                results.append(PostingList(_EMPTY_IDS))
            elif self.dense_rows[index] >= 0:
                row = self.dense_rows[index]
                if self.mask_format == "packed":
                    results.append(PackedBitmap(self.bitmaps[row]))
                else:
                    assert self.byte_masks is not None
                    results.append(ByteMask(self.byte_masks[row]))
            else:
                begin, end = self.sparse_offsets[index : index + 2]
                results.append(PostingList(self.postings[begin:end]))
        return results

    def conjunction_batch(
        self,
        query_offsets: np.ndarray,
        query_terms: np.ndarray,
        *,
        threads: int = 0,
    ) -> list[PredicateResult]:
        packed = self.mask_format == "packed"
        dense_masks = self.bitmaps if packed else self.byte_masks
        assert dense_masks is not None
        kinds, offsets, ids, rows, result_masks = evaluate_predicate_conjunctions(
            self.keys,
            self.counts,
            self.sparse_offsets,
            self.postings,
            self.dense_rows,
            dense_masks,
            self.n_docs,
            query_offsets,
            query_terms,
            packed=packed,
            threads=threads,
        )
        results: list[PredicateResult] = []
        for q, kind in enumerate(kinds):
            if kind != 1:
                results.append(PostingList(ids[offsets[q] : offsets[q + 1]]))
            elif packed:
                results.append(PackedBitmap(result_masks[rows[q]]))
            else:
                results.append(ByteMask(result_masks[rows[q]]))
        return results

    def save(self, directory: str | Path) -> None:
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        metadata = {
            "version": _VERSION,
            "n_docs": self.n_docs,
            "dense_threshold": self.dense_threshold,
        }
        (path / "index.json").write_text(json.dumps(metadata) + "\n")
        for name in self._ARRAYS:
            np.save(path / f"{name}.npy", getattr(self, name), allow_pickle=False)

    @classmethod
    def load(
        cls, directory: str | Path, *, mmap_mode: str | None = "r"
    ) -> HybridCategoricalIndex:
        path = Path(directory)
        metadata = json.loads((path / "index.json").read_text())
        if metadata.get("version") != _VERSION:
            raise ValueError("unsupported predicate-index version")
        arrays = [
            np.load(path / f"{name}.npy", mmap_mode=mmap_mode, allow_pickle=False)
            for name in cls._ARRAYS
        ]
        return cls(metadata["n_docs"], *arrays, metadata["dense_threshold"])


class SortedRangeIndex:
    """A scalar index sorted by ``(value, cluster-major document ID)``."""

    def __init__(
        self, n_docs: int, values: np.ndarray, document_ids: np.ndarray
    ) -> None:
        self.n_docs = int(n_docs)
        self.values = values
        self.document_ids = document_ids

    @classmethod
    def build(
        cls, values: np.ndarray, *, document_ids: np.ndarray | None = None
    ) -> SortedRangeIndex:
        values = np.asarray(values)
        if values.ndim != 1 or values.dtype.kind not in "iuIf" or not values.size:
            raise ValueError("values must be a nonempty numeric vector")
        n_docs = values.size
        if n_docs > np.iinfo(np.uint32).max:
            raise ValueError("uint32 postings cannot represent n_docs")

        if document_ids is None:
            ids = np.arange(n_docs, dtype=np.uint32)
        else:
            ids = np.asarray(document_ids)
            if (
                ids.shape != (n_docs,)
                or ids.dtype.kind not in "iu"
                or ids.min() < 0
                or ids.max() >= n_docs
            ):
                raise ValueError("document_ids must contain one valid ID per value")
            ids = ids.astype(np.uint32, copy=False)
        order = np.lexsort((ids, values))
        return cls(
            n_docs,
            _readonly(np.ascontiguousarray(values[order])),
            _readonly(np.ascontiguousarray(ids[order])),
        )

    def equal_batch(self, values: np.ndarray) -> list[PredicateResult]:
        values = np.asarray(values)
        starts = np.searchsorted(self.values, values, side="left")
        ends = np.searchsorted(self.values, values, side="right")
        return [
            PostingList(self.document_ids[start:end])
            for start, end in zip(starts, ends)
        ]

    def range_batch(
        self, lower: np.ndarray, upper: np.ndarray
    ) -> list[PredicateResult]:
        lower, upper = np.asarray(lower), np.asarray(upper)
        if lower.ndim != 1 or upper.shape != lower.shape or np.any(lower >= upper):
            raise ValueError("range bounds must be valid same-length vectors")
        starts = np.searchsorted(self.values, lower, side="left")
        ends = np.searchsorted(self.values, upper, side="left")
        return [
            PostingList(self.document_ids[start:end])
            for start, end in zip(starts, ends)
        ]

    def save(self, directory: str | Path) -> None:
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        (path / "index.json").write_text(
            json.dumps({"version": _VERSION, "n_docs": self.n_docs}) + "\n"
        )
        np.save(path / "values.npy", self.values, allow_pickle=False)
        np.save(path / "document_ids.npy", self.document_ids, allow_pickle=False)

    @classmethod
    def load(
        cls, directory: str | Path, *, mmap_mode: str | None = "r"
    ) -> SortedRangeIndex:
        path = Path(directory)
        metadata = json.loads((path / "index.json").read_text())
        if metadata.get("version") != _VERSION:
            raise ValueError("unsupported predicate-index version")
        values = np.load(path / "values.npy", mmap_mode=mmap_mode, allow_pickle=False)
        document_ids = np.load(
            path / "document_ids.npy", mmap_mode=mmap_mode, allow_pickle=False
        )
        return cls(metadata["n_docs"], values, document_ids)


def materialize_results(
    results: list[PredicateResult],
    n_docs: int,
    out: np.ndarray,
    *,
    threads: int = 0,
) -> np.ndarray:
    """Materialize postings or existing bitmaps into a reusable packed batch."""
    return predicate_results_to_bitmap(
        [result.ids if isinstance(result, PostingList) else None for result in results],
        [result.bits if isinstance(result, PackedBitmap) else None for result in results],
        n_docs,
        out=out[: len(results)],
        threads=threads,
    )


def materialize_byte_masks(
    results: list[PredicateResult],
    n_docs: int,
    out: np.ndarray,
    *,
    threads: int = 0,
) -> np.ndarray:
    """Materialize native byte results into a reusable byte-mask batch."""
    return predicate_results_to_mask(
        [result.ids if isinstance(result, PostingList) else None for result in results],
        [
            result.bytes if isinstance(result, ByteMask) else None
            for result in results
        ],
        n_docs,
        out=out[: len(results)],
        threads=threads,
    )


__all__ = [
    "ByteMask",
    "HybridCategoricalIndex",
    "MaskFormat",
    "PackedBitmap",
    "PostingList",
    "PredicateResult",
    "SortedRangeIndex",
    "materialize_results",
    "materialize_byte_masks",
]
