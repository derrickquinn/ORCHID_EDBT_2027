import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

import predicate_workload
from predicate_index import (
    ByteMask,
    HybridCategoricalIndex,
    materialize_byte_masks,
    materialize_results,
)
from predicate_workload import PredicateWorkload, load_or_build_predicate_workload
from src.dataset_config import DatasetConfig


def workload_masks(workload: PredicateWorkload) -> np.ndarray:
    results = workload.evaluate_batch(0, workload.n_queries, threads=2)
    output = np.empty(
        (workload.n_queries, (workload.index.n_docs + 7) // 8), dtype=np.uint8
    )
    return materialize_results(
        results, workload.index.n_docs, output, threads=2
    )


def workload_byte_masks(workload: PredicateWorkload) -> np.ndarray:
    results = workload.evaluate_batch(0, workload.n_queries, threads=2)
    output = np.empty(
        (workload.n_queries, workload.index.n_docs), dtype=np.uint8
    )
    return materialize_byte_masks(
        results, workload.index.n_docs, output, threads=2
    )


def dataset(name: str) -> DatasetConfig:
    return DatasetConfig(name=name, vec_zarr="vectors", pred_zarr="masks", nlists=1)


def write_sparse(path: Path, rows: list[list[int]], total_rows: int) -> None:
    offsets = np.zeros(total_rows + 1, dtype=np.int64)
    offsets[1 : len(rows) + 1] = np.cumsum([len(row) for row in rows])
    offsets[len(rows) + 1 :] = offsets[len(rows)]
    indices = np.asarray([term for row in rows for term in row], dtype=np.int32)
    with path.open("wb") as output:
        np.array([total_rows, 3, indices.size], dtype=np.int64).tofile(output)
        offsets.tofile(output)
        indices.tofile(output)


class PredicateWorkloadTest(unittest.TestCase):
    def test_mixed_single_term_and_conjunction_batch(self) -> None:
        document_offsets = np.array([0, 1, 2, 4, 5, 7, 8], dtype=np.int64)
        document_terms = np.array([1, 1, 1, 2, 2, 3, 1, 3], dtype=np.int64)
        index = HybridCategoricalIndex.build(
            document_offsets, document_terms, dense_threshold=3
        )
        workload = PredicateWorkload(
            index,
            "conjunction",
            np.array([0, 1, 3, 5], dtype=np.int64),
            np.array([1, 1, 2, 1, 3], dtype=np.int64),
        )
        expected = np.array(
            [
                [True, True, True, False, True, False],
                [False, False, True, False, False, False],
                [False, False, False, False, True, False],
            ]
        )
        np.testing.assert_array_equal(
            workload_masks(workload),
            np.packbits(expected, axis=1, bitorder="little"),
        )

    def test_sift_equality_reproduces_generator_and_reloads_index(self) -> None:
        n_docs, n_queries = 20, 7
        old_to_new = np.arange(n_docs, dtype=np.uint32)[::-1].copy()
        with tempfile.TemporaryDirectory() as directory:
            workload = load_or_build_predicate_workload(
                dataset("sift12"), directory, old_to_new, n_docs, n_queries
            )
            loaded = load_or_build_predicate_workload(
                dataset("sift12"), directory, old_to_new, n_docs, n_queries
            )
            byte_loaded = load_or_build_predicate_workload(
                dataset("sift12"),
                directory,
                old_to_new,
                n_docs,
                n_queries,
                mask_format="byte",
            )

        rng = np.random.default_rng(42)
        values = rng.integers(0, 12, size=n_docs)
        queries = rng.integers(0, 12, size=n_queries)
        expected = (queries[:, None] == values[None, :])[:, ::-1]
        np.testing.assert_array_equal(
            workload_masks(workload),
            np.packbits(expected, axis=1, bitorder="little"),
        )
        np.testing.assert_array_equal(workload_masks(loaded), workload_masks(workload))
        np.testing.assert_array_equal(workload_byte_masks(byte_loaded), expected)
        self.assertTrue(
            any(
                isinstance(result, ByteMask)
                for result in byte_loaded.evaluate_batch(0, n_queries)
            )
        )

    def test_sift_range_reproduces_generator(self) -> None:
        n_docs, n_queries = 31, 6
        with tempfile.TemporaryDirectory() as directory:
            workload = load_or_build_predicate_workload(
                dataset("sift_r10"),
                directory,
                np.arange(n_docs, dtype=np.uint32),
                n_docs,
                n_queries,
            )

        rng = np.random.default_rng(42)
        values = rng.integers(0, 1000, size=n_docs)
        lower = rng.integers(0, 991, size=n_queries)
        expected = (values >= lower[:, None]) & (values < lower[:, None] + 10)
        np.testing.assert_array_equal(
            workload_masks(workload),
            np.packbits(expected, axis=1, bitorder="little"),
        )

    def test_laion_alternating_uses_top_and_bottom_fields(self) -> None:
        vocabulary = [f"term-{i}" for i in range(30)]
        rankings = [vocabulary[i:] + vocabulary[:i] for i in range(4)]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            source = path / "keywords.jsonl"
            source.write_text(
                "".join(
                    json.dumps({"top_30_keywords": ranking}) + "\n"
                    for ranking in rankings
                )
            )
            queries = path / "queries.txt"
            queries.write_text("0 1\n")
            with (
                patch.object(predicate_workload, "LAION_KEYWORDS", source),
                patch.object(predicate_workload, "LAION_QUERIES", queries),
            ):
                workload = load_or_build_predicate_workload(
                    dataset("laion_all"),
                    path / "cache",
                    np.arange(4, dtype=np.uint32),
                    4,
                    2,
                )

        expected = np.array(
            [[True, False, False, False], [False, False, True, True]]
        )
        np.testing.assert_array_equal(
            workload_masks(workload),
            np.packbits(expected, axis=1, bitorder="little"),
        )

    def test_yfcc_mixed_and_single_tag_workloads(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            documents, queries = path / "documents.spmat", path / "queries.spmat"
            write_sparse(documents, [[1], [1, 2], [2], [1, 2]], 4)
            write_sparse(queries, [[1], [1, 2], [2]], 10_000)
            with (
                patch.object(predicate_workload, "YFCC_DOCUMENTS", documents),
                patch.object(predicate_workload, "YFCC_QUERIES", queries),
            ):
                mixed = load_or_build_predicate_workload(
                    dataset("yfcc"), path / "mixed", np.arange(4), 4, 2
                )
                mixed_byte = load_or_build_predicate_workload(
                    dataset("yfcc"),
                    path / "mixed",
                    np.arange(4),
                    4,
                    2,
                    mask_format="byte",
                )
                single = load_or_build_predicate_workload(
                    dataset("yfcc_single"), path / "single", np.arange(4), 4, 2
                )

        expected_mixed = np.array(
            [[True, True, False, True], [False, True, False, True]]
        )
        expected_single = np.array(
            [[True, True, False, True], [False, True, True, True]]
        )
        np.testing.assert_array_equal(
            workload_masks(mixed),
            np.packbits(expected_mixed, axis=1, bitorder="little"),
        )
        np.testing.assert_array_equal(workload_byte_masks(mixed_byte), expected_mixed)
        np.testing.assert_array_equal(
            workload_masks(single),
            np.packbits(expected_single, axis=1, bitorder="little"),
        )


if __name__ == "__main__":
    unittest.main()
