import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from predicate_index import (
    ByteMask,
    HybridCategoricalIndex,
    PackedBitmap,
    PostingList,
    SortedRangeIndex,
)


def result_ids(
    result: PostingList | PackedBitmap | ByteMask, n_docs: int
) -> np.ndarray:
    if isinstance(result, PostingList):
        return np.sort(result.ids)
    if isinstance(result, ByteMask):
        return np.flatnonzero(result.bytes)
    return np.flatnonzero(
        np.unpackbits(result.bits, bitorder="little", count=n_docs)
    )


class HybridCategoricalIndexTest(unittest.TestCase):
    def test_dataset_driven_sparse_dense_split_and_batch_lookup(self) -> None:
        # Five bitmap bytes versus four bytes per posting gives a threshold of 2.
        values = np.array([7, 7, 2, 7, *range(10, 46)], dtype=np.int64)
        index = HybridCategoricalIndex.build_scalar(values)
        dense, sparse, missing = index.equal_batch(np.array([7, 2, 1_000]))

        self.assertEqual(index.dense_threshold, 2)
        self.assertIsInstance(dense, PackedBitmap)
        self.assertIsInstance(sparse, PostingList)
        np.testing.assert_array_equal(result_ids(dense, index.n_docs), [0, 1, 3])
        np.testing.assert_array_equal(result_ids(sparse, index.n_docs), [2])
        self.assertEqual(result_ids(missing, index.n_docs).size, 0)

    def test_byte_mode_preserves_batch_result_boundary(self) -> None:
        values = np.array([7, 7, 2, 7, *range(10, 46)], dtype=np.int64)
        index = HybridCategoricalIndex.build_scalar(values)
        index.set_mask_format("byte")

        dense, sparse, missing = index.equal_batch(np.array([7, 2, 1_000]))

        self.assertIsInstance(dense, ByteMask)
        self.assertIsInstance(sparse, PostingList)
        self.assertIsInstance(missing, PostingList)
        np.testing.assert_array_equal(result_ids(dense, index.n_docs), [0, 1, 3])
        np.testing.assert_array_equal(result_ids(sparse, index.n_docs), [2])

    def test_multivalued_set_semantics_and_native_conjunctions(self) -> None:
        offsets = np.array([0, 3, 4, 7, 7, 9], dtype=np.int64)
        terms = np.array([4, 1, 4, 2, 1, 3, 4, 2, 4], dtype=np.int64)
        index = HybridCategoricalIndex.build(offsets, terms, dense_threshold=10)

        for actual, expected in zip(
            index.equal_batch(np.array([1, 2, 3, 4])),
            ([0, 2], [1, 4], [2], [0, 2, 4]),
        ):
            np.testing.assert_array_equal(result_ids(actual, index.n_docs), expected)

        query_offsets = np.array([0, 2, 4, 6], dtype=np.int64)
        query_terms = np.array([1, 4, 2, 4, 1, 99], dtype=np.int64)
        for actual, expected in zip(
            index.conjunction_batch(query_offsets, query_terms, threads=2),
            ([0, 2], [4], []),
        ):
            np.testing.assert_array_equal(result_ids(actual, index.n_docs), expected)

        dense_index = HybridCategoricalIndex.build(
            offsets, terms, dense_threshold=2
        )
        for actual, expected in zip(
            dense_index.conjunction_batch(query_offsets, query_terms, threads=2),
            ([0, 2], [4], []),
        ):
            np.testing.assert_array_equal(
                result_ids(actual, dense_index.n_docs), expected
            )

        dense_index.set_mask_format("byte")
        with patch("predicate_index.np.unpackbits", side_effect=AssertionError):
            byte_results = dense_index.conjunction_batch(
                query_offsets, query_terms, threads=2
            )
        self.assertIsInstance(byte_results[0], ByteMask)
        self.assertIsInstance(byte_results[1], ByteMask)
        for actual, expected in zip(byte_results, ([0, 2], [4], [])):
            np.testing.assert_array_equal(
                result_ids(actual, dense_index.n_docs), expected
            )

    def test_cluster_major_ids_and_persistence(self) -> None:
        internal_to_external = np.array([1, 4, 2, 0, 3, 5])
        external_to_internal = np.empty(6, dtype=np.uint32)
        external_to_internal[internal_to_external] = np.arange(6)
        index = HybridCategoricalIndex.build_scalar(
            np.array([7, 8, 7, 9, 8, 7]),
            document_ids=external_to_internal,
            dense_threshold=10,
        )

        with tempfile.TemporaryDirectory() as directory:
            index.save(Path(directory) / "tags")
            loaded = HybridCategoricalIndex.load(Path(directory) / "tags")
            result = loaded.equal_batch(np.array([7]))[0]
            assert isinstance(result, PostingList)
            selected_external = internal_to_external[result.ids]
            np.testing.assert_array_equal(np.sort(selected_external), [0, 2, 5])


class SortedRangeIndexTest(unittest.TestCase):
    def test_batched_equal_range_and_persistence(self) -> None:
        index = SortedRangeIndex.build(np.array([5, 1, 3, 1, 4, 2]))
        equal = index.equal_batch(np.array([1]))[0]
        ranges = index.range_batch(np.array([2, 4]), np.array([5, 6]))

        np.testing.assert_array_equal(result_ids(equal, index.n_docs), [1, 3])
        np.testing.assert_array_equal(result_ids(ranges[0], index.n_docs), [2, 4, 5])
        np.testing.assert_array_equal(result_ids(ranges[1], index.n_docs), [0, 4])

        with tempfile.TemporaryDirectory() as directory:
            index.save(Path(directory) / "ranges")
            loaded = SortedRangeIndex.load(Path(directory) / "ranges")
            actual = loaded.range_batch(np.array([2]), np.array([5]))[0]
            np.testing.assert_array_equal(result_ids(actual, index.n_docs), [2, 4, 5])


if __name__ == "__main__":
    unittest.main()
