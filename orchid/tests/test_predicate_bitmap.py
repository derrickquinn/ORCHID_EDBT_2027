import unittest

import numpy as np

from predicate_index import (
    ByteMask,
    PackedBitmap,
    PostingList,
    materialize_byte_masks,
    materialize_results,
)


class PredicateBitmapTest(unittest.TestCase):
    def test_materializes_mixed_results_and_reuses_output(self) -> None:
        output = np.empty((4, 3), dtype=np.uint8)
        dense = np.packbits(
            np.isin(np.arange(19), [0, 7, 18]), bitorder="little"
        )
        first = materialize_results(
            [
                PostingList(np.array([1, 3, 17], dtype=np.uint32)),
                PackedBitmap(dense),
                PostingList(np.empty(0, dtype=np.uint32)),
            ],
            19,
            output,
            threads=2,
        )
        expected = np.zeros((3, 19), dtype=np.bool_)
        expected[0, [1, 3, 17]] = True
        expected[1, [0, 7, 18]] = True
        np.testing.assert_array_equal(
            first, np.packbits(expected, axis=1, bitorder="little")
        )

        second = materialize_results(
            [PostingList(np.array([2], dtype=np.uint32))], 19, output, threads=2
        )
        expected_second = np.zeros((1, 19), dtype=np.bool_)
        expected_second[0, 2] = True
        np.testing.assert_array_equal(
            second, np.packbits(expected_second, axis=1, bitorder="little")
        )
        self.assertTrue(np.shares_memory(second, output))

    def test_rejects_wrong_bitmap_width(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "wrong width"):
            materialize_results(
                [PackedBitmap(np.zeros(2, dtype=np.uint8))],
                17,
                np.empty((1, 3), dtype=np.uint8),
            )

    def test_materializes_byte_masks_and_reuses_output(self) -> None:
        output = np.empty((4, 19), dtype=np.uint8)
        dense_values = np.isin(np.arange(19), [0, 7, 8, 18])
        actual = materialize_byte_masks(
            [
                PostingList(np.array([1, 3, 17], dtype=np.uint32)),
                ByteMask(dense_values.astype(np.uint8)),
                PostingList(np.empty(0, dtype=np.uint32)),
            ],
            19,
            output,
            threads=2,
        )
        expected = np.zeros((3, 19), dtype=np.uint8)
        expected[0, [1, 3, 17]] = 1
        expected[1] = dense_values
        np.testing.assert_array_equal(actual, expected)
        self.assertTrue(np.shares_memory(actual, output))

        materialize_byte_masks(
            [PostingList(np.array([2], dtype=np.uint32))],
            19,
            output,
            threads=2,
        )
        np.testing.assert_array_equal(output[0], np.eye(1, 19, 2, dtype=np.uint8)[0])

    def test_byte_mask_rejects_noncontiguous_output(self) -> None:
        output = np.empty((1, 38), dtype=np.uint8)[:, ::2]
        with self.assertRaisesRegex(ValueError, "C-contiguous uint8"):
            materialize_byte_masks(
                [PostingList(np.array([1], dtype=np.uint32))], 19, output
            )

    def test_byte_materializer_rejects_packed_dense_results(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "either postings or a dense"):
            materialize_byte_masks(
                [PackedBitmap(np.zeros(3, dtype=np.uint8))],
                19,
                np.empty((1, 19), dtype=np.uint8),
            )


if __name__ == "__main__":
    unittest.main()
