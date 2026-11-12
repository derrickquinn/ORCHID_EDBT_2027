import unittest
from unittest.mock import Mock, patch

import numpy as np

from predicate_index import (
    ByteMask,
    PackedBitmap,
    PostingList,
    materialize_byte_masks,
    materialize_results,
    materialize_results_gpu,
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


class PredicateGpuArgumentsTest(unittest.TestCase):
    def test_wires_mixed_results_in_kernel_order(self) -> None:
        ids = np.array([1, 3, 17], dtype=np.uint32)
        bitmap = np.array([129, 0, 4], dtype=np.uint8)
        spans = tuple(object() for _ in range(4))
        api = Mock()
        api.gpu_prep.side_effect = spans
        api.predicate_results_to_bitmap.return_value = (object(), None)
        with patch("predicate_index.gpu", api):
            result = materialize_results_gpu(
                [PostingList(ids), PackedBitmap(bitmap), PostingList(np.empty(0, dtype=np.uint32))],
                19,
            )
        inputs = [call.args[0] for call in api.gpu_prep.call_args_list]
        mask_queries = api.predicate_results_to_bitmap.call_args.args[4]
        self.assertIs(inputs[0][0], ids)
        self.assertIs(inputs[3][0], bitmap)
        for actual, expected, dtype in (
            (inputs[1], [3, 0], np.int32),
            (inputs[2], [0, 2], np.int32),
            (mask_queries, [1], np.int64),
        ):
            self.assertEqual(actual.dtype, dtype)
            np.testing.assert_array_equal(actual, expected)
        api.upload.assert_called_once_with(*spans)
        api.predicate_results_to_bitmap.assert_called_once_with(*spans, mask_queries, 3, 19)
        self.assertIs(result, api.predicate_results_to_bitmap.return_value)

    def test_missing_groups_have_typed_empty_buffers(self) -> None:
        for results in ([], [PostingList(np.array([2], dtype=np.uint32))],
                        [PackedBitmap(np.array([4, 0, 0], dtype=np.uint8))]):
            with self.subTest(results=results):
                api = Mock()
                spans = tuple(object() for _ in range(4))
                api.gpu_prep.side_effect = spans
                with patch("predicate_index.gpu", api):
                    materialize_results_gpu(results, 19)
                inputs = [call.args[0] for call in api.gpu_prep.call_args_list]
                for index, dtype in ((0, np.uint32), (3, np.uint8)):
                    if isinstance(inputs[index], np.ndarray):
                        self.assertEqual(inputs[index].size, 0)
                        self.assertEqual(inputs[index].dtype, dtype)
                for index, dtype in ((1, np.int32), (2, np.int32)):
                    self.assertEqual(inputs[index].dtype, dtype)
                mask_queries = api.predicate_results_to_bitmap.call_args.args[4]
                self.assertEqual(mask_queries.dtype, np.int64)
                np.testing.assert_array_equal(
                    mask_queries, [i for i, result in enumerate(results) if isinstance(result, PackedBitmap)])
                api.upload.assert_called_once_with(*spans)
                api.predicate_results_to_bitmap.assert_called_once_with(
                    *spans, mask_queries, len(results), 19)

    def test_rejects_byte_masks_before_upload(self) -> None:
        with patch("predicate_index.gpu") as api:
            with self.assertRaises(TypeError):
                materialize_results_gpu(
                    [ByteMask(np.zeros(19, dtype=np.uint8))], 19,
                )
            api.upload.assert_not_called()


if __name__ == "__main__":
    unittest.main()
