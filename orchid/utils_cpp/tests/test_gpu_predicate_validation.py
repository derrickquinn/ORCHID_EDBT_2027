"""Validate GPU bitmap buffer dimensions before launching the CUDA kernel."""

import unittest

import numpy as np

from utils_cpp import gpu


class GpuPredicateValidationTest(unittest.TestCase):
    def test_span_size_counts_elements(self):
        for dtype in (np.uint8, np.uint32, np.int32, np.int64, np.float32):
            for values in (np.empty(0, dtype), np.arange(6, dtype=dtype).reshape(2, 3),
                           [np.arange(3, dtype=dtype), np.arange(2, dtype=dtype)]):
                with self.subTest(dtype=dtype, values=values):
                    expected = sum(value.size for value in values) if isinstance(values, list) else values.size
                    try:
                        span = gpu.gpu_prep(values)
                        self.assertEqual(span.size, expected)
                        promise = gpu.upload(span)
                    except RuntimeError as error:
                        if "GPU backend not compiled" in str(error):
                            self.skipTest("GPU backend not compiled")
                        raise
                    self.assertEqual(span.size, expected)

    def uploaded_inputs(self, **changes):
        values = dict(
            posting_lists=np.array([1, 3], dtype=np.uint32),
            posting_sizes=np.array([2], dtype=np.int32),
            posting_queries=np.array([0], dtype=np.int32),
            dense_masks=np.array([1, 0, 4], dtype=np.uint8),
            dense_mask_queries=np.array([1], dtype=np.int64),
        )
        values.update(changes)
        mask_queries = values.pop("dense_mask_queries")
        try:
            spans = [gpu.gpu_prep(value) for value in values.values()]
            promise = gpu.upload(*spans)
        except RuntimeError as error:
            if "GPU backend not compiled" in str(error):
                self.skipTest("GPU backend not compiled")
            raise
        return [*spans, mask_queries], promise

    def test_rejects_inconsistent_span_sizes(self):
        cases = [
            ({}, -1, 19, "n_queries must be nonnegative"),
            ({}, 2, 0, "n_docs must be positive"),
            ({}, 2, -1, "n_docs must be positive"),
            ({"posting_sizes": np.array([], dtype=np.int32)}, 2, 19, "same number of elements"),
            ({}, 1, 19, "total n_queries"),
            ({}, 3, 19, "total n_queries"),
            ({"dense_masks": np.array([1, 0], dtype=np.uint8)}, 2, 19, "bytes per dense query"),
            ({"dense_masks": np.array([1, 0, 4, 0], dtype=np.uint8)}, 2, 19, "bytes per dense query"),
            ({"posting_sizes": np.array([], dtype=np.int32),
              "posting_queries": np.array([], dtype=np.int32)}, 1, 19, "posting_lists must be empty"),
            ({"dense_mask_queries": np.array([], dtype=np.int64)}, 2, 19, "total n_queries"),
            ({"dense_mask_queries": np.array([-1], dtype=np.int64)}, 2, 19, "query outside"),
            ({"dense_mask_queries": np.array([2], dtype=np.int64)}, 2, 19, "query outside"),
            ({"dense_mask_queries": np.array([[1]], dtype=np.int64)}, 2, 19, "one-dimensional"),
        ]
        for changes, n_queries, n_docs, message in cases:
            with self.subTest(changes=changes, n_queries=n_queries, n_docs=n_docs):
                spans, promise = self.uploaded_inputs(**changes)
                with self.assertRaisesRegex(ValueError, message):
                    gpu.predicate_results_to_bitmap(*spans, n_queries, n_docs)


if __name__ == "__main__":
    unittest.main()
