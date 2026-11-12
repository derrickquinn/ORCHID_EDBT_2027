"""Regression coverage for packed GPU posting-list and dense-mask batches."""

import gc
import unittest

import numpy as np

from utils_cpp import gpu


class GpuPredicateBitmapTest(unittest.IsolatedAsyncioTestCase):
    def upload(self, values, separate_uploads=False):
        try:
            spans = [gpu.gpu_prep(value) for value in values]
            if separate_uploads:
                for span in spans:
                    gpu.upload(span)
            else:
                gpu.upload(*spans)
        except RuntimeError as error:
            if "GPU backend not compiled" in str(error):
                self.skipTest("GPU backend not compiled")
            raise
        return spans

    async def check_batch(self, postings, posting_queries, masks, mask_queries, n_docs,
                    separate_uploads=False):
        n_queries = len(postings) + len(masks)
        width = (n_docs + 7) // 8
        values = [
            np.concatenate(postings).astype(np.uint32) if postings else np.empty(0, np.uint32),
            np.array([len(ids) for ids in postings], dtype=np.int32),
            np.array(posting_queries, dtype=np.int32),
            np.concatenate(masks).astype(np.uint8) if masks else np.empty(0, np.uint8),
        ]
        expected = np.zeros((n_queries, width), dtype=np.uint8)
        for ids, query in zip(postings, posting_queries):
            dense = np.zeros(n_docs, dtype=np.uint8)
            dense[np.asarray(ids, dtype=np.intp)] = 1
            expected[query] = np.packbits(dense, bitorder="little")
        for mask, query in zip(masks, mask_queries):
            expected[query] = mask
        spans = self.upload(values, separate_uploads)
        # Reusing the exact metadata catches accidental in-place prefix scans.
        for _ in range(2):
            result = gpu.predicate_results_to_bitmap(
                *spans, np.asarray(mask_queries, dtype=np.int64), n_queries, n_docs)
            actual = await result.collect_to_cpu()
            self.assertEqual(actual.size, n_queries * width)
            np.testing.assert_array_equal(actual.reshape(n_queries, width), expected)
        np.testing.assert_array_equal(await spans[1].collect_to_cpu(), values[1])

    async def test_mixed_reordered_queries_with_unaligned_rows(self):
        await self.check_batch(
            [[0, 7, 8, 18] * 300, [], [1, 2, 3, 4, 5, 6, 17] * 100],
            [3, 0, 2], [np.array([0x81, 0x40, 0x04], np.uint8)], [1], 19,
            separate_uploads=True,
        )

    async def test_posting_only_word_and_block_boundaries(self):
        for n_docs in [1, 7, 8, 9, 31, 32, 33, 257, 4097]:
            with self.subTest(n_docs=n_docs):
                await self.check_batch(
                    [np.arange(n_docs, dtype=np.uint32)[::-1], [], [0, n_docs - 1]],
                    [2, 1, 0], [], [], n_docs,
                )

    async def test_dense_only_preserves_bytes_and_query_order(self):
        await self.check_batch([], [], [np.array([255, 3, 0], np.uint8),
                                 np.array([0, 0, 255], np.uint8)], [1, 0], 19)

    async def test_posting_chunk_and_query_boundaries(self):
        sizes = [0, 1, 127, 128, 129, 0, 255, 256, 257, 0]
        postings = [(np.arange(size, dtype=np.uint32) * 7 + query) % 4097
                    for query, size in enumerate(sizes)]
        await self.check_batch(postings, list(reversed(range(len(sizes)))), [], [], 4097)

    async def test_single_posting_list_spans_blocks(self):
        await self.check_batch([np.arange(65537, dtype=np.uint32)], [0], [], [], 65537)

    async def test_empty_batch(self):
        await self.check_batch([], [], [], [], 19)

    async def test_many_queries_with_empty_postings(self):
        await self.check_batch([[], [0]] * 1000, list(range(2000)), [], [], 1)

    async def test_temporary_uploads_on_same_stream(self):
        def materialize():
            spans = self.upload([
                np.arange(4097, dtype=np.uint32), np.array([4097], np.int32),
                np.array([0], np.int32), np.empty(0, np.uint8),
            ])
            return gpu.predicate_results_to_bitmap(*spans, np.empty(0, np.int64), 1, 4097)
        result = materialize()
        gc.collect()
        expected = np.packbits(np.ones(4097, np.uint8), bitorder="little")
        np.testing.assert_array_equal(await result.collect_to_cpu(), expected)


if __name__ == "__main__":
    unittest.main()
