import unittest

import numpy as np

from utils_cpp import ids_to_bitmap


class IdsToBitmapTest(unittest.TestCase):
    def test_matches_numpy_packbits(self) -> None:
        n_docs = 19
        offsets = np.array([0, 4, 4, 8], dtype=np.int64)
        ids = np.array([0, 7, 8, 18, 1, 1, 9, 17], dtype=np.uint32)

        dense = np.zeros((3, n_docs), dtype=np.uint8)
        for q in range(3):
            dense[q, ids[offsets[q] : offsets[q + 1]]] = 1
        expected = np.packbits(dense, axis=1, bitorder="little")

        actual = ids_to_bitmap(offsets, ids, n_docs, threads=2)
        np.testing.assert_array_equal(actual, expected)

    def test_clear_false_unions_with_existing_bitmap(self) -> None:
        n_docs = 16
        out = np.zeros((2, 2), dtype=np.uint8)
        ids_to_bitmap(
            np.array([0, 1, 2]),
            np.array([0, 8], dtype=np.uint32),
            n_docs,
            out=out,
        )
        ids_to_bitmap(
            np.array([0, 1, 2]),
            np.array([7, 15], dtype=np.uint32),
            n_docs,
            out=out,
            clear=False,
        )
        np.testing.assert_array_equal(out, np.array([[129, 0], [0, 129]], dtype=np.uint8))

    def test_rejects_out_of_range_ids(self) -> None:
        with self.assertRaisesRegex(ValueError, "outside"):
            ids_to_bitmap(np.array([0, 1]), np.array([9]), n_docs=9)


if __name__ == "__main__":
    unittest.main()
