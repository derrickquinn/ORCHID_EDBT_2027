import unittest

import numpy as np
import torch

from utils_cpp import topk_small


class TopKSmallTest(unittest.TestCase):
    def test_matches_torch_for_unique_scores(self) -> None:
        rng = np.random.default_rng(20260825)
        for rows, width, k in (
            (1, 1, 1),
            (3, 7, 2),
            (5, 65, 4),
            (100, 4096, 4),
            (7, 129, 32),
            (2, 64, 64),
            (5, 257, 65),
            (3, 4096, 128),
            (2, 4096, 1024),
        ):
            scores = rng.standard_normal((rows, width), dtype=np.float32)
            # Make equality impossible even if the generator happens to repeat.
            scores += np.arange(width, dtype=np.float32) * np.float32(1e-5)
            expected = torch.topk(
                torch.from_numpy(scores), k=k, dim=1, sorted=True
            ).indices.numpy()
            actual = topk_small(scores, k, threads=4)
            np.testing.assert_array_equal(actual, expected)

    def test_ties_are_ordered_by_ascending_column(self) -> None:
        scores = np.array(
            [[1.0, 3.0, 3.0, 2.0], [5.0, 5.0, 4.0, 5.0]],
            dtype=np.float32,
        )
        actual = topk_small(scores, 3, threads=2)
        np.testing.assert_array_equal(
            actual,
            np.array([[1, 2, 3], [0, 1, 3]], dtype=np.int64),
        )

    def test_reuses_output_and_validates_inputs(self) -> None:
        scores = np.arange(24, dtype=np.float32).reshape(3, 8)
        out = np.empty((3, 4), dtype=np.int64)
        self.assertIs(topk_small(scores, 4, out=out, threads=1), out)
        with self.assertRaises(ValueError):
            topk_small(scores.astype(np.float64), 4)
        with self.assertRaises(ValueError):
            topk_small(scores[:, ::2], 4)
        with self.assertRaises(ValueError):
            topk_small(scores, 0)
        with self.assertRaises(ValueError):
            topk_small(scores, 9)


if __name__ == "__main__":
    unittest.main()
