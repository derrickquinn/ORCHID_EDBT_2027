import unittest

from optimize_orchid import _log_neighborhood


class OptimizeOrchidTest(unittest.TestCase):
    def test_neighborhood_includes_axial_and_diagonal_moves(self) -> None:
        candidates = _log_neighborhood(0.0, 0.0, 0.5, -1.0, 1.0, -1.0, 1.0)

        self.assertEqual(candidates[0], (0.0, 0.0))
        self.assertEqual(
            set(candidates),
            {
                (-0.5, -0.5),
                (-0.5, 0.0),
                (-0.5, 0.5),
                (0.0, -0.5),
                (0.0, 0.0),
                (0.0, 0.5),
                (0.5, -0.5),
                (0.5, 0.0),
                (0.5, 0.5),
            },
        )

    def test_neighborhood_deduplicates_clamped_boundaries(self) -> None:
        candidates = _log_neighborhood(-1.0, 1.0, 0.5, -1.0, 1.0, -1.0, 1.0)

        self.assertEqual(len(candidates), 4)
        self.assertEqual(candidates[0], (-1.0, 1.0))


if __name__ == "__main__":
    unittest.main()
