import unittest

import numpy as np

from src.test_ivf import TestIvfConfig
from test_orchid import _resolve_alphas


class TestOrchidGridTest(unittest.TestCase):
    def test_fixed_zero_alpha_is_evaluated_once(self) -> None:
        alphas = _resolve_alphas(TestIvfConfig(alpha=0.0))

        np.testing.assert_array_equal(alphas, np.array([0.0]))


if __name__ == "__main__":
    unittest.main()
