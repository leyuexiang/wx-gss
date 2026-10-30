"""覆盖深度标定的错配、退化与缺测拒绝路径。"""

import unittest
import numpy as np
from app.depth_alignment import fit_inverse_depth, apply_inverse_alignment, alignment_errors


class DepthAlignmentTests(unittest.TestCase):
    def test_recover_with_outliers_and_independent_targets(self):
        predicted = np.linspace(1., 4., 200)
        target = 1. / (1.7 / predicted + .2)
        noisy = target.copy()
        noisy[::13] *= 3
        fit = fit_inverse_depth(predicted, noisy)
        self.assertAlmostEqual(fit['scale'], 1.7, places=3)
        self.assertAlmostEqual(fit['offset'], .2, places=3)
        held = np.linspace(1.2, 3.8, 71)
        result = apply_inverse_alignment(held, fit)
        self.assertLess(alignment_errors(result, 1. / (1.7 / held + .2))['p90_relative_z'], .001)

    def test_degenerate_and_negative_relation_rejected(self):
        with self.assertRaisesRegex(ValueError, '跨度不足'):
            fit_inverse_depth(np.ones(50), np.ones(50))
        x = np.linspace(1., 2., 50)
        with self.assertRaisesRegex(ValueError, '方向不一致'):
            fit_inverse_depth(x, 1. / (2. - 1. / x))

    def test_insufficient_and_nonfinite_anchors_rejected(self):
        with self.assertRaisesRegex(ValueError, '锚点不足'):
            fit_inverse_depth(np.r_[np.ones(20), np.nan, 0], np.ones(22))

    def test_invalid_depth_stays_missing(self):
        result = apply_inverse_alignment(np.array([np.nan, 0, -1, 1, 4]), {'scale': 1., 'offset': -.5})
        np.testing.assert_array_equal(result, [0, 0, 0, 2, 0])
        self.assertEqual(alignment_errors(result, np.ones(5))['valid_ratio'], .2)


if __name__ == '__main__':
    unittest.main()
