"""校验训练视角全局颜色拟合的数值与失效保护。"""

import unittest
import numpy as np
from scripts.calibrate_training_color import fit_channels


class TrainingColorTests(unittest.TestCase):
    def test_recovers_known_affine_without_changing_samples(self):
        rng = np.random.default_rng(4)
        prediction = rng.uniform(.1, .7, (1000, 3))
        original = prediction.copy()
        reference = prediction * [1.1, .9, 1.2] + [.03, .04, -.02]
        np.testing.assert_allclose(fit_channels(prediction, reference), [[1.1, .03], [.9, .04], [1.2, -.02]], atol=1e-10)
        np.testing.assert_array_equal(prediction, original)

    def test_rejects_constant_or_extreme_transform(self):
        with self.assertRaises(ValueError):
            fit_channels(np.full((200, 3), .4), np.full((200, 3), .5))
        prediction = np.random.default_rng(1).uniform(.05, .2, (500, 3))
        with self.assertRaises(ValueError):
            fit_channels(prediction, prediction * 3)


if __name__ == "__main__":
    unittest.main()
