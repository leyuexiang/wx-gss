"""底圈局部恢复不能跨暗缝抓取桌面反射，不能更改顶盖和共同主体。"""

import unittest
import numpy as np
from app.foreground.priors.base_band import preserve_base_band


class BaseBandTests(unittest.TestCase):
    def example(self):
        rgb = np.full((200, 160, 3), 40, np.uint8)
        mask = np.zeros((200, 160), np.uint8)
        mask[20:150, 25:135] = 255
        rgb[148:152, 25:135] = (230, 210, 170)
        rgb[155:170, 25:135] = (240, 225, 200)
        return rgb, mask

    def test_connected_band_only_and_upper_body_unchanged(self):
        rgb, mask = self.example()
        fixed, report = preserve_base_band(rgb, mask)
        self.assertEqual(fixed[151, 80], 255)
        self.assertEqual(fixed[155, 80], 0)
        self.assertTrue((fixed >= mask).all())
        np.testing.assert_array_equal(fixed[:148], mask[:148])
        self.assertGreater(report['added_pixels'], 0)
        self.assertFalse(report['reviewed'])

    def test_bounded_extension_even_if_reflection_touches(self):
        rgb, mask = self.example()
        rgb[148:, 25:135] = (230, 210, 170)
        fixed, report = preserve_base_band(rgb, mask)
        self.assertEqual(fixed[150 + report['max_extension_pixels'], 80], 0)

    def test_dark_edge_or_green_background_unchanged(self):
        rgb, mask = self.example()
        for color in ((40, 40, 40), (70, 220, 80)):
            rgb[150:] = color
            fixed, _ = preserve_base_band(rgb, mask)
            np.testing.assert_array_equal(fixed, mask)

    def test_invalid_mask_rejected(self):
        rgb, mask = self.example()
        with self.assertRaises(ValueError):
            preserve_base_band(rgb, mask.astype(float))

    def test_narrow_connected_reflection_is_not_extended(self):
        rgb, mask = self.example()
        rgb[150:] = (40, 40, 40)
        rgb[150:160, 79:82] = (230, 210, 170)
        fixed, _ = preserve_base_band(rgb, mask)
        self.assertEqual(fixed[151, 80], 0)


if __name__ == '__main__':
    unittest.main()
