"""验证实心茶罐原型边界清理不修改原图、不误删内部白色文字和顶部金属高光。"""

import unittest

import numpy as np

from scripts.refine_subject_masks import refine_mask


class RefineSubjectMaskTests(unittest.TestCase):
    def test_detached_speck_removed_while_body_and_internal_letters_are_preserved(self):
        image = np.full((100, 100, 3), 220, dtype=np.uint8)
        image[20:85, 30:70] = [150, 30, 40]
        image[65:70, 45:55] = 245
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[20:85, 30:70] = 255
        mask[90:92, 90:92] = 255
        original_image, original_mask = image.copy(), mask.copy()
        result = refine_mask(image, mask)
        self.assertEqual(result[90, 90], 0)
        self.assertTrue((result[65:70, 45:55] == 255).all())
        np.testing.assert_array_equal(image, original_image)
        np.testing.assert_array_equal(mask, original_mask)

    def test_shallow_white_attachment_is_removed_but_gold_lid_is_kept(self):
        image = np.full((100, 100, 3), 220, dtype=np.uint8)
        image[20:85, 30:70] = [150, 30, 40]
        image[20:35, 30:70] = [230, 215, 150]
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[20:88, 30:70] = 255
        result = refine_mask(image, mask)
        self.assertTrue((result[20:35, 30:70] == 255).all())
        self.assertEqual(result[87, 50], 0)
        self.assertEqual(result[84, 50], 255)

    def test_connected_wood_background_does_not_expand_convex_hull(self):
        # 木色背景紧贴金盖时，不能因红色通道较大而把整块背景并入凸包。
        image = np.full((100, 100, 3), 220, dtype=np.uint8)
        image[5:30, 15:55] = [120, 95, 65]
        image[30:85, 30:70] = [150, 30, 40]
        image[20:35, 30:70] = [230, 215, 150]
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[20:90, 25:75] = 255
        result = refine_mask(image, mask)
        self.assertEqual(result[15, 25], 0)
        self.assertEqual(result[25, 50], 255)
        self.assertEqual(result[80, 50], 255)

    def test_white_barcode_touching_boundary_is_preserved(self):
        # 白条码与背景相连时仍属于实心罐身，凸形先验应补回浅缺口。
        image = np.full((100, 100, 3), 220, dtype=np.uint8)
        image[20:85, 30:70] = [150, 30, 40]
        image[60:70, 55:70] = 245
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[20:85, 30:70] = 255
        result = refine_mask(image, mask)
        self.assertTrue((result[60:70, 55:70] == 255).all())


if __name__ == "__main__":
    unittest.main()
