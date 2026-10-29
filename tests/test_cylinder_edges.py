"""检查圆罐候选轮廓保留盖顶并去掉桌面，不把先验输出当通用标注。"""

import unittest
import cv2
import numpy as np

from app.foreground.priors.cylinder_edges import cylinder_edge_mask


class CylinderEdgeTests(unittest.TestCase):
    def test_keep_lid_and_exclude_table(self):
        rgb = np.full((1000, 700, 3), (60, 220, 90), dtype=np.uint8)
        rgb[800:] = (50, 60, 55)
        cv2.rectangle(rgb, (190, 160), (510, 730), (220, 185, 120), -1)
        cv2.ellipse(rgb, (350, 730), (160, 65), 0, 0, 360, (240, 230, 210), -1)
        cv2.rectangle(rgb, (190, 300), (510, 720), (235, 25, 35), -1)
        cv2.ellipse(rgb, (350, 720), (160, 55), 0, 0, 180, (235, 25, 35), -1)
        cv2.ellipse(rgb, (350, 160), (160, 45), 0, 0, 360, (210, 205, 130), -1)
        mask, report = cylinder_edge_mask(rgb)
        self.assertEqual(mask.dtype, np.uint8)
        self.assertEqual(mask[140, 350], 255)
        self.assertEqual(mask[500, 350], 255)
        self.assertEqual(mask[850, 350], 0)
        self.assertEqual(mask[100, 350], 0)
        self.assertFalse(report["reviewed"])

    def test_no_localization_evidence_fails(self):
        with self.assertRaisesRegex(ValueError, "定位证据"):
            cylinder_edge_mask(np.full((300, 300, 3), (60, 220, 90), dtype=np.uint8))


if __name__ == "__main__":
    unittest.main()
