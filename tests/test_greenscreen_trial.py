"""核对绿色盖顶保留、亮绿背景排除及前景指标，避免黑底掩盖主体误差。"""

import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import cv2
import numpy as np
import torch

from app.evaluation import evaluate_views
from scripts.prepare_greenscreen_video import can_mask


class GreenScreenTests(unittest.TestCase):
    def test_keep_green_lid_artwork_but_reject_bright_green_backdrop(self):
        rgb = np.full((480, 720, 3), (90, 220, 80), dtype=np.uint8)
        rgb[170:400, 245:277] = (195, 249, 188)
        rgb[200:440, 280:440] = (250, 245, 238)
        cv2.ellipse(rgb, (360, 183), (82, 30), 0, 0, 360, (210, 155, 95), -1)
        cv2.ellipse(rgb, (360, 180), (72, 22), 0, 0, 360, (110, 170, 85), -1)
        mask = can_mask(rgb)
        self.assertEqual(mask[180, 360], 255)
        self.assertEqual(mask[300, 360], 255)
        self.assertEqual(mask[260, 260], 0)
        self.assertEqual(mask[460, 360], 0)

    def test_foreground_metric_excludes_black_area_and_reports_alpha(self):
        class Dataset:
            foreground_only = True
            scene = SimpleNamespace(names=["can.png"])

            def __len__(self):
                return 1

            def __getitem__(self, index):
                mask = torch.zeros(24, 24, dtype=torch.uint8)
                mask[:12, :12] = 255
                target = torch.zeros(3, 24, 24)
                target[:, :12, :12] = .5
                return {"image": target, "mask": mask, "camtoworld": torch.eye(4), "K": torch.eye(3)}

        params = {"means": torch.zeros(1, 3), "scales": torch.zeros(1, 3),
                  "quats": torch.tensor([[1., 0, 0, 0]]), "opacities": torch.zeros(1),
                  "sh0": torch.zeros(1, 1, 3), "shN": torch.zeros(1, 0, 3)}
        render = torch.zeros(1, 24, 24, 3)
        render[:, :12, :12] = .4
        alpha = torch.full((1, 24, 24, 1), .05)
        alpha[:, :12, :12] = .9
        with tempfile.TemporaryDirectory() as temporary, patch(
                "gsplat.rasterization", return_value=(render, alpha, {})):
            result = evaluate_views(params, Dataset(), temporary, save_images=False)
        metrics = result["foreground_averages"]["0"]
        self.assertAlmostEqual(metrics["psnr"], 20, places=4)
        self.assertGreater(result["averages"]["0"]["psnr"], metrics["psnr"] + 5)
        self.assertAlmostEqual(metrics["outside_alpha_mean"], .05, places=6)
        self.assertAlmostEqual(metrics["inside_alpha_mae"], .1, places=6)
        self.assertEqual(metrics["silhouette_iou"], 1)


if __name__ == "__main__":
    unittest.main()
