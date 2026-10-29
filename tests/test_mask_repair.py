"""覆盖被高光分开的主体和冻结监督集，防止半主体掩码再次进入训练。"""

import unittest
import cv2
import numpy as np
import tempfile
import json
from pathlib import Path

from app.foreground import get_provider
from app.capture.quality import evaluate_foreground_support
from app.capture.splits import training_indices
from app.capture.trajectory import analyze_orbit
from scripts.compare_mask_repair import verify_pair


class MaskRepairTests(unittest.TestCase):
    def test_pair_verification_rejects_different_samples(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            report = {"initialization": {"means": "same"}, "image_sha256": {"a": "same"},
                      "sampling": "same", "completed_samples": 1000}
            for label in ("old_mask", "repaired_mask"):
                p = root / label / "reproduction_audit"
                p.mkdir(parents=True)
                (p / "audit.json").write_text(json.dumps(report))
            self.assertTrue(verify_pair(root)["passed"])
            report["sampling"] = "changed"
            (root / "repaired_mask/reproduction_audit/audit.json").write_text(json.dumps(report))
            with self.assertRaises(ValueError):
                verify_pair(root)

    def test_orbit_is_scale_invariant_and_rejects_degenerate_path(self):
        angles = np.linspace(0, 4 * np.pi, 121)
        centers = np.c_[np.cos(angles), np.sin(angles), np.zeros_like(angles)]
        report = analyze_orbit(centers * 3 + [4, 5, 6])
        self.assertLess(report["radius_cv"], 1e-10)
        self.assertAlmostEqual(report["absolute_turns"], 2)
        self.assertAlmostEqual(report["max_angular_gap_degrees"], 6)
        for bad in (np.zeros((10, 3)), np.c_[angles, angles, angles]):
            with self.assertRaises(ValueError):
                analyze_orbit(bad)

    def test_vertical_highlight_must_not_remove_half_object(self):
        rgb = np.full((600, 700, 3), (80, 210, 90), dtype=np.uint8)
        cv2.rectangle(rgb, (200, 200), (500, 510), (230, 20, 30), -1)
        cv2.rectangle(rgb, (330, 200), (345, 510), (250, 250, 250), -1)
        cv2.rectangle(rgb, (200, 160), (500, 200), (210, 160, 100), -1)
        old = get_provider("red_can").coarse(rgb)
        new = get_provider("red_can_v2").coarse(rgb)
        self.assertEqual(old[350, 220], 0)
        self.assertEqual(new[350, 220], 255)
        self.assertEqual(new[350, 470], 255)
        self.assertEqual(new[550, 350], 0)
        self.assertEqual(get_provider("red_can_v2").describe()["version"], 2)

    def test_support_detects_half_mask_even_when_area_is_reasonable(self):
        support = np.zeros((100, 100), dtype=np.uint8)
        support[20:80, 20:80] = 255
        mask = support.copy()
        mask[:, :50] = 0
        self.assertFalse(evaluate_foreground_support(mask, support)["passed"])
        self.assertTrue(evaluate_foreground_support(support, support)["passed"])

    def test_frozen_split_excludes_new_frames_and_rejects_overlap(self):
        names = ["a", "b", "c", "new"]
        self.assertEqual(training_indices(names, [1], ["a", "c"]), [0, 2])
        self.assertEqual(training_indices(names, [1]), [0, 2, 3])
        for requested in (["b"], ["a", "a"], ["missing"]):
            with self.assertRaises(ValueError):
                training_indices(names, [1], requested)


if __name__ == "__main__":
    unittest.main()
