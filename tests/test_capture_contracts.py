"""通用采集协议和质量门禁的单元测试。"""

import tempfile
import unittest
from pathlib import Path

from app.capture import CaptureManifest, evaluate_mask_ratios, evaluate_reconstruction, write_protocol


class CaptureContractTests(unittest.TestCase):
    def test_manifest_requires_actual_training_size_and_consistent_counts(self):
        manifest = CaptureManifest(frame_count=120, selected_count=60, fps=30,
                                   original_size=[3840, 2160], training_size=[1629, 2000],
                                   capture_mode="turntable_object", provider="red_can")
        self.assertEqual(manifest.to_dict()["training_size"], [1629, 2000])
        for invalid in (
            CaptureManifest(frame_count=0, selected_count=0, original_size=[1, 1], training_size=[1, 1]),
            CaptureManifest(frame_count=2, selected_count=3, original_size=[1, 1], training_size=[1, 1]),
            CaptureManifest(frame_count=2, selected_count=1, original_size=[3840], training_size=[1, 1]),
        ):
            with self.assertRaises(ValueError):
                invalid.to_dict()

    def test_mask_gate_rejects_temporal_jump(self):
        report = evaluate_mask_ratios([.30, .31, .58], max_jump=.20)
        self.assertFalse(report["passed"])
        self.assertFalse(report["checks"]["temporal_stability"])
        self.assertTrue(evaluate_mask_ratios([.3, .31, .32])["passed"])
        for invalid in ([True, .3], [float("nan")], ["0.3"]):
            with self.assertRaises(ValueError):
                evaluate_mask_ratios(invalid)

    def test_reconstruction_gate_requires_coverage_and_error(self):
        good = evaluate_reconstruction({"input_images": 100, "registered_images": 96,
            "mean_reprojection_error": .7, "max_angular_gap_degrees": 20})
        self.assertTrue(good["passed"])
        bad = evaluate_reconstruction({"input_images": 100, "registered_images": 80,
            "mean_reprojection_error": .7, "max_angular_gap_degrees": 20})
        self.assertFalse(bad["passed"])
        with self.assertRaises(ValueError):
            evaluate_reconstruction({"input_images": 100, "registered_images": 100})
        with self.assertRaises(ValueError):
            evaluate_reconstruction({"input_images": 100, "registered_images": 100,
                                     "mean_reprojection_error": -1})

    def test_protocol_write_is_utf8_and_replace_is_complete(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "protocol.json"
            write_protocol(str(path), {"说明": "中文", "passed": True})
            self.assertEqual(path.read_text(encoding="utf-8")[0], "{")
            self.assertFalse(path.with_name("protocol.json.tmp").exists())


if __name__ == "__main__":
    unittest.main()
