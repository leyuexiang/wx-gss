"""接口迁移的兼容性和协议防误用检查，不依赖显卡训练。"""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from types import SimpleNamespace
import json

import numpy as np

from app.foreground import get_provider
from app.capture.provenance import freeze_training_protocol


class ForegroundProviderTests(unittest.TestCase):
    def test_unknown_provider_and_invalid_inputs_fail_explicitly(self):
        with self.assertRaises(ValueError):
            get_provider("auto")
        for name in ("white_can", "red_can", "gold_cap"):
            provider = get_provider(name)
            self.assertFalse(provider.describe()["reviewed"])
            for invalid in (np.zeros((40, 40)), np.zeros((40, 40, 4), dtype=np.uint8),
                            np.zeros((2, 2, 3), dtype=np.uint8)):
                with self.assertRaises(ValueError):
                    provider.coarse(invalid)

    def test_legacy_imports_point_to_same_implementation(self):
        from scripts.prepare_greenscreen_video import can_mask
        from scripts.prepare_red_can_capture import red_can_mask, gold_cap_mask
        for name, function in (("white_can", can_mask), ("red_can", red_can_mask), ("gold_cap", gold_cap_mask)):
            self.assertIs(get_provider(name).coarse_algorithm, function)

    def test_protocol_freezes_actual_inputs_and_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as temporary:
            task = Path(temporary)
            (task / "selection.json").write_text(json.dumps({"video_sha256": "source"}), encoding="utf-8")
            masks = task / "masks"
            masks.mkdir()
            (masks / "mask_manifest.json").write_text(json.dumps({"provider": {"name": "red_can"}}), encoding="utf-8")
            scene = SimpleNamespace(names=["one.png"], dimensions=[{"width": 123, "height": 234}])
            report = {"frames": [{"name": "one.png", "reviewed": False}], "manifest_sha256": "mask"}
            options = {"mask_dir": str(masks), "allow_unreviewed_masks": True, "foreground_only": True}
            with patch("app.dataset.ColmapScene", return_value=scene), patch(
                    "app.subject_masks.validate_mask_set", return_value=({}, report)):
                payload = freeze_training_protocol(task, options, 7000, Path(__file__).resolve().parents[1])
                self.assertEqual(payload["capture_mode"], "unknown")
                self.assertEqual(payload["actual_image_dimensions"], scene.dimensions)
                self.assertEqual(payload["foreground_provider"]["name"], "red_can")
                self.assertIn("app/trainer.py", payload["sources"])
                with self.assertRaisesRegex(ValueError, "协议已存在"):
                    freeze_training_protocol(task, options, 7000, Path(__file__).resolve().parents[1])


if __name__ == "__main__":
    unittest.main()
