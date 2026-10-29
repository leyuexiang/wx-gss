"""确认重建准备阶段可以显式启用通用质量门禁。"""

import json
import importlib.util
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from app.pipeline import _prepare_dataset


@unittest.skipUnless(importlib.util.find_spec("pycolmap"), "需要 pycolmap 运行时依赖")
class PipelineQualityGateTests(unittest.TestCase):
    def test_prepare_dataset_quality_gate_rejects_low_registration_before_copying_images(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dense = root / "dense"
            (dense / "sparse").mkdir(parents=True)
            (dense / "images").mkdir()
            for name in ("cameras.bin", "images.bin", "points3D.bin"):
                (dense / "sparse" / name).write_bytes(b"")

            class Reconstruction:
                images = {1: object()}
                points3D = {}

            with patch("pycolmap.Reconstruction", return_value=Reconstruction()), \
                    self.assertRaisesRegex(RuntimeError, "质量门禁"):
                _prepare_dataset(str(root), str(dense), 1, 20, lambda *args: None, quality_gate=True)
            self.assertFalse((root / "dataset" / "images_1").exists())
            report = json.loads((root / "reconstruction_report.json").read_text(encoding="utf-8"))
            self.assertFalse(report["quality_gate"]["passed"])


if __name__ == "__main__":
    unittest.main()
