"""检查配对汇总拒绝缺组、输入变化以及标错实际开关。"""

import json
import tempfile
import unittest
from pathlib import Path

from scripts.summarize_defog import summarize


class DefogSummaryTests(unittest.TestCase):
    def fixture(self, directory):
        root = Path(directory)
        def save(path, value):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(value), encoding="utf-8")
        save(root / "protocol.json", {"iterations": 20, "repeats": 2})
        names = [f"repeat-{repeat}-{label}" for repeat in (1, 2) for label in ("baseline", "depth", "grid", "combined")]
        save(root / "comparison_summary.json", {"runs": dict.fromkeys(names, {})})
        for name in names:
            label = name.split("-", 2)[2]
            run = root / name
            save(run / "reproduction_audit/audit.json", {"initialization": {}, "sampling": {},
                 "image_sha256": {}, "completed_samples": 20})
            save(run / "train_stats.json", {"subject_validation": {"3": {"psnr": 30., "ssim": .95}},
                 "validation": {"3": {"psnr": 29.}}, "background_validation": {"3": {"mse": .01}},
                 "train_seconds": 2., "peak_gpu_mem_gb": 1.})
            save(run / "effective_config.json", {"training": {"rigid_normalization": True},
                 "sparse_depth": {"enabled": label in ("depth", "combined")},
                 "bilateral_grid": {"enabled": label in ("grid", "combined")},
                 "image_dimensions": [], "validation_indices": [7]})
            save(run / "stage_history.json", {"annotation_sha256": "same"})
            save(run / "evaluation/validation/metrics.json", {"subject_roi": {}, "records": [
                 {"sh_degree": 3, "name": "view.jpg", "psnr": 29., "subject": {"psnr": 30.}, "background": {"mse": .01}}]})
        return root

    def test_equal_runs_aggregate_without_selecting_best(self):
        with tempfile.TemporaryDirectory() as directory:
            result = summarize(self.fixture(directory))
            self.assertEqual(result["groups"]["baseline"]["subject_psnr"]["mean"], 30.)
            self.assertEqual(result["paired"]["combined"][0]["subject_psnr_delta"], 0)

    def test_missing_group_and_mismatched_actual_flags_are_rejected(self):
        for change in ("group", "flags", "audit"):
            with self.subTest(change=change), tempfile.TemporaryDirectory() as directory:
                root = self.fixture(directory)
                if change == "group":
                    path = root / "comparison_summary.json"
                    value = json.loads(path.read_text()); value["runs"].pop("repeat-2-grid")
                elif change == "flags":
                    path = root / "repeat-2-grid/effective_config.json"
                    value = json.loads(path.read_text()); value["bilateral_grid"]["enabled"] = False
                else:
                    path = root / "repeat-2-grid/reproduction_audit/audit.json"
                    value = json.loads(path.read_text()); value["sampling"] = {"sha256": "different"}
                path.write_text(json.dumps(value), encoding="utf-8")
                with self.assertRaises(ValueError):
                    summarize(root)


if __name__ == "__main__":
    unittest.main()
