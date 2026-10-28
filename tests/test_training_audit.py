"""确认审计不会改变随机序列或初始化，并且只保存实际抽样前缀。"""

import json
import random
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from app.training_audit import TrainingAudit, array_digest
from scripts.compare_reproduction_audits import summarize


class TrainingAuditTests(unittest.TestCase):
    def test_snapshot_and_sampling_do_not_consume_random_state_or_alias_parameters(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "view.jpg").write_bytes(b"fixture")
            scene = SimpleNamespace(image_dir=directory, names=["view.jpg"],
                                    camtoworlds=np.eye(4)[None], Ks=np.eye(3)[None])
            parameter = torch.tensor([[1., 2., 3.]], requires_grad=True)
            torch_before = torch.get_rng_state().clone()
            np_before = np.random.get_state()
            python_before = random.getstate()
            audit = TrainingAudit(directory, {"means": parameter}, scene, 3)
            audit.record(4)
            audit.record(7)
            audit.save()
            torch.testing.assert_close(torch.get_rng_state(), torch_before)
            np.testing.assert_equal(np.random.get_state(), np_before)
            self.assertEqual(random.getstate(), python_before)
            np.testing.assert_array_equal(np.load(audit.root / "sampled_indices.npy"), [4, 7])
            with torch.no_grad():
                parameter.add_(10)
            with np.load(audit.root / "initialization.npz") as saved:
                np.testing.assert_array_equal(saved["param_means"], [[1., 2., 3.]])
            report = json.loads((audit.root / "audit.json").read_text(encoding="utf-8"))
            self.assertEqual(report["completed_samples"], 2)

    def test_digest_distinguishes_point_order_shape_and_dtype(self):
        array = np.arange(6, dtype=np.float32).reshape(2, 3)
        self.assertNotEqual(array_digest(array), array_digest(array[::-1]))
        self.assertNotEqual(array_digest(array), array_digest(array.reshape(3, 2)))
        self.assertNotEqual(array_digest(array), array_digest(array.astype(np.float64)))

    def test_comparison_identifies_input_changes_and_rejects_partial_trace(self):
        with tempfile.TemporaryDirectory() as directory:
            runs = [Path(directory) / name for name in ("a", "b")]
            audit = {"initialization": {"points": "same"}, "image_sha256": {"view": "same"},
                     "sampling": {"sha256": "same"}, "completed_samples": 2}
            stats = {"iterations": 2, "num_gaussians": 100, "validation": {},
                     "subject_validation": {}, "background_validation": {}}
            for root in runs:
                (root / "reproduction_audit").mkdir(parents=True)
                for path, value in (("reproduction_audit/audit.json", audit),
                                    ("train_stats.json", stats), ("effective_config.json", {})):
                    (root / path).write_text(json.dumps(value), encoding="utf-8")
            self.assertTrue(all(summarize(runs)["equal_checks"].values()))
            audit["initialization"]["points"] = "changed"
            path = runs[1] / "reproduction_audit/audit.json"
            path.write_text(json.dumps(audit), encoding="utf-8")
            self.assertFalse(summarize(runs)["equal_checks"]["initialization"])
            audit["completed_samples"] = 1
            path.write_text(json.dumps(audit), encoding="utf-8")
            with self.assertRaises(ValueError):
                summarize(runs)


if __name__ == "__main__":
    unittest.main()
