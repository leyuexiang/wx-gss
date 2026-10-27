"""核对掩码损失尺度、来源防误用、离线复查导入和矩形插值。"""

import base64
import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from app.subject_masks import validate_mask_set, validate_mask_weights, weighted_l1
from scripts.import_reviewed_masks import import_masks
from scripts.prepare_interpolated_masks import interpolate_boxes


class SubjectMaskTests(unittest.TestCase):
    def test_equal_weight_preserves_loss_and_gradients_for_all_channels(self):
        prediction = torch.arange(36, dtype=torch.float32).reshape(3, 3, 4).div(36).requires_grad_()
        target = torch.full_like(prediction, .4)
        mask = torch.zeros(3, 4)
        mask[:1] = 1
        baseline = F.l1_loss(prediction, target)
        actual = weighted_l1(prediction, target, mask, .25, .25)
        torch.testing.assert_close(actual, baseline)
        torch.testing.assert_close(torch.autograd.grad(actual, prediction)[0],
                                   torch.autograd.grad(baseline, prediction)[0])

    def test_foreground_priority_changes_gradient_ratio_without_tripling_loss(self):
        prediction = torch.ones(3, 2, 2, requires_grad=True)
        mask = torch.tensor([[1., 0.], [1., 0.]])
        loss = weighted_l1(prediction, torch.zeros_like(prediction), mask, 1.5, .25)
        self.assertAlmostEqual(loss.item(), 1.0)
        gradient, = torch.autograd.grad(loss, prediction)
        self.assertAlmostEqual((gradient[0, 0, 0] / gradient[0, 0, 1]).item(), 6.)
        for value in (float("nan"), float("inf"), 0, -1, True, 11):
            with self.subTest(value=value), self.assertRaises(ValueError):
                validate_mask_weights(value, 1.)

    def fixture(self, temporary):
        root = Path(temporary)
        images, masks = root / "images", root / "masks"
        images.mkdir(); masks.mkdir()
        Image.new("RGB", (8, 6), (100, 80, 60)).save(images / "view.jpg")
        array = np.zeros((6, 8), dtype=np.uint8); array[:, :4] = 255
        Image.fromarray(array).save(masks / "view.png")
        entry = {"mask": "view.png", "reviewed": True,
                 "image_sha256": hashlib.sha256((images / "view.jpg").read_bytes()).hexdigest(),
                 "mask_sha256": hashlib.sha256((masks / "view.png").read_bytes()).hexdigest()}
        manifest = {"schema_version": 1, "kind": "test-fixture", "frames": {"view.jpg": entry}}
        path = masks / "mask_manifest.json"
        path.write_text(json.dumps(manifest), encoding="utf-8")
        scene = SimpleNamespace(image_dir=str(images), names=["view.jpg"],
                                dimensions=[{"name": "view.jpg", "width": 8, "height": 6}])
        return scene, masks, manifest

    def test_missing_incomplete_unreviewed_or_changed_data_fails_before_training(self):
        with tempfile.TemporaryDirectory() as temporary:
            scene, masks, manifest = self.fixture(temporary)
            path = masks / "mask_manifest.json"
            paths, report = validate_mask_set(scene, masks)
            self.assertEqual(report["frames"][0]["foreground_ratio"], .5)
            self.assertEqual(len(paths), 1)
            for kind in ("missing", "incomplete", "unreviewed", "mask_hash", "image_hash", "escape"):
                changed = copy.deepcopy(manifest)
                if kind == "missing":
                    path.unlink()
                else:
                    if kind == "incomplete": changed["frames"] = {}
                    else:
                        row = changed["frames"]["view.jpg"]
                        if kind == "unreviewed": row["reviewed"] = False
                        elif kind == "mask_hash": row["mask_sha256"] = "changed"
                        elif kind == "image_hash": row["image_sha256"] = "changed"
                        else: row["mask"] = "../escape.png"
                    path.write_text(json.dumps(changed), encoding="utf-8")
                with self.subTest(kind=kind), self.assertRaises(ValueError):
                    validate_mask_set(scene, masks)
                path.write_text(json.dumps(manifest), encoding="utf-8")

    def test_invalid_mask_pixels_and_dimensions_rejected_even_with_correct_hash(self):
        with tempfile.TemporaryDirectory() as temporary:
            scene, masks, manifest = self.fixture(temporary)
            for size, color in (((8, 6), 0), ((8, 6), 255), ((8, 6), 128), ((7, 6), 255)):
                Image.new("L", size, color).save(masks / "view.png")
                manifest["frames"]["view.jpg"]["mask_sha256"] = hashlib.sha256((masks / "view.png").read_bytes()).hexdigest()
                (masks / "mask_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
                with self.subTest(size=size, color=color), self.assertRaises(ValueError):
                    validate_mask_set(scene, masks)

    def test_review_import_roundtrip_and_incomplete_review_leaves_no_partial_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            scene, masks, manifest = self.fixture(temporary)
            item = {"image_sha256": manifest["frames"]["view.jpg"]["image_sha256"], "reviewed": False,
                    "png": base64.b64encode((masks / "view.png").read_bytes()).decode("ascii")}
            source = Path(temporary) / "review.json"
            output = Path(temporary) / "imported"
            data = {"schema_version": 1, "kind": "manual-mask-review", "frames": {"view.jpg": item}}
            source.write_text(json.dumps(data), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "尚未确认"):
                import_masks(scene.image_dir, source, output)
            self.assertFalse(output.exists())
            item["reviewed"] = True
            source.write_text(json.dumps(data), encoding="utf-8")
            import_masks(scene.image_dir, source, output)
            _, report = validate_mask_set(scene, output)
            self.assertEqual(report["frames"][0]["foreground_ratio"], .5)

    def test_interpolation_between_late_keyframes_and_at_endpoints(self):
        keys = {"frame_0008.jpg": [0, 0, 10, 20], "frame_0016.jpg": [8, 8, 18, 28],
                "frame_0024.jpg": [16, 16, 26, 36]}
        result = interpolate_boxes(["frame_0001.jpg", "frame_0020.jpg", "frame_0030.jpg"], keys)
        self.assertEqual(result["frame_0001.jpg"], keys["frame_0008.jpg"])
        self.assertEqual(result["frame_0020.jpg"], [12, 12, 22, 32])
        self.assertEqual(result["frame_0030.jpg"], keys["frame_0024.jpg"])


if __name__ == "__main__":
    unittest.main()
