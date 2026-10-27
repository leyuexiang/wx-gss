"""验证参数传递、几何缩放、选帧覆盖及导出保真，避免只测试界面状态。"""

import asyncio
import json
import hashlib
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image
from plyfile import PlyData

from app.evaluation import evaluate_views, load_ply_params, resolve_roi
from app.training_monitor import best_stage, preserve_random_state, StageMonitor
from app.dataset import camera_normalization, ColmapDataset
from app.export import export_gaussians
from app.object_capture import prepare_object_views
from app.pipeline import mast3r_crop_transform, _run_colmap
from app.training_config import generate_training_config, save_training_config
from scripts.prepare_strict_validation import verify_frozen_map


class QualityPipelineTests(unittest.TestCase):
    def test_opacity_reset_fix_obeys_interval_and_updates_optimizer_parameters(self):
        from gsplat import DefaultStrategy
        from app.strategy_compat import DefaultStrategyWithOpacityReset

        opacities = torch.nn.Parameter(torch.tensor([-8., 2.]))
        params = {"opacities": opacities}
        optimizer = torch.optim.Adam([opacities])
        opacities.sum().backward()
        optimizer.step()
        optimizers = {"opacities": optimizer}
        corrected = DefaultStrategyWithOpacityReset(reset_every=3000, refine_stop_iter=7500)
        with patch.object(DefaultStrategy, "step_post_backward"):
            for step in [0, 2999, 7500, 9000]:
                previous = params["opacities"]
                corrected.step_post_backward(params, optimizers, {}, step, {})
                self.assertIs(params["opacities"], previous)
            corrected.step_post_backward(params, optimizers, {}, 3000, {})
        self.assertIsNot(params["opacities"], opacities)
        self.assertIs(optimizer.param_groups[0]["params"][0], params["opacities"])
        self.assertLess(params["opacities"].sigmoid().max().item(), .010001)
        self.assertEqual(params["opacities"][0].item(), opacities[0].item())
        for key in ["exp_avg", "exp_avg_sq"]:
            self.assertEqual(torch.count_nonzero(optimizer.state[params["opacities"]][key]).item(), 0)

    def test_stage_selection_uses_subject_instead_of_training_or_background(self):
        records = [{"completed_steps": 1000, "subject_psnr": 30., "full_psnr": 28.},
                   {"completed_steps": 2000, "subject_psnr": 25., "full_psnr": 35.},
                   {"completed_steps": 3000, "subject_psnr": 30., "full_psnr": 29.}]
        self.assertEqual(best_stage(records, "subject_psnr")["completed_steps"], 1000)
        self.assertEqual(best_stage(records, "full_psnr")["completed_steps"], 2000)
        with self.assertRaises(ValueError):
            best_stage([{"subject_psnr": float("nan")}], "subject_psnr")

    def test_stage_evaluation_preserves_training_random_sequence_even_on_error(self):
        import random

        random.seed(7)
        np.random.seed(7)
        torch.manual_seed(7)
        with self.assertRaises(RuntimeError), preserve_random_state():
            random.random(), np.random.rand(), torch.rand(4)
            raise RuntimeError("模拟评估失败")
        actual = random.random(), np.random.rand(), torch.rand(4)
        random.seed(7)
        np.random.seed(7)
        torch.manual_seed(7)
        expected = random.random(), np.random.rand(), torch.rand(4)
        self.assertEqual(actual[:2], expected[:2])
        torch.testing.assert_close(actual[2], expected[2], rtol=0, atol=0)

    def test_stage_roi_rejects_changed_reference_before_training(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            Image.new("RGB", (24, 24)).save(root / "reference.png")
            annotation = root / "roi.json"
            annotation.write_text(json.dumps({"frames": {"reference.png": [0, 0, 24, 24]},
                                              "reference_sha256": {"reference.png": "changed"}}), encoding="utf-8")
            dataset = SimpleNamespace(scene=SimpleNamespace(names=["reference.png"], image_dir=temporary,
                dimensions=[{"name": "reference.png", "width": 24, "height": 24}]))
            with self.assertRaisesRegex(ValueError, "冻结标注不一致"):
                StageMonitor(temporary, dataset, [0], 1000, 3, True, annotation)

    def test_full_and_subject_metrics_reuse_one_render_without_image_output(self):
        class Dataset:
            scene = SimpleNamespace(names=["reference.png"])

            def __len__(self):
                return 1

            def __getitem__(self, index):
                return {"image": torch.full((3, 24, 24), .5),
                        "camtoworld": torch.eye(4), "K": torch.eye(3)}

        params = {"means": torch.zeros(1, 3), "scales": torch.zeros(1, 3),
                  "quats": torch.tensor([[1., 0, 0, 0]]), "opacities": torch.zeros(1),
                  "sh0": torch.zeros(1, 1, 3), "shN": torch.zeros(1, 0, 3)}
        regions = {"reference.png": [2, 3, 12, 13]}
        rendered = torch.full((1, 24, 24, 3), .4)
        with tempfile.TemporaryDirectory() as temporary, \
                patch("gsplat.rasterization", return_value=(rendered, None, {})) as render:
            combined = evaluate_views(params, Dataset(), temporary, additional_roi=regions, save_images=False)
            self.assertEqual(render.call_count, 1)
            self.assertFalse(list(Path(temporary).glob("*.png")))
            self.assertAlmostEqual(combined["background_averages"]["0"]["mse"], .01, places=6)
            self.assertEqual(combined["records"][0]["background"]["pixels"], 24 * 24 - 12 * 13)
            cropped = evaluate_views(params, Dataset(), temporary, roi=regions, save_images=False)
            self.assertEqual(combined["subject_averages"], cropped["averages"])

    def test_byte_cache_preserves_pixels_and_stays_within_budget(self):
        with tempfile.TemporaryDirectory() as temporary:
            names = [f"{index}.png" for index in range(3)]
            for index, name in enumerate(names):
                image = np.full((96, 96, 3), 50 + index * 70, dtype=np.uint8)
                Image.fromarray(image).save(Path(temporary) / name)
            scene = SimpleNamespace(image_dir=temporary, names=names, n_images=3, has_depth=False,
                                    camtoworlds=np.repeat(np.eye(4)[None], 3, axis=0),
                                    Ks=np.repeat(np.eye(3)[None], 3, axis=0))
            budget = 96 * 96 * 3 * 2
            cached = ColmapDataset(scene, cache_bytes=budget, uint8_images=True)
            floating = ColmapDataset(scene, cache_bytes=0)
            for index in range(3):
                sample = cached[index]["image"]
                self.assertEqual(sample.dtype, torch.uint8)
                torch.testing.assert_close(sample.float() / 255.0, floating[index]["image"])
                self.assertLessEqual(cached._cached_bytes, budget)
            self.assertNotIn(0, cached._cache)

    def test_mask_cache_requires_matching_size_and_preserves_uint8(self):
        with tempfile.TemporaryDirectory() as temporary:
            image_dir, mask_dir = Path(temporary) / "images", Path(temporary) / "masks"
            image_dir.mkdir(); mask_dir.mkdir()
            Image.new("RGB", (8, 6), (80, 90, 100)).save(image_dir / "frame_0001.jpg")
            mask = np.zeros((6, 8), dtype=np.uint8)
            mask[:, :4] = 255
            Image.fromarray(mask).save(mask_dir / "frame_0001.png")
            manifest = {"schema_version": 1, "frames": {"frame_0001.jpg": {
                "mask": "frame_0001.png", "reviewed": True,
                "image_sha256": hashlib.sha256((image_dir / "frame_0001.jpg").read_bytes()).hexdigest(),
                "mask_sha256": hashlib.sha256((mask_dir / "frame_0001.png").read_bytes()).hexdigest()}}}
            (mask_dir / "mask_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            scene = SimpleNamespace(image_dir=str(image_dir), names=["frame_0001.jpg"], n_images=1,
                                    dimensions=[{"name": "frame_0001.jpg", "width": 8, "height": 6}],
                                    has_depth=False, camtoworlds=np.eye(4)[None], Ks=np.eye(3)[None])
            sample = ColmapDataset(scene, cache_bytes=1024, uint8_images=True, mask_dir=str(mask_dir))[0]
            self.assertEqual(sample["mask"].dtype, torch.uint8)
            Image.new("L", (7, 6), 255).save(mask_dir / "frame_0001.png")
            with self.assertRaises(ValueError):
                ColmapDataset(scene, cache_bytes=1024, uint8_images=True, mask_dir=str(mask_dir))[0]

    def test_strict_validation_rejects_geometry_or_color_leakage(self):
        import copy
        reference = {"points": {"1": {"xyz": [1., 2., 3.], "color": [10, 20, 30]}},
                     "cameras": {"1": [500, 250, 400]}, "poses": {"train.jpg": [[1, 0, 0, 0]]}}
        verify_frozen_map(reference, copy.deepcopy(reference))
        for field in ("point_position", "point_color", "camera", "pose"):
            changed = copy.deepcopy(reference)
            if field == "point_position":
                changed["points"]["1"]["xyz"][0] += 0.01
            elif field == "point_color":
                changed["points"]["1"]["color"][0] += 1
            elif field == "camera":
                changed["cameras"]["1"][0] += 1
            else:
                changed["poses"]["train.jpg"][0][3] += 0.01
            with self.subTest(field=field), self.assertRaises(ValueError):
                verify_frozen_map(reference, changed)

    def test_validation_camera_does_not_change_frozen_training_normalization(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "scene_normalization.json"
            path.write_text(json.dumps({"center": [1, 2, 3], "scale": 0.5}), encoding="utf-8")
            center, scale = camera_normalization(np.array([[100, 200, 300]]), temporary)
            np.testing.assert_array_equal(center, [1, 2, 3])
            self.assertEqual(scale, 0.5)
            path.write_text(json.dumps({"center": [1, 2, 3], "scale": -1}), encoding="utf-8")
            with self.assertRaises(ValueError):
                camera_normalization(np.zeros((1, 3)), temporary)

    def test_subject_roi_matches_reference_names_and_rejects_invalid_boxes(self):
        regions = {"first.jpg": [10, 20, 80, 120], "second.jpg": [50, 70, 30, 40]}
        self.assertEqual(resolve_roi(regions, "second.jpg", 100, 200), (50, 70, 30, 40))
        for roi in (regions, [0, 0, 101, 20], [1.5, 0, 10, 20], [0, 0, 0, 20]):
            with self.subTest(roi=roi), self.assertRaises(ValueError):
                resolve_roi(roi, "missing.jpg", 100, 200)

    def test_frontend_aliases_and_independent_depth_switch(self):
        config = generate_training_config({"advanced": {
            "gradientClipping": False, "sizeFiltering": False, "antiAliasing": False,
            "depthRegularization": False, "depthWeight": 0.2, "maxGaussians": 300000,
            "positionLR": 0.0003, "strategy": "default", "shDegree": 2,
        }})
        self.assertFalse(config["denoise"]["gradient_clipping"]["enable"])
        self.assertFalse(config["denoise"]["size_filtering"]["enable"])
        self.assertFalse(config["denoise"]["antialiasing"])
        self.assertFalse(config["depth"]["enable"])
        self.assertEqual(config["training"]["cap_max"], 300000)
        self.assertEqual(config["training"]["position_lr"], 0.0003)
        self.assertEqual(config["denoise"]["regularization"]["position_variance_weight"], 0)

    def test_invalid_parameters_rejected_before_queue(self):
        for values in ({"positionLR": float("nan")}, {"shDegree": True},
                       {"validationEvery": 1}, {"strategy": "unsupported"},
                       {"checkpointEvery": 1000}, {"checkpointEvery": 5, "validationEvery": 8},
                       {"maxSizeThreshold": 0.01, "minSizeThreshold": 0.02}):
            with self.subTest(values=values), self.assertRaises(ValueError):
                generate_training_config({"advanced": values})

    def test_square_crop_intrinsics_are_not_stretched(self):
        sx, sy, left, top, _, _ = mast3r_crop_transform(1000, 1000, 512, 384)
        self.assertEqual(top, 64)
        self.assertAlmostEqual((192 + top) * sy, 500)
        self.assertAlmostEqual(sy, sx)

    def test_high_quality_uses_constrained_camera_without_changing_standard(self):
        high = generate_training_config({"qualityMode": "high"})
        standard = generate_training_config({})
        self.assertEqual(high["colmap"]["camera_model"], "SIMPLE_RADIAL")
        self.assertEqual(high["colmap"]["init_min_tri_angle"], 4)
        self.assertEqual(standard["colmap"]["camera_model"], "OPENCV")
        self.assertEqual(high["training"]["strategy"], "default")
        self.assertEqual(standard["training"]["strategy"], "mcmc")

    def test_non_multiple_dimensions_restore_original_center(self):
        sx, sy, left, top, width, height = mast3r_crop_transform(721, 1001, 368, 512)
        self.assertAlmostEqual(sx * width, 721)
        self.assertAlmostEqual(sy * height, 1001)
        self.assertGreaterEqual(left, 0)

    def test_colmap_uses_largest_component_instead_of_first(self):
        with tempfile.TemporaryDirectory() as temporary:
            sparse = Path(temporary) / "colmap/sparse"
            for name in ("0", "1"):
                (sparse / name).mkdir(parents=True)
                (sparse / name / "images.bin").touch()

            def reconstruction(path):
                count = 2 if Path(path).name == "0" else 60
                return SimpleNamespace(images=list(range(count)), points3D=list(range(count * 10)))

            with patch("app.pipeline._run") as commands, patch("pycolmap.Reconstruction", side_effect=reconstruction):
                _run_colmap(temporary, 60, lambda *args: None, {"camera_model": "SIMPLE_RADIAL"})
            undistort_command = commands.call_args.args[0]
            position = undistort_command.index("--input_path")
            self.assertEqual(Path(undistort_command[position + 1]).name, "1")

    def test_high_quality_views_keep_temporal_coverage(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            source.mkdir()
            rng = np.random.default_rng(10)
            for index in range(80):
                Image.fromarray(rng.integers(0, 255, (96, 96, 3), dtype="uint8")).save(source / f"{index:04d}.png")
            result = prepare_object_views(str(source), str(root / "selected"), 60, False, max_views=200)
            self.assertEqual(result["selected_count"], 60)
            indices = [frame["source_index"] for frame in result["frames"]]
            self.assertEqual(indices, sorted(indices))
            self.assertLess(indices[0], 2)
            self.assertGreaterEqual(indices[-1], 78)

    def test_raw_export_preserves_sh_and_training_parameters(self):
        with tempfile.TemporaryDirectory() as temporary:
            count = 4
            params = {
                "means": torch.arange(count * 3, dtype=torch.float32).reshape(count, 3),
                "scales": torch.zeros(count, 3),
                "quats": torch.tensor([[1., 0, 0, 0]] * count),
                "opacities": torch.tensor([-5., -2., 2., 5.]),
                "sh0": torch.zeros(count, 1, 3),
                "shN": torch.arange(count * 45, dtype=torch.float32).reshape(count, 15, 3) / 1000,
            }
            exported = export_gaussians(params, temporary, cleanup_threshold=0.5,
                                        denoise_config={"distance_percentile": 100})
            self.assertEqual(exported, 2)
            self.assertEqual(len(params["means"]), count)
            raw = load_ply_params(str(Path(temporary) / "point_cloud_raw.ply"), device="cpu")
            torch.testing.assert_close(raw["shN"], params["shN"])
            self.assertEqual(len(PlyData.read(str(Path(temporary) / "point_cloud_raw.ply"))["vertex"]), 4)
            self.assertEqual((Path(temporary) / "point_cloud.splat").stat().st_size, 2 * 32)

    def test_high_quality_task_propagates_requested_parameters(self):
        from app import management_api as api
        with tempfile.TemporaryDirectory() as temporary, patch.object(api, "TASKS_DIR", temporary):
            task_id = "a" * 32
            root = Path(temporary) / task_id
            (root / "raw_images").mkdir(parents=True)
            generation = {"qualityMode": "high", "sceneType": "object", "advanced": {
                "maxGaussians": 300000, "depthRegularization": False, "positionLR": 0.0003}}
            result = {"selected_count": 60, "input_count": 120, "median_focus": 200, "frames": []}
            with patch.object(api, "prepare_object_views", return_value=result), \
                 patch.object(api.celery_app, "send_task") as submit:
                asyncio.run(api.process_generation_task(task_id, str(root / "raw_images"),
                            15000, 1, generation, False, {}, False, False, 60, True))
            self.assertEqual(submit.call_args.kwargs["kwargs"]["sfm_backend"], "colmap")
            self.assertEqual(submit.call_args.kwargs["kwargs"]["cap_max"], 300000)
            saved = json.loads((root / "training_config.json").read_text(encoding="utf-8"))
            self.assertEqual(saved["training"]["position_lr"], 0.0003)
            self.assertTrue((root / "frame_manifest.json").is_file())


if __name__ == "__main__":
    unittest.main()
