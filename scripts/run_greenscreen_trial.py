"""隔离运行绿幕转台实验：主体特征重建、同变换掩码及前景透明度训练。"""

import argparse
import hashlib
import json
import shutil
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def prepare(source, task, status, reuse_task=None):
    """原图参与特征提取，掩码只排除背景；训练掩码使用完全相同的去畸变变换。"""
    import pycolmap
    from app.pipeline import _run, _run_colmap, _prepare_dataset
    from app.subject_masks import validate_mask_set

    names = sorted(path.name for path in (source / "images").glob("*.png"))
    dimensions = []
    for name in names:
        with Image.open(source / "images" / name) as image:
            dimensions.append({"width": image.width, "height": image.height})
    paths, report = validate_mask_set(SimpleNamespace(names=names, dimensions=dimensions,
        image_dir=str(source / "images")), source / "feature_masks", allow_unreviewed=True)
    task.mkdir(parents=True, exist_ok=False)
    shutil.copytree(source / "images", task / "images")
    shutil.copy2(source / "selection.json", task / "selection.json")
    source_protocol = json.loads((source / "selection.json").read_text(encoding="utf-8"))
    source_masks = json.loads((source / "feature_masks/mask_manifest.json").read_text(encoding="utf-8"))
    # 新准备器若提供明确失败的质量检查，不能继续执行昂贵的几何和训练。
    if source_masks.get("quality", {}).get("passed") is False:
        raise ValueError("前景质量检查未通过，拒绝继续重建")
    (task / "sfm_masks").mkdir()
    (task / "foreground_original").mkdir()
    for name in names:
        with Image.open(paths[name]) as image:
            mask = np.array(image)
        Image.fromarray(mask).save(task / "foreground_original" / name)
        # 腐蚀 20 像素排除轮廓及背景交界特征，避免静止绿布主导转台位姿。
        inner = cv2.erode(mask, np.ones((41, 41), np.uint8))
        Image.fromarray(inner).save(task / "sfm_masks" / (name + ".png"))
    (task / "input_mask_validation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    if reuse_task is None:
        feature_groups = []
        for group in source_protocol.get("feature_groups", []):
            list_path = task / Path(group["image_list"]).name
            list_path.write_text((source / group["image_list"]).read_text(encoding="utf-8"), encoding="utf-8")
            feature_groups.append({"image_list": str(list_path), "camera_params": group["camera_params"]})
        colmap_options = {
            "camera_model": "SIMPLE_RADIAL", "matcher": "exhaustive", "init_min_tri_angle": 4,
            "max_image_size": 2000, "mask_path": str(task / "sfm_masks"), "merge_components": True,
        }
        if feature_groups:
            colmap_options["feature_groups"] = feature_groups
        dense = _run_colmap(str(task), len(names), status, colmap_options)
    else:
        # 复用同一批图已验证的相机，逐文件核对摘要，避免再次提取、匹配和重建。
        for name in names:
            if hashlib.sha256((task / "images" / name).read_bytes()).digest() != hashlib.sha256(
                    (reuse_task / "images" / name).read_bytes()).digest():
                raise ValueError(f"复用重建的原图不一致：{name}")
        model = reuse_task / "colmap/merged_refined"
        shutil.copytree(model, task / "colmap/merged_refined")
        (task / "reconstruction_source.json").write_text(json.dumps({"task": str(reuse_task),
            "model": str(model), "verification": "全部原图字节摘要一致"}, ensure_ascii=False, indent=2), encoding="utf-8")
        dense = str(task / "colmap/dense")
        _run(["colmap", "image_undistorter", "--image_path", str(task / "images"),
              "--input_path", str(task / "colmap/merged_refined"), "--output_path", dense,
              "--output_type", "COLMAP", "--max_image_size", "2000"],
             str(task / "colmap/colmap.log"), "undistort_reused_model")
    _, registered, ratio = _prepare_dataset(str(task), dense, 1, len(names), status, quality_gate=True)
    candidates = []
    for path in (task / "colmap/sparse").glob("*"):
        if (path / "images.bin").exists():
            model = pycolmap.Reconstruction(str(path))
            candidates.append((len(model.images), len(model.points3D), str(path)))
    # 有效合并结果已完成联合优化，图像与掩码必须使用相同的这份原始相机。
    refined = task / "colmap/merged_refined"
    model_path = str(refined) if (refined / "images.bin").exists() else max(candidates)[2]
    _run(["colmap", "image_undistorter", "--image_path", str(task / "foreground_original"),
          "--input_path", model_path, "--output_path", str(task / "mask_undistorted"),
          "--output_type", "COLMAP", "--max_image_size", "2000"],
         str(task / "colmap/colmap.log"), "mask_undistort")
    masks = task / "subject_masks"
    masks.mkdir()
    entries = {}
    for path in sorted((task / "dataset/images_1").iterdir()):
        with Image.open(task / "mask_undistorted/images" / path.name) as image:
            # 插值产生的灰度按固定阈值二值化，尺寸和相机与训练参考一致。
            mask = (np.array(image.convert("L")) >= 128).astype(np.uint8) * 255
        destination = masks / (path.name + ".png")
        Image.fromarray(mask).save(destination)
        entries[path.name] = {"mask": destination.name, "reviewed": False,
            "image_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "mask_sha256": hashlib.sha256(destination.read_bytes()).hexdigest()}
    (masks / "mask_manifest.json").write_text(json.dumps({"schema_version": 1,
        "kind": "foreground-with-colmap-undistortion", "provider": source_masks.get("provider", "legacy_unspecified"),
        "source_kind": source_masks.get("kind", "unspecified"), "frames": entries}, indent=2), encoding="utf-8")
    status("prepared", "prepared", .4, f"独立绿幕实验已准备：{registered}/{len(names)} 张，注册率 {ratio:.1%}")


def train_trial(task, iterations, status):
    """质量门槛通过后才训练；仅颜色留出，不宣称完全独立的新视角验证。"""
    from app.trainer import train
    from app.training_config import generate_training_config, save_training_config

    report = json.loads((task / "reconstruction_report.json").read_text(encoding="utf-8"))
    if report["registration_ratio"] < .9:
        raise ValueError("注册率不足 90%，拒绝把局部分量作为完整绿幕模型训练")
    output = task / "output"
    config = generate_training_config({"qualityMode": "high", "advanced": {
        "validationEvery": 8, "opacityThreshold": .005, "imageCacheMB": 256}}, scene_type="object")
    options = config["training"]
    options.update(rigid_normalization=True, mask_dir=str(task / "subject_masks"),
                   allow_unreviewed_masks=True, foreground_only=True, silhouette_weight=.1)
    from app.capture.provenance import freeze_training_protocol
    freeze_training_protocol(task, options, iterations, Path(__file__).resolve().parents[1])
    output.mkdir(exist_ok=False)
    save_training_config(str(task), config)
    save_training_config(str(task), {"iterations": iterations, "downsample": 1,
        "cap_max": options["cap_max"], "sfm_backend": "colmap", "requested_config": config}, "effective_config.json")
    stats = train(str(task / "dataset"), str(output), iterations, 1, options["cap_max"],
        lambda step, total, message: status("training", "training", .4 + .59 * step / total, message),
        auto_cleanup=True, cleanup_threshold=.005, denoise_config=config["denoise"], training_options=options)
    # 与网页下载保持一致，只打包结果和配置，不重复包含原视频与全部帧。
    with zipfile.ZipFile(output / "result.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        for name in ("point_cloud.ply", "point_cloud.splat", "train_stats.json", "effective_config.json",
                     "mask_validation.json"):
            archive.write(output / name, name)
    status("done", "done", 1., "绿幕前景实验训练完成，自动掩码仍需边缘复查")
    return stats


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_dir")
    parser.add_argument("task_dir", help="准备阶段必须为不存在的新目录")
    parser.add_argument("--phase", choices=("prepare", "train", "all"), default="prepare")
    parser.add_argument("--iterations", type=int, default=7000)
    parser.add_argument("--reuse-task", help="可选复用任务中的已合并相机，所有原图摘要必须一致")
    args = parser.parse_args()
    if not 100 <= args.iterations <= 30000:
        parser.error("训练次数需在 100–30000 之间")
    source, task = Path(args.source_dir).resolve(), Path(args.task_dir).resolve()
    # 拒绝覆盖必须发生在状态异常处理之前，避免误把历史成功任务标为失败。
    if args.phase in ("prepare", "all") and task.exists():
        raise ValueError("准备阶段必须使用不存在的新任务目录")
    if args.phase == "train" and (task / "output").exists():
        raise ValueError("训练输出已存在，请保留原实验并使用独立目录")
    from app.training_monitor import write_json_atomic
    created = datetime.now(timezone.utc).isoformat()

    def status(state, stage, progress, message):
        write_json_atomic(task / "status.json", {"task_id": task.name, "status": state, "stage": stage,
            "progress": progress, "message": message, "model_name": "tea-can-green-screen-foreground",
            "created_at": created, "is_video": True, "scene_type": "object",
            "params": {"iterations": args.iterations, "downsample": 1, "sfm_backend": "colmap"}})
        print(message, flush=True)

    try:
        if args.phase in ("prepare", "all"):
            prepare(source, task, status, Path(args.reuse_task).resolve() if args.reuse_task else None)
        if args.phase in ("train", "all"):
            train_trial(task, args.iterations, status)
    except Exception as error:
        if (task / "status.json").exists():
            status("failed", "failed", 0, str(error))
        raise


if __name__ == "__main__":
    main()
