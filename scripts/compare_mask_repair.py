"""固定共享几何和共同监督视角，仅比较旧掩码与修正掩码；短训练不自动发布。"""

import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def verify_pair(output):
    """按实际初始化、图像摘要和全部抽样核对可比性，不声称显卡逐位确定。"""
    from app.training_monitor import write_json_atomic
    root = Path(output)
    a, b = [json.loads((root / label / "reproduction_audit/audit.json").read_text())
            for label in ("old_mask", "repaired_mask")]
    checks = {key: a[key] == b[key] for key in ("initialization", "image_sha256", "sampling", "completed_samples")}
    report = {"passed": all(checks.values()), "checks": checks,
              "scope": "固定输入、初始化和抽样；掩码不同，显卡反向数值可能波动"}
    write_json_atomic(root / "pair_verification.json", report)
    if not report["passed"]:
        raise ValueError("掩码配对输入审计未通过")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("shared_task")
    parser.add_argument("baseline_task")
    parser.add_argument("repaired_source")
    parser.add_argument("output")
    parser.add_argument("--iterations", type=int, default=1000)
    args = parser.parse_args()
    if not 600 <= args.iterations <= 7000:
        parser.error("训练次数应在 600–7000 范围内")
    import pycolmap
    from app.pipeline import _run
    from app.trainer import train
    from app.dataset import ColmapScene, ColmapDataset
    from app.evaluation import load_ply_params, evaluate_views
    from app.capture.provenance import file_digest
    from app.training_monitor import write_json_atomic

    task, baseline, source, output = map(Path, (args.shared_task, args.baseline_task, args.repaired_source, args.output))
    source_manifest = json.loads((source / "feature_masks/mask_manifest.json").read_text())
    if not source_manifest.get("quality", {}).get("passed"):
        raise ValueError("修正掩码的前景覆盖检查必须通过")
    output.mkdir(parents=True, exist_ok=False)
    originals = output / "original_masks"
    originals.mkdir()
    for name, entry in source_manifest["frames"].items():
        with Image.open(source / "feature_masks" / entry["mask"]) as image:
            image.save(originals / name)
    models = [path for path in (task / "colmap/sparse").glob("*") if (path / "images.bin").exists()]
    refined = task / "colmap/merged_refined"
    def model_score(path):
        """与流水线的最大分量规则一致，同注册数时不能误用另一份几何。"""
        reconstruction = pycolmap.Reconstruction(str(path))
        return len(reconstruction.images), len(reconstruction.points3D), str(path)
    selected = refined if (refined / "images.bin").exists() else max(models,
        key=model_score)
    _run(["colmap", "image_undistorter", "--image_path", str(originals), "--input_path", str(selected),
          "--output_path", str(output / "mask_undistorted"), "--output_type", "COLMAP", "--max_image_size", "2000"],
         str(output / "mask_undistort.log"), "mask_undistort")
    scene = ColmapScene(str(task / "dataset"), 1, rigid_normalization=True)
    masks = output / "repaired_masks"
    masks.mkdir()
    entries = {}
    for name in scene.names:
        with Image.open(output / "mask_undistorted/images" / name) as image:
            mask = (np.array(image.convert("L")) >= 128).astype(np.uint8) * 255
        path = masks / (name + ".png")
        Image.fromarray(mask).save(path)
        entries[name] = {"mask": path.name, "reviewed": False, "mask_sha256": file_digest(path),
            "image_sha256": file_digest(task / "dataset/images_1" / name)}
    write_json_atomic(masks / "mask_manifest.json", {"schema_version": 1, "frames": entries,
        "kind": "red_can_v2_undistorted", "provider": source_manifest["provider"]})
    old_actual = json.loads((baseline / "output/effective_config.json").read_text())
    old_config = json.loads((baseline / "training_config.json").read_text())
    dimensions = old_actual["image_dimensions"]
    train_names = [dimensions[i]["name"] for i in old_actual["train_indices"]]
    val_names = [dimensions[i]["name"] for i in old_actual["validation_indices"]]
    indices = [i for i, name in enumerate(scene.names) if name in val_names]
    # 同一份共享几何、原图和初始化点云；只更换颜色及轮廓监督掩码。
    options = copy.deepcopy(old_config["training"])
    options.update(training_names=train_names, validation_names=val_names, validation_every=0,
        foreground_only=True, rigid_normalization=True, allow_unreviewed_masks=True,
        refine_stop_ratio=.8, audit_trace=True, checkpoint_every=0)
    write_json_atomic(output / "protocol.json", {"iterations": args.iterations,
        "shared_task": str(task), "baseline_names_source": str(baseline), "training_names": train_names,
        "validation_names": val_names, "training_options": options,
        "scope": "同几何同图同监督集合的掩码对照；验证参考统一为修正后的自动掩码，非人工真值；短训练不发布"})
    records = []
    for label, mask_dir in (("old_mask", task / "subject_masks"), ("repaired_mask", masks)):
        destination = output / label
        destination.mkdir()
        actual_options = dict(options, mask_dir=str(mask_dir))
        print('START', label, flush=True)
        stats = train(str(task / "dataset"), str(destination), args.iterations, 1, old_actual["cap_max"],
            lambda step, total, message: print(label, message, flush=True), auto_cleanup=True,
            cleanup_threshold=.005, denoise_config=old_config["denoise"], training_options=actual_options)
        reference = ColmapDataset(scene, cache_bytes=128 * 1024**2, uint8_images=True, mask_dir=str(masks),
                                  allow_unreviewed_masks=True, foreground_only=True)
        params = load_ply_params(str(destination / "point_cloud.ply"))
        metrics = evaluate_views(params, reference, str(destination / "common_reference"), indices=indices,
                                 split="fixed_common_names_corrected_auto_mask_reference")
        del params, reference
        record = {"name": label, "points": stats["num_gaussians"], "foreground": metrics["foreground_averages"],
                  "full": metrics["averages"], "seconds": stats["train_seconds"]}
        records.append(record)
        write_json_atomic(output / "comparison.json", records)
        print(json.dumps(record), flush=True)
    print(json.dumps(verify_pair(output)), flush=True)


if __name__ == "__main__":
    main()
