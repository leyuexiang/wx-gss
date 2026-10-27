"""将已验证的阶段作为独立模型发布，保留原模型及完整训练时程说明。"""

import argparse
import json
import shutil
import sys
import uuid
import zipfile
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_task")
    parser.add_argument("experiment_dir")
    parser.add_argument("--name", required=True)
    parser.add_argument("--step", type=int, help="默认使用验证最佳阶段")
    args = parser.parse_args()
    from app import config
    from app.dataset import ColmapScene, ColmapDataset
    from app.evaluation import evaluate_views, load_ply_params
    from app.export import export_gaussians
    from app.training_monitor import write_json_atomic

    source, experiment = Path(args.source_task).resolve(), Path(args.experiment_dir).resolve()
    # 长实验完成后再发布，避免把运行中的阶段误当成最终选择结论。
    final_stats = json.loads((experiment / "train_stats.json").read_text(encoding="utf-8"))
    history = json.loads((experiment / "stage_history.json").read_text(encoding="utf-8"))
    selected = history["best_stage"] if args.step is None else next(
        row for row in history["records"] if row["completed_steps"] == args.step)
    actual = json.loads((experiment / "effective_config.json").read_text(encoding="utf-8"))
    annotation_path = experiment / "monitor_reference_roi.json"
    regions = json.loads(annotation_path.read_text(encoding="utf-8"))["frames"] if annotation_path.exists() else None
    task_id = uuid.uuid4().hex
    task = Path(config.TASKS_DIR) / task_id
    task.mkdir(parents=True, exist_ok=False)
    ignored = {"output", "status.json", "effective_config.json", "quality_job.json"}
    for item in source.iterdir():
        if item.name in ignored:
            continue
        target = task / item.name
        if item.is_dir():
            shutil.copytree(item, target)
        else:
            shutil.copy2(item, target)
    output = task / "output"
    output.mkdir()
    training_config = json.loads((task / "training_config.json").read_text(encoding="utf-8"))
    training_config["training"] = actual["training"]
    write_json_atomic(task / "training_config.json", training_config)
    stage_path = (experiment / selected["model"]).resolve()
    if experiment not in stage_path.parents:
        raise ValueError("阶段模型超出实验目录")
    params = load_ply_params(str(stage_path))
    denoise = actual["denoise"]
    point_count = export_gaussians(params, str(output), auto_cleanup=denoise.get("enable", True),
                                   cleanup_threshold=denoise.get("opacity_threshold", .005), denoise_config=denoise)
    del params
    scene = ColmapScene(str(task / "dataset"), actual["downsample"])
    dataset = ColmapDataset(scene, cache_bytes=64 * 1024**2, uint8_images=True)
    delivered = load_ply_params(str(output / "point_cloud.ply"))
    metrics = evaluate_views(delivered, dataset, str(output / "evaluation/validation"),
                             indices=actual["validation_indices"], additional_roi=regions,
                             split="validation_used_for_stage_selection", antialiased=actual["antialiased"])
    stats = {"iterations": selected["completed_steps"], "scheduled_iterations": actual["iterations"],
             "num_gaussians": point_count, "num_gaussians_before_cleanup": selected["point_count"],
             "n_images": len(scene.names), "n_training_images": len(actual["train_indices"]),
             "n_validation_images": len(actual["validation_indices"]), "downsample": actual["downsample"],
             "seed": actual["training"]["seed"], "strategy": actual["training"]["strategy"],
             "sh_degree": selected["evaluation_sh_degree"], "antialiased": actual["antialiased"],
             "image_dimensions": scene.dimensions, "validation": metrics["averages"],
             "subject_validation": metrics.get("subject_averages"),
             "full_run_train_seconds": final_stats["train_seconds"],
             "selection_scope": history["selection_scope"],
             "monitoring": {"selected_stage": selected, "experiment": str(experiment)}}
    write_json_atomic(output / "train_stats.json", stats)
    write_json_atomic(output / "effective_config.json", {**actual, "exported_completed_steps": selected["completed_steps"]})
    write_json_atomic(task / "effective_config.json", {"iterations": selected["completed_steps"],
        "scheduled_iterations": actual["iterations"], "downsample": actual["downsample"],
        "cap_max": actual["cap_max"], "sfm_backend": "colmap", "requested_config": training_config,
        "source_task": source.name, "experiment": str(experiment), "stage": selected})
    for name in ["stage_history.json", "best_stage.json", "monitor_reference_roi.json"]:
        if (experiment / name).exists():
            shutil.copy2(experiment / name, output / name)
    write_json_atomic(output / "stage_provenance.json", {"source_task": source.name,
        "experiment": str(experiment), "selected_model": str(stage_path),
        "selection_metric": history["selection_metric"], "annotation_sha256": history["annotation_sha256"],
        "scope": "验证集选优结果，不作为独立测试集泛化结论"})
    with zipfile.ZipFile(output / "result.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(output.glob("*.json")):
            archive.write(path, path.name)
        for name in ["point_cloud.ply", "point_cloud.splat"]:
            archive.write(output / name, name)
        archive.write(output / "evaluation/validation/metrics.json", "evaluation/validation/metrics.json")
    write_json_atomic(task / "status.json", {"task_id": task_id, "status": "done", "stage": "done",
        "progress": 1, "message": "固定验证集选出的阶段模型，保留完整时程记录", "model_name": args.name,
        "created_at": datetime.now(timezone.utc).isoformat(), "is_video": True, "scene_type": "object",
        "params": {"iterations": selected["completed_steps"], "downsample": actual["downsample"],
                   "cap_max": actual["cap_max"], "sfm_backend": "colmap", "strategy": actual["training"]["strategy"]}})
    print(json.dumps({"task_id": task_id, "step": selected["completed_steps"],
                      "validation": metrics["averages"], "subject": metrics.get("subject_averages")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
