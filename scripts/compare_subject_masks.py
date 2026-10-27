"""在同一准备好的数据集上顺序执行无掩码与掩码加权训练，保存对照协议。"""

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task_dir")
    parser.add_argument("mask_dir")
    parser.add_argument("output_dir")
    parser.add_argument("--subject-roi", required=True)
    parser.add_argument("--iterations", type=int, default=7000)
    parser.add_argument("--checkpoint-every", type=int, default=1000)
    parser.add_argument("--foreground-weight", type=float, default=1.5)
    parser.add_argument("--background-weight", type=float, default=.25)
    parser.add_argument("--allow-unreviewed-masks", action="store_true",
                        help="允许显式自动原型探索或合成流程检查，不代表人工掩码验收")
    args = parser.parse_args()
    from app.dataset import ColmapDataset, ColmapScene
    from app.subject_masks import validate_mask_weights
    from app.training_monitor import StageMonitor, write_json_atomic

    source, output, masks = Path(args.task_dir).resolve(), Path(args.output_dir).resolve(), Path(args.mask_dir).resolve()
    if output == source or source in output.parents or output.exists():
        raise ValueError("对照输出必须是源任务之外的新目录")
    if args.iterations < 1 or args.checkpoint_every < 1:
        raise ValueError("迭代次数和阶段间隔必须大于零")
    config = json.loads((source / "training_config.json").read_text(encoding="utf-8"))
    runtime = json.loads((source / "effective_config.json").read_text(encoding="utf-8"))
    if config["training"].get("mask_dir"):
        raise ValueError("源配置已包含掩码，不能作为无掩码基线")
    validate_mask_weights(args.foreground_weight, args.background_weight)
    scene = ColmapScene(str(source / "dataset"), runtime["downsample"])
    dataset = ColmapDataset(scene, mask_dir=str(masks), allow_unreviewed_masks=args.allow_unreviewed_masks)
    options = config["training"]
    if options.get("validation_names") is not None:
        if not set(options["validation_names"]) <= set(scene.names):
            raise ValueError("源配置包含未定位的验证视角")
        validation = [i for i, name in enumerate(scene.names) if name in options["validation_names"]]
    else:
        stride = options.get("validation_every", 0)
        validation = list(range(stride - 1, len(scene.names), stride)) if stride >= 3 else []
    if not validation or len(scene.names) - len(validation) < 2:
        raise ValueError("源数据需要固定验证视角及至少两张训练图")
    geometry_files = sorted((source / "dataset/sparse/0").glob("*.bin"))
    fingerprints = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in geometry_files}
    output.mkdir(parents=True)
    # 提前核对冻结的主体矩形和参考图摘要，禁止长训练结束后才发现评估区域不匹配。
    StageMonitor(output, dataset, validation, args.checkpoint_every, options.get("sh_degree", 3),
                 config["denoise"].get("antialiasing", True), args.subject_roi)
    protocol = {"source_task": source.name, "iterations": args.iterations,
                "mask_validation": dataset.mask_report, "geometry_sha256": fingerprints,
                "difference": {"mask_l1_weights": [args.foreground_weight, args.background_weight]},
                "scope": "同数据、种子、训练预算和初始化；仅改变 L1 像素权重，SSIM 保持全图。验证集用于选优。"}
    write_json_atomic(output / "comparison_protocol.json", protocol)
    script = str(Path(__file__).with_name("compare_strategy.py"))
    results = {}
    for label in ("baseline", "weighted"):
        command = [sys.executable, "-u", script, str(source), str(output / label),
                   "--iterations", str(args.iterations), "--strategy", options.get("strategy", "default"),
                   "--checkpoint-every", str(args.checkpoint_every), "--subject-roi", str(Path(args.subject_roi).resolve())]
        if label == "weighted":
            command += ["--mask-dir", str(masks), "--mask-foreground-weight", str(args.foreground_weight),
                        "--mask-background-weight", str(args.background_weight)]
            if args.allow_unreviewed_masks:
                command.append("--allow-unreviewed-masks")
        subprocess.run(command, check=True)
        stats = json.loads((output / label / "train_stats.json").read_text(encoding="utf-8"))
        results[label] = {key: stats[key] for key in ("iterations", "num_gaussians", "validation",
                                                    "subject_validation", "background_validation")}
    if any(hashlib.sha256(Path(path).read_bytes()).hexdigest() != fingerprint for path, fingerprint in fingerprints.items()):
        raise RuntimeError("对照期间源相机或初始化点云被修改")
    write_json_atomic(output / "comparison_summary.json", results)
    print(json.dumps(results, ensure_ascii=False))


if __name__ == "__main__":
    main()
