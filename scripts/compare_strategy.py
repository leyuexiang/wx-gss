"""固定数据、留出视角和训练预算，独立对照增密策略或可选掩码权重。"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task_dir")
    parser.add_argument("output_dir")
    parser.add_argument("--strategy", choices=("default", "mcmc"), default="default")
    parser.add_argument("--iterations", type=int)
    parser.add_argument("--rigid-normalization", action="store_true", help="仅缩放相机中心，需新建同约定基线")
    parser.add_argument("--sparse-depth", action="store_true", help="启用可靠观测的逆深度监督")
    parser.add_argument("--sparse-depth-weight", type=float, default=.01)
    parser.add_argument("--bilateral-grid", action="store_true", help="启用训练视角局部仿射光度补偿")
    parser.add_argument("--audit-trace", action="store_true", help="额外保存真实初始化和抽样序列，用于隔离复现诊断")
    parser.add_argument("--refine-stop", type=int, help="固定增密停止步号，其余训练时程保持原样")
    parser.add_argument("--checkpoint-every", type=int, default=0)
    parser.add_argument("--subject-roi", help="冻结的参考图矩形及摘要文件")
    parser.add_argument("--mask-dir", help="可选主体掩码目录；仅用于隔离实验")
    parser.add_argument("--allow-unreviewed-masks", action="store_true", help="允许自动原型探索或合成流程检查，不代表掩码已复查")
    parser.add_argument("--mask-foreground-weight", type=float, default=1.5)
    parser.add_argument("--mask-background-weight", type=float, default=0.25)
    parser.add_argument("--fix-opacity-reset", action="store_true", help="单独验证已确认的依赖重置条件修正")
    args = parser.parse_args()
    from app.trainer import train
    root, output = Path(args.task_dir).resolve(), Path(args.output_dir).resolve()
    if output == root or root in output.parents:
        raise ValueError("策略对照必须使用原任务之外的目录")
    output.mkdir(parents=True, exist_ok=False)
    config = json.loads((root / "training_config.json").read_text(encoding="utf-8"))
    runtime = json.loads((root / "effective_config.json").read_text(encoding="utf-8"))
    options = dict(config["training"], strategy=args.strategy)
    options.update(sparse_depth=args.sparse_depth, sparse_depth_weight=args.sparse_depth_weight,
                   bilateral_grid=args.bilateral_grid, rigid_normalization=args.rigid_normalization)
    if args.audit_trace:
        options["audit_trace"] = True
    iterations = args.iterations or runtime["iterations"]
    if args.refine_stop is not None:
        if not 500 < args.refine_stop < iterations:
            raise ValueError("增密停止步号必须大于 500 且小于总训练次数")
        options["refine_stop_ratio"] = args.refine_stop / iterations
    if args.checkpoint_every:
        options["checkpoint_every"] = args.checkpoint_every
    if args.subject_roi:
        options["monitor_roi_path"] = str(Path(args.subject_roi).resolve())
    if args.mask_dir:
        options["mask_dir"] = str(Path(args.mask_dir).resolve())
        options["mask_foreground_weight"] = args.mask_foreground_weight
        options["mask_background_weight"] = args.mask_background_weight
        options["allow_unreviewed_masks"] = args.allow_unreviewed_masks
    if args.fix_opacity_reset:
        if args.strategy != "default":
            raise ValueError("不透明度重置修正仅适用于原版风格增密策略")
        options["fix_opacity_reset"] = True
    stats = train(
        str(root / "dataset"), str(output), iterations,
        runtime["downsample"], runtime["cap_max"],
        lambda step, total, message: print(message, flush=True),
        auto_cleanup=config["denoise"]["enable"],
        cleanup_threshold=config["denoise"]["opacity_threshold"],
        enable_depth_reg=config["depth"]["enable"], depth_reg_weight=config["depth"]["weight"],
        denoise_config=config["denoise"], training_options=options,
    )
    print(json.dumps(stats, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
