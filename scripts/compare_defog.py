"""顺序执行四组去杂云对照和配对重复，短训练仅用于链路验证。"""

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source")
    parser.add_argument("output")
    parser.add_argument("--subject-roi", required=True)
    parser.add_argument("--iterations", type=int, default=7000)
    parser.add_argument("--repeats", type=int, default=2)
    args = parser.parse_args()
    if args.iterations < 1 or args.repeats < 2:
        raise ValueError("需要正数预算及至少两次配对重复")
    source, output = Path(args.source).resolve(), Path(args.output).resolve()
    if output == source or source in output.parents or output.exists():
        raise ValueError("必须使用源任务之外的新目录")
    roi = Path(args.subject_roi).resolve()
    frozen = list((source / "dataset/sparse/0").glob("*.bin")) + list((source / "dataset/images_1").glob("*")) + [roi, source / "training_config.json", source / "effective_config.json"]
    fingerprints = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in frozen if path.is_file()}
    output.mkdir(parents=True)
    protocol = {"iterations": args.iterations, "repeats": args.repeats, "input_sha256": fingerprints,
                "image": "3dgs:quality-mask-experiment-20261027", "rigid_normalization": True,
                "depth_weight": .01, "grid_shape": [8, 8, 4], "tv_weight": 10., "grid_lr": .002,
                "scope": "配对重复，统一刚性相机约定，不使用验证图拟合校色；短预算不代表去雾效果",
                "code_sha256": {str(p.relative_to(Path(__file__).resolve().parents[1])): hashlib.sha256(p.read_bytes()).hexdigest()
                                   for folder in ("app", "scripts") for p in (Path(__file__).resolve().parents[1] / folder).glob("*.py")}}
    (output / "protocol.json").write_text(json.dumps(protocol, ensure_ascii=False, indent=2), encoding="utf-8")
    results = {}
    reference_audit = None
    for repeat in range(1, args.repeats + 1):
        for label, flags in (("baseline", []), ("depth", ["--sparse-depth"]),
                             ("grid", ["--bilateral-grid"]), ("combined", ["--sparse-depth", "--bilateral-grid"])):
            name = f"repeat-{repeat}-{label}"
            command = [sys.executable, "-u", str(Path(__file__).with_name("compare_strategy.py")),
                str(source), str(output / name), "--iterations", str(args.iterations),
                "--strategy", "default", "--checkpoint-every", str(min(1000, args.iterations)),
                "--subject-roi", str(roi), "--audit-trace", "--rigid-normalization", *flags]
            print(f"开始 {name}，预算 {args.iterations}", flush=True)
            with (output / f"{name}.log").open("x", encoding="utf-8") as log:
                subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
            audit = json.loads((output / name / "reproduction_audit/audit.json").read_text())
            signature = {k: audit[k] for k in ("initialization", "sampling", "image_sha256", "completed_samples")}
            if reference_audit is not None and signature != reference_audit:
                raise RuntimeError("配对运行的实际初始化或抽样不一致，不发布混合对照汇总")
            reference_audit = signature
            if any(hashlib.sha256(Path(path).read_bytes()).hexdigest() != digest for path, digest in fingerprints.items()):
                raise RuntimeError("输入或配置在对照过程中发生变化")
            stats = json.loads((output / name / "train_stats.json").read_text())
            results[name] = {key: stats[key] for key in ("iterations", "validation", "subject_validation", "background_validation", "train_seconds", "peak_gpu_mem_gb")}
            (output / "completed_runs.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"完成 {name}", flush=True)
    (output / "comparison_summary.json").write_text(json.dumps({"scope": protocol["scope"], "runs": results}, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
