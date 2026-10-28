"""对比真实初始化、输入和抽样记录，避免把固定种子等同于逐位复现。"""

import argparse
import json
from pathlib import Path


def summarize(runs):
    loaded = []
    for directory in runs:
        root = Path(directory)
        audit = json.loads((root / "reproduction_audit/audit.json").read_text(encoding="utf-8"))
        stats = json.loads((root / "train_stats.json").read_text(encoding="utf-8"))
        config = json.loads((root / "effective_config.json").read_text(encoding="utf-8"))
        if audit["completed_samples"] != stats["iterations"]:
            raise ValueError("抽样记录不完整，不能作为已完成的复现实验")
        loaded.append((root, audit, stats, config))
    if len(loaded) < 2:
        raise ValueError("至少需要两次已完成训练")
    reference = loaded[0]
    checks = {key: all(row[1][key] == reference[1][key] for row in loaded[1:])
              for key in ("initialization", "image_sha256", "sampling", "completed_samples")}
    checks["effective_config"] = all(row[3] == reference[3] for row in loaded[1:])
    results = {str(root): {key: stats[key] for key in
               ("iterations", "num_gaussians", "validation", "subject_validation", "background_validation")}
               for root, _, stats, _ in loaded}
    return {"equal_checks": checks, "runs": results,
            "scope": "仅诊断这些运行；即使初始化与抽样一致，也不保证显卡算子逐位确定。历史无审计运行不能据此归因。"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output")
    parser.add_argument("runs", nargs="+")
    args = parser.parse_args()
    report = summarize(args.runs)
    # 禁止覆盖上一轮诊断，保留复现差异的原始证据。
    with Path(args.output).open("x", encoding="utf-8") as destination:
        json.dump(report, destination, ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
