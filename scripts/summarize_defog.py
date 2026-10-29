"""汇总完整四组配对实验的最终交付结果，不挑最高分重复或阶段。"""

import argparse
import json
import statistics
from pathlib import Path


def summarize(root):
    root = Path(root)
    protocol = json.loads((root / "protocol.json").read_text(encoding="utf-8"))
    complete = json.loads((root / "comparison_summary.json").read_text(encoding="utf-8"))
    expected = {f"repeat-{repeat}-{label}" for repeat in range(1, protocol["repeats"] + 1)
                for label in ("baseline", "depth", "grid", "combined")}
    if set(complete["runs"]) != expected:
        raise ValueError("四组重复结果不完整，不能输出最终质量汇总")
    groups, details, reference, reference_views = {}, {}, None, None
    for name in sorted(expected):
        run = root / name
        audit = json.loads((run / "reproduction_audit/audit.json").read_text(encoding="utf-8"))
        signature = {key: audit[key] for key in ("initialization", "sampling", "image_sha256", "completed_samples")}
        if audit["completed_samples"] != protocol["iterations"] or (reference is not None and signature != reference):
            raise ValueError("初始化、原图或完整抽样不一致")
        reference = signature
        stats = json.loads((run / "train_stats.json").read_text(encoding="utf-8"))
        metrics = json.loads((run / "evaluation/validation/metrics.json").read_text(encoding="utf-8"))
        config = json.loads((run / "effective_config.json").read_text(encoding="utf-8"))
        history = json.loads((run / "stage_history.json").read_text(encoding="utf-8"))
        view_signature = {"dimensions": config["image_dimensions"], "indices": config["validation_indices"],
                          "roi": metrics["subject_roi"], "annotation": history["annotation_sha256"]}
        if reference_views is not None and view_signature != reference_views:
            raise ValueError("验证视角、尺寸或标注不一致")
        reference_views = view_signature
        label = name.split("-", 2)[2]
        if not config["training"].get("rigid_normalization") or bool(config["sparse_depth"]["enabled"]) != (label in ("depth", "combined")) or bool(config["bilateral_grid"]["enabled"]) != (label in ("grid", "combined")):
            raise ValueError("分组标签与实际开关或相机约定不一致")
        records = [row for row in metrics["records"] if row["sh_degree"] == 3]
        values = {"subject_psnr": stats["subject_validation"]["3"]["psnr"],
                  "subject_ssim": stats["subject_validation"]["3"]["ssim"],
                  "full_psnr": stats["validation"]["3"]["psnr"],
                  "background_mse": stats["background_validation"]["3"]["mse"],
                  "worst_subject_psnr": min(row["subject"]["psnr"] for row in records),
                  "train_seconds": stats["train_seconds"], "peak_gpu_mem_gb": stats["peak_gpu_mem_gb"]}
        details[name] = {"values": values, "views": {row["name"]: {"full_psnr": row["psnr"],
                          "subject_psnr": row["subject"]["psnr"], "background_mse": row["background"]["mse"]}
                          for row in records}}
        groups.setdefault(label, []).append(values)
    aggregated = {label: {key: {"mean": statistics.mean(row[key] for row in rows),
                                  "min": min(row[key] for row in rows), "max": max(row[key] for row in rows)}
                          for key in rows[0]} for label, rows in groups.items()}
    paired = {}
    for label in ("depth", "grid", "combined"):
        deltas = []
        for repeat in range(1, protocol["repeats"] + 1):
            candidate = details[f"repeat-{repeat}-{label}"]["values"]
            baseline = details[f"repeat-{repeat}-baseline"]["values"]
            deltas.append({"repeat": repeat, "subject_psnr_delta": candidate["subject_psnr"] - baseline["subject_psnr"],
                           "full_psnr_delta": candidate["full_psnr"] - baseline["full_psnr"],
                           "background_mse_ratio": candidate["background_mse"] / baseline["background_mse"]})
        paired[label] = deltas
    return {"scope": "最终清理模型、未经验证图颜色拟合；两次重复是初步稳定性检查，不构成统计显著性结论",
            "groups": aggregated, "paired": paired, "runs": details}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment")
    parser.add_argument("output")
    args = parser.parse_args()
    report = summarize(args.experiment)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    (output / "quality-summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = ["# 去杂云配对长训练结果", "", report["scope"], "",
             "| 组别 | 主体 PSNR 均值 [范围] | 主体 SSIM | 全图 PSNR | 背景 MSE | 最差主体视角均值 |",
             "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for label in ("baseline", "depth", "grid", "combined"):
        row = report["groups"][label]
        subject = row["subject_psnr"]
        lines.append(f"| {label} | {subject['mean']:.4f} [{subject['min']:.4f}, {subject['max']:.4f}] | "
                     f"{row['subject_ssim']['mean']:.4f} | {row['full_psnr']['mean']:.4f} | "
                     f"{row['background_mse']['mean']:.8f} | {row['worst_subject_psnr']['mean']:.4f} |")
    (output / "quality-summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(report["groups"], ensure_ascii=False))


if __name__ == "__main__":
    main()
