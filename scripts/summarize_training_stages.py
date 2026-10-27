"""汇总固定验证集的训练阶段曲线；不把选优数据误标成独立测试集。"""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir")
    parser.add_argument("runs", nargs="+", help="标签=实验目录")
    args = parser.parse_args()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    experiments = {}
    reference = None
    for entry in args.runs:
        label, path = entry.split("=", 1)
        root = Path(path)
        history = json.loads((root / "stage_history.json").read_text(encoding="utf-8"))
        effective = json.loads((root / "effective_config.json").read_text(encoding="utf-8"))
        fingerprint = {"annotation": history["annotation_sha256"],
                       "dimensions": effective["image_dimensions"],
                       "indices": effective["validation_indices"]}
        if reference is not None and reference != fingerprint:
            raise ValueError("阶段曲线的验证图、尺寸或主体标注不一致")
        reference = fingerprint
        rows = history["records"]
        if not rows:
            raise ValueError(f"{label} 尚无完整阶段")
        complete = (root / "train_stats.json").is_file()
        experiments[label] = {"root": str(root), "complete": complete,
                              "refine_stop_iter": effective["training"]["refine_stop_iter"],
                              "opacity_reset_fixed": effective["training"].get("fix_opacity_reset", False),
                              "mask_loss": effective.get("mask_loss"),
                              "selection_metric": history["selection_metric"],
                              "best_stage": history["best_stage"], "last_stage": rows[-1],
                              "records": rows}
    report = {"scope": "固定验证集用于选阶段与时程，不是独立测试集",
              "reference": reference, "experiments": experiments}
    (output / "summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = ["# 阶段监控结果", "", report["scope"], "",
             "| 实验 | 状态 | 最佳步数 | 主体 PSNR | 主体 SSIM | 全图 PSNR | 最后阶段主体 PSNR |",
             "| --- | --- | --- | --- | --- | --- | --- |"]
    for label, data in experiments.items():
        best, last = data["best_stage"], data["last_stage"]
        subject = f"{best['subject_psnr']:.4f}" if "subject_psnr" in best else "—"
        subject_ssim = f"{best['subject_ssim']:.4f}" if "subject_ssim" in best else "—"
        final_subject = f"{last['subject_psnr']:.4f}" if "subject_psnr" in last else "—"
        lines.append(f"| {label} | {'完成' if data['complete'] else '运行中'} | {best['completed_steps']} | "
                     f"{subject} | {subject_ssim} | {best['full_psnr']:.4f} | {final_subject} |")
    (output / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # 静态科学曲线使用标准绘图库，坐标标记使用英文以免服务器中文字体缺失。
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
    for axis, metric, title in zip(axes.flat,
            ["subject_psnr", "full_psnr", "subject_ssim", "point_count"],
            ["Subject PSNR (dB)", "Full-frame PSNR (dB)", "Subject SSIM", "Gaussian count"]):
        for label, data in experiments.items():
            rows = [row for row in data["records"] if metric in row]
            # 使用输入标签区分方案；掩码对照的增密停止步号相同，不能只按步号命名。
            line, = axis.plot([row["completed_steps"] for row in rows],
                              [row[metric] for row in rows], marker=".", label=label)
            axis.axvline(data["refine_stop_iter"], color=line.get_color(), alpha=.35, linestyle="--")
        axis.set(title=title, xlabel="Completed steps")
        axis.grid(alpha=.2)
        axis.legend()
    figure.suptitle("Fixed validation views for stage selection")
    figure.savefig(output / "training-stages.png", dpi=170)
    plt.close(figure)
    print(json.dumps({label: {key: data[key] for key in ("complete", "best_stage")}
                      for label, data in experiments.items()}, ensure_ascii=False))


if __name__ == "__main__":
    main()
