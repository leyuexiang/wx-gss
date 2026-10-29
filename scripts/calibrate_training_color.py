"""用固定训练视角拟合一次全局逐通道颜色变换，另存诊断，不读取验证图拟合。"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def fit_channels(prediction, reference):
    """全局逐通道仿射拟合，拒绝病态或过强变换，不为单张验证图定制。"""
    transforms = []
    for channel in range(3):
        x, y = prediction[:, channel], reference[:, channel]
        valid = (x > .02) & (x < .98) & (y > .02) & (y < .98)
        x, y = x[valid], y[valid]
        if len(x) < 100 or np.var(x) < 1e-5:
            raise ValueError("训练颜色样本不足或方差过小")
        design = np.stack((x, np.ones_like(x)), axis=1)
        gain, bias = np.linalg.lstsq(design, y, rcond=None)[0]
        if not np.isfinite([gain, bias]).all() or not .5 <= gain <= 2 or abs(bias) > .25:
            raise ValueError("颜色变换过强，拒绝自动应用")
        transforms.append([float(gain), float(bias)])
    return np.asarray(transforms)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_task")
    parser.add_argument("experiment")
    parser.add_argument("output")
    args = parser.parse_args()
    source, experiment, output = Path(args.source_task), Path(args.experiment), Path(args.output)
    config = json.loads((experiment / "effective_config.json").read_text(encoding="utf-8"))
    json.loads((experiment / "train_stats.json").read_text(encoding="utf-8"))
    train, validation = config["train_indices"], config["validation_indices"]
    indices = [train[i] for i in np.linspace(0, len(train) - 1, min(6, len(train)), dtype=int)]
    if set(indices) & set(validation):
        raise ValueError("颜色校准混入验证图")
    references, predictions, provenance = [], [], []
    for index in indices:
        entry = config["image_dimensions"][index]
        path = experiment / "evaluation/training" / f"view_{index:04d}.png"
        with Image.open(path) as image:
            width, height = entry["width"], entry["height"]
            if image.size != (width * 2, height + 24):
                raise ValueError("训练渲染面板格式与预期不一致")
            pixels = np.asarray(image.convert("RGB"))[24:]
            # 固定规则稀疏取样控制内存，既不修改随机序列也不按验证结果挑样本。
            references.append(pixels[::8, :width:8].reshape(-1, 3) / 255.)
            predictions.append(pixels[::8, width::8].reshape(-1, 3) / 255.)
        provenance.append({"name": entry["name"], "index": index,
                           "panel_sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    affine = fit_channels(np.concatenate(predictions), np.concatenate(references))
    output.mkdir(parents=True, exist_ok=False)
    from app.evaluation import load_ply_params, evaluate_views
    from app.dataset import ColmapScene, ColmapDataset
    from app.export import export_ply, export_splat_from_ply
    import torch
    params = load_ply_params(str(experiment / "point_cloud.ply"))
    gain = torch.as_tensor(affine[:, 0], dtype=params["sh0"].dtype, device=params["sh0"].device)
    bias = torch.as_tensor(affine[:, 1], dtype=params["sh0"].dtype, device=gain.device)
    # 球谐表示 C = 0.5 + SH：全局颜色仿射变换可解析写回全部阶数。
    params["sh0"] = params["sh0"] * gain + (gain * .5 + bias - .5)[None, None] / .28209479177387814
    params["shN"] = params["shN"] * gain
    export_ply(params, str(output / "point_cloud.ply"))
    export_splat_from_ply(str(output / "point_cloud.ply"), str(output / "point_cloud.splat"))
    roi = json.loads((experiment / "monitor_reference_roi.json").read_text(encoding="utf-8"))["frames"]
    dataset = ColmapDataset(ColmapScene(str(source / "dataset"), config["downsample"],
                           rigid_normalization=config["training"].get("rigid_normalization", False)), uint8_images=True)
    metrics = evaluate_views(params, dataset, str(output / "evaluation/validation"), indices=validation,
                             antialiased=config["antialiased"], additional_roi=roi, split="training_only_global_color_calibration")
    report = {"scope": "仅六张预先固定训练渲染面板拟合全局颜色；独立诊断，不替代原四组结果",
              "source_model_sha256": hashlib.sha256((experiment / "point_cloud.ply").read_bytes()).hexdigest(),
              "gain_bias": affine.tolist(), "training_references": provenance,
              "validation_used_for_fitting": False, "full": metrics["averages"],
              "subject": metrics["subject_averages"], "background": metrics["background_averages"]}
    (output / "color_calibration.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("gain_bias", "full", "subject", "background")}))


if __name__ == "__main__":
    main()
