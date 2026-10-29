"""固定相机的真实图像评估；训练重建指标与留出视角指标分别标记。"""

import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw


def resolve_roi(roi, name, width, height):
    """按参考图名称使用预先冻结的矩形，禁止缺失标注时悄悄改用全图指标。"""
    if roi is None:
        return None
    box = roi.get(name) if isinstance(roi, dict) else roi
    if not isinstance(box, (list, tuple)) or len(box) != 4:
        raise ValueError(f"{name} 缺少四个整数构成的主体区域")
    if any(isinstance(value, bool) or not isinstance(value, int) for value in box):
        raise ValueError("主体区域必须使用整数像素坐标")
    x, y, crop_width, crop_height = box
    if min(x, y) < 0 or min(crop_width, crop_height) < 1 or x + crop_width > width or y + crop_height > height:
        raise ValueError("主体评估区域超出图像范围")
    return tuple(box)


def load_ply_params(path: str, device="cuda") -> dict:
    """恢复标准 PLY 中的完整球谐系数，避免通过固定颜色格式评估。"""
    from plyfile import PlyData

    vertex = PlyData.read(path)["vertex"]
    names = vertex.data.dtype.names

    def values(fields):
        array = np.stack([vertex[field] for field in fields], axis=1).copy()
        return torch.as_tensor(array, dtype=torch.float32, device=device)

    rest = sorted((name for name in names if name.startswith("f_rest_")),
                  key=lambda name: int(name.rsplit("_", 1)[1]))
    count = len(vertex)
    return {
        "means": values(["x", "y", "z"]),
        "scales": values([f"scale_{i}" for i in range(3)]),
        "quats": values([f"rot_{i}" for i in range(4)]),
        "opacities": values(["opacity"]).squeeze(1),
        "sh0": values([f"f_dc_{i}" for i in range(3)])[:, None],
        "shN": values(rest).reshape(count, 3, -1).transpose(1, 2) if rest
        else torch.empty((count, 0, 3), device=device),
    }


def _image_metrics(prediction, target):
    """统一全图与主体的量化方式，保证同一次渲染的两组指标可追溯。"""
    from pytorch_msssim import ssim

    mse = F.mse_loss(prediction, target).item()
    window = min(11, *target.shape[1:])
    window = window if window % 2 else window - 1
    return {
        "psnr": float(-10 * math.log10(max(mse, 1e-12))),
        "ssim": float(ssim(prediction[None], target[None], data_range=1.0,
                           size_average=True, win_size=window)),
    }


@torch.no_grad()
def evaluate_views(params: dict, dataset, output_dir: str, indices=None,
                   degrees=None, split="training_reconstruction", antialiased=True,
                   roi=None, additional_roi=None, save_images=True) -> dict:
    """在一致的相机和像素尺寸下渲染，输出指标、逐视角记录和对照图片。"""
    from gsplat import rasterization

    if roi is not None and additional_roi is not None:
        raise ValueError("全图附加主体评估与仅主体评估不能同时指定")

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    if indices is None:
        indices = np.linspace(0, len(dataset) - 1, min(6, len(dataset)), dtype=int).tolist()
    indices = list(dict.fromkeys(int(index) for index in indices))
    if not indices or any(index < 0 or index >= len(dataset) for index in indices):
        raise ValueError("评估视角为空或超出数据集范围")
    features = torch.cat([params["sh0"], params["shN"]], dim=1)
    max_degree = math.isqrt(features.shape[1]) - 1
    degrees = sorted(set(min(int(degree), max_degree) for degree in (degrees or [max_degree])))
    if degrees[0] < 0:
        raise ValueError("球谐阶数不能小于零")
    device = params["means"].device
    # 激活参数只计算一次，评估不修改训练参数或优化器。
    scales = params["scales"].exp()
    opacities = params["opacities"].sigmoid()
    rotations = F.normalize(params["quats"], dim=-1)
    records = []
    for index in indices:
        data = dataset[index]
        target = data["image"].to(device=device, dtype=torch.float32)
        if data["image"].dtype == torch.uint8:
            target.div_(255.0)
        height, width = target.shape[1:]
        box = resolve_roi(roi, dataset.scene.names[index], width, height)
        subject_box = resolve_roi(additional_roi, dataset.scene.names[index], width, height)
        tiles = [Image.fromarray((target.permute(1, 2, 0).cpu().numpy() * 255).clip(0, 255).astype("uint8"))] if save_images else []
        panels = ["reference"]
        for degree in degrees:
            rendered, alphas, _ = rasterization(
                means=params["means"], scales=scales, quats=rotations,
                opacities=opacities, colors=features,
                viewmats=torch.linalg.inv(data["camtoworld"].to(device))[None],
                Ks=data["K"].to(device)[None], width=width, height=height,
                sh_degree=degree, packed=True,
                rasterize_mode="antialiased" if antialiased else "classic",
            )
            prediction = rendered[0].permute(2, 0, 1).clamp(0, 1)
            measured_prediction, measured_target = prediction, target
            if box is not None:
                x, y, crop_width, crop_height = box
                measured_prediction = prediction[:, y:y + crop_height, x:x + crop_width]
                measured_target = target[:, y:y + crop_height, x:x + crop_width]
            record = {
                "index": index, "name": dataset.scene.names[index],
                "width": width, "height": height, "sh_degree": degree,
                "roi": box,
                **_image_metrics(measured_prediction, measured_target),
            }
            if getattr(dataset, "foreground_only", False):
                # 黑色背景占比较大会抬高全图分数，必须额外报告真正前景及轮廓质量。
                foreground = data["mask"].to(device) > 0
                alpha = alphas[0, :, :, 0]
                mse = (prediction - target).square().mean(dim=0)[foreground].mean().item()
                union = ((alpha >= .5) | foreground).sum().clamp_min(1)
                record["foreground"] = {
                    "psnr": float(-10 * math.log10(max(mse, 1e-12))),
                    "outside_alpha_mean": float(alpha[~foreground].mean()),
                    "inside_alpha_mae": float((1 - alpha[foreground]).abs().mean()),
                    "silhouette_iou": float(((alpha >= .5) & foreground).sum() / union),
                }
            if subject_box is not None:
                x, y, crop_width, crop_height = subject_box
                record["subject"] = {"roi": subject_box, **_image_metrics(
                    prediction[:, y:y + crop_height, x:x + crop_width],
                    target[:, y:y + crop_height, x:x + crop_width])}
                # 矩形外像素单独计算误差，避免主体变好掩盖背景回退；不将其当作漂浮点数量。
                background = torch.ones((height, width), dtype=torch.bool, device=device)
                background[y:y + crop_height, x:x + crop_width] = False
                if background.any():
                    squared_error = (prediction - target).square().mean(dim=0)
                    mse = squared_error[background].mean().item()
                    record["background"] = {"region": "outside_subject_rectangle",
                                            "pixels": int(background.sum()), "mse": mse,
                                            "psnr": float(-10 * math.log10(max(mse, 1e-12)))}
            records.append(record)
            if not save_images:
                continue
            tiles.append(Image.fromarray((prediction.permute(1, 2, 0).cpu().numpy() * 255).astype("uint8")))
            panels.append(f"{'ROI ' if box else ''}SH {degree}: PSNR {record['psnr']:.2f}, SSIM {record['ssim']:.3f}")
        if not save_images:
            continue
        canvas = Image.new("RGB", (width * len(tiles), height + 24), (25, 25, 25))
        draw = ImageDraw.Draw(canvas)
        for panel_index, (tile, label) in enumerate(zip(tiles, panels)):
            canvas.paste(tile, (panel_index * width, 24))
            draw.text((panel_index * width + 4, 5), label, fill="white")
            if box is not None:
                x, y, crop_width, crop_height = box
                draw.rectangle((panel_index * width + x, y + 24,
                                panel_index * width + x + crop_width - 1, y + crop_height + 23),
                               outline=(40, 220, 90), width=2)
        canvas.save(output / f"view_{index:04d}.png")
    result = {
        "split": split, "roi": roi, "antialiased": antialiased,
        "region_kind": "fixed_reference_rectangle" if roi is not None else "full_frame",
        "indices": indices, "records": records,
        "reference_kind": ("foreground_black_background" if getattr(dataset, "foreground_only", False)
                           else "original_full_frame"),
        "averages": {str(degree): {
            metric: float(np.mean([record[metric] for record in records if record["sh_degree"] == degree]))
            for metric in ("psnr", "ssim")
        } for degree in degrees},
    }
    if getattr(dataset, "foreground_only", False):
        result["foreground_averages"] = {str(degree): {
            metric: float(np.mean([record["foreground"][metric] for record in records
                                   if record["sh_degree"] == degree]))
            for metric in ("psnr", "outside_alpha_mean", "inside_alpha_mae", "silhouette_iou")
        } for degree in degrees}
    if additional_roi is not None:
        result["subject_roi"] = additional_roi
        result["subject_averages"] = {str(degree): {
            metric: float(np.mean([record["subject"][metric] for record in records
                                   if record["sh_degree"] == degree]))
            for metric in ("psnr", "ssim")
        } for degree in degrees}
        background_records = [record for record in records if "background" in record]
        result["background_averages"] = {str(degree): {
            metric: float(np.mean([record["background"][metric] for record in background_records
                                   if record["sh_degree"] == degree]))
            for metric in ("mse", "psnr")
        } for degree in degrees if any(record["sh_degree"] == degree for record in background_records)}
    (output / "metrics.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result
