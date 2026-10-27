"""校验主体掩码及输入来源，防止错误图像、尺寸或未复查原型进入训练。"""

import hashlib
import io
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image


def read_binary_mask(content, width, height, name):
    """只接受明确的黑白前景标注，禁止颜色转换或缩放掩盖标注错误。"""
    with Image.open(io.BytesIO(content)) as image:
        if image.format != "PNG" or image.mode != "L" or image.size != (width, height):
            raise ValueError(f"主体掩码必须为同尺寸灰度 PNG：{name}")
        mask = np.array(image, dtype=np.uint8)
    if not np.all((mask == 0) | (mask == 255)):
        raise ValueError(f"主体掩码必须仅含 0 和 255：{name}")
    count = int(np.count_nonzero(mask))
    if count == 0 or count == mask.size:
        raise ValueError(f"主体掩码必须同时包含前景和背景：{name}")
    return mask


def validate_mask_set(scene, mask_dir, allow_unreviewed=False):
    """在初始化优化器之前检查全数据集，一次记录文件摘要和覆盖率。"""
    root = Path(mask_dir).resolve()
    manifest_path = root / "mask_manifest.json"
    if not manifest_path.is_file():
        raise ValueError("主体掩码缺少 mask_manifest.json 来源清单")
    content = manifest_path.read_bytes()
    manifest = json.loads(content.decode("utf-8"))
    if manifest.get("schema_version") != 1 or not isinstance(manifest.get("frames"), dict):
        raise ValueError("主体掩码清单格式无效")
    entries = manifest["frames"]
    if set(entries) != set(scene.names):
        raise ValueError("主体掩码清单必须完整对应全部数据集视角")
    paths, records = {}, []
    for name, size in zip(scene.names, scene.dimensions):
        item = entries[name]
        relative = item.get("mask", "")
        path = (root / relative).resolve()
        if not relative or not path.is_relative_to(root) or path.suffix.lower() != ".png":
            raise ValueError(f"主体掩码路径无效：{name}")
        if not path.is_file():
            raise ValueError(f"主体掩码缺少视角：{name}")
        if item.get("reviewed") is not True and not allow_unreviewed:
            raise ValueError(f"主体掩码尚未逐帧复查：{name}")
        image_path = Path(scene.image_dir) / name
        if hashlib.sha256(image_path.read_bytes()).hexdigest() != item.get("image_sha256"):
            raise ValueError(f"主体掩码参考图摘要不一致：{name}")
        mask_content = path.read_bytes()
        mask_hash = hashlib.sha256(mask_content).hexdigest()
        if mask_hash != item.get("mask_sha256"):
            raise ValueError(f"主体掩码文件摘要不一致：{name}")
        mask = read_binary_mask(mask_content, size["width"], size["height"], name)
        paths[name] = str(path)
        records.append({"name": name, "mask": relative, "mask_sha256": mask_hash,
                        "image_sha256": item["image_sha256"], "reviewed": item.get("reviewed") is True,
                        "foreground_ratio": float(np.count_nonzero(mask) / mask.size)})
    return paths, {"manifest_sha256": hashlib.sha256(content).hexdigest(),
                   "kind": manifest.get("kind", "unspecified"),
                   "allow_unreviewed": bool(allow_unreviewed), "frames": records}


def validate_mask_weights(foreground, background):
    """命令行绕过界面配置时也必须拒绝非有限值、布尔值和无效权重。"""
    for value in (foreground, background):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 < value <= 10:
            raise ValueError("主体掩码损失权重必须为大于 0 且不超过 10 的有限数字")


def weighted_l1(prediction, target, mask, foreground, background):
    """权重同时按像素和颜色通道归一化，等权时与普通平均绝对误差一致。"""
    if prediction.shape != target.shape or mask.shape != prediction.shape[1:]:
        raise ValueError("主体加权损失的图像或掩码尺寸不匹配")
    weight = background + (foreground - background) * mask
    # 先平均颜色通道，避免仅除以空间权重导致 RGB 损失错误放大三倍。
    error = (prediction - target).abs().mean(dim=0)
    return (error * weight).sum() / weight.sum().clamp_min(1e-12)
