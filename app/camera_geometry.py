"""训练器和接口共用的相机坐标变换，不依赖显卡训练框架。"""

import json
import os

import numpy as np


def normalize_camera_translations(camtoworlds, center, scale):
    """世界尺度变化只作用于相机中心，旋转必须仍是正交矩阵。"""
    result = camtoworlds.copy()
    result[:, :3, 3] = (result[:, :3, 3] - center) * scale
    return result


def camera_normalization(positions, dataset_dir):
    """严格留出协议可固定训练相机的归一化，防止验证相机影响坐标尺度。"""
    manifest = os.path.join(dataset_dir, "scene_normalization.json")
    if os.path.isfile(manifest):
        with open(manifest, encoding="utf-8") as source:
            stored = json.load(source)
        center = np.asarray(stored["center"], dtype=np.float64)
        scale = float(stored["scale"])
        if center.shape != (3,) or not np.isfinite(center).all() or not np.isfinite(scale) or scale <= 0:
            raise ValueError("训练相机归一化记录无效")
        return center, scale
    center = positions.mean(axis=0)
    max_dist = np.linalg.norm(positions - center, axis=-1).max()
    return center, 1.0 / (max_dist * 1.05) if max_dist > 1e-6 else 1.0
