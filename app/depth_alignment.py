"""深度诊断的稳健逆深度对齐；不负责启用训练。"""

import numpy as np


def fit_inverse_depth(predicted, target, min_samples=30):
    """拟合目标逆深度 = a / 预测深度 + b，拒绝无跨度或反向关系。"""
    predicted, target = np.asarray(predicted), np.asarray(target)
    if predicted.shape != target.shape or predicted.ndim != 1:
        raise ValueError("深度锚点必须是一维同形数组")
    valid = np.isfinite(predicted) & np.isfinite(target) & (predicted > 0) & (target > 0)
    x, y = 1. / predicted[valid], 1. / target[valid]
    if len(x) < min_samples:
        raise ValueError("可靠深度锚点不足")
    center, spread = np.median(x), np.percentile(x, 90) - np.percentile(x, 10)
    if spread < max(1e-8, abs(center) * 1e-4):
        raise ValueError("预测深度跨度不足，不能标定尺度和偏移")
    # 标准化避免近似平面上斜率与偏移的数值病态；稳健权重抑制少量错配。
    design = np.c_[(x - center) / spread, np.ones(len(x))]
    weights = np.ones(len(x))
    for _ in range(12):
        coef = np.linalg.lstsq(design * np.sqrt(weights[:, None]), y * np.sqrt(weights), rcond=None)[0]
        residual = design @ coef - y
        sigma = max(1e-9, 1.4826 * np.median(np.abs(residual - np.median(residual))))
        weights = np.minimum(1., 1.345 * sigma / np.maximum(np.abs(residual), 1e-12))
    scale, offset = coef[0] / spread, coef[1] - coef[0] * center / spread
    if not np.isfinite([scale, offset]).all() or scale <= 0:
        raise ValueError("预测深度与稀疏几何方向不一致")
    return {"scale": float(scale), "offset": float(offset), "samples": len(x),
            "inverse_depth_span": float(spread)}


def apply_inverse_alignment(depth, fit):
    """仅变换有限正深度；负逆深度和缺测不插补为真值。"""
    depth = np.asarray(depth)
    valid = np.isfinite(depth) & (depth > 0)
    inverse = np.zeros(depth.shape, dtype=np.float64)
    inverse[valid] = fit["scale"] / depth[valid] + fit["offset"]
    valid &= np.isfinite(inverse) & (inverse > 1e-8)
    result = np.zeros(depth.shape, dtype=np.float32)
    result[valid] = 1. / inverse[valid]
    return result


def alignment_errors(prediction, target):
    """以留出的空间块锚点检查相对 Z 误差；缺测也计入失败比例。"""
    prediction, target = np.asarray(prediction), np.asarray(target)
    valid = np.isfinite(prediction) & (prediction > 0) & np.isfinite(target) & (target > 0)
    errors = np.abs(prediction[valid] - target[valid]) / target[valid]
    return {"samples": int(len(target)), "valid_samples": int(valid.sum()),
            "valid_ratio": float(valid.mean()) if len(valid) else 0.,
            "median_relative_z": float(np.median(errors)) if len(errors) else None,
            "p90_relative_z": float(np.percentile(errors, 90)) if len(errors) else None}
