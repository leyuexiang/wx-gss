"""转台候选轨迹的圆周诊断，不把拟合良好当作真实几何精度。"""

import numpy as np


def analyze_orbit(centers):
    """按输入顺序拟合主平面和圆，输出相对误差及角度覆盖，不负责识别拍摄类型。"""
    points = np.asarray(centers, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) < 4 or not np.isfinite(points).all():
        raise ValueError("轨迹需要至少四个有限三维相机中心")
    origin = points.mean(axis=0)
    _, singular, axes = np.linalg.svd(points - origin, full_matrices=False)
    if singular[1] <= max(singular[0] * 1e-6, 1e-12):
        raise ValueError("相机轨迹退化为点或直线，无法评估圆周覆盖")
    plane = (points - origin) @ axes[:2].T
    solution, _, _, _ = np.linalg.lstsq(np.c_[2 * plane, np.ones(len(plane))],
                                       (plane ** 2).sum(axis=1), rcond=None)
    radial = plane - solution[:2]
    radii = np.linalg.norm(radial, axis=1)
    angles = np.unwrap(np.arctan2(radial[:, 1], radial[:, 0]))
    ordered = np.sort(np.mod(angles, 2 * np.pi))
    gaps = np.diff(np.r_[ordered, ordered[0] + 2 * np.pi])
    return {"plane_residual_ratio": float(singular[-1] / singular[0]),
            "radius_cv": float(radii.std() / radii.mean()),
            "absolute_turns": float(abs(angles[-1] - angles[0]) / (2 * np.pi)),
            "max_angular_gap_degrees": float(np.degrees(gaps.max())),
            "max_step_degrees": float(np.degrees(np.abs(np.diff(angles))).max()),
            "scope": "仅适用于圆周转台候选轨迹；无真实标定，不能据此证明三维正确"}
