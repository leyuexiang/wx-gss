"""可靠稀疏观测的投影与逆深度监督，使用训练器一致的归一化坐标。"""

import numpy as np
import torch
import torch.nn.functional as F


def project_depth(points, camtoworld, intrinsic, width, height):
    """只监督图内且相机前方的点；相机轴向 Z 与欧氏距离不可混用。"""
    worldtocam = np.linalg.inv(camtoworld)
    camera = points @ worldtocam[:3, :3].T + worldtocam[:3, 3]
    projected = camera @ intrinsic.T
    valid = np.isfinite(camera).all(axis=1) & (camera[:, 2] > 1e-6)
    xy = projected[:, :2] / np.maximum(camera[:, 2:3], 1e-6)
    valid &= np.isfinite(xy).all(axis=1) & (xy[:, 0] >= 0) & (xy[:, 0] <= width - 1)
    valid &= (xy[:, 1] >= 0) & (xy[:, 1] <= height - 1)
    return xy[valid].astype(np.float32), camera[valid, 2].astype(np.float32)


def inverse_depth_loss(rendered, xy, target):
    """在真实观测位置双线性采样；不将无监督区域补成稠密深度。"""
    if len(target) == 0:
        return rendered.sum() * 0
    height, width = rendered.shape
    grid = torch.stack((xy[:, 0] * 2 / max(1, width - 1) - 1,
                        xy[:, 1] * 2 / max(1, height - 1) - 1), dim=-1)
    prediction = F.grid_sample(rendered[None, None], grid[None, :, None], align_corners=True).reshape(-1)
    # 极小正数截断避免除零；空深度仍产生有限惩罚，不静默跳过整帧。
    return (prediction.clamp_min(1e-4).reciprocal() - target.clamp_min(1e-4).reciprocal()).abs().mean()
