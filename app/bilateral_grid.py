"""按双边网格的局部仿射模型实现训练外观补偿，不改变导出高斯颜色格式。"""

import torch
from torch import nn
import torch.nn.functional as F


class BilateralGrid(nn.Module):
    def __init__(self, views, shape=(8, 8, 4)):
        super().__init__()
        width, height, intensity = shape
        # 常数恒等变换不消耗随机数，保证开关网格不会改变高斯初始化。
        grid = torch.zeros(views, 12, intensity, height, width)
        grid[:, [0, 5, 10]] = 1
        self.grid = nn.Parameter(grid)
        self._coordinates = {}

    def forward(self, rgb, index):
        height, width = rgb.shape[1:]
        key = (height, width, rgb.device, rgb.dtype)
        if key not in self._coordinates:
            self._coordinates.clear()
            # 缓存固定位置网格；亮度坐标依赖当前渲染，必须每步重新构建。
            y, x = torch.meshgrid(torch.linspace(-1, 1, height, device=rgb.device, dtype=rgb.dtype),
                                  torch.linspace(-1, 1, width, device=rgb.device, dtype=rgb.dtype), indexing="ij")
            self._coordinates[key] = (x, y)
        x, y = self._coordinates[key]
        guide = rgb.mean(dim=0).clamp(0, 1) * 2 - 1
        coordinates = torch.stack((x, y, guide), dim=-1)[None, None]
        affine = F.grid_sample(self.grid[index:index + 1], coordinates,
                               padding_mode="border", align_corners=True).reshape(3, 4, height, width)
        return (affine[:, :3] * rgb[None]).sum(dim=1) + affine[:, 3]

    def smoothness(self, index):
        # 每步只正则当前训练图，避免验证图网格参与训练和全图网格重复计算。
        grid = self.grid[index:index + 1]
        return sum((grid.diff(dim=dim)).square().mean() for dim in (2, 3, 4))
