"""显式前景适配器注册表，不提供未经验证的自动颜色选择。"""

from dataclasses import dataclass
from typing import Callable, Protocol

import numpy as np


class ForegroundProvider(Protocol):
    """统一输入 RGB 八位图，输出同尺寸二值前景；粗分割供抽帧评分复用。"""

    def coarse(self, rgb: np.ndarray) -> np.ndarray: ...
    def segment(self, rgb: np.ndarray) -> np.ndarray: ...
    def describe(self) -> dict: ...


@dataclass(frozen=True)
class PriorProvider:
    """仅包装现有先验，迁移接口时不改变阈值、尺度或图割流程。"""

    name: str
    scope: str
    coarse_algorithm: Callable

    def coarse(self, rgb):
        """在算法运行前拒绝通道、位深和空图错误。"""
        if not isinstance(rgb, np.ndarray) or rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
            raise ValueError("前景输入必须为 RGB 八位图")
        if min(rgb.shape[:2]) < 16:
            raise ValueError("前景图像尺寸不足")
        mask = self.coarse_algorithm(rgb)
        if mask.shape != rgb.shape[:2] or mask.dtype != np.uint8 or not np.all((mask == 0) | (mask == 255)):
            raise ValueError("适配器必须返回同尺寸二值八位掩码")
        if not 0 < np.count_nonzero(mask) < mask.size:
            raise ValueError("适配器必须同时保留前景和背景")
        return mask

    def segment(self, rgb):
        """保持历史半尺寸图割细化，不在迁移过程中更改轮廓。"""
        from .priors.white_can import refine_can_mask
        return refine_can_mask(rgb, self.coarse(rgb))

    def describe(self):
        """未人工复查的算法输出不能伪装成已确认的标注。"""
        return {"name": self.name, "version": 1, "kind": "color_shape_prior",
                "scope": self.scope, "reviewed": False}


def get_provider(name) -> ForegroundProvider:
    """显式选择专用插件；未知或所谓通用模式直接失败，避免静默回退。"""
    if name == "white_can":
        from .priors.white_can import can_mask
        return PriorProvider(name, "中央白色圆罐、金边；绿色盖顶需保留", can_mask)
    if name in ("red_can", "gold_cap"):
        from .priors.red_can import red_can_mask, gold_cap_mask
        return PriorProvider(name, "红色罐身与金盖" if name == "red_can" else "灰色桌面上的金属圆面",
                             red_can_mask if name == "red_can" else gold_cap_mask)
    raise ValueError(f"不支持的前景适配器：{name}；通用分割尚未接入")
