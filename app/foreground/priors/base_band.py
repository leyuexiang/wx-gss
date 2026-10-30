"""圆罐候选底边的有限亮带恢复，仅用于现有固定素材的独立实验。"""

import cv2
import numpy as np


def preserve_base_band(rgb, mask):
    """只延伸与底边相连的亮带；不跨越暗缝寻找桌面上的第二条反射。"""
    if (rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3
            or mask.shape != rgb.shape[:2] or mask.dtype != np.uint8
            or not np.isin(mask, (0, 255)).all() or not mask.any()):
        raise ValueError('底圈恢复需要同尺寸 RGB 八位图和非空二值掩码')
    y, x = np.nonzero(mask)
    width = int(x.max() - x.min() + 1)
    limit = max(2, int(round(width * .025)))
    # 金属亮圈可能是银白或暖金色，排除绿色/蓝色背景；颜色只作局部证据。
    red, green, blue = rgb.astype(np.int16).transpose(2, 0, 1)
    value = rgb.max(axis=2)
    bright = ((value >= 135) & (red >= green - 18) & (red >= blue - 10)
              & (rgb.min(axis=2) >= 70))
    bright = cv2.medianBlur(bright.astype(np.uint8), 3).astype(bool)
    columns = np.arange(mask.shape[1])
    exists = (mask > 0).any(axis=0)
    bottom = mask.shape[0] - 1 - np.argmax((mask > 0)[::-1], axis=0)
    # 只检查原掩码最下方附近，防止弧形侧边被当作罐底向外扩张。
    active = exists & (bottom >= y.max() - .20 * width)
    result = mask.copy()
    # 同时推进所有列，每列遇到第一个暗像素即停止，不允许跳过暗缝。
    for delta in range(1, limit + 1):
        rows = np.minimum(bottom + delta, mask.shape[0] - 1)
        active &= (bottom + delta < mask.shape[0]) & bright[rows, columns]
        result[rows[active], columns[active]] = 255
    # 逐列亮度可能把窄反光误作金属边，延伸量取邻域中值上限，只撤回孤立尖刺。
    extension = (result > mask).sum(axis=0).astype(np.float32)
    kernel = max(3, int(round(width * .05)) | 1)
    padded = np.pad(extension, kernel // 2, mode='edge')
    smooth = np.median(np.lib.stride_tricks.sliding_window_view(padded, kernel), axis=1)
    extension = np.minimum(extension, smooth)
    result = mask.copy()
    for delta in range(1, limit + 1):
        selected = exists & (extension >= delta) & (bottom + delta < mask.shape[0])
        result[bottom[selected] + delta, columns[selected]] = 255
    added = result > mask
    return result, {'added_pixels': int(added.sum()), 'max_extension_pixels': limit,
                    'columns_extended': int(added.any(axis=0).sum()), 'reviewed': False,
                    'scope': '底边连续亮带局部恢复；专用颜色证据，非人工真值'}
