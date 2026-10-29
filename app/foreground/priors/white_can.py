"""素材专用的颜色与形状先验，保持历史算法，不作为通用分割默认值。"""

import cv2
import numpy as np


def can_mask(rgb):
    """先定位中央白色罐身，再用金色轮廓恢复盖子，保留盖顶的绿色图案。"""
    height, width = rgb.shape[:2]
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    red, green, blue = rgb.astype(np.int16).transpose(2, 0, 1)
    # 高光绿布也可能低饱和，额外排除绿色偏色，避免凸包把背景连到罐身。
    neutral = (green <= red + 8) & (green <= blue + 16)
    white = ((hsv[:, :, 1] < 70) & (hsv[:, :, 2] > 120) & neutral).astype(np.uint8) * 255
    region = np.zeros_like(white)
    region[int(.15 * height):int(.94 * height), int(.28 * width):int(.72 * width)] = 255
    white &= region
    white = cv2.morphologyEx(white, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    contours, _ = cv2.findContours(white, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError("中央没有找到白色罐身")
    body = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(body)
    if w * h < width * height * .025:
        raise ValueError("主体面积不足，拒绝继续生成错误掩码")
    roi = np.zeros_like(white)
    roi[max(0, int(y - h * .42)):min(height, y + h + 8), max(0, x - 16):min(width, x + w + 16)] = 255
    gold = ((hsv[:, :, 0] < 40) & (hsv[:, :, 1] > 45) & (hsv[:, :, 2] > 70)
            & (red >= green + 3))
    body_only = np.zeros_like(white)
    cv2.drawContours(body_only, [body], -1, 255, cv2.FILLED)
    # 盖顶的金边可能被绿色图案隔开，保留上方分离的金边，罐身仅取主连通分量。
    gold[y + 20:] = False
    candidate = (((body_only > 0) | gold) & (roi > 0)).astype(np.uint8) * 255
    candidate = cv2.morphologyEx(candidate, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    yy, xx = np.nonzero(candidate)
    hull = cv2.convexHull(np.stack((xx, yy), axis=1).astype(np.int32))
    result = np.zeros_like(white)
    cv2.drawContours(result, [hull], -1, 255, cv2.FILLED)
    return result


def refine_can_mask(rgb, seed):
    """只在粗轮廓附近用图割细化，恢复绿色盖顶和圆形边缘，不允许背景任意扩张。"""
    small = cv2.resize(rgb, None, fx=.5, fy=.5)
    scaled = cv2.resize(seed, (small.shape[1], small.shape[0]), interpolation=cv2.INTER_NEAREST)
    outer = cv2.dilate(scaled, np.ones((29, 29), np.uint8))
    inner = cv2.erode(scaled, np.ones((21, 21), np.uint8))
    labels = np.full_like(scaled, cv2.GC_BGD)
    labels[outer > 0] = cv2.GC_PR_BGD
    labels[scaled > 0] = cv2.GC_PR_FGD
    labels[inner > 0] = cv2.GC_FGD
    cv2.setRNGSeed(42)
    cv2.grabCut(small, labels, None, np.zeros((1, 65)), np.zeros((1, 65)), 3, cv2.GC_INIT_WITH_MASK)
    foreground = ((labels == cv2.GC_FGD) | (labels == cv2.GC_PR_FGD)).astype(np.uint8) * 255
    return cv2.resize(foreground, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_NEAREST)
