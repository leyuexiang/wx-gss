"""素材专用的颜色与形状先验，保持历史算法，不作为通用分割默认值。"""

import cv2
import numpy as np


def red_can_mask(rgb):
    """以红色罐身定位金盖区域，排除绿色背景和底部转台反射。"""
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    red = (((hsv[:, :, 0] < 16) | (hsv[:, :, 0] > 165))
           & (hsv[:, :, 1] > 100) & (hsv[:, :, 2] > 65)).astype(np.uint8) * 255
    red = cv2.morphologyEx(red, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    contours, _ = cv2.findContours(red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError("没有找到红色罐身")
    body = max(contours, key=cv2.contourArea)
    x, y, width, height = cv2.boundingRect(body)
    if width * height < rgb.shape[0] * rgb.shape[1] * .02:
        raise ValueError("红色主体过小，拒绝生成掩码")
    r, g, b = rgb.astype(np.int16).transpose(2, 0, 1)
    warm = (r >= g - 20) & (r >= b + 8) & (hsv[:, :, 2] > 60)
    roi = np.zeros_like(red)
    roi[max(0, int(y - .52 * height)):min(rgb.shape[0], int(y + 1.08 * height)),
        max(0, int(x - .08 * width)):min(rgb.shape[1], int(x + 1.08 * width))] = 255
    body_only = np.zeros_like(red)
    cv2.drawContours(body_only, [body], -1, 255, cv2.FILLED)
    candidate = ((warm | (body_only > 0)) & (roi > 0)).astype(np.uint8) * 255
    candidate = cv2.morphologyEx(candidate, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    contours, _ = cv2.findContours(candidate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    hull = cv2.convexHull(max(contours, key=cv2.contourArea))
    mask = np.zeros_like(red)
    cv2.drawContours(mask, [hull], -1, 255, cv2.FILLED)
    return mask


def gold_cap_mask(rgb):
    """仅用于深灰桌面上的金属顶底照片，保留圆面内的印刷文字。"""
    r, g, b = rgb.astype(np.int16).transpose(2, 0, 1)
    seed = ((r >= g - 15) & (r >= b + 12) & (r > 85)).astype(np.uint8) * 255
    seed = cv2.morphologyEx(seed, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    contours, _ = cv2.findContours(seed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError("补拍照片没有找到金属圆面")
    contour = max(contours, key=cv2.contourArea)
    if len(contour) < 5 or cv2.contourArea(contour) < rgb.shape[0] * rgb.shape[1] * .05:
        raise ValueError("金属圆面面积或边界不足")
    mask = np.zeros_like(seed)
    cv2.ellipse(mask, cv2.fitEllipse(contour), 255, cv2.FILLED)
    return mask
