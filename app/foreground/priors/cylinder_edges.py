"""红罐的边缘引导圆柱轮廓候选；没有逐像素真值，不能作为通用分割。"""

import cv2
import numpy as np


def cylinder_edge_mask(rgb):
    """以红色主体定位，拟合真实边缘并补齐圆弧，返回掩码和不确定性记录。"""
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    red = (((hsv[:, :, 0] < 16) | (hsv[:, :, 0] > 165))
           & (hsv[:, :, 1] > 100) & (hsv[:, :, 2] > 65)).astype(np.uint8) * 255
    # 仅定位时连起细竖条，避免条码将罐身分成两个区域。
    red = cv2.morphologyEx(red, cv2.MORPH_CLOSE, np.ones((31, 31), np.uint8))
    contours, _ = cv2.findContours(red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError("圆柱轮廓缺少红色定位证据")
    x, y, width, height = cv2.boundingRect(max(contours, key=cv2.contourArea))
    h, w = rgb.shape[:2]
    if min(width, height) < 40 or width * height < w * h * .02:
        raise ValueError("圆柱主体过小，拒绝拟合边缘")
    values = rgb.astype(np.float32)
    excess = cv2.GaussianBlur(values[:, :, 1] - values[:, :, 0], (5, 5), 0)
    gx = cv2.Sobel(excess, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(excess, cv2.CV_32F, 0, 1, ksize=3)
    gray = cv2.GaussianBlur(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32), (5, 5), 0)
    brightness_gradient = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)

    def side_edge(left):
        anchor = x if left else x + width
        lo, hi = max(1, int(anchor - .13 * width)), min(w - 1, int(anchor + .13 * width))
        rows = np.linspace(max(1, y - .14 * height), max(2, y - .04 * height), 25, dtype=int)
        if hi <= lo or rows.max() >= h:
            raise ValueError("盖沿边缘搜索超出图像范围")
        sign = -1 if left else 1
        return float(np.median([lo + np.argmax(sign * gx[row, lo:hi]) for row in rows]))

    left, right = side_edge(True), side_edge(False)
    cx, radius = (left + right) / 2, (right - left) / 2
    if not .35 * width < radius < .7 * width:
        raise ValueError("盖沿宽度与罐身定位不一致")

    def curve(top):
        columns = np.linspace(cx - .85 * radius, cx + .85 * radius, 95).astype(int)
        lo = max(2, int(y - .42 * height)) if top else max(2, int(y + height - .22 * width))
        hi = min(h - 2, int(y - .08 * height)) if top else min(h - 2, int(y + height + .07 * width))
        if hi <= lo or columns.min() < 0 or columns.max() >= w:
            raise ValueError("圆弧搜索超出图像范围")
        gradient = gy if top else brightness_gradient
        locations = np.array([lo + np.argmax(-gradient[lo:hi, col]) for col in columns])
        u = (columns - cx) / radius
        design = np.c_[np.ones(len(u)), u, 1 - np.sqrt(1 - u * u)]
        good = np.ones(len(u), dtype=bool)
        for _ in range(5):
            if good.sum() < 20:
                raise ValueError("可用圆弧边缘不足，拒绝继续拟合")
            coefficients = np.linalg.lstsq(design[good], locations[good], rcond=None)[0]
            residual = np.abs(design @ coefficients - locations)
            good = residual < max(5, float(np.median(residual)) * 2)
        return coefficients, int(good.sum())

    top, top_inliers = curve(True)
    bottom, bottom_inliers = curve(False)
    fallback = not (.15 < top[2] / radius < .35 and abs(top[1] / radius) <= .07)
    if fallback:
        # 当前固定拍摄圆罐的形状先验；必须记录启用，不能解释为真实观测。
        top[1], top[2] = -.022 * radius, .245 * radius
        # 反光可能让算法错把盖顶前沿当作外沿；不能只修曲率而保留错误的高度。
        expected_top = y - .29 * height
        if abs(top[0] - expected_top) > .035 * height:
            top[0] = max(0, expected_top)
    bottom[2] = np.clip(bottom[2], -.5 * radius, -.05 * radius)
    rows, left_side, right_side = [], [], []
    for row in np.linspace(y + .15 * height, y + .7 * height, 40, dtype=int):
        columns = np.flatnonzero(red[row])
        columns = columns[(columns > x - .1 * width) & (columns < x + 1.1 * width)]
        if len(columns) > width * .35:
            rows.append(row)
            left_side.append(columns.min())
            right_side.append(columns.max())
    if len(rows) < 20:
        raise ValueError("罐身侧边证据不足")
    def robust_side(side):
        """条码白底不是物体侧边，用一致直线剔除被白底推向罐内的红色边缘。"""
        vertical, horizontal = np.array(rows), np.array(side)
        # 固定最多 40 个样本，批量评估直线假设，避免每帧执行嵌套 Python 循环。
        first, second = np.triu_indices(len(rows), 1)
        valid = vertical[second] != vertical[first]
        first, second = first[valid], second[valid]
        if not len(first):
            raise ValueError("罐身侧边缺少可用的纵向跨度")
        slopes = (horizontal[second] - horizontal[first]) / (vertical[second] - vertical[first])
        intercepts = horizontal[first] - slopes * vertical[first]
        residual = np.abs(horizontal[None] - slopes[:, None] * vertical[None] - intercepts[:, None])
        good = residual < max(3, .01 * width)
        counts = good.sum(axis=1)
        errors = (residual * good).sum(axis=1) / np.maximum(counts, 1)
        best = good[np.lexsort((errors, -counts))[0]]
        if best.sum() < 12:
            raise ValueError("罐身侧边缺少一致直线证据")
        return np.linalg.lstsq(np.c_[vertical[best], np.ones(best.sum())], horizontal[best], rcond=None)[0]

    base_y = bottom[0] + bottom[2]
    base_left = float(robust_side(left_side) @ [base_y, 1])
    base_right = float(robust_side(right_side) @ [base_y, 1])
    base_cx, base_radius = (base_left + base_right) / 2, (base_right - base_left) / 2
    if not .3 * width < base_radius < .7 * width or bottom[0] <= top[0] + .5 * height:
        raise ValueError("圆柱上下边缘次序或宽度异常")
    u = np.linspace(-1, 1, 200)
    q = 1 - np.sqrt(1 - u * u)
    upper = np.c_[cx + u * radius, top[0] + top[1] * u + top[2] * q]
    lower = np.c_[base_cx + u * base_radius, bottom[0] + bottom[1] * u + bottom[2] * q]
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [np.rint(np.concatenate((upper, lower[::-1]))).astype(np.int32)], 255)
    return mask, {"top_shape_prior_used": bool(fallback), "top_inliers": top_inliers,
                  "bottom_inliers": bottom_inliers, "reviewed": False,
                  "scope": "圆罐几何候选，包含截面先验，不等同于人工标注"}
