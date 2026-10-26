"""生成合成"照片"用于冒烟测试：精确针孔投影渲染带纹理的 3D 点场景，
SIFT 可提取大量特征，COLMAP 应能重建出环绕轨迹。

用法: python3 gen_synth_dataset.py <out_dir> [n_views]
"""

import math
import os
import sys
import zipfile

import numpy as np
from PIL import Image

W, H = 800, 600
FX = FY = 800.0
CX, CY = W / 2, H / 2


def look_at(eye, target=(0, 0, 0), up=(0, -1, 0)):
    """返回 world->camera 旋转矩阵（OpenCV 约定：z 朝前）。"""
    eye = np.asarray(eye, float)
    z = np.asarray(target, float) - eye
    z /= np.linalg.norm(z)
    x = np.cross(z, np.asarray(up, float))
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    return np.stack([x, y, z])  # R: world->cam


def render(points, colors, R, t, radius=2):
    """z-buffer 点渲染，返回 HxWx3 uint8。"""
    pts_cam = (R @ points.T).T + t
    z = pts_cam[:, 2]
    valid = z > 0.1
    pts_cam, cols, z = pts_cam[valid], colors[valid], z[valid]
    u = FX * pts_cam[:, 0] / z + CX
    v = FY * pts_cam[:, 1] / z + CY

    img = np.full((H, W, 3), 245, np.uint8)
    zbuf = np.full((H, W), 1e9)
    order = np.argsort(-z)
    for i in order:
        ui, vi = int(round(u[i])), int(round(v[i]))
        if not (radius <= ui < W - radius and radius <= vi < H - radius):
            continue
        for du in range(-radius, radius + 1):
            for dv in range(-radius, radius + 1):
                if z[i] < zbuf[vi + dv, ui + du]:
                    zbuf[vi + dv, ui + du] = z[i]
                    img[vi + dv, ui + du] = cols[i]
    return img


def main(out_dir, n_views=36):
    rng = np.random.default_rng(42)
    os.makedirs(out_dir, exist_ok=True)

    pts_list, col_list = [], []
    # 随机彩色点云（高饱和色 → SIFT 特征）
    p = rng.uniform(-1, 1, (4000, 3))
    c = rng.integers(0, 256, (4000, 3), dtype=np.uint8)
    pts_list.append(p); col_list.append(c)
    # 地面棋盘格
    xs = np.linspace(-2, 2, 25)
    gx, gz = np.meshgrid(xs, xs)
    gy = np.full_like(gx, -1.2)
    grid = np.stack([gx, gy, gz], -1).reshape(-1, 3)
    checker = ((np.arange(25)[:, None] + np.arange(25)[None, :]) % 2).reshape(-1)
    grid_col = np.repeat(np.where(checker[:, None], 30, 220).astype(np.uint8), 3, axis=1)
    pts_list.append(grid); col_list.append(grid_col)
    # 一个彩色立方体顶点网格
    cube = np.array([[i, j, k] for i in np.linspace(-.5, .5, 8)
                     for j in np.linspace(-.5, .5, 8)
                     for k in np.linspace(-.5, .5, 8)])
    cube += np.array([0.0, 0.0, 0.0])
    cube_col = ((cube + 0.5) * 255).astype(np.uint8)
    pts_list.append(cube); col_list.append(cube_col)

    points = np.concatenate(pts_list)
    colors = np.concatenate(col_list)

    zip_path = os.path.join(out_dir, "synth_photos.zip")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_STORED) as zf:
        for i in range(n_views):
            ang = 2 * math.pi * i / n_views
            elev = 0.15 * math.sin(ang * 3)
            eye = [3.2 * math.cos(ang), 0.5 + elev, 3.2 * math.sin(ang)]
            R = look_at(eye)
            t = -R @ np.asarray(eye)
            img = render(points, colors, R, t)
            name = f"IMG_{i:04d}.jpg"
            path = os.path.join(out_dir, name)
            Image.fromarray(img).save(path, quality=95)
            zf.write(path, name)
            os.remove(path)
    print(f"generated {n_views} views -> {zip_path}")


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 36)
