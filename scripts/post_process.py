"""后处理：去除离群高斯点和低不透明度点。

使用方法：
    python post_process.py point_cloud.ply output_cleaned.ply
"""

import argparse
import numpy as np
from plyfile import PlyData, PlyElement


def remove_outliers(xyz, opacity, opacity_threshold=0.1, distance_percentile=95):
    """
    去除离群点和低不透明度点。

    参数：
    - opacity_threshold: 不透明度阈值（0-1），低于此值的点被删除
    - distance_percentile: 距离百分位数，超过此距离的点被视为离群点
    """
    # 1. 过滤低不透明度点
    mask = opacity >= opacity_threshold

    # 2. 过滤离群点（基于到中心的距离）
    center = np.median(xyz, axis=0)
    distances = np.linalg.norm(xyz - center, axis=1)
    dist_threshold = np.percentile(distances, distance_percentile)
    mask &= (distances <= dist_threshold)

    return mask


def clean_ply(input_path, output_path, opacity_threshold=0.1, distance_percentile=95):
    """清理 PLY 文件，去除杂点。"""
    plydata = PlyData.read(input_path)
    vertex = plydata['vertex']

    # 提取位置和不透明度
    xyz = np.stack([vertex['x'], vertex['y'], vertex['z']], axis=1)
    opacity = vertex['opacity']

    # 应用 sigmoid 转换（PLY 中存储的是 logit）
    opacity = 1 / (1 + np.exp(-opacity))

    # 过滤
    mask = remove_outliers(xyz, opacity, opacity_threshold, distance_percentile)

    print(f"原始点数: {len(xyz)}")
    print(f"保留点数: {mask.sum()}")
    print(f"删除点数: {(~mask).sum()} ({(~mask).sum() / len(xyz) * 100:.1f}%)")

    # 创建新的 PLY
    cleaned_vertex = vertex[mask]
    cleaned_plydata = PlyData([PlyElement.describe(cleaned_vertex, 'vertex')])
    cleaned_plydata.write(output_path)

    print(f"已保存到: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="清理 3DGS PLY 文件，去除杂点")
    parser.add_argument("input", help="输入 PLY 文件")
    parser.add_argument("output", help="输出 PLY 文件")
    parser.add_argument("--opacity", type=float, default=0.1,
                        help="不透明度阈值 (默认 0.1)")
    parser.add_argument("--distance", type=float, default=95,
                        help="距离百分位数 (默认 95)")

    args = parser.parse_args()
    clean_ply(args.input, args.output, args.opacity, args.distance)
