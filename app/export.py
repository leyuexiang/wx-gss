"""把训练好的高斯参数导出为标准 3DGS PLY 和 .splat 两种格式。"""

import numpy as np
import torch

SH_C0 = 0.28209479177387814


def _to_numpy(params: dict):
    def t(name):
        return params[name].detach().cpu().numpy().astype(np.float32)
    means = t("means")                        # [N,3]
    scales = t("scales")                      # [N,3] log
    quats = t("quats")                        # [N,4] wxyz
    opacities = t("opacities")[:, None]       # [N,1] logit
    sh0 = t("sh0")                            # [N,1,3]
    shN = t("shN")                            # [N,K,3]
    return means, scales, quats, opacities, sh0, shN


def export_ply(params: dict, path: str):
    means, scales, quats, opacities, sh0, shN = _to_numpy(params)
    N = means.shape[0]
    f_dc = sh0.reshape(N, 3)
    f_rest = shN.transpose(0, 2, 1).reshape(N, -1)   # Inria 布局：通道优先
    normals = np.zeros((N, 3), dtype=np.float32)

    attrs = [means, normals, f_dc, f_rest, opacities, scales, quats]
    names = (
        ["x", "y", "z", "nx", "ny", "nz"]
        + [f"f_dc_{i}" for i in range(f_dc.shape[1])]
        + [f"f_rest_{i}" for i in range(f_rest.shape[1])]
        + ["opacity"]
        + [f"scale_{i}" for i in range(3)]
        + [f"rot_{i}" for i in range(4)]
    )
    data = np.concatenate(attrs, axis=1).astype(np.float32)

    header = (
        "ply\nformat binary_little_endian 1.0\n"
        f"element vertex {N}\n"
        + "".join(f"property float {n}\n" for n in names)
        + "end_header\n"
    )
    with open(path, "wb") as f:
        f.write(header.encode())
        data.tofile(f)


def export_splat(params: dict, path: str):
    means, scales, quats, opacities, sh0, _ = _to_numpy(params)
    N = means.shape[0]
    rgb = np.clip(sh0.reshape(N, 3) * SH_C0 + 0.5, 0, 1)
    alpha = 1.0 / (1.0 + np.exp(-opacities))
    color = (np.concatenate([rgb, alpha], axis=1) * 255).astype(np.uint8)
    rot = quats / np.linalg.norm(quats, axis=1, keepdims=True)
    rot_u8 = np.clip(rot * 128 + 128, 0, 255).astype(np.uint8)   # wxyz
    scale_lin = np.exp(scales)

    buf = np.zeros((N, 32), dtype=np.uint8)
    buf[:, 0:12] = np.frombuffer(np.ascontiguousarray(means, dtype="<f4").tobytes(), dtype=np.uint8).reshape(N, 12)
    buf[:, 12:24] = np.frombuffer(np.ascontiguousarray(scale_lin, dtype="<f4").tobytes(), dtype=np.uint8).reshape(N, 12)
    buf[:, 24:28] = color
    buf[:, 28:32] = rot_u8
    with open(path, "wb") as f:
        f.write(buf.tobytes())


def export_splat_from_ply(ply_path: str, splat_path: str):
    """从已清理的 PLY 导出查看文件，保证浏览器看到的点与交付模型一致。"""
    from plyfile import PlyData

    vertex = PlyData.read(ply_path)["vertex"]
    count = len(vertex)
    means = np.stack([vertex[name] for name in ("x", "y", "z")], axis=1).astype("<f4")
    scales = np.exp(np.stack([vertex[f"scale_{i}"] for i in range(3)], axis=1)).astype("<f4")
    sh0 = np.stack([vertex[f"f_dc_{i}"] for i in range(3)], axis=1)
    rgb = np.clip(sh0 * SH_C0 + 0.5, 0, 1)
    opacity = np.asarray(vertex["opacity"])
    alpha = 1.0 / (1.0 + np.exp(-opacity))
    color = (np.concatenate([rgb, alpha[:, None]], axis=1) * 255).astype(np.uint8)
    rot = np.stack([vertex[f"rot_{i}"] for i in range(4)], axis=1)
    rot /= np.clip(np.linalg.norm(rot, axis=1, keepdims=True), 1e-8, None)
    rot_u8 = np.clip(rot * 128 + 128, 0, 255).astype(np.uint8)

    buf = np.empty((count, 32), dtype=np.uint8)
    buf[:, 0:12] = means.view(np.uint8).reshape(count, 12)
    buf[:, 12:24] = scales.view(np.uint8).reshape(count, 12)
    buf[:, 24:28] = color
    buf[:, 28:32] = rot_u8
    with open(splat_path, "wb") as output:
        output.write(buf.tobytes())


def clean_gaussians(ply_path: str, opacity_threshold: float = 0.1, distance_percentile: float = 95):
    """
    清理高斯点：删除低不透明度和离群点

    参数:
    - ply_path: PLY 文件路径（会被覆盖）
    - opacity_threshold: 不透明度阈值（0.0-1.0），低于此值的点被删除
    - distance_percentile: 距离百分位数（0-100），超过此距离的点被视为离群点

    返回: (保留数量, 删除数量)
    """
    from plyfile import PlyData, PlyElement

    # 读取 PLY
    plydata = PlyData.read(ply_path)
    vertex = plydata['vertex']

    # 提取数据
    xyz = np.stack([vertex['x'], vertex['y'], vertex['z']], axis=1)
    if len(xyz) == 0:
        raise ValueError("模型没有高斯点，不能执行清理")
    opacity_logit = vertex['opacity']

    # 应用 sigmoid 转换
    opacity = 1.0 / (1.0 + np.exp(-opacity_logit))

    # 过滤 1: 不透明度
    mask_opacity = opacity >= opacity_threshold

    # 过滤 2: 距离中心（删除离群点）
    center = np.median(xyz, axis=0)
    distances = np.linalg.norm(xyz - center, axis=1)
    dist_threshold = np.percentile(distances, distance_percentile)
    mask_distance = distances <= dist_threshold

    # 组合过滤
    mask = mask_opacity & mask_distance
    if not mask.any():
        raise ValueError("清理条件删除了全部高斯点，请降低过滤阈值")

    kept = mask.sum()
    removed = (~mask).sum()

    print(f"清理前: {len(xyz)} 个高斯")
    print(f"  - 不透明度过滤: 删除 {(~mask_opacity).sum()} 个 (< {opacity_threshold})")
    print(f"  - 距离过滤: 删除 {(~mask_distance).sum()} 个 (> {distance_percentile}%)")
    print(f"清理后: 保留 {kept} 个，删除 {removed} 个 ({removed / len(xyz) * 100:.1f}%)")

    # 保存清理后的模型
    cleaned_vertex = vertex[mask]
    cleaned_plydata = PlyData([PlyElement.describe(cleaned_vertex, 'vertex')])
    cleaned_plydata.write(ply_path)

    return kept, removed


def export_gaussians(params: dict, out_dir: str, auto_cleanup: bool = True,
                    cleanup_threshold: float = 0.1, denoise_config: dict = None):
    """
    导出高斯模型（PLY + SPLAT 格式）

    参数:
    - params: 训练的高斯参数
    - out_dir: 输出目录
    - auto_cleanup: 是否自动清理杂点
    - cleanup_threshold: 清理阈值（不透明度）
    - denoise_config: 去噪配置（包含尺寸过滤参数）
    """
    import os
    import torch

    # 解析去噪配置
    if denoise_config is None:
        denoise_config = {}

    # 在任何尺寸或离群过滤之前留存完整结果，便于同一次训练的后处理对照。
    raw_path = os.path.join(out_dir, "point_cloud_raw.ply")
    export_ply(params, raw_path)
    raw_count = int(params["means"].shape[0])
    removed_by_size = 0

    size_filtering = denoise_config.get('size_filtering', {})
    enable_size_filtering = denoise_config.get('enable', True) and size_filtering.get('enable', False)
    min_size_threshold = size_filtering.get('min_size', 0.0001)
    max_size_threshold = size_filtering.get('max_size', 0.1)

    # 应用尺寸过滤（在导出前）
    if enable_size_filtering:
        scales = torch.exp(params["scales"])  # [N, 3] 线性尺度
        scale_norms = scales.norm(dim=1)  # [N] 尺寸范数

        size_mask = (scale_norms >= min_size_threshold) & (scale_norms <= max_size_threshold)
        removed_by_size = (~size_mask).sum().item()

        if removed_by_size > 0:
            print(f"尺寸过滤: 删除 {removed_by_size} 个高斯 (< {min_size_threshold} 或 > {max_size_threshold})")

            # 过滤参数
            if not size_mask.any():
                raise ValueError("尺寸过滤删除了全部高斯点，请调整尺寸阈值")
            # 仅生成导出视图，不修改训练参数，保证后续评估与原始模型一致。
            params = {key: value[size_mask] for key, value in params.items()}

    # 导出 PLY
    ply_path = os.path.join(out_dir, "point_cloud.ply")
    export_ply(params, ply_path)
    print(f"导出 PLY: {ply_path}")

    # 自动清理
    if auto_cleanup:
        print(f"执行自动清理（阈值: {cleanup_threshold}）...")
        kept, removed = clean_gaussians(
            ply_path, opacity_threshold=cleanup_threshold,
            distance_percentile=denoise_config.get("distance_percentile", 95),
        )

    # 已清理 PLY 时必须从它导出 SPLAT，否则查看器仍会显示被删掉的杂点。
    splat_path = os.path.join(out_dir, "point_cloud.splat")
    if auto_cleanup:
        export_splat_from_ply(ply_path, splat_path)
    else:
        export_splat(params, splat_path)
    print(f"导出 SPLAT: {splat_path}")
    exported_count = int(kept) if auto_cleanup else int(params["means"].shape[0])
    # 逐阶段数量可以识别清理是否过强，不能把删点比例直接当作质量提升。
    import json
    report = {"before_cleanup": raw_count, "removed_by_size": removed_by_size,
              "after_size_filter": int(params["means"].shape[0]), "exported_count": exported_count,
              "auto_cleanup": auto_cleanup, "opacity_threshold": cleanup_threshold,
              "distance_percentile": denoise_config.get("distance_percentile", 95)}
    with open(os.path.join(out_dir, "cleanup_report.json"), "w", encoding="utf-8") as output:
        json.dump(report, output, ensure_ascii=False, indent=2)
    return exported_count
