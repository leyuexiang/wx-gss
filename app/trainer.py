"""gsplat 训练主循环（MCMC 策略，packed 模式省显存）。"""

import json
import math
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

SH_C0 = 0.28209479177387814


def _logit(x: float) -> float:
    return math.log(x / (1 - x))


def pearson_depth_loss(pred_depth: torch.Tensor, gt_depth: torch.Tensor) -> torch.Tensor:
    """Pearson correlation depth loss (对尺度不敏感，适合少视角场景)。

    Args:
        pred_depth: [H, W] 渲染深度
        gt_depth: [H', W'] MASt3R 深度（可能尺寸不同）

    Returns:
        loss: 标量，1 - correlation (最小化此值 = 最大化相关性)
    """
    # 处理尺寸不匹配：插值 gt 到 pred 的尺寸
    if pred_depth.shape != gt_depth.shape:
        gt_depth = F.interpolate(
            gt_depth[None, None],  # [1, 1, H', W']
            size=pred_depth.shape,
            mode='bilinear',
            align_corners=False
        )[0, 0]  # [H, W]

    mask = gt_depth > 1e-3
    if mask.sum() < 100:
        return torch.tensor(0.0, device=pred_depth.device)

    pred = pred_depth[mask]
    gt = gt_depth[mask]

    # 标准化
    pred_mean = pred.mean()
    gt_mean = gt.mean()
    pred_std = pred.std() + 1e-8
    gt_std = gt.std() + 1e-8

    pred_norm = (pred - pred_mean) / pred_std
    gt_norm = (gt - gt_mean) / gt_std

    # Pearson correlation
    correlation = (pred_norm * gt_norm).mean()
    return 1.0 - correlation


def _init_gaussians(points: np.ndarray, colors: np.ndarray, sh_degree: int, device):
    from scipy.spatial import cKDTree

    N = points.shape[0]
    if N < 100:
        raise RuntimeError(f"COLMAP 稀疏点过少（{N}），无法初始化高斯")

    dists, _ = cKDTree(points).query(points, k=4)
    mean_dist = np.clip(dists[:, 1:].mean(axis=1), 1e-7, None)
    scales = torch.log(torch.from_numpy(mean_dist).float())[:, None].repeat(1, 3)

    rgbs = torch.from_numpy(colors).float()
    sh0 = ((rgbs - 0.5) / SH_C0)[:, None, :]
    shN = torch.zeros((N, (sh_degree + 1) ** 2 - 1, 3))

    params = torch.nn.ParameterDict({
        "means": torch.from_numpy(points).float(),
        "scales": scales,
        "quats": F.normalize(torch.rand(N, 4), dim=-1),
        "opacities": torch.full((N,), _logit(0.1)),
        "sh0": sh0,
        "shN": shN,
    }).to(device)
    return params


def train(dataset_dir: str, out_dir: str, iterations: int, downsample: int,
          cap_max: int, progress_cb, auto_cleanup: bool = True,
          cleanup_threshold: float = 0.1, enable_depth_reg: bool = False,
          depth_reg_weight: float = 0.1, enable_pose_opt: bool = False,
          pose_opt_start: int = 500, pose_opt_lr: float = 1e-5,
          denoise_config: dict = None) -> dict:
    from gsplat import MCMCStrategy, rasterization
    from pytorch_msssim import SSIM

    from app.dataset import ColmapDataset, ColmapScene

    device = torch.device("cuda")
    sh_degree = 3

    # 解析去噪配置
    if denoise_config is None:
        denoise_config = {}

    enable_denoising = denoise_config.get('enable', True)
    opacity_threshold = denoise_config.get('opacity_threshold', 0.005)
    denoise_strength = denoise_config.get('denoise_strength', 50)

    size_filtering = denoise_config.get('size_filtering', {})
    enable_size_filtering = size_filtering.get('enable', True)
    min_size_threshold = size_filtering.get('min_size', 0.0001)
    max_size_threshold = size_filtering.get('max_size', 0.1)

    gradient_clipping = denoise_config.get('gradient_clipping', {})
    enable_gradient_clipping = gradient_clipping.get('enable', True)
    max_grad_norm = gradient_clipping.get('max_norm', 1.0)

    regularization = denoise_config.get('regularization', {})
    smoothness_weight = regularization.get('smoothness_weight', 0.01)
    depth_weight_override = regularization.get('depth_weight', None)

    # 如果去噪配置中指定了深度权重，使用配置值
    if depth_weight_override is not None and depth_weight_override > 0:
        enable_depth_reg = True
        depth_reg_weight = depth_weight_override

    enable_antialiasing = denoise_config.get('antialiasing', True)

    scene = ColmapScene(dataset_dir, factor=downsample)
    dataset = ColmapDataset(scene)

    params = _init_gaussians(scene.points, scene.points_rgb, sh_degree, device)

    lr_means = 1.6e-4 * scene.scene_scale
    optimizers = {
        "means": torch.optim.Adam([params["means"]], lr=lr_means, eps=1e-15),
        "scales": torch.optim.Adam([params["scales"]], lr=5e-3, eps=1e-15),
        "quats": torch.optim.Adam([params["quats"]], lr=1e-3, eps=1e-15),
        "opacities": torch.optim.Adam([params["opacities"]], lr=5e-2, eps=1e-15),
        "sh0": torch.optim.Adam([params["sh0"]], lr=2.5e-3, eps=1e-15),
        "shN": torch.optim.Adam([params["shN"]], lr=2.5e-3 / 20, eps=1e-15),
    }
    scheduler = torch.optim.lr_scheduler.ExponentialLR(
        optimizers["means"], gamma=0.01 ** (1.0 / iterations)
    )

    # MCMC 策略优化：减少杂点生成
    strategy = MCMCStrategy(
        cap_max=cap_max,
        refine_start_iter=500,
        refine_stop_iter=int(iterations * 0.5),  # 从 0.6 改为 0.5，提前停止细化
        refine_every=100,
        min_opacity=0.01,  # 从 0.005 提高到 0.01，过滤低不透明度点
        verbose=False,
    )
    strategy.check_sanity(params, optimizers)
    strategy_state = strategy.initialize_state()

    ssim_fn = SSIM(data_range=1.0, size_average=True, channel=3)
    n_images = len(dataset)
    t0 = time.time()
    last_loss = 0.0

    # 检查是否有深度数据
    has_depth = scene.has_depth and enable_depth_reg
    if has_depth:
        print(f"深度正则化已启用: weight={depth_reg_weight}, 退火到 {iterations//2} 步")

    # 初始化位姿优化（如果启用）
    poses = None
    pose_optimizer = None
    if enable_pose_opt:
        from app.pose_opt import CameraPose, pose_regularization_loss
        poses = CameraPose(torch.from_numpy(scene.camtoworlds)).to(device)
        pose_optimizer = torch.optim.Adam(poses.parameters(), lr=pose_opt_lr)
        print(f"位姿优化已启用: start={pose_opt_start}, lr={pose_opt_lr}")

    for step in range(iterations):
        data = dataset[np.random.randint(n_images)]
        image = data["image"].to(device)                       # [3,H,W]
        H, W = image.shape[1], image.shape[2]

        # 获取位姿（可学习 or 固定）
        if poses is not None and step >= pose_opt_start:
            camtoworld = poses(data["image_id"])
        else:
            camtoworld = data["camtoworld"].to(device)

        viewmat = torch.linalg.inv(camtoworld)[None]  # [1,4,4]
        Ks = data["K"][None].to(device)                        # [1,3,3]

        sh_degree_now = min(step // 1000, sh_degree)
        colors = torch.cat([params["sh0"], params["shN"]], dim=1)

        # 启用深度渲染（如果需要深度正则化）
        render_mode = "RGB+ED" if (has_depth and "depth" in data) else "RGB"

        renders, _, info = rasterization(
            means=params["means"],
            quats=F.normalize(params["quats"], dim=-1),
            # gsplat rasterization 期望激活后的值：scales 线性、opacities (0,1)；
            # 参数内部按 log / logit 存储（与导出格式一致）
            scales=torch.exp(params["scales"]),
            opacities=torch.sigmoid(params["opacities"]),
            colors=colors,
            viewmats=viewmat,
            Ks=Ks,
            width=W,
            height=H,
            sh_degree=sh_degree_now,
            packed=True,
            sparse_grad=False,
            rasterize_mode="antialiased" if enable_antialiasing else "classic",
            render_mode=render_mode,
        )

        # 处理渲染输出
        # gsplat 实际返回: renders[0] shape [H, W, X]
        # X=3(RGB) or 4(RGB+ED)
        rendered = renders[0]  # [H, W, X]

        if render_mode == "RGB+ED":
            # RGB+ED 模式: X=4，最后一个通道是深度
            pred_rgb = rendered[:, :, :3].permute(2, 0, 1)  # [3, H, W]
            rendered_depth = rendered[:, :, 3]               # [H, W]
        else:
            # RGB 模式: X=3
            pred_rgb = rendered.permute(2, 0, 1)            # [3, H, W]
            rendered_depth = None

        l1 = F.l1_loss(pred_rgb, image)
        ssim_val = ssim_fn(pred_rgb[None], image[None])
        photo_loss = 0.8 * l1 + 0.2 * (1.0 - ssim_val)

        # 深度正则化 loss（退火调度：前半段强正则，后半段弱化）
        depth_loss_val = 0.0
        if has_depth and "depth" in data and rendered_depth is not None:
            gt_depth = data["depth"].to(device)
            depth_loss_val = pearson_depth_loss(rendered_depth, gt_depth)
            # 退火权重：step=0 → weight, step=iterations/2 → 0
            depth_weight = depth_reg_weight * max(0.0, 1.0 - step / (iterations * 0.5))
        else:
            depth_weight = 0.0

        # 位姿正则化 loss（防止偏离初始估计太远）
        pose_reg_loss_val = 0.0
        if poses is not None and step >= pose_opt_start:
            pose_reg_loss_val = pose_regularization_loss(poses, weight_rot=0.01, weight_trans=0.001)

        # 平滑正则化 loss（去噪配置）
        smoothness_loss_val = 0.0
        if enable_denoising and smoothness_weight > 0:
            # 计算相邻高斯点的位置和尺度差异
            means = params["means"]
            scales = torch.exp(params["scales"])

            # 简单实现：计算均值方差作为平滑度指标
            # 更复杂的实现可以使用 KNN 或空间哈希
            mean_var = torch.var(means, dim=0).mean()
            scale_var = torch.var(scales, dim=0).mean()
            smoothness_loss_val = mean_var + scale_var

        # 总loss
        loss = photo_loss + depth_weight * depth_loss_val + pose_reg_loss_val + smoothness_weight * smoothness_loss_val
        loss.backward()
        last_loss = loss.item()

        # 梯度裁剪（去噪配置）
        if enable_gradient_clipping:
            for param_group in optimizers.values():
                for param in param_group.param_groups[0]['params']:
                    if param.grad is not None:
                        torch.nn.utils.clip_grad_norm_([param], max_grad_norm)

        # 更新高斯参数
        for opt in optimizers.values():
            opt.step()
            opt.zero_grad(set_to_none=True)
        scheduler.step()

        # 更新位姿（如果启用且已到优化阶段）
        if pose_optimizer is not None and step >= pose_opt_start:
            pose_optimizer.step()
            pose_optimizer.zero_grad(set_to_none=True)

        strategy.step_post_backward(
            params, optimizers, strategy_state, step, info,
            lr=scheduler.get_last_lr()[0],
        )

        if step % 200 == 0 or step == iterations - 1:
            progress_cb(step, iterations,
                        f"训练 {step}/{iterations} loss={last_loss:.4f} "
                        f"高斯数={params['means'].shape[0]} "
                        f"显存={torch.cuda.max_memory_allocated() / 2**30:.1f}GB")

    num_gs = params["means"].shape[0]
    mem_gb = torch.cuda.max_memory_allocated() / 2**30

    # 保存优化后的位姿（如果启用了位姿优化）
    if poses is not None:
        optimized_poses = poses.get_all_poses().detach().cpu().numpy()
        np.save(os.path.join(out_dir, "optimized_poses.npy"), optimized_poses)
        print(f"已保存优化后的位姿: {optimized_poses.shape}")

    from app.export import export_gaussians
    # 统计交付模型实际包含的点数，避免清理前的数量误导质量判断。
    exported_count = export_gaussians(params, out_dir, auto_cleanup=auto_cleanup,
                                      cleanup_threshold=cleanup_threshold, denoise_config=denoise_config)

    stats = {
        "iterations": iterations,
        "num_gaussians": exported_count,
        "num_gaussians_before_cleanup": num_gs,
        "final_loss": round(last_loss, 5),
        "train_seconds": round(time.time() - t0, 1),
        "peak_gpu_mem_gb": round(mem_gb, 2),
        "n_images": n_images,
        "downsample": downsample,
    }
    with open(os.path.join(out_dir, "train_stats.json"), "w") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    return stats
