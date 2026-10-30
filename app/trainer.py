"""gsplat 训练、可选增密策略和固定视角图像评估。"""

import json
import math
import os
import random
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
          denoise_config: dict = None, training_options: dict = None) -> dict:
    from gsplat import DefaultStrategy, MCMCStrategy, rasterization
    from pytorch_msssim import SSIM

    from app.dataset import ColmapDataset, ColmapScene

    device = torch.device("cuda")
    options = training_options or {}
    sh_degree = options.get("sh_degree", 3)
    seed = options.get("seed", 42)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.reset_peak_memory_stats()

    # 解析去噪配置
    if denoise_config is None:
        denoise_config = {}

    enable_denoising = denoise_config.get('enable', True)
    opacity_threshold = denoise_config.get('opacity_threshold', 0.005)
    gradient_clipping = denoise_config.get('gradient_clipping', {})
    enable_gradient_clipping = gradient_clipping.get('enable', True)
    max_grad_norm = gradient_clipping.get('max_norm', 1.0)

    regularization = denoise_config.get('regularization', {})
    # 两种全局约束分别记录，默认关闭会收缩几何的全局位置方差惩罚。
    position_weight = regularization.get("position_variance_weight", 0.0)
    scale_weight = regularization.get("scale_variance_weight", regularization.get("smoothness_weight", 0.0))

    enable_antialiasing = denoise_config.get('antialiasing', True)

    sparse_enabled = options.get("sparse_depth", False)
    bilateral_enabled = options.get("bilateral_grid", False)
    sparse_weight = options.get("sparse_depth_weight", .01)
    observation_dir = options.get("depth_observation_dir")
    observation_weight = options.get("depth_observation_weight", .005)
    if observation_dir and (isinstance(observation_weight, bool) or not isinstance(observation_weight, (int, float))
                            or not math.isfinite(observation_weight) or not 0 < observation_weight <= .1):
        raise ValueError("外部深度权重必须在 (0, 0.1] 内")
    if observation_dir and (sparse_enabled or enable_depth_reg or enable_pose_opt):
        raise ValueError("外部深度观测不能与其他深度分支或位姿微调混用")
    if not math.isfinite(sparse_weight) or not 0 < sparse_weight <= 1:
        raise ValueError("稀疏深度权重必须在 (0, 1] 内")
    if sparse_enabled and (enable_depth_reg or enable_pose_opt):
        raise ValueError("稀疏深度实验暂不与稠密深度或位姿微调混用")
    rigid_normalization = options.get("rigid_normalization", False)
    if (sparse_enabled or observation_dir) and not rigid_normalization:
        raise ValueError("稀疏深度实验必须显式启用刚性相机归一化，并重建同约定基线")
    scene = ColmapScene(dataset_dir, factor=downsample, sparse_depth=sparse_enabled,
                        rigid_normalization=rigid_normalization)
    cache_bytes = options.get("image_cache_mb", 256) * 1024**2
    mask_dir = options.get("mask_dir")
    foreground_only = options.get("foreground_only", False)
    silhouette_weight = options.get("silhouette_weight", .1)
    if foreground_only and (not mask_dir or bilateral_enabled):
        raise ValueError("前景独立训练需要掩码，暂不与双边网格混用")
    if foreground_only and (isinstance(silhouette_weight, bool) or not isinstance(silhouette_weight, (int, float))
                            or not math.isfinite(silhouette_weight) or not 0 < silhouette_weight <= 1):
        raise ValueError("轮廓监督权重必须在 (0, 1] 内")
    allow_unreviewed_masks = options.get("allow_unreviewed_masks", False)
    dataset = ColmapDataset(scene, cache_bytes=cache_bytes, uint8_images=True, mask_dir=mask_dir,
                            allow_unreviewed_masks=allow_unreviewed_masks, foreground_only=foreground_only)
    mask_foreground_weight = options.get("mask_foreground_weight", 1.5)
    mask_background_weight = options.get("mask_background_weight", 0.25)
    if mask_dir:
        from app.subject_masks import validate_mask_weights, weighted_l1, silhouette_loss
        validate_mask_weights(mask_foreground_weight, mask_background_weight)
        from app.training_monitor import write_json_atomic
        write_json_atomic(os.path.join(out_dir, "mask_validation.json"), dataset.mask_report)

    initial_indices = np.arange(len(scene.points))
    if len(initial_indices) > cap_max:
        initial_indices = np.random.choice(initial_indices, cap_max, replace=False)
    params = _init_gaussians(scene.points[initial_indices], scene.points_rgb[initial_indices], sh_degree, device)

    lr_means = options.get("position_lr", 1.6e-4) * scene.scene_scale
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

    strategy_name = options.get("strategy", "mcmc")
    strategy_options = {"refine_start_iter": 500,
                        "refine_stop_iter": int(iterations * options.get("refine_stop_ratio", 0.5)),
                        "refine_every": options.get("densify_interval", 100), "verbose": False}
    if strategy_name == "default":
        strategy_type = DefaultStrategy
        if options.get("fix_opacity_reset", False):
            from app.strategy_compat import DefaultStrategyWithOpacityReset
            strategy_type = DefaultStrategyWithOpacityReset
        strategy = strategy_type(**strategy_options, reset_every=options.get("opacity_reset_interval", 3000))
    elif strategy_name == "mcmc":
        strategy = MCMCStrategy(**strategy_options, cap_max=cap_max, min_opacity=opacity_threshold)
    else:
        raise ValueError(f"不支持的训练策略：{strategy_name}")
    strategy.check_sanity(params, optimizers)
    strategy_state = strategy.initialize_state(scene_scale=scene.scene_scale) if strategy_name == "default" else strategy.initialize_state()

    ssim_fn = SSIM(data_range=1.0, size_average=True, channel=3)
    n_images = len(dataset)
    stride = options.get("validation_every", 0)
    validation_indices = list(range(stride - 1, n_images, stride)) if stride else []
    if options.get("validation_names") is not None:
        # 独立重建协议按文件名固定划分，避免注册顺序变化使验证图进入训练。
        requested_names = set(options["validation_names"])
        unknown = requested_names - set(scene.names)
        if unknown:
            raise ValueError(f"验证视角未成功定位：{sorted(unknown)}")
        validation_indices = [index for index, name in enumerate(scene.names) if name in requested_names]
    validation_set = set(validation_indices)
    from app.capture.splits import training_indices
    train_indices = training_indices(scene.names, validation_indices, options.get("training_names"))
    if len(train_indices) < 2:
        raise ValueError("留出验证视角后至少需要两张训练图像")
    bilateral = None
    if bilateral_enabled:
        from app.bilateral_grid import BilateralGrid
        bilateral = BilateralGrid(n_images).to(device)
        bilateral_optimizer = torch.optim.Adam(bilateral.parameters(), lr=.002)
    if sparse_enabled:
        from app.sparse_depth import inverse_depth_loss
        coverage = {scene.names[i]: len(scene.sparse_observations[i][1]) for i in range(n_images)}
        if not any(coverage[scene.names[i]] for i in train_indices):
            raise ValueError("训练集没有可靠稀疏深度观测")
        with open(os.path.join(out_dir, "sparse_depth_report.json"), "w", encoding="utf-8") as output:
            json.dump({"counts": coverage, "min_track_length": 3, "max_reprojection_error": 2.,
                       "depth_kind": "normalized_camera_z", "supervised_indices": train_indices}, output, indent=2)
    observations, observation_report = {}, None
    if observation_dir:
        from app.depth_observations import load_depth_observations
        from app.sparse_depth import inverse_depth_loss
        loaded, observation_report = load_depth_observations(observation_dir, scene, train_indices, dataset)
        # 在训练前一次传入显卡；不在每次采样时解压观测或创建相同张量。
        observations = {i: (torch.from_numpy(xy).to(device), torch.from_numpy(z).to(device))
                        for i, (xy, z) in loaded.items()}
    observation_history = {'steps': 0, 'loss_sum': 0., 'loss_max': 0., 'by_view': {}}
    effective = {
        "iterations": iterations, "downsample": downsample, "cap_max": cap_max,
        "training": {**options, "strategy": strategy_name, "seed": seed, "sh_degree": sh_degree,
                     "position_lr": lr_means, **strategy_options},
        "denoise": denoise_config, "depth_enabled": bool(scene.has_depth and enable_depth_reg),
        "depth_reg_weight": depth_reg_weight, "pose_opt_enabled": enable_pose_opt,
        "antialiased": enable_antialiasing, "image_dimensions": scene.dimensions,
        "image_cache_mb": cache_bytes // 1024**2, "cached_image_dtype": "uint8",
        "mask_dir": mask_dir, "mask_loss": {
            "scope": ("前景黑底参考、全图颜色与结构损失及轮廓透明度监督" if foreground_only
                      else "仅对 L1 像素误差加权，SSIM 保持全图；不是背景分割导出"),
            "manifest_sha256": dataset.mask_report["manifest_sha256"],
            "foreground_weight": mask_foreground_weight,
            "background_weight": mask_background_weight,
            "foreground_only": foreground_only, "silhouette_weight": silhouette_weight if foreground_only else 0,
        } if mask_dir else None,
        "train_indices": train_indices, "validation_indices": validation_indices,
        "validation_scope": options.get("validation_scope", "仅留出颜色监督；相机重建和初始点云可能使用全部输入，不能称为完全未见视角"),
        "strategy_implementation": type(strategy).__name__,
        "sparse_depth": {"enabled": sparse_enabled, "weight": sparse_weight},
        "depth_observations": {"enabled": bool(observation_dir), "weight": observation_weight,
                               "report": observation_report},
        "bilateral_grid": {"enabled": bilateral_enabled, "shape": [8, 8, 4], "tv_weight": 10.,
                           "lr": .002, "evaluation": "不使用验证图拟合校色；评估未经网格处理的高斯模型"},
    }
    with open(os.path.join(out_dir, "effective_config.json"), "w", encoding="utf-8") as output:
        json.dump(effective, output, ensure_ascii=False, indent=2)
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

    monitor = None
    if options.get("checkpoint_every", 0):
        if poses is not None:
            raise ValueError("阶段监控暂不支持位姿微调，请使用固定相机对照")
        from app.training_monitor import StageMonitor
        monitor = StageMonitor(out_dir, dataset, validation_indices, options["checkpoint_every"],
                               sh_degree, enable_antialiasing, options.get("monitor_roi_path"))

    audit = None
    if options.get("audit_trace", False):
        from app.training_audit import TrainingAudit
        audit = TrainingAudit(out_dir, params, scene, iterations)

    for step in range(iterations):
        sampled_index = int(np.random.choice(train_indices))
        if audit is not None:
            audit.record(sampled_index)
        data = dataset[sampled_index]
        # 有界八位缓存减少重复解码，显卡端转浮点避免每一步在 CPU 分配整张浮点图。
        image = data["image"].to(device=device, dtype=torch.float32, non_blocking=True).div_(255.0)
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
        render_mode = "RGB+ED" if (sparse_enabled or observation_dir or options.get("diagnostic_depth_render", False)
                                   or (has_depth and "depth" in data)) else "RGB"

        renders, alphas, info = rasterization(
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
        # 原版增密策略需要在反向传播前保留屏幕空间梯度。
        strategy.step_pre_backward(params, optimizers, strategy_state, step, info)

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

        # 光度补偿仅作用于训练颜色损失，深度保持原渲染值。
        if bilateral is not None:
            pred_rgb = bilateral(pred_rgb, data["image_id"])
        if mask_dir:
            mask = data["mask"].to(device=device, dtype=torch.float32).div_(255.0)
            # 前景独立模式使用黑底参考，另加透明度约束；普通加权模式保留原图。
            l1 = (F.l1_loss(pred_rgb, image) if foreground_only else
                  weighted_l1(pred_rgb, image, mask, mask_foreground_weight, mask_background_weight))
        else:
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
        if enable_denoising and (position_weight > 0 or scale_weight > 0):
            if position_weight > 0:
                smoothness_loss_val += position_weight * torch.var(params["means"], dim=0).mean()
            if scale_weight > 0:
                smoothness_loss_val += scale_weight * torch.var(params["scales"].exp(), dim=0).mean()

        # 总loss
        loss = photo_loss + depth_weight * depth_loss_val + pose_reg_loss_val + smoothness_loss_val
        if foreground_only:
            loss = loss + silhouette_weight * silhouette_loss(alphas[0, :, :, 0], mask)
        if sparse_enabled:
            loss = loss + sparse_weight * inverse_depth_loss(rendered_depth,
                data["sparse_xy"].to(device), data["sparse_depth"].to(device))
        if sampled_index in observations:
            xy, z = observations[sampled_index]
            observed_loss = inverse_depth_loss(rendered_depth, xy, z)
            if not torch.isfinite(observed_loss):
                raise ValueError("外部深度损失非有限，停止本轮实验")
            loss = loss + observation_weight * observed_loss
            value = float(observed_loss.detach())
            observation_history['steps'] += 1
            observation_history['loss_sum'] += value
            observation_history['loss_max'] = max(observation_history['loss_max'], value)
            name = scene.names[sampled_index]
            observation_history['by_view'][name] = observation_history['by_view'].get(name, 0) + 1
        if bilateral is not None:
            loss = loss + 10 * bilateral.smoothness(data["image_id"])
        loss.backward()
        if bilateral is not None:
            bilateral_optimizer.step()
            bilateral_optimizer.zero_grad(set_to_none=True)
        last_loss = loss.item()

        # 梯度裁剪（去噪配置）
        if enable_gradient_clipping:
            for param_group in optimizers.values():
                for param in param_group.param_groups[0]['params']:
                    if param.grad is not None:
                        torch.nn.utils.clip_grad_norm_([param], max_grad_norm)

        if strategy_name == "default":
            # 原版策略先根据本次反向传播增密，再让优化器更新变换后的参数和状态。
            strategy.step_post_backward(params, optimizers, strategy_state, step, info, packed=True)
            if len(params["means"]) > cap_max:
                from gsplat.strategy.ops import remove
                keep = torch.topk(params["opacities"].detach(), cap_max).indices
                remove_mask = torch.ones(len(params["means"]), dtype=torch.bool, device=device)
                remove_mask[keep] = False
                remove(params, optimizers, strategy_state, remove_mask)

        # 更新高斯参数
        for opt in optimizers.values():
            opt.step()
            opt.zero_grad(set_to_none=True)
        scheduler.step()

        # 更新位姿（如果启用且已到优化阶段）
        if pose_optimizer is not None and step >= pose_opt_start:
            pose_optimizer.step()
            pose_optimizer.zero_grad(set_to_none=True)

        if strategy_name == "mcmc":
            strategy.step_post_backward(params, optimizers, strategy_state, step, info,
                                        lr=scheduler.get_last_lr()[0])

        # 用已完成步数命名阶段；策略仍使用从零开始的步号，避免增密时程偏移。
        if monitor is not None and monitor.due(step + 1, iterations):
            stage = monitor.record(params, step + 1, scheduler.get_last_lr()[0], sh_degree_now)
            if audit is not None:
                audit.save()
            progress_cb(step + 1, iterations,
                        f"阶段 {step + 1} 验证全图PSNR={stage['full_psnr']:.4f} "
                        f"主体PSNR={stage.get('subject_psnr', float('nan')):.4f}")

        if step % 200 == 0 or step == iterations - 1:
            progress_cb(step, iterations,
                        f"训练 {step}/{iterations} loss={last_loss:.4f} "
                        f"高斯数={params['means'].shape[0]} "
                        f"显存={torch.cuda.max_memory_allocated() / 2**30:.1f}GB")

    if audit is not None:
        audit.save()
    if observation_dir:
        if not observation_history['steps']:
            raise ValueError("本轮未实际采样任何外部深度视角")
        observation_history['mean_loss'] = observation_history['loss_sum'] / observation_history['steps']
        with open(os.path.join(out_dir, 'depth_observation_usage.json'), 'w', encoding='utf-8') as output:
            json.dump(observation_history, output, ensure_ascii=False, indent=2)
    if bilateral is not None:
        # 网格是独立外观参数，不将其伪装为标准 PLY 的一部分。
        torch.save(bilateral.state_dict(), os.path.join(out_dir, "bilateral_grid.pt"))
    num_gs = params["means"].shape[0]
    mem_gb = torch.cuda.max_memory_allocated() / 2**30

    # 保存优化后的位姿（如果启用了位姿优化）
    if poses is not None:
        optimized_poses = poses.get_all_poses().detach().cpu().numpy()
        np.save(os.path.join(out_dir, "optimized_poses.npy"), optimized_poses)
        print(f"已保存优化后的位姿: {optimized_poses.shape}")
        scene.camtoworlds = optimized_poses
        dataset = ColmapDataset(scene, cache_bytes=cache_bytes, uint8_images=True, mask_dir=mask_dir,
                                allow_unreviewed_masks=allow_unreviewed_masks, foreground_only=foreground_only)

    from app.export import export_gaussians
    # 统计交付模型实际包含的点数，避免清理前的数量误导质量判断。
    exported_count = export_gaussians(params, out_dir, auto_cleanup=auto_cleanup,
                                      cleanup_threshold=cleanup_threshold, denoise_config=denoise_config)
    training_seconds = time.time() - t0
    # 评估另读交付模型前释放优化器，避免同时保留两份模型及其动量占满显存。
    del params, optimizers, scheduler, strategy, strategy_state, loss, renders, info, colors
    torch.cuda.empty_cache()

    # 导出模型用相同相机和尺寸评估；训练重建与留出颜色监督分别标记。
    from app.evaluation import evaluate_views, load_ply_params
    evaluated_params = load_ply_params(os.path.join(out_dir, "point_cloud.ply"))
    preview_indices = np.linspace(0, len(train_indices) - 1, min(6, len(train_indices)), dtype=int)
    evaluation = evaluate_views(evaluated_params, dataset, os.path.join(out_dir, "evaluation", "training"),
                                indices=[train_indices[index] for index in preview_indices],
                                antialiased=enable_antialiasing)
    validation = None
    if validation_indices:
        validation = evaluate_views(evaluated_params, dataset, os.path.join(out_dir, "evaluation", "validation"),
                                    indices=validation_indices, split="heldout_rgb_supervision",
                                    antialiased=enable_antialiasing,
                                    additional_roi=monitor.regions if monitor is not None else None)

    stats = {
        "iterations": iterations,
        "num_gaussians": exported_count,
        "num_gaussians_before_cleanup": num_gs,
        "final_loss": round(last_loss, 5),
        "train_seconds": round(training_seconds, 1),
        "peak_gpu_mem_gb": round(mem_gb, 2),
        "n_images": n_images,
        "downsample": downsample,
        "seed": seed, "strategy": strategy_name, "sh_degree": sh_degree,
        "image_dimensions": scene.dimensions, "n_training_images": len(train_indices),
        "n_validation_images": len(validation_indices), "antialiased": enable_antialiasing,
        "evaluation": evaluation["averages"],
        "validation": validation["averages"] if validation else None,
        "reference_kind": "foreground_black_background" if foreground_only else "original_full_frame",
        "foreground_validation": validation.get("foreground_averages") if validation else None,
    }
    if monitor is not None:
        if validation is not None:
            stats["subject_validation"] = validation.get("subject_averages")
            stats["background_validation"] = validation.get("background_averages")
        stats["monitoring"] = {"history": "stage_history.json", "best_stage": monitor.records and
                               json.loads((monitor.root / "best_stage.json").read_text(encoding="utf-8")),
                               "seconds_included_in_train_seconds": round(monitor.seconds, 3)}
    with open(os.path.join(out_dir, "train_stats.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    return stats
