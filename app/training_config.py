"""校验并统一界面参数；保存请求配置和训练器实际使用的配置。"""

import json
import math
import os
from typing import Dict


def _number(values, key, default, minimum, maximum, integer=False):
    """拒绝布尔、非有限数和越界值，避免任务排队后才暴露参数错误。"""
    value = values.get(key, default)
    expected = int if integer else (int, float)
    if isinstance(value, bool) or not isinstance(value, expected) or not math.isfinite(value):
        raise ValueError(f"{key} 必须是有效数字")
    if not minimum <= value <= maximum:
        raise ValueError(f"{key} 必须在 {minimum}–{maximum} 之间")
    return value


def _boolean(values, key, default, alias=None):
    value = values.get(key, values.get(alias, default) if alias else default)
    if not isinstance(value, bool):
        raise ValueError(f"{key} 必须是布尔值")
    return value


def generate_training_config(config_dict: Dict, frame_count=None, resolution=None,
                             scene_type="outdoor") -> Dict:
    """复用既有入口，推荐信息不覆盖用户明确设置的参数。"""
    from app.config import COLMAP_USE_GPU, DEFAULT_CAP_MAX

    advanced = config_dict.get("advanced", {})
    colmap = config_dict.get("colmap", {})
    if not isinstance(advanced, dict) or not isinstance(colmap, dict):
        raise ValueError("高级配置和重建配置必须是对象")
    # 高质量预设使用本次同数据留出对照胜出的策略，明确请求仍可覆盖。
    high_quality = config_dict.get("qualityMode") == "high"
    strategy = advanced.get("strategy", "default" if high_quality else "mcmc")
    if strategy not in ("mcmc", "default"):
        raise ValueError("训练策略只支持 mcmc 或 default")
    matcher = colmap.get("matcher", "auto")
    if matcher not in ("auto", "sequential", "exhaustive"):
        raise ValueError("匹配方式只支持自动、顺序或穷举")
    feature = colmap.get("featureExtractor", "sift-gpu" if COLMAP_USE_GPU else "sift")
    if feature not in ("sift", "sift-gpu"):
        raise ValueError("不支持的特征提取器")
    camera_model = colmap.get("cameraModel", "SIMPLE_RADIAL" if high_quality else "OPENCV")
    if camera_model not in ("OPENCV", "PINHOLE", "SIMPLE_RADIAL"):
        raise ValueError("相机模型只支持 OPENCV、PINHOLE 或 SIMPLE_RADIAL")
    enabled = _boolean(advanced, "enableDenoising", True)
    # 全局位置方差默认关闭，避免把场景收缩误当作局部去噪。
    scale_weight = advanced.get("scaleVarianceWeight", advanced.get("smoothnessWeight", 0.0))
    regularization = {
        "position_variance_weight": _number(advanced, "positionVarianceWeight", 0.0, 0, 1),
        "scale_variance_weight": _number({"scale": scale_weight}, "scale", 0.0, 0, 1),
    }
    depth_enabled = _boolean(advanced, "depthRegularization", False, "enableDepthRegularization")
    minimum_size = _number(advanced, "minSizeThreshold", 0.0002, 0, 1)
    maximum_size = _number(advanced, "maxSizeThreshold", 2.0, 0.0001, 100)
    if minimum_size >= maximum_size:
        raise ValueError("最小尺寸阈值必须小于最大尺寸阈值")
    result = {
        "schema_version": 1, "scene_type": scene_type,
        "denoise": {
            "enable": enabled,
            "opacity_threshold": _number(advanced, "opacityThreshold", 0.005, 0.0001, 0.5),
            "size_filtering": {
                "enable": enabled and _boolean(advanced, "sizeFiltering", False, "enableSizeFiltering"),
                "min_size": minimum_size, "max_size": maximum_size,
            },
            "gradient_clipping": {
                "enable": _boolean(advanced, "gradientClipping", True, "enableGradientClipping"),
                "max_norm": _number(advanced, "gradientClipValue", 1.0, 0.01, 100),
            },
            "regularization": regularization,
            "antialiasing": _boolean(advanced, "antiAliasing", True, "enableAntialiasing"),
            "distance_percentile": _number(advanced, "distancePercentile", 100.0, 50, 100),
        },
        "colmap": {
            "feature_extractor": feature, "matcher": matcher,
            "use_gpu": COLMAP_USE_GPU and feature == "sift-gpu",
            "camera_model": camera_model,
            "init_min_tri_angle": _number(advanced, "initMinTriAngle", 4.0 if high_quality else 16.0, 2, 32),
            "max_image_size": _number(advanced, "maxImageSize", 2000, 512, 8192, True),
        },
        "training": {
            "position_lr": _number(advanced, "positionLR", 0.00016, 0.000001, 0.01),
            "densify_interval": _number(advanced, "densifyInterval", 100, 10, 1000, True),
            "opacity_reset_interval": _number(advanced, "opacityReset", 3000, 100, 10000, True),
            "sh_degree": _number(advanced, "shDegree", 3, 0, 3, True),
            "cap_max": _number(advanced, "maxGaussians", DEFAULT_CAP_MAX, 100000, 1500000, True),
            "strategy": strategy, "seed": _number(advanced, "seed", 42, 0, 2147483647, True),
            "image_cache_mb": _number(advanced, "imageCacheMB", 256, 0, 2048, True),
            "checkpoint_every": _number(advanced, "checkpointEvery", 0, 0, 30000, True),
            "validation_every": _number(advanced, "validationEvery", 0, 0, 100, True),
            "refine_stop_ratio": _number(advanced, "refineStopRatio", 0.5, 0.1, 0.95),
        },
        "depth": {"enable": depth_enabled,
                  "weight": _number(advanced, "depthWeight", 0.1, 0, 1)},
    }
    stride = result["training"]["validation_every"]
    if 0 < stride < 3:
        raise ValueError("留出间隔应为 0（关闭）或至少 3")
    interval = result["training"]["checkpoint_every"]
    if interval and (interval < 100 or not stride):
        raise ValueError("阶段保存间隔至少为 100，且需要启用留出验证")
    return result


def save_training_config(task_dir: str, config: Dict, filename="training_config.json"):
    """先写临时文件再替换，防止工作进程读到不完整的配置。"""
    path = os.path.join(task_dir, filename)
    temporary = path + ".tmp"
    with open(temporary, "w", encoding="utf-8") as output:
        json.dump(config, output, indent=2, ensure_ascii=False, allow_nan=False)
    os.replace(temporary, path)
