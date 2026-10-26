"""
训练配置生成模块 - 将前端配置转换为训练参数
"""
import json
from typing import Dict


def generate_training_config(
    config_dict: Dict,
    frame_count: int = None,
    resolution: str = None,
    scene_type: str = "outdoor"
) -> Dict:
    """
    生成完整的训练配置，整合去噪、高级参数等

    Args:
        config_dict: 前端传递的配置字典
        frame_count: 提取的帧数（用于智能推荐）
        resolution: 分辨率字符串（用于智能推荐）
        scene_type: 场景类型

    Returns:
        完整的训练配置字典
    """
    advanced = config_dict.get('advanced', {})

    # 去噪配置
    enable_denoising = advanced.get('enableDenoising', True)
    denoise_config = {
        'enable': enable_denoising,
        'opacity_threshold': advanced.get('opacityThreshold', 0.005),
        'denoise_strength': advanced.get('denoiseStrength', 50),
        'size_filtering': {
            'enable': advanced.get('enableSizeFiltering', True),
            'min_size': advanced.get('minSizeThreshold', 0.0001),
            'max_size': advanced.get('maxSizeThreshold', 0.1)
        },
        'gradient_clipping': {
            'enable': advanced.get('enableGradientClipping', True),
            'max_norm': advanced.get('gradientClipValue', 1.0)
        },
        'regularization': {
            'smoothness_weight': advanced.get('smoothnessWeight', 0.01),
            'depth_weight': advanced.get('depthWeight', 0.1) if advanced.get('enableDepthRegularization', True) else 0.0
        },
        'antialiasing': advanced.get('enableAntialiasing', True)
    }

    # COLMAP 配置
    colmap = config_dict.get('colmap', {})
    colmap_config = {
        'feature_extractor': colmap.get('featureExtractor', 'sift'),
        'matcher': colmap.get('matcher', 'sequential'),
        'use_gpu': colmap.get('featureExtractor') == 'sift-gpu'
    }

    # 3DGS 训练参数
    training_config = {
        'position_lr': advanced.get('positionLR', 0.00016),
        'densify_interval': advanced.get('densifyInterval', 100),
        'opacity_reset_interval': advanced.get('opacityReset', 3000),
        'sh_degree': advanced.get('shDegree', 3),
        'cap_max': scene_type == 'object' and 500000 or (scene_type == 'indoor' and 800000 or 1200000)
    }

    # 智能推荐（如果有帧数和分辨率信息）
    if frame_count and resolution:
        from app.frame_analysis import recommend_training_params
        recommendations = recommend_training_params(frame_count, resolution, scene_type)

        # 将推荐应用到配置中（如果用户未明确配置）
        if 'iterations' not in config_dict or config_dict.get('useRecommended', False):
            training_config['iterations_recommended'] = recommendations['iterations']

        training_config['sfm_backend_recommended'] = recommendations['sfm_backend']
        training_config['cap_max'] = recommendations['cap_max']

    return {
        'denoise': denoise_config,
        'colmap': colmap_config,
        'training': training_config,
        'scene_type': scene_type
    }


def save_training_config(task_dir: str, config: Dict):
    """
    保存训练配置到任务目录

    Args:
        task_dir: 任务目录路径
        config: 配置字典
    """
    import os
    config_path = os.path.join(task_dir, 'training_config.json')
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
