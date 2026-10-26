"""
抽帧结果分析模块
"""
import os
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List


def analyze_extracted_frames(frame_dir: str) -> Dict:
    """
    分析提取的视频帧，返回详细统计信息

    Args:
        frame_dir: 帧图像目录

    Returns:
        分析结果字典
    """
    frames = sorted(Path(frame_dir).glob("frame_*.jpg"))

    if not frames:
        return {
            'frame_count': 0,
            'error': '未找到提取的帧'
        }

    frame_count = len(frames)
    resolutions = []
    brightness_values = []
    file_sizes = []

    # 采样分析（全部帧数较多时采样，避免耗时过长）
    sample_size = min(50, frame_count)
    sample_indices = np.linspace(0, frame_count - 1, sample_size, dtype=int)

    for idx in sample_indices:
        frame_path = str(frames[idx])

        # 读取图像
        img = cv2.imread(frame_path)
        if img is None:
            continue

        # 分辨率
        h, w = img.shape[:2]
        resolutions.append((w, h))

        # 亮度（转灰度后求平均）
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        brightness = np.mean(gray)
        brightness_values.append(brightness)

        # 文件大小
        file_size = os.path.getsize(frame_path)
        file_sizes.append(file_size)

    # 统计结果
    if not resolutions:
        return {
            'frame_count': frame_count,
            'error': '无法读取帧图像'
        }

    # 分辨率分析
    resolution_counts = {}
    for res in resolutions:
        res_str = f"{res[0]}x{res[1]}"
        resolution_counts[res_str] = resolution_counts.get(res_str, 0) + 1

    most_common_res = max(resolution_counts.items(), key=lambda x: x[1])[0]

    # 亮度分析
    avg_brightness = np.mean(brightness_values)
    brightness_std = np.std(brightness_values)

    # 文件大小分析
    avg_file_size = np.mean(file_sizes)
    total_size = sum(os.path.getsize(str(f)) for f in frames)

    # 质量评估
    quality_issues = []

    if avg_brightness < 50:
        quality_issues.append("整体亮度偏低，可能影响特征提取")
    elif avg_brightness > 200:
        quality_issues.append("整体亮度偏高，可能存在过曝")

    if brightness_std > 50:
        quality_issues.append("亮度变化较大，建议在稳定光照下拍摄")

    return {
        'frame_count': frame_count,
        'resolution': most_common_res,
        'resolution_distribution': resolution_counts,
        'avg_brightness': round(avg_brightness, 1),
        'brightness_std': round(brightness_std, 1),
        'avg_file_size_kb': round(avg_file_size / 1024, 1),
        'total_size_mb': round(total_size / (1024 * 1024), 2),
        'quality_issues': quality_issues,
        'sampled_frames': sample_size
    }


def recommend_training_params(frame_count: int, resolution: str, scene_type: str) -> Dict:
    """
    根据抽帧结果推荐训练参数

    Args:
        frame_count: 提取的帧数
        resolution: 分辨率字符串 (如 "1920x1080")
        scene_type: 场景类型 (outdoor/indoor/object)

    Returns:
        推荐的训练参数
    """
    # 解析分辨率
    try:
        width, height = map(int, resolution.split('x'))
    except:
        width, height = 1920, 1080

    # 基于帧数推荐迭代次数和 SfM 后端
    if frame_count < 30:
        iterations = 3000
        sfm_backend = "mast3r"
        reason = f"少视角（{frame_count}帧），使用 MASt3R 后端"
    elif frame_count < 80:
        iterations = 7000
        sfm_backend = "colmap"
        reason = f"中等视角数（{frame_count}帧），标准训练配置"
    else:
        iterations = min(15000, frame_count * 100)
        sfm_backend = "colmap"
        reason = f"密集视角（{frame_count}帧），增加迭代次数提升质量"

    # 基于分辨率推荐降采样
    if width >= 3840:
        downsample = 4
        resolution_note = "4K 分辨率，降采样到 1/4 节省显存"
    elif width >= 1920:
        downsample = 2
        resolution_note = "1080p 分辨率，降采样到 1/2 平衡质量和速度"
    else:
        downsample = 1
        resolution_note = f"{resolution} 分辨率，使用原始分辨率"

    # 基于场景类型调整
    scene_configs = {
        'object': {
            'iterations_multiplier': 1.5,
            'cap_max': 500000,
            'note': "物体扫描需要更多迭代获得精细细节"
        },
        'indoor': {
            'iterations_multiplier': 1.0,
            'cap_max': 800000,
            'note': "室内场景标准配置"
        },
        'outdoor': {
            'iterations_multiplier': 1.0,
            'cap_max': 1200000,
            'note': "室外场景，增加点云容量应对大场景"
        }
    }

    scene_config = scene_configs.get(scene_type, scene_configs['outdoor'])
    iterations = int(iterations * scene_config['iterations_multiplier'])

    return {
        'iterations': iterations,
        'resolution': downsample,
        'sfm_backend': sfm_backend,
        'cap_max': scene_config['cap_max'],
        'reason': reason,
        'resolution_note': resolution_note,
        'scene_note': scene_config['note']
    }
