"""
相机运动分析模块 - 光流分析和运动模糊检测
"""
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional


def analyze_camera_motion(frame_dir: str, sample_rate: int = 5) -> Dict:
    """
    使用光流法分析相机运动特征

    Args:
        frame_dir: 帧图像目录
        sample_rate: 采样率（每 N 帧分析一次，降低计算量）

    Returns:
        运动分析结果字典
    """
    frames = sorted(Path(frame_dir).glob("frame_*.jpg"))

    if len(frames) < 2:
        return {
            'error': '帧数不足，无法分析运动',
            'frame_count': len(frames)
        }

    motion_magnitudes = []
    motion_directions = []
    fast_motion_frames = []
    prev_gray = None

    # 采样分析（避免处理所有帧）
    sampled_indices = list(range(0, len(frames), sample_rate))

    for i, idx in enumerate(sampled_indices[:-1]):
        frame1_path = str(frames[idx])
        frame2_path = str(frames[sampled_indices[i + 1]])

        # 读取帧
        img1 = cv2.imread(frame1_path, cv2.IMREAD_GRAYSCALE)
        img2 = cv2.imread(frame2_path, cv2.IMREAD_GRAYSCALE)

        if img1 is None or img2 is None:
            continue

        # 缩小尺寸加速计算
        img1_small = cv2.resize(img1, (320, 240))
        img2_small = cv2.resize(img2, (320, 240))

        # 计算稠密光流（Farneback 方法）
        flow = cv2.calcOpticalFlowFarneback(
            img1_small, img2_small, None,
            pyr_scale=0.5, levels=3, winsize=15,
            iterations=3, poly_n=5, poly_sigma=1.2, flags=0
        )

        # 计算运动幅度和方向
        magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])

        # 平均运动幅度
        avg_magnitude = np.mean(magnitude)
        motion_magnitudes.append(avg_magnitude)

        # 主导运动方向（圆周平均）
        avg_angle = np.arctan2(np.mean(np.sin(angle)), np.mean(np.cos(angle)))
        motion_directions.append(np.degrees(avg_angle))

        # 检测快速运动（幅度超过阈值）
        if avg_magnitude > 5.0:  # 阈值可调
            fast_motion_frames.append({
                'frame_index': idx,
                'frame_name': frames[idx].name,
                'magnitude': round(avg_magnitude, 2)
            })

    if not motion_magnitudes:
        return {
            'error': '光流计算失败',
            'frame_count': len(frames)
        }

    # 统计分析
    avg_motion = np.mean(motion_magnitudes)
    max_motion = np.max(motion_magnitudes)
    motion_std = np.std(motion_magnitudes)

    # 运动稳定性评分（0-100，越高越稳定）
    stability_score = max(0, 100 - motion_std * 10)

    # 运动类型判断
    if avg_motion < 2.0:
        motion_type = "静态或缓慢移动"
    elif avg_motion < 5.0:
        motion_type = "中等速度移动"
    else:
        motion_type = "快速移动"

    # 运动一致性（方向变化）
    direction_changes = 0
    for i in range(1, len(motion_directions)):
        angle_diff = abs(motion_directions[i] - motion_directions[i-1])
        if angle_diff > 180:
            angle_diff = 360 - angle_diff
        if angle_diff > 45:  # 方向变化超过 45 度
            direction_changes += 1

    consistency_score = max(0, 100 - (direction_changes / len(motion_directions) * 100))

    return {
        'frame_count': len(frames),
        'analyzed_pairs': len(motion_magnitudes),
        'motion_statistics': {
            'avg_magnitude': round(avg_motion, 2),
            'max_magnitude': round(max_motion, 2),
            'std_magnitude': round(motion_std, 2)
        },
        'motion_type': motion_type,
        'stability_score': round(stability_score, 1),
        'consistency_score': round(consistency_score, 1),
        'fast_motion_frames': fast_motion_frames,
        'motion_magnitudes': [round(m, 2) for m in motion_magnitudes],
        'motion_directions': [round(d, 1) for d in motion_directions]
    }


def detect_motion_blur(frame_path: str, motion_magnitude: float = 0.0, blur_threshold: float = 100.0) -> Tuple[bool, float, Dict]:
    """
    检测运动模糊（结合光流和拉普拉斯方差）

    Args:
        frame_path: 图像文件路径
        motion_magnitude: 运动幅度（来自光流分析）
        blur_threshold: 模糊阈值

    Returns:
        (is_motion_blurred, blur_score, details)
    """
    try:
        img = cv2.imread(frame_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return False, 0.0, {'error': '无法读取图像'}

        # 计算拉普拉斯方差（静态模糊）
        laplacian = cv2.Laplacian(img, cv2.CV_64F)
        variance = laplacian.var()

        # 综合判断运动模糊
        # 如果运动幅度大且拉普拉斯方差低，则可能是运动模糊
        is_blurred = variance < blur_threshold
        has_motion = motion_magnitude > 5.0
        is_motion_blurred = is_blurred and has_motion

        details = {
            'laplacian_variance': round(variance, 2),
            'motion_magnitude': round(motion_magnitude, 2),
            'is_static_blur': is_blurred and not has_motion,
            'is_motion_blur': is_motion_blurred
        }

        # 综合模糊评分（0-100，越高越清晰）
        blur_score = min(100, variance / 2.0)
        if has_motion:
            blur_score *= 0.8  # 运动降低评分

        return is_motion_blurred, round(blur_score, 1), details

    except Exception as e:
        return False, 0.0, {'error': str(e)}


def generate_shooting_suggestions(motion_analysis: Dict) -> List[str]:
    """
    基于运动分析生成拍摄建议

    Args:
        motion_analysis: 运动分析结果

    Returns:
        建议列表
    """
    suggestions = []

    if 'error' in motion_analysis:
        return ["无法分析运动，请确保视频包含足够帧数"]

    stats = motion_analysis.get('motion_statistics', {})
    avg_motion = stats.get('avg_magnitude', 0)
    stability_score = motion_analysis.get('stability_score', 100)
    consistency_score = motion_analysis.get('consistency_score', 100)
    fast_frames = motion_analysis.get('fast_motion_frames', [])

    # 运动速度建议
    if avg_motion > 8.0:
        suggestions.append("⚠️ 相机移动过快，建议降低移动速度以减少运动模糊")
    elif avg_motion < 1.0:
        suggestions.append("⚠️ 相机移动过慢，可能导致视角覆盖不足，建议适当增加移动速度")
    else:
        suggestions.append("✓ 相机移动速度适中")

    # 稳定性建议
    if stability_score < 60:
        suggestions.append("⚠️ 相机抖动较大，建议使用三脚架或稳定器")
    elif stability_score < 80:
        suggestions.append("💡 相机稳定性一般，可以使用稳定器改善")
    else:
        suggestions.append("✓ 相机稳定性良好")

    # 一致性建议
    if consistency_score < 60:
        suggestions.append("⚠️ 相机运动方向变化频繁，建议保持平滑的运动轨迹")
    elif consistency_score < 80:
        suggestions.append("💡 相机运动较为平滑，但仍有改进空间")
    else:
        suggestions.append("✓ 相机运动轨迹平滑")

    # 快速运动帧建议
    if len(fast_frames) > 5:
        suggestions.append(f"⚠️ 检测到 {len(fast_frames)} 个快速运动帧，可能影响重建质量")

    # 总体评价
    overall_score = (stability_score + consistency_score) / 2
    if overall_score >= 80:
        suggestions.append("✓ 整体拍摄质量优秀，适合 3DGS 重建")
    elif overall_score >= 60:
        suggestions.append("💡 拍摄质量良好，但仍有优化空间")
    else:
        suggestions.append("⚠️ 建议重新拍摄以获得更好的重建效果")

    return suggestions


def analyze_coverage(frame_dir: str) -> Dict:
    """
    分析视角覆盖度（简单实现：基于图像相似度）

    Args:
        frame_dir: 帧图像目录

    Returns:
        覆盖度分析结果
    """
    frames = sorted(Path(frame_dir).glob("frame_*.jpg"))

    if len(frames) < 10:
        return {
            'coverage_score': 0,
            'warning': '帧数过少，无法评估覆盖度'
        }

    # 简单实现：采样帧并计算特征点分布
    sample_size = min(20, len(frames))
    sample_indices = np.linspace(0, len(frames) - 1, sample_size, dtype=int)

    orb = cv2.ORB_create(nfeatures=500)
    feature_counts = []

    for idx in sample_indices:
        img = cv2.imread(str(frames[idx]), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue

        keypoints = orb.detect(img, None)
        feature_counts.append(len(keypoints))

    if not feature_counts:
        return {
            'coverage_score': 0,
            'warning': '特征检测失败'
        }

    # 特征点数量和分布作为覆盖度指标
    avg_features = np.mean(feature_counts)
    feature_std = np.std(feature_counts)

    # 覆盖度评分（0-100）
    # 特征点多且分布均匀 = 覆盖度好
    coverage_score = min(100, avg_features / 10)
    if feature_std / avg_features > 0.5:
        coverage_score *= 0.8  # 分布不均降低评分

    return {
        'coverage_score': round(coverage_score, 1),
        'avg_features': round(avg_features, 1),
        'feature_distribution': 'uniform' if feature_std / avg_features < 0.3 else 'uneven',
        'recommendation': '覆盖度良好' if coverage_score > 70 else '建议增加更多视角'
    }
