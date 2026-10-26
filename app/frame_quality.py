"""
帧质量检测模块 - 模糊检测和重复检测
"""
import cv2
import numpy as np
from pathlib import Path
from typing import Optional


def detect_blur(image_path: str, threshold: float = 100.0) -> tuple[bool, float]:
    """
    使用拉普拉斯方差检测图像模糊度

    Args:
        image_path: 图像文件路径
        threshold: 模糊阈值，低于此值认为图像模糊（默认 100.0）

    Returns:
        (is_blurry, variance): 是否模糊和拉普拉斯方差值
    """
    try:
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return False, 0.0

        # 计算拉普拉斯方差
        laplacian = cv2.Laplacian(img, cv2.CV_64F)
        variance = laplacian.var()

        is_blurry = variance < threshold
        return is_blurry, round(variance, 2)

    except Exception as e:
        print(f"模糊检测失败 {image_path}: {str(e)}")
        return False, 0.0


def detect_duplicate(img1_path: str, img2_path: str, threshold: float = 0.95) -> tuple[bool, float]:
    """
    使用感知哈希检测两张图像是否重复

    Args:
        img1_path: 第一张图像路径
        img2_path: 第二张图像路径
        threshold: 相似度阈值，高于此值认为重复（默认 0.95）

    Returns:
        (is_duplicate, similarity): 是否重复和相似度
    """
    try:
        img1 = cv2.imread(img1_path)
        img2 = cv2.imread(img2_path)

        if img1 is None or img2 is None:
            return False, 0.0

        # 缩放到 8x8 进行感知哈希
        img1_small = cv2.resize(img1, (8, 8), interpolation=cv2.INTER_AREA)
        img2_small = cv2.resize(img2, (8, 8), interpolation=cv2.INTER_AREA)

        # 转换为灰度
        img1_gray = cv2.cvtColor(img1_small, cv2.COLOR_BGR2GRAY)
        img2_gray = cv2.cvtColor(img2_small, cv2.COLOR_BGR2GRAY)

        # 计算相似度（基于像素差异）
        diff = cv2.absdiff(img1_gray, img2_gray)
        similarity = 1.0 - (np.mean(diff) / 255.0)

        is_duplicate = similarity > threshold
        return is_duplicate, round(similarity, 3)

    except Exception as e:
        print(f"重复检测失败 {img1_path} vs {img2_path}: {str(e)}")
        return False, 0.0


async def filter_frames(
    frame_dir: str,
    remove_blurry: bool = False,
    remove_duplicates: bool = False,
    blur_threshold: float = 100.0,
    duplicate_threshold: float = 0.95,
    progress_callback: Optional[callable] = None
) -> dict:
    """
    过滤帧目录中的模糊和重复帧

    Args:
        frame_dir: 帧图像目录
        remove_blurry: 是否移除模糊帧
        remove_duplicates: 是否移除重复帧
        blur_threshold: 模糊检测阈值
        duplicate_threshold: 重复检测阈值
        progress_callback: 进度回调函数 callback(current, total, message)

    Returns:
        过滤统计信息
    """
    frames = sorted(Path(frame_dir).glob("frame_*.jpg"))
    total_frames = len(frames)

    if total_frames == 0:
        return {
            'total_frames': 0,
            'valid_frames': 0,
            'removed_blur': 0,
            'removed_duplicate': 0
        }

    valid_frames = []
    removed_blur = 0
    removed_duplicate = 0
    prev_frame = None

    for i, frame_path in enumerate(frames):
        frame_str = str(frame_path)

        # 模糊检测
        if remove_blurry:
            is_blurry, variance = detect_blur(frame_str, blur_threshold)
            if is_blurry:
                frame_path.unlink()  # 删除模糊帧
                removed_blur += 1
                if progress_callback:
                    await progress_callback(i + 1, total_frames, f"移除模糊帧: {frame_path.name} (方差={variance})")
                continue

        # 重复检测（只与前一帧比较）
        if remove_duplicates and prev_frame:
            is_duplicate, similarity = detect_duplicate(prev_frame, frame_str, duplicate_threshold)
            if is_duplicate:
                frame_path.unlink()  # 删除重复帧
                removed_duplicate += 1
                if progress_callback:
                    await progress_callback(i + 1, total_frames, f"移除重复帧: {frame_path.name} (相似度={similarity})")
                continue

        # 保留此帧
        valid_frames.append(frame_path)
        prev_frame = frame_str

        if progress_callback and (i + 1) % 10 == 0:
            await progress_callback(i + 1, total_frames, f"已处理 {i + 1}/{total_frames} 帧")

    return {
        'total_frames': total_frames,
        'valid_frames': len(valid_frames),
        'removed_blur': removed_blur,
        'removed_duplicate': removed_duplicate
    }
