"""单物体快速重建的输入整理：分段选清晰视角并可选居中裁切。"""

import os
import shutil
import zipfile
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
MIN_OBJECT_IMAGES = 8
MAX_OBJECT_INPUTS = 500
MAX_OBJECT_VIEWS = 15


def extract_photo_archive(archive_path: str, output_dir: str) -> list[str]:
    """按压缩包顺序提取照片，避免文件名重复和路径穿越。"""
    os.makedirs(output_dir, exist_ok=True)
    paths = []
    with zipfile.ZipFile(archive_path) as archive:
        entries = sorted(
            (entry for entry in archive.infolist()
             if not entry.is_dir() and Path(entry.filename).suffix.lower() in IMAGE_EXTENSIONS
             and not entry.filename.startswith("__MACOSX/")),
            key=lambda entry: entry.filename,
        )
        if not MIN_OBJECT_IMAGES <= len(entries) <= MAX_OBJECT_INPUTS:
            raise ValueError(f"单物体照片需 {MIN_OBJECT_IMAGES}-{MAX_OBJECT_INPUTS} 张，当前 {len(entries)} 张")
        if sum(entry.file_size for entry in entries) > 2 * 1024**3:
            raise ValueError("照片解压后不能超过 2GB")
        for index, entry in enumerate(entries, start=1):
            destination = os.path.join(output_dir, f"source_{index:04d}{Path(entry.filename).suffix.lower()}")
            with archive.open(entry) as source, open(destination, "wb") as target:
                shutil.copyfileobj(source, target, length=4 * 1024 * 1024)
            paths.append(destination)
    return paths


def _focus_details(path: str, square_crop: bool):
    """只对缩小的主体区域计算清晰度，减少远处背景对选帧的干扰。"""
    with Image.open(path) as source:
        image = ImageOps.exif_transpose(source).convert("L")
        if square_crop:
            side = min(image.size)
            left = (image.width - side) // 2
            top = (image.height - side) // 2
            image = image.crop((left, top, left + side, top + side))
        image.thumbnail((384, 384), Image.Resampling.BILINEAR)
        gray = np.asarray(image)
    if min(gray.shape) < 64:
        raise ValueError(f"图片分辨率过低：{os.path.basename(path)}")
    height, width = gray.shape
    roi = gray[height // 5:height * 4 // 5, width // 5:width * 4 // 5]
    return float(cv2.Laplacian(roi, cv2.CV_64F).var()), cv2.resize(roi, (64, 64), interpolation=cv2.INTER_AREA)


def _focus_score(path: str, square_crop: bool) -> float:
    """保持旧评分接口，选帧流程同时复用缩略图判断近乎相同的画面。"""
    return _focus_details(path, square_crop)[0]


def prepare_object_views(source_dir: str, output_dir: str, target_views: int = 12,
                         square_crop: bool = True, max_views: int = MAX_OBJECT_VIEWS) -> dict:
    """按拍摄顺序分段选图；每段取主体最清晰的一张，保留环绕视角覆盖。"""
    files = sorted(
        str(path) for path in Path(source_dir).iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not MIN_OBJECT_IMAGES <= len(files) <= MAX_OBJECT_INPUTS:
        raise ValueError(f"单物体输入需 {MIN_OBJECT_IMAGES}-{MAX_OBJECT_INPUTS} 张，当前 {len(files)} 张")
    count = min(max(int(target_views), MIN_OBJECT_IMAGES), max_views, len(files))
    details = [_focus_details(path, square_crop) for path in files]
    scores = [item[0] for item in details]
    selected = []
    duplicate_fallbacks = []
    for bucket in np.array_split(np.arange(len(files)), count):
        ranked = sorted((int(index) for index in bucket), key=lambda index: scores[index], reverse=True)
        best = ranked[0]
        if selected:
            # 只排除近乎逐像素相同的候选，避免把正常的高重叠相邻视角误删。
            alternatives = [index for index in ranked if float(np.abs(
                details[index][1].astype(np.float32) - details[selected[-1]][1]).mean()) >= 0.5]
            if alternatives:
                best = alternatives[0]
            else:
                duplicate_fallbacks.append(len(selected))
        selected.append(int(best))

    os.makedirs(output_dir, exist_ok=True)
    for order, index in enumerate(selected, start=1):
        with Image.open(files[index]) as source:
            image = ImageOps.exif_transpose(source).convert("RGB")
            if square_crop:
                side = min(image.size)
                left = (image.width - side) // 2
                top = (image.height - side) // 2
                image = image.crop((left, top, left + side, top + side))
            image.save(os.path.join(output_dir, f"frame_{order:04d}.jpg"), quality=95)
    return {
        "input_count": len(files),
        "selected_count": len(selected),
        "median_focus": round(float(np.median([scores[index] for index in selected])), 1),
        "selected_names": [os.path.basename(files[index]) for index in selected],
        "selection_method": "temporal_buckets_focus_and_near_duplicate_check",
        "duplicate_fallback_indices": duplicate_fallbacks,
        "frames": [{"name": f"frame_{order:04d}.jpg", "source": os.path.basename(files[index]),
                    "source_index": index, "focus_score": round(scores[index], 3)}
                   for order, index in enumerate(selected, start=1)],
    }
