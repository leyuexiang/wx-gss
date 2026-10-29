"""当前白色圆罐绿幕视频的独立选帧和自动轮廓原型，不作为通用分割真值。"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def can_mask(rgb):
    """先定位中央白色罐身，再用金色轮廓恢复盖子，保留盖顶的绿色图案。"""
    height, width = rgb.shape[:2]
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    red, green, blue = rgb.astype(np.int16).transpose(2, 0, 1)
    # 高光绿布也可能低饱和，额外排除绿色偏色，避免凸包把背景连到罐身。
    neutral = (green <= red + 8) & (green <= blue + 16)
    white = ((hsv[:, :, 1] < 70) & (hsv[:, :, 2] > 120) & neutral).astype(np.uint8) * 255
    region = np.zeros_like(white)
    region[int(.15 * height):int(.94 * height), int(.28 * width):int(.72 * width)] = 255
    white &= region
    white = cv2.morphologyEx(white, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    contours, _ = cv2.findContours(white, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError("中央没有找到白色罐身")
    body = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(body)
    if w * h < width * height * .025:
        raise ValueError("主体面积不足，拒绝继续生成错误掩码")
    roi = np.zeros_like(white)
    roi[max(0, int(y - h * .42)):min(height, y + h + 8), max(0, x - 16):min(width, x + w + 16)] = 255
    gold = ((hsv[:, :, 0] < 40) & (hsv[:, :, 1] > 45) & (hsv[:, :, 2] > 70)
            & (red >= green + 3))
    body_only = np.zeros_like(white)
    cv2.drawContours(body_only, [body], -1, 255, cv2.FILLED)
    # 盖顶的金边可能被绿色图案隔开，保留上方分离的金边，罐身仅取主连通分量。
    gold[y + 20:] = False
    candidate = (((body_only > 0) | gold) & (roi > 0)).astype(np.uint8) * 255
    candidate = cv2.morphologyEx(candidate, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    yy, xx = np.nonzero(candidate)
    hull = cv2.convexHull(np.stack((xx, yy), axis=1).astype(np.int32))
    result = np.zeros_like(white)
    cv2.drawContours(result, [hull], -1, 255, cv2.FILLED)
    return result


def refine_can_mask(rgb, seed):
    """只在粗轮廓附近用图割细化，恢复绿色盖顶和圆形边缘，不允许背景任意扩张。"""
    small = cv2.resize(rgb, None, fx=.5, fy=.5)
    scaled = cv2.resize(seed, (small.shape[1], small.shape[0]), interpolation=cv2.INTER_NEAREST)
    outer = cv2.dilate(scaled, np.ones((29, 29), np.uint8))
    inner = cv2.erode(scaled, np.ones((21, 21), np.uint8))
    labels = np.full_like(scaled, cv2.GC_BGD)
    labels[outer > 0] = cv2.GC_PR_BGD
    labels[scaled > 0] = cv2.GC_PR_FGD
    labels[inner > 0] = cv2.GC_FGD
    cv2.setRNGSeed(42)
    cv2.grabCut(small, labels, None, np.zeros((1, 65)), np.zeros((1, 65)), 3, cv2.GC_INIT_WITH_MASK)
    foreground = ((labels == cv2.GC_FGD) | (labels == cv2.GC_PR_FGD)).astype(np.uint8) * 255
    return cv2.resize(foreground, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_NEAREST)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video")
    parser.add_argument("output")
    parser.add_argument("--views", type=int, default=60)
    parser.add_argument("--candidates-per-view", type=int, default=4)
    args = parser.parse_args()
    if not 12 <= args.views <= 240 or not 1 <= args.candidates_per_view <= 4:
        parser.error("视角数需在 12–240 之间，每段候选帧数需在 1–4 之间")
    video, root = Path(args.video), Path(args.output)
    root.mkdir(parents=True, exist_ok=False)
    images, masks = root / "images", root / "feature_masks"
    images.mkdir()
    masks.mkdir()
    cap = cv2.VideoCapture(str(video))
    count, fps = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), cap.get(cv2.CAP_PROP_FPS)
    if not cap.isOpened() or fps <= 0 or count < args.views * args.candidates_per_view:
        raise ValueError("视频不可读取或有效帧数不足")
    targets = np.linspace(0, count - 2, args.views * args.candidates_per_view, dtype=int)
    # 固定分段保持时间覆盖，在主体外接框评估清晰度，避免全图绿布纹理主导打分。
    selected, candidates = [], []
    for target in targets:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(target))
        ok, bgr = cap.read()
        if not ok:
            raise ValueError("视频帧解码失败")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        mask = can_mask(rgb)
        x, y, w, h = cv2.boundingRect(mask)
        gray = cv2.cvtColor(bgr[y:y+h, x:x+w], cv2.COLOR_BGR2GRAY)
        score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        candidates.append((score, int(target), rgb, mask))
        if len(candidates) == args.candidates_per_view:
            selected.append(max(candidates, key=lambda row: row[0]))
            candidates = []
    cap.release()
    records, manifest = [], {}
    for index, (score, frame, rgb, mask) in enumerate(selected, 1):
        mask = refine_can_mask(rgb, mask)
        name = f"frame_{index:04d}.png"
        Image.fromarray(rgb).save(images / name)
        Image.fromarray(mask).save(masks / (name + ".png"))
        records.append({"name": name, "source_frame": frame, "seconds": frame / fps, "focus": score,
                        "foreground_ratio": float((mask > 0).mean())})
        manifest[name] = {"mask": name + ".png", "reviewed": False,
                         "image_sha256": hashlib.sha256((images / name).read_bytes()).hexdigest(),
                         "mask_sha256": hashlib.sha256((masks / (name + ".png")).read_bytes()).hexdigest()}
        if (index - 1) % 15 == 0:
            contact = Image.new("RGB", (1200, 900), (25, 28, 35))
        preview = rgb.copy()
        preview[mask == 0] = (preview[mask == 0] * .2).astype(np.uint8)
        tile = Image.fromarray(preview)
        tile.thumbnail((235, 265))
        position = (index - 1) % 15
        left, top = (position % 5) * 240, (position // 5) * 300
        contact.paste(tile, (left + (240 - tile.width) // 2, top + 25))
        ImageDraw.Draw(contact).text((left + 4, top + 4), f"{index}: {frame / fps:.1f}s", fill="white")
        if index % 15 == 0 or index == len(selected):
            contact.save(root/f"mask-contact-{(index - 1)//15 + 1}.jpg")
    (masks / "mask_manifest.json").write_text(json.dumps({"schema_version": 1,"kind": "automatic-white-can-green-screen-prototype", "frames": manifest},indent=2),encoding="utf-8")
    (root / "selection.json").write_text(json.dumps({"video_sha256": hashlib.sha256(video.read_bytes()).hexdigest(),
        "fps": fps,"frames": count,"selected": records,"scope": "当前圆罐形状与颜色先验，未人工像素级复查"},ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps({"selected":len(records),"coverage":[min(x['foreground_ratio'] for x in records),max(x['foreground_ratio'] for x in records)]}))


if __name__ == "__main__":
    main()
