"""红罐绿幕视频与顶底补拍照片的独立准备，使用颜色先验而非通用分割。"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageOps

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.prepare_greenscreen_video import refine_can_mask


def red_can_mask(rgb):
    """以红色罐身定位金盖区域，排除绿色背景和底部转台反射。"""
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    red = (((hsv[:, :, 0] < 16) | (hsv[:, :, 0] > 165))
           & (hsv[:, :, 1] > 100) & (hsv[:, :, 2] > 65)).astype(np.uint8) * 255
    red = cv2.morphologyEx(red, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    contours, _ = cv2.findContours(red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError("没有找到红色罐身")
    body = max(contours, key=cv2.contourArea)
    x, y, width, height = cv2.boundingRect(body)
    if width * height < rgb.shape[0] * rgb.shape[1] * .02:
        raise ValueError("红色主体过小，拒绝生成掩码")
    r, g, b = rgb.astype(np.int16).transpose(2, 0, 1)
    warm = (r >= g - 20) & (r >= b + 8) & (hsv[:, :, 2] > 60)
    roi = np.zeros_like(red)
    roi[max(0, int(y - .52 * height)):min(rgb.shape[0], int(y + 1.08 * height)),
        max(0, int(x - .08 * width)):min(rgb.shape[1], int(x + 1.08 * width))] = 255
    body_only = np.zeros_like(red)
    cv2.drawContours(body_only, [body], -1, 255, cv2.FILLED)
    candidate = ((warm | (body_only > 0)) & (roi > 0)).astype(np.uint8) * 255
    candidate = cv2.morphologyEx(candidate, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    contours, _ = cv2.findContours(candidate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    hull = cv2.convexHull(max(contours, key=cv2.contourArea))
    mask = np.zeros_like(red)
    cv2.drawContours(mask, [hull], -1, 255, cv2.FILLED)
    return mask


def gold_cap_mask(rgb):
    """仅用于深灰桌面上的金属顶底照片，保留圆面内的印刷文字。"""
    r, g, b = rgb.astype(np.int16).transpose(2, 0, 1)
    seed = ((r >= g - 15) & (r >= b + 12) & (r > 85)).astype(np.uint8) * 255
    seed = cv2.morphologyEx(seed, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    contours, _ = cv2.findContours(seed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError("补拍照片没有找到金属圆面")
    contour = max(contours, key=cv2.contourArea)
    if len(contour) < 5 or cv2.contourArea(contour) < rgb.shape[0] * rgb.shape[1] * .05:
        raise ValueError("金属圆面面积或边界不足")
    mask = np.zeros_like(seed)
    cv2.ellipse(mask, cv2.fitEllipse(contour), 255, cv2.FILLED)
    return mask


def smaller(rgb, long_edge=1600):
    scale = min(1., long_edge / max(rgb.shape[:2]))
    return cv2.resize(rgb, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA), scale


def digest(path):
    """流式摘要避免额外复制整段高分辨率视频。"""
    value = hashlib.sha256()
    with open(path, "rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video")
    parser.add_argument("output")
    parser.add_argument("--top", required=True)
    parser.add_argument("--bottom", required=True)
    parser.add_argument("--views", type=int, default=120)
    args = parser.parse_args()
    if not 20 <= args.views <= 200:
        parser.error("视频视角数需在 20–200 之间")
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=False)
    images, masks = root / "images", root / "feature_masks"
    images.mkdir()
    masks.mkdir()
    cap = cv2.VideoCapture(args.video)
    count, fps = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), cap.get(cv2.CAP_PROP_FPS)
    if not cap.isOpened() or count < 2 * args.views or fps <= 0:
        raise ValueError("视频不能解码或帧数不足")
    selected, candidates, bounds = [], [], []
    for target in np.linspace(0, count - 2, 2 * args.views, dtype=int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(target))
        ok, bgr = cap.read()
        if not ok:
            raise ValueError(f"视频帧读取失败：{target}")
        # 当前文件像素画面横倒，固定顺时针旋转后罐盖朝上，不对各帧随意旋转。
        rgb = cv2.cvtColor(cv2.rotate(bgr, cv2.ROTATE_90_CLOCKWISE), cv2.COLOR_BGR2RGB)
        full_height, full_width = rgb.shape[:2]
        small, scale = smaller(rgb)
        mask = red_can_mask(small)
        gray = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY)
        interior = cv2.erode(mask, np.ones((9, 9), np.uint8)) > 0
        score = float(cv2.Laplacian(gray, cv2.CV_32F)[interior].var())
        x, y, w, h = cv2.boundingRect(mask)
        candidates.append({"source_frame": int(target), "seconds": target / fps, "focus": score,
            "bounds": [int(x / scale), int(y / scale), int((x + w) / scale), int((y + h) / scale)]})
        if len(candidates) == 2:
            winner = max(candidates, key=lambda row: row["focus"])
            selected.append(winner)
            bounds.append(winner["bounds"])
            candidates = []
    boxes = np.array(bounds)
    left, top = max(0, int(boxes[:, 0].min()) - 48), max(0, int(boxes[:, 1].min()) - 48)
    right, bottom = min(full_width, int(boxes[:, 2].max()) + 48), min(full_height, int(boxes[:, 3].max()) + 48)
    # 全部视频帧使用同一裁切，主点按原画面中心平移，焦距初值仍使用原画面尺寸。
    video_params = [1.2 * max(full_width, full_height), full_width / 2 - left, full_height / 2 - top, 0.]
    records, entries, groups = [], {}, []

    def save(name, rgb, mask, record):
        Image.fromarray(rgb).save(images / name)
        Image.fromarray(mask).save(masks / (name + ".png"))
        entries[name] = {"mask": name + ".png", "reviewed": False,
            "image_sha256": digest(images / name), "mask_sha256": digest(masks / (name + ".png"))}
        records.append({"name": name, **record, "foreground_ratio": float((mask > 0).mean())})
        j = len(records) - 1
        if j % 15 == 0:
            save.contact = Image.new("RGB", (1500, 1500), (25, 25, 25))
        ys, xs = np.nonzero(mask)
        preview = rgb.copy()
        preview[mask == 0] //= 5
        tile = Image.fromarray(preview).crop((max(0, xs.min() - 15), max(0, ys.min() - 15),
            min(rgb.shape[1], xs.max() + 16), min(rgb.shape[0], ys.max() + 16)))
        tile.thumbnail((290, 455))
        x, y = (j % 5) * 300, ((j % 15) // 5) * 500
        save.contact.paste(tile, (x + (300 - tile.width) // 2, y + 30))
        ImageDraw.Draw(save.contact).text((x + 5, y + 5), name, fill="white")
        save.contact.save(root / f"mask-contact-{j // 15 + 1}.jpg")

    video_names = []
    for i, row in enumerate(selected, 1):
        cap.set(cv2.CAP_PROP_POS_FRAMES, row["source_frame"])
        ok, bgr = cap.read()
        if not ok:
            raise ValueError("选中帧无法再次解码")
        rgb = cv2.cvtColor(cv2.rotate(bgr, cv2.ROTATE_90_CLOCKWISE), cv2.COLOR_BGR2RGB)[top:bottom, left:right]
        small, _ = smaller(rgb)
        mask = refine_can_mask(small, red_can_mask(small))
        mask = cv2.resize(mask, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_NEAREST)
        name = f"frame_{i:04d}.png"
        save(name, rgb, mask, {**row, "kind": "video"})
        video_names.append(name)
        if i % 20 == 0:
            print(f"视频帧已保存 {i}/{len(selected)}", flush=True)
    cap.release()
    (root / "video_names.txt").write_text("\n".join(video_names) + "\n", encoding="utf-8")
    groups.append({"image_list": "video_names.txt", "camera_params": video_params})
    for kind, source in (("top", args.top), ("bottom", args.bottom)):
        with Image.open(source) as image:
            rgb = np.array(ImageOps.exif_transpose(image).convert("RGB"))
        small, ratio = smaller(rgb)
        seed = refine_can_mask(small, gold_cap_mask(small))
        mask = cv2.resize(seed, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_NEAREST)
        x, y, w, h = cv2.boundingRect(mask)
        x0, y0, x1, y1 = max(0, x - 80), max(0, y - 80), min(rgb.shape[1], x + w + 80), min(rgb.shape[0], y + h + 80)
        original_h, original_w = rgb.shape[:2]
        rgb, mask = rgb[y0:y1, x0:x1], mask[y0:y1, x0:x1]
        cropped_h, cropped_w = rgb.shape[:2]
        # 补拍仍保留足够像素匹配细纹，最终训练由同一去畸变上限控制。
        rgb, scale = smaller(rgb, 2400)
        mask = cv2.resize(mask, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_NEAREST)
        sx, sy = rgb.shape[1] / cropped_w, rgb.shape[0] / cropped_h
        if abs(sx - sy) > .001:
            raise ValueError("补拍缩放出现异常")
        params = [1.2 * max(original_w, original_h) * scale,
            (original_w / 2 - x0) * sx, (original_h / 2 - y0) * sy, 0.]
        name = f"supplement_{kind}.png"
        save(name, rgb, mask, {"kind": kind, "source_sha256": digest(source),
            "crop": [x0, y0, x1, y1], "exif_orientation_applied": True})
        (root / f"{kind}_names.txt").write_text(name + "\n", encoding="utf-8")
        groups.append({"image_list": f"{kind}_names.txt", "camera_params": params})
    (masks / "mask_manifest.json").write_text(json.dumps({"schema_version": 1,
        "kind": "red-can-and-gold-caps-automatic-prototype", "frames": entries}, indent=2), encoding="utf-8")
    protocol = {"video_sha256": digest(args.video), "rotation": "90_degrees_clockwise",
        "video_original_upright_size": [full_width, full_height], "video_crop": [left, top, right, bottom],
        "fps": fps, "frames": count, "selected": records, "feature_groups": groups,
        "scope": "当前红罐颜色与形状先验；补拍照片必须独立定位后才能参与训练"}
    (root / "selection.json").write_text(json.dumps(protocol, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"images": len(records), "crop": protocol["video_crop"]}), flush=True)


if __name__ == "__main__":
    main()
