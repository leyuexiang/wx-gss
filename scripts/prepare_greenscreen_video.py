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

# 保留历史函数导出，供旧脚本和测试继续使用。
from app.foreground.priors.white_can import can_mask, refine_can_mask
from app.foreground import get_provider


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
    provider = get_provider("white_can")
    for target in targets:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(target))
        ok, bgr = cap.read()
        if not ok:
            raise ValueError("视频帧解码失败")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        mask = provider.coarse(rgb)
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
    (masks / "mask_manifest.json").write_text(json.dumps({"schema_version": 1,"kind": "automatic-white-can-green-screen-prototype", "provider": provider.describe(), "frames": manifest},indent=2),encoding="utf-8")
    (root / "selection.json").write_text(json.dumps({"video_sha256": hashlib.sha256(video.read_bytes()).hexdigest(),
        "fps": fps,"frames": count,"selected": records,"scope": "当前圆罐形状与颜色先验，未人工像素级复查"},ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps({"selected":len(records),"coverage":[min(x['foreground_ratio'] for x in records),max(x['foreground_ratio'] for x in records)]}))


if __name__ == "__main__":
    main()
