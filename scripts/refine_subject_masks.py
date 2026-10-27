"""使用红色罐身与金色罐盖的颜色线索修正实心茶罐原型，仍标记为未复查。"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.subject_masks import read_binary_mask


def largest_solid_component(mask):
    """当前对象为实心罐体：保留最大外轮廓，填回内部文字/高光小孔。"""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError("原型掩码没有前景")
    result = np.zeros_like(mask)
    cv2.drawContours(result, [max(contours, key=cv2.contourArea)], -1, 255, cv2.FILLED)
    return result


def refine_mask(rgb, original):
    """颜色线索只适用于当前红金茶罐，不能当作通用物体分割或人工真值。"""
    component = largest_solid_component(original)
    x, y, width, height = cv2.boundingRect(component)
    image_height, image_width = original.shape
    padding = max(10, int(round(max(width, height) * .12)))
    left, top = max(0, x - padding), max(0, y - padding)
    right, bottom = min(image_width, x + width + padding), min(image_height, y + height + padding)
    # 扩大搜索范围以恢复被插值矩形截掉的罐盖；桌面通常呈低色差，不作为颜色种子。
    # 木色椅背也满足“红色通道更大”，改用色相区分红罐身、偏黄金盖和偏橙木色。
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    hue, saturation, value = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    colored = (saturation >= 25) & (value >= 25)
    seed = (colored & ((hue <= 10) | (hue >= 170) |
                       ((hue >= 22) & (hue <= 40)))).astype(np.uint8) * 255
    restricted = np.zeros_like(original)
    restricted[top:bottom, left:right] = seed[top:bottom, left:right]
    restricted = cv2.morphologyEx(restricted, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    # 连接罐盖与罐身，填回实心物体上的内部白色文字和镜面高光，保留外轮廓。
    radius = max(1, int(round(min(width, height) * .008)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
    restricted = cv2.morphologyEx(restricted, cv2.MORPH_CLOSE, kernel)
    component = largest_solid_component(restricted)
    contours, _ = cv2.findContours(component, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # 当前罐体轮廓近似凸形，用凸包补回与外部相连的白色条码和镜面高光缺口。
    # 这属于当前物体的形状先验，不可用于凹形、镂空或透明物体。
    hull = cv2.convexHull(max(contours, key=cv2.contourArea))
    result = np.zeros_like(original)
    cv2.drawContours(result, [hull], -1, 255, cv2.FILLED)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image_dir")
    parser.add_argument("prototype_dir")
    parser.add_argument("output_dir")
    args = parser.parse_args()
    images, source, output = Path(args.image_dir), Path(args.prototype_dir), Path(args.output_dir)
    if output.exists():
        raise ValueError("输出目录已存在，拒绝覆盖已有修正结果")
    names = sorted(path.name for path in images.glob("*.jpg"))
    if not names:
        raise ValueError("没有找到茶罐图像")
    output.mkdir(parents=True)
    entries, records, pages = {}, [], []
    for index, name in enumerate(names):
        image_bytes = (images / name).read_bytes()
        with Image.open(images / name) as image:
            rgb = np.array(image.convert("RGB"))
        height, width = rgb.shape[:2]
        prototype = (source / (Path(name).stem + ".png")).read_bytes()
        original = read_binary_mask(prototype, width, height, name)
        refined = refine_mask(rgb, original)
        mask_name = Path(name).stem + ".png"
        Image.fromarray(refined).save(output / mask_name)
        saved = (output / mask_name).read_bytes()
        read_binary_mask(saved, width, height, name)
        entries[name] = {"mask": mask_name, "reviewed": False,
                         "image_sha256": hashlib.sha256(image_bytes).hexdigest(),
                         "mask_sha256": hashlib.sha256(saved).hexdigest()}
        removed = int(np.count_nonzero((original > 0) & (refined == 0)))
        added = int(np.count_nonzero((original == 0) & (refined > 0)))
        fraction = (removed + added) / max(1, np.count_nonzero(original))
        records.append({"name": name, "prototype_sha256": hashlib.sha256(prototype).hexdigest(),
                        "removed_pixels": removed, "added_pixels": added,
                        "changed_fraction_of_original_foreground": fraction,
                        "foreground_ratio": float(np.count_nonzero(refined) / refined.size)})
        preview = rgb.copy()
        preview[refined == 0] = (preview[refined == 0] * .3).astype(np.uint8)
        preview[(original > 0) & (refined == 0)] = (230, 70, 170)
        # 预览同时覆盖原轮廓与新增轮廓，避免恢复的罐盖在诊断图中再次被裁掉。
        x, y, box_width, box_height = cv2.boundingRect(cv2.bitwise_or(original, refined))
        preview = Image.fromarray(preview).crop((max(0, x - 40), max(0, y - 40),
                                                 min(width, x + box_width + 40), min(height, y + box_height + 40)))
        preview.thumbnail((224, 352))
        tile = Image.new("RGB", (232, 384), (20, 25, 35))
        tile.paste(preview, ((232 - preview.width) // 2, 28))
        ImageDraw.Draw(tile).text((4, 4), f"{name} change {fraction:.1%}", fill="white")
        pages.append(tile)
        if len(pages) == 15 or index == len(names) - 1:
            contact = Image.new("RGB", (232 * 5, 384 * 3), (20, 25, 35))
            for position, panel in enumerate(pages):
                contact.paste(panel, ((position % 5) * 232, (position // 5) * 384))
            contact.save(output / f"contact_{index // 15 + 1}.png")
            pages = []
    report = {"kind": "automatic-hue-convex-can-prototype", "frames": records,
              "scope": "实心茶罐专用自动原型；未人工复查，不可解释为分割真值"}
    (output / "mask_manifest.json").write_text(json.dumps({"schema_version": 1,
        "kind": report["kind"], "frames": entries}, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "refinement_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"frames": len(records), "changed_frames": sum(row["removed_pixels"] + row["added_pixels"] > 0 for row in records),
                      "max_change": max(row["changed_fraction_of_original_foreground"] for row in records)}))


if __name__ == "__main__":
    main()
