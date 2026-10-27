"""根据冻结主体矩形生成可复查的 GrabCut 前景掩码原型。"""

import argparse
import hashlib
import json
import math
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw


def _load_annotations(path):
    source = Path(path)
    data = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(data.get("frames"), dict) or not isinstance(data.get("reference_sha256"), dict):
        raise ValueError("标注必须包含 frames 和 reference_sha256")
    return data


def _grabcut(image, box, iterations):
    height, width = image.shape[:2]
    x, y, box_width, box_height = box
    right, bottom = x + box_width, y + box_height
    if x < 0 or y < 0 or right > width or bottom > height:
        raise ValueError("主体矩形超出图像")
    mask = np.full((height, width), cv2.GC_BGD, dtype=np.uint8)
    mask[y:bottom, x:right] = cv2.GC_PR_BGD
    # 矩形内缩后作为确定前景，避免把矩形边缘的桌面带入前景。
    inset_x = max(2, int(box_width * 0.10))
    inset_y = max(2, int(box_height * 0.06))
    inner_x, inner_y = x + inset_x, y + inset_y
    inner_right = max(inner_x + 1, right - inset_x)
    inner_bottom = max(inner_y + 1, bottom - inset_y)
    mask[inner_y:inner_bottom, inner_x:inner_right] = cv2.GC_PR_FGD
    # 罐身和罐盖中心区域种子，允许 GrabCut 从矩形边缘收缩背景。
    seed_x = x + int(box_width * 0.22)
    seed_y = y + int(box_height * 0.12)
    seed_right = max(seed_x + 1, right - int(box_width * 0.22))
    seed_bottom = max(seed_y + 1, bottom - int(box_height * 0.12))
    mask[seed_y:seed_bottom, seed_x:seed_right] = cv2.GC_FGD
    background_model = np.zeros((1, 65), np.float64)
    foreground_model = np.zeros((1, 65), np.float64)
    cv2.grabCut(image, mask, None, background_model, foreground_model, iterations, cv2.GC_INIT_WITH_MASK)
    return np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image_dir")
    parser.add_argument("annotations")
    parser.add_argument("output_dir")
    parser.add_argument("--iterations", type=int, default=5)
    args = parser.parse_args()
    if args.iterations < 1 or args.iterations > 20:
        raise ValueError("GrabCut 迭代次数必须在 1–20 之间")
    image_dir, output = Path(args.image_dir), Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    annotations = _load_annotations(args.annotations)
    records, tiles = [], []
    for name, box in annotations["frames"].items():
        path = image_dir / name
        if not path.is_file():
            raise ValueError(f"缺少参考图：{name}")
        actual_hash = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual_hash != annotations["reference_sha256"].get(name):
            raise ValueError(f"参考图摘要不一致：{name}")
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"无法读取参考图：{name}")
        mask = _grabcut(image, box, args.iterations)
        mask_path = output / f"{Path(name).stem}.png"
        Image.fromarray(mask).save(mask_path)
        ratio = float(np.count_nonzero(mask)) / float(mask.size)
        records.append({"name": name, "mask": mask_path.name, "foreground_ratio": ratio,
                        "box": box, "image_sha256": actual_hash})
        preview = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        preview[mask == 0] = (preview[mask == 0] * 0.18).astype(np.uint8)
        preview = Image.fromarray(preview)
        draw = ImageDraw.Draw(preview)
        x, y, width, height = box
        draw.rectangle((x, y, x + width - 1, y + height - 1), outline=(40, 220, 90), width=4)
        preview.thumbnail((270, 480))
        tile = Image.new("RGB", (280, 510), "black")
        tile.paste(preview, (0, 25))
        ImageDraw.Draw(tile).text((3, 4), f"{name} fg={ratio:.1%}", fill="white")
        tiles.append(tile)
    contact = Image.new("RGB", (280 * 4, 510 * math.ceil(len(tiles) / 4)), "black")
    for index, tile in enumerate(tiles):
        contact.paste(tile, ((index % 4) * 280, (index // 4) * 510))
    contact.save(output / "mask-contact.png")
    report = {"annotation_sha256": hashlib.sha256(Path(args.annotations).read_bytes()).hexdigest(),
              "algorithm": "opencv-grabcut", "iterations": args.iterations, "records": records,
              "scope": "矩形初始化的前景原型，不是人工分割真值；接入训练前需人工检查边缘"}
    (output / "mask_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"frames": len(records), "foreground_ratio": {
        record["name"]: round(record["foreground_ratio"], 4) for record in records}}, ensure_ascii=False))


if __name__ == "__main__":
    main()
