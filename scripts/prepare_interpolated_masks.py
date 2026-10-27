"""用七个冻结矩形插值出全序列掩码实验，输出前景覆盖率和接触图。"""

import argparse
import json
import math
import hashlib
import sys
from bisect import bisect_left
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.prepare_subject_masks import _grabcut


def interpolate_boxes(names, key_boxes):
    keys = sorted((int(name.split("_")[-1].split(".")[0]), np.asarray(box, dtype=float))
                  for name, box in key_boxes.items())
    if not keys:
        raise ValueError("插值需要至少一个参考矩形")
    positions = [item[0] for item in keys]
    result = {}
    for name in names:
        index = int(name.split("_")[-1].split(".")[0])
        if index <= keys[0][0]:
            box = keys[0][1]
        elif index >= keys[-1][0]:
            box = keys[-1][1]
        else:
            right_index = bisect_left(positions, index)
            left, right = keys[right_index - 1], keys[right_index]
            ratio = (index - left[0]) / (right[0] - left[0])
            box = left[1] * (1 - ratio) + right[1] * ratio
        result[name] = [int(round(value)) for value in box]
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image_dir")
    parser.add_argument("annotations")
    parser.add_argument("output_dir")
    parser.add_argument("--iterations", type=int, default=5)
    args = parser.parse_args()
    image_dir, annotation_path, output = Path(args.image_dir), Path(args.annotations), Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    annotations = json.loads(annotation_path.read_text(encoding="utf-8"))
    if not 1 <= args.iterations <= 20:
        raise ValueError("掩码迭代次数必须在 1–20 之间")
    # 验证关键帧来源，不能把另一组去畸变图上的矩形直接用于当前数据。
    for name in annotations["frames"]:
        if hashlib.sha256((image_dir / name).read_bytes()).hexdigest() != annotations["reference_sha256"].get(name):
            raise ValueError(f"参考图摘要不一致：{name}")
    names = sorted(path.name for path in image_dir.glob("*.jpg"))
    boxes = interpolate_boxes(names, annotations["frames"])
    records, tiles, entries = [], [], {}
    for name in names:
        image = cv2.imread(str(image_dir / name), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"无法读取图像：{name}")
        mask = _grabcut(image, boxes[name], args.iterations)
        Image.fromarray(mask).save(output / f"{Path(name).stem}.png")
        mask_file = output / f"{Path(name).stem}.png"
        entries[name] = {"mask": mask_file.name, "reviewed": False,
                         "image_sha256": hashlib.sha256((image_dir / name).read_bytes()).hexdigest(),
                         "mask_sha256": hashlib.sha256(mask_file.read_bytes()).hexdigest()}
        ratio = float(np.count_nonzero(mask)) / float(mask.size)
        records.append({"name": name, "box": boxes[name], "foreground_ratio": ratio})
        if name in annotations["frames"]:
            preview = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            preview[mask == 0] = (preview[mask == 0] * 0.18).astype(np.uint8)
            preview = Image.fromarray(preview)
            x, y, width, height = boxes[name]
            ImageDraw.Draw(preview).rectangle((x, y, x + width - 1, y + height - 1), outline=(40, 220, 90), width=4)
            preview.thumbnail((270, 480))
            tile = Image.new("RGB", (280, 510), "black")
            tile.paste(preview, (0, 25))
            ImageDraw.Draw(tile).text((3, 4), f"{name} fg={ratio:.1%}", fill="white")
            tiles.append(tile)
    contact = Image.new("RGB", (280 * 4, 510 * math.ceil(len(tiles) / 4)), "black")
    for index, tile in enumerate(tiles):
        contact.paste(tile, ((index % 4) * 280, (index // 4) * 510))
    contact.save(output / "mask-contact.png")
    report = {"source_annotations": str(annotation_path), "key_frames": len(annotations["frames"]),
              "frames": len(records), "algorithm": "interpolated-box-grabcut",
              "iterations": args.iterations, "records": records,
              "scope": "训练实验原型，不是人工分割真值；接入默认流程前必须人工检查"}
    (output / "mask_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "mask_manifest.json").write_text(json.dumps({"schema_version": 1,
        "kind": "interpolated-box-grabcut-prototype", "frames": entries}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"frames": len(records), "min_foreground_ratio": min(x["foreground_ratio"] for x in records),
                      "max_foreground_ratio": max(x["foreground_ratio"] for x in records)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
