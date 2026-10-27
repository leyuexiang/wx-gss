"""审计固定主体矩形内外的亮度变化，不直接修改训练图像。"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image_dir")
    parser.add_argument("annotations")
    parser.add_argument("output")
    args = parser.parse_args()
    image_dir, annotation_path = Path(args.image_dir), Path(args.annotations)
    annotations = json.loads(annotation_path.read_text(encoding="utf-8"))
    rows = []
    for name, box in annotations["frames"].items():
        image_path = image_dir / name
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"无法读取图像：{image_path}")
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        x, y, width, height = box
        roi = gray[y:y + height, x:x + width]
        inner = roi[int(height * .15):int(height * .85), int(width * .15):int(width * .85)]
        outside = np.ones(gray.shape, dtype=bool)
        outside[y:y + height, x:x + width] = False
        background = gray[outside]
        rows.append({
            "name": name,
            "full_mean": float(gray.mean()),
            "roi_mean": float(roi.mean()),
            "inner_mean": float(inner.mean()),
            "background_mean": float(background.mean()),
            "roi_p05": float(np.percentile(roi, 5)),
            "roi_p95": float(np.percentile(roi, 95)),
        })
    report = {
        "rows": rows,
        "inner_luminance_range": max(row["inner_mean"] for row in rows) - min(row["inner_mean"] for row in rows),
        "background_luminance_range": max(row["background_mean"] for row in rows) - min(row["background_mean"] for row in rows),
        "scope": "亮度审计，不是颜色真值；不对训练图像做校正",
    }
    Path(args.output).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
