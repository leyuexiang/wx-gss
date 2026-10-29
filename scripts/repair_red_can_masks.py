"""从冻结选帧建立第二版红罐掩码，保留原图与旧掩码供配对对照。"""

import argparse
import json
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.capture.provenance import file_digest
from app.capture.quality import evaluate_foreground_support
from app.foreground import get_provider


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source")
    parser.add_argument("output")
    args = parser.parse_args()
    source, output = Path(args.source), Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    shutil.copytree(source / "images", output / "images")
    masks = output / "feature_masks"
    masks.mkdir()
    manifest = json.loads((source / "feature_masks/mask_manifest.json").read_text())
    provider = get_provider("red_can_v2")
    report = []
    for name, entry in manifest["frames"].items():
        old = source / "feature_masks" / entry["mask"]
        destination = masks / entry["mask"]
        if name.startswith("frame_"):
            rgb = np.array(Image.open(source / "images" / name).convert("RGB"))
            scale = min(1., 1600 / max(rgb.shape[:2]))
            small = cv2.resize(rgb, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            mask = provider.segment(small)
            # 向内收缩后的粗轮廓用于发现半主体截断；只是专用先验，不冒充人工复查。
            support = cv2.erode(provider.coarse(small), np.ones((21, 21), np.uint8))
            with Image.open(old) as image:
                old_mask = cv2.resize(np.array(image), (small.shape[1], small.shape[0]), interpolation=cv2.INTER_NEAREST)
            record = {"name": name, "support_kind": "red_can_v2_eroded_prior",
                "old": evaluate_foreground_support(old_mask, support),
                "new": evaluate_foreground_support(mask, support)}
            report.append(record)
            mask = cv2.resize(mask, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_NEAREST)
            Image.fromarray(mask).save(destination)
        else:
            shutil.copy2(old, destination)
        entry["mask_sha256"] = file_digest(destination)
        entry["reviewed"] = False
    manifest["provider"] = {"video": provider.describe(), "supplements": get_provider("gold_cap").describe()}
    manifest["quality"] = {"passed": all(row["new"]["passed"] for row in report),
        "scope": "仅检查专用先验覆盖，不证明背景干净或标注准确", "support_report": "support_report.json"}
    (masks / "mask_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    (output / "support_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    for path in source.iterdir():
        if path.name == "selection.json" or path.name.endswith("_names.txt"):
            shutil.copy2(path, output / path.name)
    print(json.dumps({"old_failed": sum(not r["old"]["passed"] for r in report),
        "new_failed": sum(not r["new"]["passed"] for r in report), "frames": len(report)}), flush=True)


if __name__ == "__main__":
    main()
