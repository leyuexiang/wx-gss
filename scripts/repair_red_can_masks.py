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
    parser.add_argument("--edge-guided", action="store_true", help="使用圆罐边缘候选；记录几何先验，尚非通用分割")
    parser.add_argument("--preserve-base-band", action="store_true", help="只恢复已有轮廓底边相连的有限金属亮带")
    args = parser.parse_args()
    source, output = Path(args.source), Path(args.output)
    if args.preserve_base_band and not args.edge_guided:
        parser.error('底圈亮带恢复仅允许与边缘候选一起使用')
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
            edge_report = None
            if args.edge_guided:
                from app.foreground.priors.cylinder_edges import cylinder_edge_mask
                mask, edge_report = cylinder_edge_mask(small)
                if args.preserve_base_band:
                    from app.foreground.priors.base_band import preserve_base_band
                    mask, band_report = preserve_base_band(small, mask)
                    edge_report['base_band'] = band_report
            else:
                mask = provider.segment(small)
            # 向内收缩后的粗轮廓用于发现半主体截断；只是专用先验，不冒充人工复查。
            support = cv2.erode(provider.coarse(small), np.ones((21, 21), np.uint8))
            if args.edge_guided:
                # 移除的转台不能当作“必须保留”的证据，覆盖检查只使用可靠红色罐身。
                hsv = cv2.cvtColor(small, cv2.COLOR_RGB2HSV)
                evidence = (((hsv[:, :, 0] < 16) | (hsv[:, :, 0] > 165))
                            & (hsv[:, :, 1] > 100) & (hsv[:, :, 2] > 65))
                support[~evidence] = 0
                # 转台会反射出红色，颜色不能作为底部真值；证据门禁仅检查罐身中段遗漏。
                _, sy, _, sh = cv2.boundingRect(provider.coarse(small))
                support[:int(sy + .2 * sh)] = 0
                support[int(sy + .8 * sh):] = 0
            with Image.open(old) as image:
                old_mask = cv2.resize(np.array(image), (small.shape[1], small.shape[0]), interpolation=cv2.INTER_NEAREST)
            record = {"name": name, "support_kind": "red_can_v2_eroded_prior",
                "old": evaluate_foreground_support(old_mask, support),
                "new": evaluate_foreground_support(mask, support)}
            report.append(record)
            if edge_report is not None:
                record["edge_guidance"] = edge_report
                record["support_kind"] = "eroded_prior_red_middle_body_only"
            mask = cv2.resize(mask, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_NEAREST)
            Image.fromarray(mask).save(destination)
        else:
            shutil.copy2(old, destination)
        entry["mask_sha256"] = file_digest(destination)
        entry["reviewed"] = False
    manifest["provider"] = {"video": provider.describe(), "supplements": get_provider("gold_cap").describe()}
    if args.edge_guided:
        manifest["provider"]["video"] = {"name": "red_can_edge_guided", "version": 1,
            "kind": "geometry_color_prior", "reviewed": False,
            "scope": "固定拍摄圆罐边缘候选；异常上沿使用截面先验"}
        if args.preserve_base_band:
            manifest['provider']['video'].update(version=2, base_band_recovery=True)
    manifest["quality"] = {"passed": all(row["new"]["passed"] for row in report),
        "scope": "仅检查专用先验覆盖，不证明背景干净或标注准确", "support_report": "support_report.json"}
    if args.edge_guided:
        manifest["quality"].update({"scope": "仅检查罐身中段覆盖，盖顶及罐底仍需独立复查",
            "shape_prior_frames": sum(r["edge_guidance"]["top_shape_prior_used"] for r in report),
            "outline_reviewed": False})
    (masks / "mask_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    (output / "support_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    for path in source.iterdir():
        if path.name == "selection.json" or path.name.endswith("_names.txt"):
            shutil.copy2(path, output / path.name)
    print(json.dumps({"old_failed": sum(not r["old"]["passed"] for r in report),
        "new_failed": sum(not r["new"]["passed"] for r in report), "frames": len(report)}), flush=True)


if __name__ == "__main__":
    main()
