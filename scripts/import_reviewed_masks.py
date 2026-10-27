"""导入逐帧复查文件，校验来源后生成可用于训练的掩码目录与摘要清单。"""

import argparse
import base64
import hashlib
import io
import json
import shutil
import sys
import tempfile
from pathlib import Path

from PIL import Image
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.subject_masks import read_binary_mask


def import_masks(image_dir, review_file, output_dir):
    images, output = Path(image_dir), Path(output_dir)
    data = json.loads(Path(review_file).read_text(encoding="utf-8"))
    if data.get("schema_version") != 1 or data.get("kind") != "manual-mask-review":
        raise ValueError("复查文件格式无效")
    names = {path.name for path in images.iterdir() if path.suffix.lower() in (".jpg", ".jpeg", ".png")}
    if not names or set(data.get("frames", {})) != names:
        raise ValueError("复查文件必须包含全部原图视角")
    if output.exists():
        raise ValueError("输出掩码目录已存在，拒绝覆盖")
    output.parent.mkdir(parents=True, exist_ok=True)
    # 全部校验成功才发布目录，导入失败不会留下可误用的半套掩码。
    with tempfile.TemporaryDirectory(prefix="mask-import-", dir=output.parent) as temporary:
        staging, entries = Path(temporary) / "masks", {}
        staging.mkdir()
        for name in sorted(names):
            item, path = data["frames"][name], images / name
            if item.get("reviewed") is not True:
                raise ValueError(f"视角尚未确认：{name}")
            if hashlib.sha256(path.read_bytes()).hexdigest() != item.get("image_sha256"):
                raise ValueError(f"原图摘要不一致：{name}")
            content = base64.b64decode(item["png"], validate=True)
            with Image.open(path) as image:
                width, height = image.size
            # 浏览器输出 RGBA；先检查是二值不透明像素，再无损写为训练使用的灰度 PNG。
            with Image.open(io.BytesIO(content)) as mask_image:
                rgba = np.array(mask_image.convert("RGBA"))
                if mask_image.format != "PNG" or mask_image.size != (width, height):
                    raise ValueError(f"掩码尺寸不一致：{name}")
            if not ((rgba[:, :, :3] == rgba[:, :, :1]).all() and (rgba[:, :, 3] == 255).all()):
                raise ValueError(f"掩码必须是黑白不透明图像：{name}")
            mask_name = name + ".mask.png"
            destination = staging / mask_name
            Image.fromarray(rgba[:, :, 0]).save(destination)
            saved = destination.read_bytes()
            read_binary_mask(saved, width, height, name)
            entries[name] = {"mask": mask_name, "reviewed": True,
                             "image_sha256": item["image_sha256"],
                             "mask_sha256": hashlib.sha256(saved).hexdigest()}
        (staging / "mask_manifest.json").write_text(json.dumps({"schema_version": 1,
            "kind": "manual-mask-review", "frames": entries}, ensure_ascii=False, indent=2), encoding="utf-8")
        shutil.move(str(staging), str(output))
    return {"frames": len(entries), "output": str(output)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image_dir")
    parser.add_argument("review_file")
    parser.add_argument("output_dir")
    args = parser.parse_args()
    print(json.dumps(import_masks(args.image_dir, args.review_file, args.output_dir), ensure_ascii=False))


if __name__ == "__main__":
    main()
