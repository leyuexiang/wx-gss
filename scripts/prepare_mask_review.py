"""生成包含全部原图和掩码的离线逐帧修正页面，不修改源图片或原型掩码。"""

import argparse
import base64
import hashlib
import json
import sys
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.subject_masks import read_binary_mask


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image_dir")
    parser.add_argument("mask_dir")
    parser.add_argument("output_html")
    args = parser.parse_args()
    images, masks, output = Path(args.image_dir), Path(args.mask_dir), Path(args.output_html)
    if output.exists():
        raise ValueError("复查页面已存在，请使用新文件名")
    frames = {}
    for path in sorted(images.iterdir()):
        if path.suffix.lower() not in (".jpg", ".jpeg", ".png"):
            continue
        with Image.open(path) as image:
            width, height = image.size
            mime = Image.MIME[image.format]
        mask_bytes = (masks / (path.stem + ".png")).read_bytes()
        read_binary_mask(mask_bytes, width, height, path.name)
        source = path.read_bytes()
        frames[path.name] = {"width": width, "height": height,
                             "image_sha256": hashlib.sha256(source).hexdigest(),
                             "image": f"data:{mime};base64," + base64.b64encode(source).decode("ascii"),
                             "png": base64.b64encode(mask_bytes).decode("ascii")}
    if not frames:
        raise ValueError("未找到可复查的图像")
    template = Path(__file__).with_name("mask_review.html").read_text(encoding="utf-8")
    bundle = json.dumps({"frames": frames}, ensure_ascii=False).replace("<", "\\u003c")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(template.replace("__MASK_REVIEW_BUNDLE__", bundle), encoding="utf-8")
    print(json.dumps({"frames": len(frames), "output": str(output), "bytes": output.stat().st_size}))


if __name__ == "__main__":
    main()
