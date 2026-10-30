"""核验底圈恢复只增加下沿像素，并生成全量底缘叠加图供助手或用户检查。"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.capture.provenance import file_digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('baseline', 'candidate', 'output'):
        parser.add_argument(key)
    args = parser.parse_args()
    old, new, output = map(Path, (args.baseline, args.candidate, args.output))
    output.mkdir(parents=True, exist_ok=False)
    a, b = [json.loads((p / 'feature_masks/mask_manifest.json').read_text(encoding='utf-8')) for p in (old, new)]
    if set(a['frames']) != set(b['frames']):
        raise ValueError('候选帧集合改变')
    names = sorted(n for n in b['frames'] if n.startswith('frame_'))
    records, supplements = [], []
    for name in sorted(b['frames']):
        if file_digest(old / 'images' / name) != file_digest(new / 'images' / name):
            raise ValueError('底圈实验不允许修改原图')
        if not name.startswith('frame_'):
            supplements.append(file_digest(old / 'feature_masks' / a['frames'][name]['mask']) ==
                               file_digest(new / 'feature_masks' / b['frames'][name]['mask']))
    for page in range((len(names) + 29) // 30):
        canvas = Image.new('RGB', (1200, 600), (20, 20, 20))
        draw = ImageDraw.Draw(canvas)
        for k, name in enumerate(names[page * 30:(page + 1) * 30]):
            rgb = np.array(Image.open(new / 'images' / name).convert('RGB'))
            before = np.array(Image.open(old / 'feature_masks' / a['frames'][name]['mask']))
            after = np.array(Image.open(new / 'feature_masks' / b['frames'][name]['mask']))
            if before.shape != after.shape or not np.isin(after, (0, 255)).all():
                raise ValueError('候选尺寸或二值格式异常')
            y, x = np.nonzero(before)
            added, removed = after > before, before > after
            upper_added = int(added[:int(y.max() - .20 * (x.max() - x.min()))].sum())
            records.append({'name': name, 'added': int(added.sum()), 'removed': int(removed.sum()),
                            'upper_added': upper_added})
            for m, color in ((before, (0, 70, 255)), (after, (255, 80, 0))):
                contour, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(rgb, contour, -1, color, 2)
            box = (max(0, x.min() - 12), max(0, y.max() - 95), min(rgb.shape[1], x.max() + 12),
                   min(rgb.shape[0], y.max() + 50))
            tile = Image.fromarray(rgb).crop(box).resize((200, 96))
            col, row = k % 6, k // 6
            canvas.paste(tile, (col * 200, row * 120 + 24))
            draw.text((col * 200 + 3, row * 120 + 5), name, fill='white')
        canvas.save(output / f'base-band-page-{page + 1}.jpg', quality=95)
    summary = {'frames': records, 'all_monotonic': all(r['removed'] == 0 for r in records),
               'upper_unchanged': all(r['upper_added'] == 0 for r in records),
               'supplements_unchanged': all(supplements), 'total_added': sum(r['added'] for r in records),
               'candidate_manifest_sha256': file_digest(new / 'feature_masks/mask_manifest.json'),
               'baseline_manifest_sha256': file_digest(old / 'feature_masks/mask_manifest.json'),
               'generator_sha256': file_digest(__file__), 'visual_reviewed': False,
               'scope': '单调性及来源检查不证明轮廓准确；蓝为原轮廓，橙为恢复轮廓'}
    (output / 'audit.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    if not all(summary[k] for k in ('all_monotonic', 'upper_unchanged', 'supplements_unchanged')):
        raise ValueError('候选违反仅恢复下沿的约定')
    print(json.dumps({k: v for k, v in summary.items() if k != 'frames'}), flush=True)


if __name__ == '__main__':
    main()
