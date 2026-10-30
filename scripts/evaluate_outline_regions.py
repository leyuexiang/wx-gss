"""在统一自动轮廓参考下区分盖顶补回、罐底移除和共同罐身区域的误差。"""

import argparse
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image


def region_masks(old, candidate):
    """区域定义固定于训练结果之前；不能用整图背景占比掩盖顶底差异。"""
    if old.shape != candidate.shape or old.ndim != 2 or not candidate.any():
        raise ValueError('顶底区域需要同尺寸非空候选掩码')
    y = np.flatnonzero(candidate.any(axis=1))
    rows = np.arange(candidate.shape[0])[:, None]
    height = max(int(y[-1] - y[0]), 1)
    return {'added_cap': candidate & ~old & (rows <= y[0] + .35 * height),
            'removed_base': old & ~candidate & (rows >= y[0] + .80 * height),
            'common_body': old & candidate & (rows >= y[0] + .30 * height) & (rows <= y[0] + .75 * height)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('baseline_masks')
    parser.add_argument('trial')
    args = parser.parse_args()
    trial, baseline = Path(args.trial), Path(args.baseline_masks)
    groups = ('old_mask', 'repaired_mask')
    metrics = json.loads((trial / groups[0] / 'common_reference/metrics.json').read_text(encoding='utf-8'))
    records, totals = [], {g: {} for g in groups}
    for record in metrics['records']:
        name, w, h = record['name'], record['width'], record['height']
        with Image.open(baseline / (name + '.png')) as image:
            old = np.array(image) > 127
        with Image.open(trial / 'repaired_masks' / (name + '.png')) as image:
            candidate = np.array(image) > 127
        regions = region_masks(old, candidate)
        for group in groups:
            with Image.open(trial / group / 'common_reference' / f"view_{record['index']:04d}.png") as image:
                # 两组均读取同一候选掩码参考，禁止各自使用自己的标注给自己评分。
                target = np.asarray(image.crop((0, 24, w, h + 24)), dtype=np.float32) / 255.
                prediction = np.asarray(image.crop((w, 24, 2 * w, h + 24)), dtype=np.float32) / 255.
            error = ((prediction - target) ** 2).mean(axis=-1)
            for key, mask in regions.items():
                pixels, square_sum = int(mask.sum()), float(error[mask].sum(dtype=np.float64))
                item = totals[group].setdefault(key, {'pixels': 0, 'square_sum': 0.})
                item['pixels'] += pixels
                item['square_sum'] += square_sum
                records.append({'name': name, 'group': group, 'region': key, 'pixels': pixels,
                                'mse': square_sum / pixels if pixels else None})
    for group in totals.values():
        for item in group.values():
            item['mse'] = item['square_sum'] / item['pixels'] if item['pixels'] else None
            item['psnr'] = -10 * math.log10(max(item['mse'], 1e-12)) if item['pixels'] else None
    result = {'totals': totals, 'per_view': records,
              'scope': '候选自动轮廓的同区域像素加权诊断；由八位评估面板计算，非人工真值，不能替代边界确认',
              'regions': {'added_cap': '新增前景且位于候选高度上方35%',
                          'removed_base': '移除前景且位于候选高度下方20%',
                          'common_body': '两组共同前景且位于候选高度30%-75%'}}
    (trial / 'outline_regions.json').write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps(totals), flush=True)


if __name__ == '__main__':
    main()
