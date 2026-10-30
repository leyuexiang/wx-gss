"""从实际去畸变训练图生成少量深度并审计尺度；仅诊断，不启动训练。"""

import argparse
import json
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.capture.provenance import file_digest
from app.depth_alignment import fit_inverse_depth, apply_inverse_alignment, alignment_errors


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('task')
    parser.add_argument('split_protocol')
    parser.add_argument('mask_dir')
    parser.add_argument('output')
    parser.add_argument('--reuse-depth', action='store_true', help='仅重审已有深度，仍核验输入摘要')
    args = parser.parse_args()
    from app.dataset import ColmapScene
    from app.pipeline import _run_mast3r
    from app.config import MAST3R_CKPT

    task, split_path, mask_dir, output = map(Path, (args.task, args.split_protocol, args.mask_dir, args.output))
    split = json.loads(split_path.read_text(encoding='utf-8'))
    scene = ColmapScene(str(task / 'dataset'), 1, sparse_depth=True, rigid_normalization=True)
    # 历史共同训练集缺少部分注册帧，必须从实际冻结名单等距取样，不能硬编码帧号。
    available = sorted(set(split['training_names']) & set(scene.names))
    available = [name for name in available if name.startswith('frame_')]
    if len(available) < 8:
        raise ValueError('冻结训练集的诊断视角不足 8 个')
    names = [available[i] for i in np.linspace(0, len(available) - 1, 8, dtype=int)]
    if not set(names) <= set(split['training_names']) or set(names) & set(split['validation_names']):
        raise ValueError('深度诊断只允许使用冻结的训练视角')
    inputs = {name: file_digest(task / 'dataset/images_1' / name) for name in names}
    masks = {name: file_digest(mask_dir / (name + '.png')) for name in names}
    protocol = {'image_sha256': inputs, 'mask_sha256': masks, 'names': names,
        'checkpoint_sha256': file_digest(MAST3R_CKPT), 'split_protocol_sha256': file_digest(split_path),
        'sparse_sha256': {p.name: file_digest(p) for p in (task / 'dataset/sparse/0').glob('*.bin')},
        'sources': {p: file_digest(Path(__file__).resolve().parents[1] / p) for p in
                    ('scripts/probe_depth_alignment.py', 'app/depth_alignment.py', 'app/pipeline.py', 'app/dataset.py')},
        'image_space': '实际去畸变训练图；MASt3R 输入与像素参考逐字节一致',
        'target_space': '刚性相机归一化后的相机轴向 Z；非米制承诺',
        'mask_scope': '修正后的自动掩码，侵蚀并限定中段；非人工真值',
        'geometry_scope': '原稀疏地图可能包含验证图，仅冻结颜色监督留出，不是独立几何测试',
        'thresholds': {'min_fit_anchors': 30, 'min_check_anchors': 10,
                       'check_median_relative_z': .10, 'check_p90_relative_z': .25}}
    if args.reuse_depth:
        saved = json.loads((output / 'protocol.json').read_text(encoding='utf-8'))
        for key in ('image_sha256', 'mask_sha256', 'names', 'checkpoint_sha256', 'split_protocol_sha256', 'sparse_sha256'):
            if saved[key] != protocol[key]:
                raise ValueError(f'诊断输入改变，不能复用深度：{key}')
    else:
        output.mkdir(parents=True, exist_ok=False)
        (output / 'images').mkdir()
        write_json(output / 'protocol.json', protocol)
        for name in names:
            shutil.copy2(task / 'dataset/images_1' / name, output / 'images' / name)
        _run_mast3r(str(output), names, 4, lambda stage, sub, progress, message: print(message, flush=True))
    depth_fingerprints = {name: file_digest(output / 'dataset/depths' / (Path(name).stem + '.npy')) for name in names}
    previous_report = output / 'alignment_report.json'
    if args.reuse_depth and previous_report.exists():
        previous = json.loads(previous_report.read_text(encoding='utf-8'))
        # 旧版无输出摘要只作迁移诊断；建立摘要后，重审拒绝被替换的缓存深度。
        if previous.get('depth_sha256') is not None and previous['depth_sha256'] != depth_fingerprints:
            raise ValueError('缓存深度已改变，请另建诊断目录')
    (output / 'aligned_diagnostic').mkdir(exist_ok=True)
    records, views = [], []
    for name in names:
        i = scene.names.index(name)
        depth = np.load(output / 'dataset/depths' / (Path(name).stem + '.npy'))
        h, w = depth.shape
        dim = scene.dimensions[i]
        K = scene.Ks[i].copy()
        K[0] *= w / dim['width']
        K[1] *= h / dim['height']
        mask = cv2.imread(str(mask_dir / (name + '.png')), cv2.IMREAD_GRAYSCALE)
        if mask is None or mask.shape != (dim['height'], dim['width']):
            raise ValueError(f'掩码与实际训练图不匹配：{name}')
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
        support = cv2.erode(mask, np.ones((15, 15), np.uint8)) > 127
        # 只检查罐身内部，避免将未复查盖沿、转台或反光纳入深度标定。
        ys = np.flatnonzero(support.any(axis=1))
        if len(ys):
            support[:int(ys[0] + .2 * (ys[-1] - ys[0]))] = False
            support[int(ys[0] + .8 * (ys[-1] - ys[0])):] = False
        xy, z = scene.sparse_observations[i]
        xy = xy * [w / dim['width'], h / dim['height']]
        pixel = np.rint(xy).astype(int)
        valid = (pixel[:, 0] >= 0) & (pixel[:, 0] < w) & (pixel[:, 1] >= 0) & (pixel[:, 1] < h)
        pixel, z = pixel[valid], z[valid]
        valid = support[pixel[:, 1], pixel[:, 0]]
        pixel, z = pixel[valid], z[valid]
        # 一个预测像素只保留一个观测，避免低分辨率取整使锚点数虚高。
        _, unique = np.unique(pixel[:, 1] * w + pixel[:, 0], return_index=True)
        pixel, z = pixel[unique], z[unique]
        predicted = depth[pixel[:, 1], pixel[:, 0]]
        valid = np.isfinite(predicted) & (predicted > 0)
        pixel, z, predicted = pixel[valid], z[valid], predicted[valid]
        # 固定空间块留出检验锚点，不使用同一批拟合点报告验收误差。
        check = ((pixel[:, 0] // 16 + pixel[:, 1] // 16) % 4 == 0)
        record = {'name': name, 'depth_shape': [h, w], 'anchor_count': len(z),
                  'fit_count': int((~check).sum()), 'check_count': int(check.sum()), 'passed': False}
        aligned = np.zeros_like(depth)
        try:
            if check.sum() < 10:
                raise ValueError('空间留出检查锚点不足')
            fit = fit_inverse_depth(predicted[~check], z[~check])
            aligned = apply_inverse_alignment(depth, fit)
            errors = alignment_errors(aligned[pixel[check, 1], pixel[check, 0]], z[check])
            # 物体很薄时整张常数深度也可能取得很小的相对 Z 误差，必须提供平面基线。
            constant = alignment_errors(np.full(check.sum(), np.median(z[~check])), z[check])
            record.update(fit=fit, check_errors=errors, constant_depth_check_errors=constant,
                          target_z_span_p10_p90=float(np.percentile(z, 90) - np.percentile(z, 10)))
            record['shape_better_than_constant'] = bool(errors['p90_relative_z'] is not None
                and errors['p90_relative_z'] < .8 * constant['p90_relative_z'])
            record['passed'] = bool(errors['valid_ratio'] == 1 and errors['median_relative_z'] <= .10
                                    and errors['p90_relative_z'] <= .25)
            if not record['passed']:
                record['reason'] = '空间留出锚点误差未达标'
        except ValueError as exc:
            record['reason'] = str(exc)
        np.savez_compressed(output / 'aligned_diagnostic' / (Path(name).stem + '.npz'), depth=aligned, support=support)
        views.append((aligned, support, K, scene.camtoworlds[i]))
        records.append(record)
    consistency = []
    for i in range(len(views)):
        source, support, K, c2w = views[i]
        yy, xx = np.mgrid[0:source.shape[0]:8, 0:source.shape[1]:8]
        valid = support[yy, xx] & (source[yy, xx] > 0)
        xx, yy = xx[valid], yy[valid]
        rays = np.c_[xx, yy, np.ones(len(xx))] @ np.linalg.inv(K).T
        world = (rays * source[yy, xx, None]) @ c2w[:3, :3].T + c2w[:3, 3]
        # 所有方向都记录，防止只挑选相似视角掩盖不一致；遮挡未显式求解，不能冒充置信度。
        for j, (target, target_support, target_K, target_c2w) in enumerate(views):
            if i == j:
                continue
            w2c = np.linalg.inv(target_c2w)
            camera = world @ w2c[:3, :3].T + w2c[:3, 3]
            projected = camera @ target_K.T
            positive = camera[:, 2] > 1e-6
            uv = projected[:, :2] / np.maximum(camera[:, 2:3], 1e-6)
            in_bounds = positive & (uv[:, 0] >= 0) & (uv[:, 0] < target.shape[1] - 1) & (uv[:, 1] >= 0) & (uv[:, 1] < target.shape[0] - 1)
            uv, camera, target_world = uv[in_bounds], camera[in_bounds], world[in_bounds]
            pixel = np.rint(uv).astype(int)
            good = target_support[pixel[:, 1], pixel[:, 0]]
            pixel, camera, target_world = pixel[good], camera[good], target_world[good]
            errors = alignment_errors(target[pixel[:, 1], pixel[:, 0]], camera[:, 2])
            source_ray = c2w[:3, 3] - target_world
            target_ray = target_c2w[:3, 3] - target_world
            cosine = (source_ray * target_ray).sum(axis=1) / np.maximum(
                np.linalg.norm(source_ray, axis=1) * np.linalg.norm(target_ray, axis=1), 1e-12)
            near = cosine >= np.cos(np.deg2rad(20.))
            # 小夹角单独报告，减少背面遮挡的混淆；仍不能把它当作显式可见性求解。
            near_errors = alignment_errors(target[pixel[near, 1], pixel[near, 0]], camera[near, 2])
            consistency.append({'source': names[i], 'target': names[j], **errors,
                                'source_samples': int(len(world)),
                                'near_angle_support_ratio': float(near.sum() / max(len(world), 1)),
                                'ray_angle_le20_errors': near_errors})
    report = {'frames': records, 'accepted_views': sum(row['passed'] for row in records),
              'depth_sha256': depth_fingerprints,
              'shape_better_views': sum(row.get('shape_better_than_constant', False) for row in records),
              'shape_check_scope': '新增常数深度基线诊断：留出锚点 P90 误差至少降低 20%，未作为训练启用条件',
              'audit_source_sha256': {p: file_digest(Path(__file__).resolve().parents[1] / p) for p in
                  ('scripts/probe_depth_alignment.py', 'app/depth_alignment.py')},
              'cross_view_diagnostic': consistency, 'training_enabled': False,
              'scope': '单视角尺度初筛，跨视角误差包含遮挡；未审阅前景且未生成训练可信区域，不可直接启用稠密损失'}
    write_json(output / 'alignment_report.json', report)
    # 共享色标仅用于数值查看；黑色表示不在本轮检查的罐身内部，不表示物体破损。
    positive = [depth[support & (depth > 0)] for depth, support, _, _ in views]
    values = np.concatenate(positive)
    if len(values):
        lo, hi = np.percentile(values, [1, 99])
        cards = []
        for name, record, (depth, support, _, _) in zip(names, records, views):
            scaled = np.clip((depth - lo) / max(hi - lo, 1e-8) * 255, 0, 255).astype(np.uint8)
            colored = cv2.applyColorMap(255 - scaled, cv2.COLORMAP_TURBO)
            colored[~support | (depth <= 0)] = 0
            rgb = cv2.imread(str(output / 'images' / name))
            card = np.zeros((235, 320, 3), dtype=np.uint8)
            card[35:, :160] = cv2.resize(rgb, (160, 200), interpolation=cv2.INTER_AREA)
            card[35:, 160:] = cv2.resize(colored, (160, 200), interpolation=cv2.INTER_NEAREST)
            label = name + (' aligned' if record['passed'] else ' rejected')
            cv2.putText(card, label, (5, 22), cv2.FONT_HERSHEY_SIMPLEX, .43, (255, 255, 255), 1, cv2.LINE_AA)
            cards.append(card)
        sheet = np.vstack([np.hstack(cards[i:i + 2]) for i in range(0, len(cards), 2)])
        cv2.imwrite(str(output / 'alignment-contact.png'), sheet)
    print(json.dumps({'accepted_views': report['accepted_views'], 'frames': records}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
