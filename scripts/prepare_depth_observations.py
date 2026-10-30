"""用小夹角、深度与双向像素检查构造稀疏/稠密来源的同区域实验观测。"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.capture.provenance import file_digest
from app.depth_alignment import apply_inverse_alignment
from app.depth_observations import scene_fingerprint


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('task', 'diagnostic', 'mask_dir', 'output'):
        parser.add_argument(key)
    args = parser.parse_args()
    from app.dataset import ColmapScene
    task, diagnostic, masks, output = map(Path, (args.task, args.diagnostic, args.mask_dir, args.output))
    protocol = json.loads((diagnostic / 'protocol.json').read_text(encoding='utf-8'))
    report = json.loads((diagnostic / 'alignment_report.json').read_text(encoding='utf-8'))
    scene = ColmapScene(str(task / 'dataset'), 1, sparse_depth=True, rigid_normalization=True)
    actual_sparse = {p.name: file_digest(p) for p in (task / 'dataset/sparse/0').glob('*.bin')}
    if protocol['sparse_sha256'] != actual_sparse:
        raise ValueError('诊断稀疏地图发生变化')
    output.mkdir(parents=True, exist_ok=False)
    for kind in ('sparse', 'dense', 'trust_masks'):
        (output / kind).mkdir()
    names, views = [], []
    for row in report['frames']:
        if not row['passed'] or not row.get('shape_better_than_constant'):
            continue
        name = row['name']
        if (protocol['image_sha256'][name] != file_digest(task / 'dataset/images_1' / name)
                or protocol['mask_sha256'][name] != file_digest(masks / (name + '.png'))):
            raise ValueError('诊断图像或掩码发生变化')
        raw = diagnostic / 'dataset/depths' / (Path(name).stem + '.npy')
        if report['depth_sha256'][name] != file_digest(raw):
            raise ValueError('原始预测深度发生变化')
        depth = apply_inverse_alignment(np.load(raw), row['fit'])
        h, w = depth.shape
        i = scene.names.index(name)
        dim = scene.dimensions[i]
        full_mask = cv2.imread(str(masks / (name + '.png')), cv2.IMREAD_GRAYSCALE)
        if full_mask is None or full_mask.shape != (dim['height'], dim['width']):
            raise ValueError('掩码与实际图像像素空间不一致')
        support = cv2.erode(cv2.resize(full_mask, (w, h), interpolation=cv2.INTER_NEAREST),
                            np.ones((15, 15), np.uint8)) > 127
        yy = np.flatnonzero(support.any(axis=1))
        if len(yy):
            support[:int(yy[0] + .2 * (yy[-1] - yy[0]))] = False
            support[int(yy[0] + .8 * (yy[-1] - yy[0])):] = False
        support &= np.isfinite(depth) & (depth > 0)
        K = scene.Ks[i].copy()
        K[0] *= w / dim['width']
        K[1] *= h / dim['height']
        names.append(name)
        views.append((depth, support, K, scene.camtoworlds[i]))
    entries = {'sparse': {}, 'dense': {}}
    coverage = []
    for name, (depth, support, K, c2w) in zip(names, views):
        y, x = np.nonzero(support)
        world = (np.c_[x, y, np.ones(len(x))] @ np.linalg.inv(K).T * depth[y, x, None]) @ c2w[:3, :3].T + c2w[:3, 3]
        verified = np.zeros(len(x), dtype=np.int16)
        for target_name, (target, target_support, tK, tc2w) in zip(names, views):
            if target_name == name:
                continue
            w2c = np.linalg.inv(tc2w)
            cam = world @ w2c[:3, :3].T + w2c[:3, 3]
            a, b = c2w[:3, 3] - world, tc2w[:3, 3] - world
            cosine = (a * b).sum(axis=1) / np.maximum(np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1), 1e-12)
            projected = cam @ tK.T
            uv = projected[:, :2] / np.maximum(cam[:, 2:3], 1e-8)
            eligible = ((cam[:, 2] > 0) & (cosine >= np.cos(np.deg2rad(20))) & (uv[:, 0] >= 0)
                        & (uv[:, 0] < target.shape[1] - 1) & (uv[:, 1] >= 0) & (uv[:, 1] < target.shape[0] - 1))
            ids = np.flatnonzero(eligible)
            px = np.rint(uv[ids]).astype(int)
            tz = target[px[:, 1], px[:, 0]]
            good = target_support[px[:, 1], px[:, 0]] & (tz > 0)
            good &= np.abs(tz - cam[ids, 2]) / cam[ids, 2] <= .01
            # 用目标预测反投影回源图，防止只比较轴向距离而忽略横向漂移。
            back_world = (np.c_[px, np.ones(len(px))] @ np.linalg.inv(tK).T * tz[:, None]) @ tc2w[:3, :3].T + tc2w[:3, 3]
            source_w2c = np.linalg.inv(c2w)
            back_cam = back_world @ source_w2c[:3, :3].T + source_w2c[:3, 3]
            back = back_cam @ K.T
            back_xy = back[:, :2] / np.maximum(back_cam[:, 2:3], 1e-8)
            good &= (back_cam[:, 2] > 0) & (np.linalg.norm(back_xy - np.c_[x[ids], y[ids]], axis=1) <= 1.5)
            verified[ids[good]] += 1
        trusted = np.zeros(support.shape, dtype=np.uint8)
        trusted[y[verified > 0], x[verified > 0]] = 255
        cv2.imwrite(str(output / 'trust_masks' / (name + '.png')), trusted)
        count = int((verified > 0).sum())
        ratio = count / max(len(x), 1)
        record = {'name': name, 'support_pixels': len(x), 'trusted_pixels': count, 'coverage': ratio, 'used': False}
        coverage.append(record)
        if count < 300 or ratio < .30:
            continue
        i = scene.names.index(name)
        dim = scene.dimensions[i]
        sy, sx = np.nonzero(trusted)
        dense_xy = np.c_[sx * dim['width'] / depth.shape[1], sy * dim['height'] / depth.shape[0]]
        dense_z = depth[sy, sx]
        sparse_xy, sparse_z = scene.sparse_observations[i]
        small_xy = np.rint(sparse_xy * [depth.shape[1] / dim['width'], depth.shape[0] / dim['height']]).astype(int)
        inside = ((small_xy >= 0).all(axis=1) & (small_xy[:, 0] < depth.shape[1]) & (small_xy[:, 1] < depth.shape[0]))
        good = np.zeros(len(sparse_z), dtype=bool)
        good[inside] = trusted[small_xy[inside, 1], small_xy[inside, 0]] > 0
        sparse_xy, sparse_z = sparse_xy[good], sparse_z[good]
        if len(sparse_z) < 30:
            record['reason'] = '同区域稀疏观测不足，不加入任何一组'
            continue
        record.update(used=True, sparse_anchors=len(sparse_z))
        for kind, xy, z in (('dense', dense_xy, dense_z), ('sparse', sparse_xy, sparse_z)):
            # 固定均匀取样，上限控制显存；不消耗训练随机序列。
            sample = np.linspace(0, len(z) - 1, min(4096, len(z)), dtype=int)
            path = output / kind / (Path(name).stem + '.npz')
            np.savez_compressed(path, xy=xy[sample].astype(np.float32), depth=z[sample].astype(np.float32))
            entries[kind][name] = {'file': path.name, 'sha256': file_digest(path),
                                   'image_sha256': protocol['image_sha256'][name],
                                   'mask_sha256': protocol['mask_sha256'][name]}
    if not entries['dense']:
        raise ValueError('没有通过双向检查与同区域稀疏检查的视角')
    scope = '自动掩码中段、至少一个辅助视角一致；非实测深度、非原生置信度，仍可能存在共同预测偏差'
    for kind in entries:
        write_json(output / kind / 'manifest.json', {'kind': kind, 'scene': scene_fingerprint(scene),
            'depth_kind': 'normalized_camera_z', 'frames': entries[kind], 'scope': scope,
            'diagnostic_report_sha256': file_digest(diagnostic / 'alignment_report.json'),
            'generator_sha256': file_digest(__file__)})
    summary = {'frames': coverage, 'supervised_names': list(entries['dense']), 'scope': scope,
        'thresholds': {'max_ray_angle_degrees': 20, 'max_relative_z_error': .01,
            'max_roundtrip_pixels': 1.5, 'min_coverage': .30, 'min_pixels': 300, 'max_observations_per_view': 4096}}
    write_json(output / 'coverage.json', summary)
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
