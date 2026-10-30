"""固定相机、掩码和抽样，对照无深度、同区域稀疏深度与可信区域预测深度。"""

import argparse
import copy
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.capture.provenance import file_digest


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('task', 'baseline_task', 'mask_pair', 'observations', 'output'):
        parser.add_argument(key)
    parser.add_argument('--iterations', type=int, default=1000)
    args = parser.parse_args()
    if not 600 <= args.iterations <= 7000:
        parser.error('对照训练次数必须在 600–7000 之间')
    from app.trainer import train
    task, baseline, mask_pair, observations, output = map(Path,
        (args.task, args.baseline_task, args.mask_pair, args.observations, args.output))
    protocol = json.loads((mask_pair / 'protocol.json').read_text(encoding='utf-8'))
    old_config = json.loads((baseline / 'training_config.json').read_text(encoding='utf-8'))
    effective = json.loads((baseline / 'output/effective_config.json').read_text(encoding='utf-8'))
    manifests = {kind: json.loads((observations / kind / 'manifest.json').read_text(encoding='utf-8'))
                 for kind in ('sparse', 'dense')}
    if set(manifests['sparse']['frames']) != set(manifests['dense']['frames']):
        raise ValueError('稀疏和稠密观测必须监督相同视角')
    output.mkdir(parents=True, exist_ok=False)
    options = copy.deepcopy(old_config['training'])
    options.update(training_names=protocol['training_names'], validation_names=protocol['validation_names'],
        validation_every=0, foreground_only=True, rigid_normalization=True, allow_unreviewed_masks=True,
        refine_stop_ratio=.8, audit_trace=True, checkpoint_every=0, mask_dir=str(mask_pair / 'repaired_masks'),
        sparse_depth=False, bilateral_grid=False, diagnostic_depth_render=True)
    options.pop('depth_observation_dir', None)
    source_root = Path(__file__).resolve().parents[1]
    source_files = ['app/trainer.py', 'app/dataset.py', 'app/depth_observations.py', 'app/sparse_depth.py',
                    'scripts/compare_depth_observations.py', 'scripts/prepare_depth_observations.py']
    write_json(output / 'protocol.json', {'iterations': args.iterations, 'seed': options.get('seed', 42),
        'training_options': options, 'depth_weight': .005,
        'observation_manifests': {kind: file_digest(observations / kind / 'manifest.json') for kind in manifests},
        'mask_manifest_sha256': file_digest(mask_pair / 'repaired_masks/mask_manifest.json'),
        'source_sha256': {name: file_digest(source_root / name) for name in source_files},
        'scope': '所有组固定 RGB+ED 渲染；深度只监督相同四视角可信中段，采样点密度和深度来源不同；自动掩码参考、1000 步流程对照不证明长训练改善'})
    records, audits = [], []
    for kind in ('none', 'sparse', 'dense'):
        destination = output / kind
        destination.mkdir()
        actual = dict(options)
        if kind != 'none':
            actual.update(depth_observation_dir=str(observations / kind), depth_observation_weight=.005)
        print('START', kind, flush=True)
        stats = train(str(task / 'dataset'), str(destination), args.iterations, 1, effective['cap_max'],
            lambda step, total, message: print(kind, message, flush=True), auto_cleanup=True,
            cleanup_threshold=.005, denoise_config=old_config['denoise'], training_options=actual)
        audit = json.loads((destination / 'reproduction_audit/audit.json').read_text(encoding='utf-8'))
        audits.append(audit)
        usage_path = destination / 'depth_observation_usage.json'
        usage = json.loads(usage_path.read_text(encoding='utf-8')) if usage_path.exists() else None
        record = {'kind': kind, 'points': stats['num_gaussians'], 'seconds': stats['train_seconds'],
                  'full_validation': stats['validation'], 'foreground_validation': stats['foreground_validation'],
                  'depth_usage': usage}
        records.append(record)
        write_json(output / 'comparison.json', records)
        print(json.dumps(record), flush=True)
    checks = {key: all(audit[key] == audits[0][key] for audit in audits[1:]) for key in
              ('initialization', 'image_sha256', 'sampling', 'completed_samples')}
    # 输入一致性只证明可比性，不保证显卡反向计算逐位一致，也不代替质量验收。
    paired = {'passed': all(checks.values()), 'checks': checks}
    write_json(output / 'pair_verification.json', paired)
    if not paired['passed']:
        raise ValueError('三组初始化、图像或抽样审计不一致')
    if any(file_digest(source_root / name) != sha for name, sha in
           json.loads((output / 'protocol.json').read_text(encoding='utf-8'))['source_sha256'].items()):
        raise ValueError('对照训练期间活动源码发生变化')
    print(json.dumps(paired), flush=True)


if __name__ == '__main__':
    main()
