"""加载经过一致性筛选的可选深度观测，严格绑定训练图、掩码和相机坐标。"""

import json
from pathlib import Path

import numpy as np

from app.capture.provenance import file_digest
from app.training_audit import array_digest


def scene_fingerprint(scene):
    """归一化旋转、平移、内参和初始化点均参与绑定，禁止跨地图复用尺度。"""
    return {'names': scene.names, 'dimensions': scene.dimensions,
            'cameras': array_digest(scene.camtoworlds), 'intrinsics': array_digest(scene.Ks),
            'points': array_digest(scene.points)}


def load_depth_observations(directory, scene, train_indices, dataset):
    """一次性校验并载入小规模采样点；未列出的视角明确不使用深度监督。"""
    root = Path(directory)
    path = root / 'manifest.json'
    manifest = json.loads(path.read_text(encoding='utf-8'))
    if manifest.get('depth_kind') != 'normalized_camera_z' or manifest.get('scene') != scene_fingerprint(scene):
        raise ValueError('外部深度与当前归一化相机不匹配')
    entries = manifest.get('frames', {})
    allowed = {scene.names[i] for i in train_indices}
    if not entries or not set(entries) <= allowed:
        raise ValueError('外部深度只能监督冻结训练视角，且至少需要一个视角')
    result = {}
    for name, entry in entries.items():
        index = scene.names.index(name)
        if (entry['image_sha256'] != file_digest(Path(scene.image_dir) / name)
                or entry['mask_sha256'] != dataset.mask_fingerprints.get(name)):
            raise ValueError('外部深度绑定的图像或前景掩码已改变')
        filename = entry['file']
        if Path(filename).name != filename or '/' in filename or '\\' in filename:
            raise ValueError('深度观测文件必须位于观测目录内')
        if file_digest(root / filename) != entry['sha256']:
            raise ValueError('深度观测文件摘要不匹配')
        with np.load(root / filename, allow_pickle=False) as arrays:
            xy, depth = arrays['xy'].astype(np.float32), arrays['depth'].astype(np.float32)
        dim = scene.dimensions[index]
        if (depth.ndim != 1 or xy.ndim != 2 or xy.shape != (len(depth), 2) or not 30 <= len(depth) <= 4096
                or not np.isfinite(xy).all() or not np.isfinite(depth).all() or (depth <= 0).any()
                or (xy < 0).any() or (xy[:, 0] > dim['width'] - 1).any()
                or (xy[:, 1] > dim['height'] - 1).any()):
            raise ValueError('深度观测的数量、坐标或数值无效')
        result[index] = (xy, depth)
    return result, {'manifest_sha256': file_digest(path), 'kind': manifest.get('kind'),
                    'counts': {scene.names[i]: len(v[1]) for i, v in result.items()},
                    'depth_kind': manifest['depth_kind'], 'scope': manifest.get('scope')}
