"""轮廓候选的目视检查记录；允许短实验，不冒充人工逐像素真值。"""

import json
from pathlib import Path

from app.capture.provenance import file_digest


def validate_outline_review(source, review_file):
    """候选变化后必须重新检查，不能以中段颜色覆盖通过替代顶底轮廓检查。"""
    root, path = Path(source), Path(review_file)
    manifest_path = root / 'feature_masks/mask_manifest.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    review = json.loads(path.read_text(encoding='utf-8'))
    if (review.get('kind') != 'assistant_contact_sheet_visual_check'
            or review.get('decision') != 'accepted_for_short_experiment'
            or review.get('human_reviewed') is not False):
        raise ValueError('轮廓记录只能授权自动候选的短实验，不能标为人工确认')
    if review.get('manifest_sha256') != file_digest(manifest_path):
        raise ValueError('轮廓候选清单已改变，需要重新检查')
    names = {name for name in manifest['frames'] if name.startswith('frame_')}
    if not names or set(review.get('frames', {})) != names:
        raise ValueError('目视检查必须覆盖全部视频视角')
    for name in names:
        item, entry = review['frames'][name], manifest['frames'][name]
        image_sha = file_digest(root / 'images' / name)
        mask_sha = file_digest(root / 'feature_masks' / entry['mask'])
        if (item.get('image_sha256') != image_sha or item.get('mask_sha256') != mask_sha
                or entry.get('image_sha256') != image_sha or entry.get('mask_sha256') != mask_sha):
            raise ValueError('目视检查后的原图或轮廓候选发生变化')
    return {'review_sha256': file_digest(path), 'frames': len(names), 'kind': review['kind'],
            'human_reviewed': False, 'scope': review.get('scope'), 'limitations': review.get('limitations')}
