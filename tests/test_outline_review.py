"""候选目视检查记录不能覆盖缺失帧、变更后的轮廓或假冒人工确认。"""

import json
import tempfile
import unittest
from pathlib import Path

from app.capture.provenance import file_digest
from app.foreground.review import validate_outline_review


class OutlineReviewTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / 'images').mkdir()
        (self.root / 'feature_masks').mkdir()
        (self.root / 'images/frame_0001.png').write_bytes(b'image')
        (self.root / 'feature_masks/mask.png').write_bytes(b'mask')
        entry = {'mask': 'mask.png', 'image_sha256': file_digest(self.root / 'images/frame_0001.png'),
                 'mask_sha256': file_digest(self.root / 'feature_masks/mask.png')}
        self.manifest = self.root / 'feature_masks/mask_manifest.json'
        self.manifest.write_text(json.dumps({'frames': {'frame_0001.png': entry}}), encoding='utf-8')
        self.review = {'kind': 'assistant_contact_sheet_visual_check', 'human_reviewed': False,
            'decision': 'accepted_for_short_experiment', 'manifest_sha256': file_digest(self.manifest),
            'frames': {'frame_0001.png': entry}}
        self.path = self.root / 'review.json'
        self.save()

    def save(self):
        self.path.write_text(json.dumps(self.review), encoding='utf-8')

    def test_accept_short_experiment_without_human_truth(self):
        report = validate_outline_review(self.root, self.path)
        self.assertEqual(report['frames'], 1)
        self.assertFalse(report['human_reviewed'])

    def test_missing_frame_rejected(self):
        self.review['frames'] = {}
        self.save()
        with self.assertRaisesRegex(ValueError, '全部视频视角'):
            validate_outline_review(self.root, self.path)

    def test_modified_mask_rejected(self):
        (self.root / 'feature_masks/mask.png').write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError, '发生变化'):
            validate_outline_review(self.root, self.path)

    def test_human_truth_cannot_be_claimed(self):
        self.review['human_reviewed'] = True
        self.save()
        with self.assertRaisesRegex(ValueError, '人工确认'):
            validate_outline_review(self.root, self.path)


if __name__ == '__main__':
    unittest.main()
