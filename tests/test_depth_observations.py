"""外部深度必须拒绝验证泄漏、坐标错配和缓存篡改，并保持有效梯度。"""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from app.capture.provenance import file_digest
from app.depth_observations import load_depth_observations, scene_fingerprint
from app.sparse_depth import inverse_depth_loss


class DepthObservationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / 'a.png').write_bytes(b'image')
        self.scene = SimpleNamespace(names=['a.png'], dimensions=[{'name': 'a.png', 'width': 8, 'height': 8}],
            camtoworlds=np.eye(4)[None], Ks=np.eye(3)[None], points=np.zeros((3, 3)), image_dir=str(self.root))
        self.dataset = SimpleNamespace(mask_fingerprints={'a.png': 'mask-hash'})
        xy = np.c_[np.arange(30) % 6, np.arange(30) // 6].astype(np.float32)
        np.savez(self.root / 'a.npz', xy=xy, depth=np.ones(30, dtype=np.float32))
        self.manifest = {'depth_kind': 'normalized_camera_z', 'scene': scene_fingerprint(self.scene), 'frames': {
            'a.png': {'file': 'a.npz', 'sha256': file_digest(self.root / 'a.npz'),
                      'image_sha256': file_digest(self.root / 'a.png'), 'mask_sha256': 'mask-hash'}}}
        self.save()

    def save(self):
        (self.root / 'manifest.json').write_text(json.dumps(self.manifest), encoding='utf-8')

    def load(self, indices=None):
        return load_depth_observations(self.root, self.scene, [0] if indices is None else indices, self.dataset)

    def test_valid_input_and_finite_depth_gradient(self):
        observations, report = self.load()
        self.assertEqual(report['counts']['a.png'], 30)
        xy, z = observations[0]
        rendered = torch.full((8, 8), 2., requires_grad=True)
        loss = inverse_depth_loss(rendered, torch.from_numpy(xy), torch.from_numpy(z))
        loss.backward()
        self.assertTrue(torch.isfinite(rendered.grad).all())
        self.assertGreater(rendered.grad.abs().sum().item(), 0)

    def test_validation_view_rejected(self):
        with self.assertRaisesRegex(ValueError, '冻结训练视角'):
            self.load([])

    def test_camera_or_mask_change_rejected(self):
        self.scene.camtoworlds[0, 0, 3] = 1
        with self.assertRaisesRegex(ValueError, '相机不匹配'):
            self.load()
        self.scene.camtoworlds[0, 0, 3] = 0
        self.dataset.mask_fingerprints['a.png'] = 'changed'
        with self.assertRaisesRegex(ValueError, '掩码已改变'):
            self.load()

    def test_file_tamper_rejected(self):
        (self.root / 'a.npz').write_bytes(b'corrupt')
        with self.assertRaisesRegex(ValueError, '摘要不匹配'):
            self.load()

    def test_invalid_values_and_bounds_rejected(self):
        for xy, z in ((np.zeros((30, 2)), np.full(30, np.nan)),
                      (np.full((30, 2), 9.), np.ones(30)),
                      (np.zeros((30, 2)), np.array(1.))):
            with self.subTest(shape=z.shape):
                np.savez(self.root / 'a.npz', xy=xy, depth=z)
                self.manifest['frames']['a.png']['sha256'] = file_digest(self.root / 'a.npz')
                self.save()
                with self.assertRaisesRegex(ValueError, '数值无效'):
                    self.load()


if __name__ == '__main__':
    unittest.main()
