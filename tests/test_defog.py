"""验证去杂云实验的投影、尺度和光度补偿，不以点数替代质量。"""

import unittest
import numpy as np
import torch

from app.sparse_depth import project_depth, inverse_depth_loss
from app.bilateral_grid import BilateralGrid
from app.camera_geometry import normalize_camera_translations


class DefogTests(unittest.TestCase):
    def test_world_normalization_keeps_camera_rigid_and_depth_scaled(self):
        camera = np.eye(4)[None]
        camera[0, :3, 3] = [1, 2, 3]
        points = np.array([[2., 2., 5.]])
        xy, z = project_depth(points, camera[0], np.eye(3), 10, 10)
        center, scale = np.array([.3, .4, .5]), .2
        normalized = normalize_camera_translations(camera, center, scale)
        xy2, z2 = project_depth((points - center) * scale, normalized[0], np.eye(3), 10, 10)
        np.testing.assert_allclose(xy2, xy)
        np.testing.assert_allclose(z2, z * scale)
        np.testing.assert_allclose(normalized[0, :3, :3].T @ normalized[0, :3, :3], np.eye(3))
        np.testing.assert_allclose(camera[0, :3, 3], [1, 2, 3])
    def test_projection_uses_camera_z_and_rejects_behind_and_outside(self):
        points = np.array([[2., 0., 2.], [0, 0, -1], [30, 0, 1], [np.nan, 0, 1]])
        xy, depth = project_depth(points, np.eye(4), np.eye(3), 10, 10)
        np.testing.assert_allclose(xy, [[1, 0]])
        np.testing.assert_allclose(depth, [2])
        # 世界坐标和相机平移同倍缩放，投影不变、轴向深度同步缩放。
        xy2, depth2 = project_depth(points * 3, np.eye(4), np.eye(3), 10, 10)
        np.testing.assert_allclose(xy, xy2)
        np.testing.assert_allclose(depth2, depth * 3)

    def test_depth_loss_exact_sample_gradient_and_empty(self):
        rendered = torch.full((4, 4), 2., requires_grad=True)
        xy = torch.tensor([[1., 1.], [2., 2.]])
        self.assertEqual(inverse_depth_loss(rendered, xy, torch.tensor([2., 2.])).item(), 0)
        loss = inverse_depth_loss(rendered, xy, torch.tensor([4., 4.]))
        self.assertAlmostEqual(loss.item(), .25)
        loss.backward()
        self.assertGreater(rendered.grad.abs().sum(), 0)
        self.assertEqual(inverse_depth_loss(rendered, xy[:0], torch.empty(0)).item(), 0)

    def test_bilateral_identity_rng_and_view_isolation(self):
        rng = torch.get_rng_state().clone()
        module = BilateralGrid(3)
        torch.testing.assert_close(torch.get_rng_state(), rng)
        rgb = torch.rand(3, 7, 9, requires_grad=True)
        torch.testing.assert_close(module(rgb, 1), rgb)
        self.assertEqual(module.smoothness(1).item(), 0)
        with torch.no_grad():
            module.grid[1, [3, 7, 11]] = .1
        torch.testing.assert_close(module(rgb, 1), rgb + .1)
        torch.testing.assert_close(module(rgb, 0), rgb)
        module(rgb, 1).mean().backward()
        self.assertGreater(module.grid.grad[1].abs().sum(), 0)
        self.assertEqual(module.grid.grad[0].abs().sum(), 0)


if __name__ == "__main__":
    unittest.main()
