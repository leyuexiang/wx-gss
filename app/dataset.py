"""用 pycolmap 读取稀疏重建结果，构造 gsplat 训练数据集。

目录布局约定：
    dataset_dir/
        sparse/0/{cameras.bin, images.bin, points3D.bin}  # 已去畸变（PINHOLE）
        images_{factor}/*.png                              # 已按 factor 降采样
"""

import os

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


def _quat_to_rotmat(qvec) -> np.ndarray:
    w, x, y, z = qvec
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def _image_w2c(img) -> np.ndarray:
    """兼容新旧 pycolmap API，返回 world→camera 4x4。

    pycolmap 4.x 里 cam_from_world 是方法（Rigid3d 需调用后取），
    旧版是属性，两种都兼容。
    """
    cfw = getattr(img, "cam_from_world", None)
    if cfw is not None:
        m = cfw() if callable(cfw) else cfw
        m = np.asarray(m.matrix(), dtype=np.float64)
        if m.shape == (3, 4):                       # pycolmap 4.x 返回 3x4
            m = np.vstack([m, [0, 0, 0, 1]])
        return m
    w2c = np.eye(4)
    w2c[:3, :3] = _quat_to_rotmat(np.asarray(img.qvec, dtype=np.float64))
    w2c[:3, 3] = np.asarray(img.tvec, dtype=np.float64)
    return w2c


class ColmapScene:
    def __init__(self, dataset_dir: str, factor: int):
        import pycolmap

        rec = pycolmap.Reconstruction(os.path.join(dataset_dir, "sparse", "0"))
        if len(rec.images) == 0:
            raise RuntimeError("COLMAP 模型为空")

        self.image_dir = os.path.join(dataset_dir, f"images_{factor}")
        if not os.path.isdir(self.image_dir):
            raise RuntimeError(f"缺少 {self.image_dir}")

        # 检测深度图目录（MASt3R 后端会生成）
        self.depth_dir = os.path.join(dataset_dir, "depths")
        self.has_depth = os.path.isdir(self.depth_dir)

        names, camtoworlds, Ks = [], [], []
        for img_id, img in sorted(rec.images.items()):
            cam = rec.cameras[img.camera_id]
            params = np.asarray(cam.params, dtype=np.float64)
            if params.shape[0] >= 4:   # PINHOLE / OPENCV
                fx, fy, cx, cy = params[0], params[1], params[2], params[3]
            else:                       # SIMPLE_PINHOLE
                fx = fy = params[0]
                cx, cy = params[1], params[2]
            scale = 1.0 / factor
            Ks.append([
                [fx * scale, 0, cx * scale],
                [0, fy * scale, cy * scale],
                [0, 0, 1],
            ])
            camtoworlds.append(np.linalg.inv(_image_w2c(img)))
            names.append(img.name)

        camtoworlds = np.stack(camtoworlds)

        # 场景归一化：相机中心平移到原点，缩放到单位球内
        center = camtoworlds[:, :3, 3].mean(axis=0)
        max_dist = np.linalg.norm(camtoworlds[:, :3, 3] - center, axis=-1).max()
        scene_scale = 1.0 / (max_dist * 1.05) if max_dist > 1e-6 else 1.0
        T = np.eye(4)
        T[:3, 3] = -center
        S = np.eye(4) * scene_scale
        S[3, 3] = 1.0
        transform = S @ T
        camtoworlds = transform[None] @ camtoworlds

        pts, cols = [], []
        for _, p in rec.points3D.items():
            pts.append(p.xyz)
            cols.append(p.color)
        points = np.asarray(pts, dtype=np.float64)
        points_rgb = np.asarray(cols, dtype=np.float32) / 255.0
        points = (transform[:3, :3] @ points.T).T + transform[:3, 3]

        self.names = names
        self.camtoworlds = camtoworlds.astype(np.float32)
        self.Ks = np.asarray(Ks, dtype=np.float32)
        self.points = points.astype(np.float32)
        self.points_rgb = points_rgb.astype(np.float32)
        self.scene_scale = 1.0  # 已归一化到单位尺度
        self.n_images = len(names)


class ColmapDataset(Dataset):
    def __init__(self, scene: ColmapScene):
        self.scene = scene

    def __len__(self):
        return self.scene.n_images

    def __getitem__(self, idx):
        path = os.path.join(self.scene.image_dir, self.scene.names[idx])
        img = Image.open(path).convert("RGB")
        arr = torch.from_numpy(np.asarray(img, dtype=np.float32) / 255.0).permute(2, 0, 1)

        data = {
            "image": arr,                                            # [3,H,W]
            "camtoworld": torch.from_numpy(self.scene.camtoworlds[idx]),
            "K": torch.from_numpy(self.scene.Ks[idx]),
            "image_id": idx,
        }

        # 加载深度图（如果存在）
        if self.scene.has_depth:
            name = self.scene.names[idx]
            depth_path = os.path.join(self.scene.depth_dir, os.path.splitext(name)[0] + ".npy")
            if os.path.exists(depth_path):
                depth = np.load(depth_path)
                data["depth"] = torch.from_numpy(depth)  # [H,W]

        return data
