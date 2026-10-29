"""用 pycolmap 读取稀疏重建结果，构造 gsplat 训练数据集。

目录布局约定：
    dataset_dir/
        sparse/0/{cameras.bin, images.bin, points3D.bin}  # 已去畸变（PINHOLE）
        images_{factor}/*.png                              # 已按 factor 降采样
"""

import os
import hashlib
import io
from collections import OrderedDict

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from app.camera_geometry import camera_normalization


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
    def __init__(self, dataset_dir: str, factor: int, sparse_depth=False, rigid_normalization=False):
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

        names, camtoworlds, Ks, dimensions = [], [], [], []
        for img_id, img in sorted(rec.images.items()):
            cam = rec.cameras[img.camera_id]
            params = np.asarray(cam.params, dtype=np.float64)
            if params.shape[0] >= 4:   # PINHOLE / OPENCV
                fx, fy, cx, cy = params[0], params[1], params[2], params[3]
            else:                       # SIMPLE_PINHOLE
                fx = fy = params[0]
                cx, cy = params[1], params[2]
            # 使用实际像素尺寸换算内参，处理奇数尺寸取整后的横纵比例差异。
            with Image.open(os.path.join(self.image_dir, img.name)) as image:
                width, height = image.size
            sx, sy = width / cam.width, height / cam.height
            dimensions.append({"name": img.name, "width": width, "height": height})
            Ks.append([
                [fx * sx, 0, cx * sx],
                [0, fy * sy, cy * sy],
                [0, 0, 1],
            ])
            camtoworlds.append(np.linalg.inv(_image_w2c(img)))
            names.append(img.name)

        camtoworlds = np.stack(camtoworlds)

        # 场景归一化：相机中心平移到原点，缩放到单位球内
        center, scene_scale = camera_normalization(camtoworlds[:, :3, 3], dataset_dir)
        T = np.eye(4)
        T[:3, 3] = -center
        S = np.eye(4) * scene_scale
        S[3, 3] = 1.0
        transform = S @ T
        if rigid_normalization:
            from app.camera_geometry import normalize_camera_translations
            camtoworlds = normalize_camera_translations(camtoworlds, center, scene_scale)
        else:
            # 历史模型保留原坐标约定，新深度实验显式使用刚性相机归一化。
            camtoworlds = transform[None] @ camtoworlds

        pts, cols = [], []
        for _, p in rec.points3D.items():
            pts.append(p.xyz)
            cols.append(p.color)
        points = np.asarray(pts, dtype=np.float64)
        points_rgb = np.asarray(cols, dtype=np.float32) / 255.0
        points = (transform[:3, :3] @ points.T).T + transform[:3, 3]

        self.names = names
        self.dimensions = dimensions
        self.camtoworlds = camtoworlds.astype(np.float32)
        self.Ks = np.asarray(Ks, dtype=np.float32)
        self.points = points.astype(np.float32)
        self.points_rgb = points_rgb.astype(np.float32)
        self.scene_scale = 1.0  # 已归一化到单位尺度
        self.n_images = len(names)
        self.sparse_observations = None
        if sparse_depth:
            from app.sparse_depth import project_depth
            self.sparse_observations = []
            # 只使用实际观测轨迹，避免把被遮挡点投影为监督；阈值固定并在报告披露。
            for index, (_, image) in enumerate(sorted(rec.images.items())):
                reliable = []
                for observation in image.points2D:
                    if not observation.has_point3D():
                        continue
                    point = rec.points3D[observation.point3D_id]
                    if point.track.length() >= 3 and np.isfinite(point.error) and point.error <= 2.0:
                        reliable.append(point.xyz)
                world = np.asarray(reliable, dtype=np.float64).reshape(-1, 3)
                world = world @ transform[:3, :3].T + transform[:3, 3]
                size = dimensions[index]
                self.sparse_observations.append(project_depth(world, self.camtoworlds[index],
                    self.Ks[index], size["width"], size["height"]))


class ColmapDataset(Dataset):
    def __init__(self, scene: ColmapScene, cache_bytes: int = 64 * 1024**2,
                 uint8_images=False, mask_dir=None, allow_unreviewed_masks=False, foreground_only=False):
        self.scene = scene
        self.uint8_images = uint8_images
        self.mask_dir = os.path.abspath(mask_dir) if mask_dir else None
        self.foreground_only = foreground_only
        if foreground_only and not self.mask_dir:
            raise ValueError("前景独立训练必须提供完整主体掩码")
        if self.mask_dir and not os.path.isdir(self.mask_dir):
            raise ValueError(f"主体掩码目录不存在：{self.mask_dir}")
        self.mask_paths, self.mask_report = {}, None
        if self.mask_dir:
            from app.subject_masks import validate_mask_set
            self.mask_paths, self.mask_report = validate_mask_set(scene, self.mask_dir, allow_unreviewed_masks)
        self.mask_fingerprints = {record["name"]: record["mask_sha256"] for record in
                                  (self.mask_report["frames"] if self.mask_report else [])}
        # 有界缓存减少随机训练反复解码，避免高分辨率数据全部常驻内存。
        self.cache_bytes = cache_bytes
        self._cache = OrderedDict()
        self._cached_bytes = 0

    def __len__(self):
        return self.scene.n_images

    def __getitem__(self, idx):
        if idx in self._cache:
            self._cache.move_to_end(idx)
            return self._cache[idx]
        path = os.path.join(self.scene.image_dir, self.scene.names[idx])
        with Image.open(path) as img:
            if self.uint8_images:
                # 训练缓存保留八位像素，内存占用为浮点图的 1/4，归一化交给显卡。
                arr = torch.from_numpy(np.array(img.convert("RGB"), dtype=np.uint8)).permute(2, 0, 1).contiguous()
            else:
                arr = torch.from_numpy(np.array(img.convert("RGB"), dtype=np.float32) / 255.0).permute(2, 0, 1)

        data = {
            "image": arr,                                            # [3,H,W]
            "camtoworld": torch.from_numpy(self.scene.camtoworlds[idx]),
            "K": torch.from_numpy(self.scene.Ks[idx]),
            "image_id": idx,
        }
        if getattr(self.scene, "sparse_observations", None) is not None:
            xy, depths = self.scene.sparse_observations[idx]
            data["sparse_xy"] = torch.from_numpy(xy)
            data["sparse_depth"] = torch.from_numpy(depths)

        if self.mask_dir:
            # 掩码只作为可选实验输入；缺少任一帧时直接失败，避免静默退回无掩码训练。
            mask_path = self.mask_paths[self.scene.names[idx]]
            with open(mask_path, "rb") as mask_source:
                mask_content = mask_source.read()
            if hashlib.sha256(mask_content).hexdigest() != self.mask_fingerprints[self.scene.names[idx]]:
                raise ValueError(f"训练开始后主体掩码发生变化：{self.scene.names[idx]}")
            with Image.open(io.BytesIO(mask_content)) as mask_image:
                mask = np.array(mask_image.convert("L"), dtype=np.uint8)
            if mask.shape != (self.scene.dimensions[idx]["height"], self.scene.dimensions[idx]["width"]):
                raise ValueError(f"主体掩码尺寸不匹配：{self.scene.names[idx]}")
            data["mask"] = torch.from_numpy(mask).contiguous()
            if self.foreground_only:
                # 仅修改内存中的训练参考，保留原始文件与摘要；阶段监控也使用同一参考约定。
                data["image"].masked_fill_(data["mask"][None] == 0, 0)

        # 加载深度图（如果存在）
        if self.scene.has_depth:
            name = self.scene.names[idx]
            depth_path = os.path.join(self.scene.depth_dir, os.path.splitext(name)[0] + ".npy")
            if os.path.exists(depth_path):
                depth = np.load(depth_path)
                data["depth"] = torch.from_numpy(depth)  # [H,W]

        byte_size = sum(value.numel() * value.element_size() for value in data.values() if torch.is_tensor(value))
        if byte_size <= self.cache_bytes:
            while self._cache and self._cached_bytes + byte_size > self.cache_bytes:
                _, previous = self._cache.popitem(last=False)
                self._cached_bytes -= sum(value.numel() * value.element_size() for value in previous.values()
                                          if torch.is_tensor(value))
            self._cache[idx] = data
            self._cached_bytes += byte_size
        return data
