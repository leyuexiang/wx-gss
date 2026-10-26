"""可学习的相机位姿模块，用于位姿联合优化。

使用 6D 连续旋转表示 + 平移向量，避免四元数/轴角的奇异性问题。
参考: On the Continuity of Rotation Representations (CVPR 2019)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CameraPose(nn.Module):
    """可学习的相机位姿 (SE(3))，支持梯度优化。

    使用 6D 连续旋转表示，相比四元数/轴角更稳定。
    """

    def __init__(self, camtoworlds: torch.Tensor):
        """初始化相机位姿参数。

        Args:
            camtoworlds: [N, 4, 4] 初始 camera-to-world 变换矩阵
        """
        super().__init__()
        assert camtoworlds.shape[1:] == (4, 4), "Expected [N, 4, 4] camtoworlds"

        N = camtoworlds.shape[0]
        R = camtoworlds[:, :3, :3]  # [N, 3, 3] 旋转
        t = camtoworlds[:, :3, 3]   # [N, 3] 平移

        # 6D 旋转表示: 取旋转矩阵的前两列 (6 个值)
        rot_6d = self._rotmat_to_6d(R)
        self.rot_6d = nn.Parameter(rot_6d)  # [N, 6]
        self.trans = nn.Parameter(t)         # [N, 3]

        # 保存初始值（用于正则化）
        self.register_buffer('rot_6d_init', self.rot_6d.data.clone())
        self.register_buffer('trans_init', self.trans.data.clone())
        self.register_buffer('n_cameras', torch.tensor(N))

    def _rotmat_to_6d(self, R: torch.Tensor) -> torch.Tensor:
        """旋转矩阵 → 6D 连续表示（前两列向量）。

        Args:
            R: [N, 3, 3] 旋转矩阵

        Returns:
            [N, 6] 前两列展平 [x1,x2,x3, y1,y2,y3]
        """
        # R[:,:,0] 是第一列，R[:,:,1] 是第二列
        col1 = R[:, :, 0]  # [N, 3]
        col2 = R[:, :, 1]  # [N, 3]
        return torch.cat([col1, col2], dim=-1)  # [N, 6]

    def _6d_to_rotmat(self, rot_6d: torch.Tensor) -> torch.Tensor:
        """6D 连续表示 → 旋转矩阵 (Gram-Schmidt 正交化)。

        Args:
            rot_6d: [N, 6] 6D 旋转表示 [x1,x2,x3, y1,y2,y3]

        Returns:
            [N, 3, 3] 正交旋转矩阵
        """
        x = rot_6d[:, :3]  # 第一列 [N, 3]
        y = rot_6d[:, 3:]  # 第二列 [N, 3]

        # Gram-Schmidt 正交化
        x = F.normalize(x, dim=-1)
        # 投影：y - (y·x)x
        y = y - (x * y).sum(dim=-1, keepdim=True) * x
        y = F.normalize(y, dim=-1)
        # 第三列：叉积
        z = torch.linalg.cross(x, y, dim=-1)

        # 组装旋转矩阵：[x, y, z] 作为列向量
        return torch.stack([x, y, z], dim=2)  # [N, 3, 3]

    def forward(self, idx: int) -> torch.Tensor:
        """获取第 idx 个相机的 camera-to-world 矩阵。

        Args:
            idx: 相机索引 (0 到 N-1)

        Returns:
            [4, 4] camera-to-world 变换矩阵
        """
        R = self._6d_to_rotmat(self.rot_6d[idx:idx+1])  # [1, 3, 3]
        t = self.trans[idx:idx+1]                       # [1, 3]

        c2w = torch.eye(4, device=R.device, dtype=R.dtype)
        c2w[:3, :3] = R[0]
        c2w[:3, 3] = t[0]
        return c2w

    def get_all_poses(self) -> torch.Tensor:
        """获取所有相机位姿。

        Returns:
            [N, 4, 4] camera-to-world 变换矩阵
        """
        N = self.rot_6d.shape[0]
        R = self._6d_to_rotmat(self.rot_6d)  # [N, 3, 3]
        t = self.trans                        # [N, 3]

        c2w = torch.eye(4, device=R.device, dtype=R.dtype)[None].repeat(N, 1, 1)
        c2w[:, :3, :3] = R
        c2w[:, :3, 3] = t
        return c2w


def pose_regularization_loss(poses: CameraPose, weight_rot: float = 0.01,
                            weight_trans: float = 0.001) -> torch.Tensor:
    """位姿正则化损失，惩罚与初始位姿的偏差。

    Args:
        poses: CameraPose 模块
        weight_rot: 旋转正则化权重
        weight_trans: 平移正则化权重

    Returns:
        正则化损失（标量）
    """
    # 旋转正则化: L2 距离
    rot_delta = poses.rot_6d - poses.rot_6d_init
    rot_loss = (rot_delta ** 2).mean()

    # 平移正则化: L2 距离
    trans_delta = poses.trans - poses.trans_init
    trans_loss = (trans_delta ** 2).mean()

    return weight_rot * rot_loss + weight_trans * trans_loss
