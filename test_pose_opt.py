#!/usr/bin/env python3
"""位姿优化模块单元测试"""

import sys
sys.path.insert(0, "/app")

import torch
import numpy as np
from app.pose_opt import CameraPose, pose_regularization_loss


def test_6d_rotation_conversion():
    """测试 6D 旋转表示的往返转换精度"""
    print("=" * 60)
    print("测试 6D 旋转表示往返转换")
    print("=" * 60)

    # 生成随机旋转矩阵
    N = 10
    angles = torch.randn(N, 3) * 0.5  # 小角度

    # 构造旋转矩阵（简化版，用于测试）
    R_original = []
    for i in range(N):
        # 绕 z 轴旋转
        theta = angles[i, 2]
        c, s = torch.cos(theta), torch.sin(theta)
        Rz = torch.tensor([
            [c, -s, 0],
            [s, c, 0],
            [0, 0, 1]
        ], dtype=torch.float32)
        R_original.append(Rz)

    R_original = torch.stack(R_original)

    # 构造假的 camtoworlds
    c2w = torch.eye(4)[None].repeat(N, 1, 1)
    c2w[:, :3, :3] = R_original

    # 往返转换
    poses = CameraPose(c2w)
    R_reconstructed = poses._6d_to_rotmat(poses.rot_6d)

    # 检查误差
    error = (R_original - R_reconstructed).abs().max().item()
    print(f"最大重建误差: {error:.6f}")

    if error < 1e-5:
        print("✅ 6D 旋转转换测试通过")
    else:
        print(f"❌ 6D 旋转转换误差过大: {error}")
        return False

    return True


def test_pose_forward():
    """测试位姿前向传播"""
    print("\n" + "=" * 60)
    print("测试位姿前向传播")
    print("=" * 60)

    N = 5
    c2w_init = torch.eye(4)[None].repeat(N, 1, 1)
    c2w_init[:, :3, 3] = torch.randn(N, 3)  # 随机平移

    poses = CameraPose(c2w_init)

    # 获取单个位姿
    c2w_0 = poses(0)
    print(f"相机 0 位姿形状: {c2w_0.shape}")
    assert c2w_0.shape == (4, 4), "位姿形状错误"

    # 检查是否接近初始值
    error = (c2w_0 - c2w_init[0]).abs().max().item()
    print(f"与初始值误差: {error:.6f}")

    if error < 1e-5:
        print("✅ 位姿前向传播测试通过")
    else:
        print(f"❌ 位姿前向传播误差过大: {error}")
        return False

    return True


def test_pose_gradient():
    """测试位姿梯度反传"""
    print("\n" + "=" * 60)
    print("测试位姿梯度反传")
    print("=" * 60)

    N = 3
    c2w_init = torch.eye(4)[None].repeat(N, 1, 1)
    c2w_init[:, :3, 3] = torch.randn(N, 3)

    poses = CameraPose(c2w_init)
    optimizer = torch.optim.Adam(poses.parameters(), lr=0.01)

    # 模拟损失：让平移接近目标
    target = torch.tensor([1.0, 2.0, 3.0])

    for _ in range(10):
        c2w = poses(0)
        loss = ((c2w[:3, 3] - target) ** 2).sum()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    # 检查是否收敛
    final_trans = poses(0)[:3, 3]
    error = (final_trans - target).abs().max().item()
    print(f"优化后误差: {error:.6f}")

    if error < 0.5:
        print("✅ 位姿梯度反传测试通过")
    else:
        print(f"❌ 位姿优化未收敛: {error}")
        return False

    return True


def test_pose_regularization():
    """测试位姿正则化损失"""
    print("\n" + "=" * 60)
    print("测试位姿正则化损失")
    print("=" * 60)

    N = 5
    c2w_init = torch.eye(4)[None].repeat(N, 1, 1)
    c2w_init[:, :3, 3] = torch.randn(N, 3)

    poses = CameraPose(c2w_init)

    # 初始状态：正则化损失应该为 0
    reg_loss_init = pose_regularization_loss(poses)
    print(f"初始正则化损失: {reg_loss_init.item():.6f}")

    if reg_loss_init.item() > 1e-6:
        print(f"❌ 初始正则化损失应为 0: {reg_loss_init.item()}")
        return False

    # 修改参数
    poses.trans.data += 0.1
    reg_loss_modified = pose_regularization_loss(poses)
    print(f"修改后正则化损失: {reg_loss_modified.item():.6f}")

    if reg_loss_modified.item() < 1e-6:
        print("❌ 修改后正则化损失应该增加")
        return False

    print("✅ 位姿正则化测试通过")
    return True


def test_orthogonality():
    """测试重建的旋转矩阵是否正交"""
    print("\n" + "=" * 60)
    print("测试旋转矩阵正交性")
    print("=" * 60)

    N = 10
    c2w = torch.eye(4)[None].repeat(N, 1, 1)
    c2w[:, :3, :3] = torch.randn(N, 3, 3)  # 非正交矩阵

    poses = CameraPose(c2w)
    R = poses._6d_to_rotmat(poses.rot_6d)

    # 检查正交性: R @ R^T = I
    RRt = R @ R.transpose(1, 2)
    I = torch.eye(3)[None].repeat(N, 1, 1)
    ortho_error = (RRt - I).abs().max().item()

    print(f"正交性误差: {ortho_error:.6f}")

    if ortho_error < 1e-5:
        print("✅ 旋转矩阵正交性测试通过")
    else:
        print(f"❌ 旋转矩阵不正交: {ortho_error}")
        return False

    return True


if __name__ == "__main__":
    all_passed = True

    all_passed &= test_6d_rotation_conversion()
    all_passed &= test_pose_forward()
    all_passed &= test_pose_gradient()
    all_passed &= test_pose_regularization()
    all_passed &= test_orthogonality()

    print("\n" + "=" * 60)
    if all_passed:
        print("✅ 所有测试通过")
    else:
        print("❌ 部分测试失败")
        sys.exit(1)
    print("=" * 60)
