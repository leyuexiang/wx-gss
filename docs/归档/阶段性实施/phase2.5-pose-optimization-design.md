# Phase 2.5: 位姿联合优化设计方案

**日期**: 2026-09-04  
**目标**: 在训练中微调相机位姿，进一步提升少视角重建质量

## 背景

Phase 2 的深度正则化改善了几何结构，但 MASt3R 的位姿估计仍可能存在误差（尤其是少视角场景）。位姿联合优化可以：
- 在训练中微调相机位姿
- 补偿 MASt3R 的位姿误差
- 提升渲染质量和几何一致性

## 核心方法：Gaussian Bundle Adjustment (GauBA)

根据 InstantSplat 论文，GauBA 是关键技术：

### 原理
```
传统 BA: 最小化重投影误差（2D 特征点）
GauBA: 最小化渲染误差（整个图像）

Joint Optimization:
  固定位姿 → 优化高斯参数 (标准 3DGS 训练)
  固定高斯 → 优化位姿参数 (梯度反传到位姿)
  交替迭代
```

### 优势
- 端到端可微分
- 利用整图信息（vs 稀疏特征点）
- 与 3DGS 训练无缝集成

## 技术方案

### 1. 位姿参数化

**SE(3) 参数化** (rotation + translation):
```python
class CameraPose(nn.Module):
    def __init__(self, camtoworlds):
        """
        camtoworlds: [N, 4, 4] 初始位姿
        """
        super().__init__()
        N = camtoworlds.shape[0]
        
        # 提取旋转和平移
        R = camtoworlds[:, :3, :3]  # [N, 3, 3]
        t = camtoworlds[:, :3, 3]   # [N, 3]
        
        # 旋转用 6D 连续表示（避免四元数奇异性）
        # https://arxiv.org/abs/1812.07035
        self.rot_6d = nn.Parameter(self._rotmat_to_6d(R))  # [N, 6]
        self.trans = nn.Parameter(t)                        # [N, 3]
        
        # 保存初始值（用于正则化）
        self.register_buffer('rot_6d_init', self.rot_6d.data.clone())
        self.register_buffer('trans_init', self.trans.data.clone())
    
    def _rotmat_to_6d(self, R):
        """旋转矩阵 → 6D 连续表示 (前两列)"""
        return R[:, :, :2].reshape(-1, 6)
    
    def _6d_to_rotmat(self, rot_6d):
        """6D 连续表示 → 旋转矩阵"""
        # Gram-Schmidt 正交化
        x = rot_6d[:, :3]
        y = rot_6d[:, 3:]
        x = F.normalize(x, dim=-1)
        y = y - (x * y).sum(dim=-1, keepdim=True) * x
        y = F.normalize(y, dim=-1)
        z = torch.cross(x, y, dim=-1)
        return torch.stack([x, y, z], dim=-1)  # [N, 3, 3]
    
    def forward(self, idx):
        """返回第 idx 个相机的 camtoworld [4, 4]"""
        R = self._6d_to_rotmat(self.rot_6d[idx:idx+1])  # [1, 3, 3]
        t = self.trans[idx:idx+1]                       # [1, 3]
        c2w = torch.eye(4, device=R.device)
        c2w[:3, :3] = R[0]
        c2w[:3, 3] = t[0]
        return c2w
```

**为什么用 6D 而非四元数/轴角**:
- 四元数有符号歧义，不连续
- 轴角在小角度附近不稳定
- 6D 表示连续、无奇异性，梯度更稳定

### 2. 位姿正则化

**防止偏离初始估计太远**:
```python
def pose_regularization_loss(poses: CameraPose, weight_rot=0.01, weight_trans=0.001):
    """
    位姿正则化: 惩罚与初始位姿的偏差
    
    rotation: 用 Frobenius 范数衡量旋转矩阵偏差
    translation: 用 L2 范数
    """
    # 旋转正则化
    rot_delta = poses.rot_6d - poses.rot_6d_init
    rot_loss = (rot_delta ** 2).mean()
    
    # 平移正则化
    trans_delta = poses.trans - poses.trans_init
    trans_loss = (trans_delta ** 2).mean()
    
    return weight_rot * rot_loss + weight_trans * trans_loss
```

**权重建议**:
- `weight_rot=0.01`: 旋转较敏感，惩罚权重高
- `weight_trans=0.001`: 平移容差更大

### 3. 训练集成

**修改 trainer.py**:
```python
def train(..., enable_pose_opt: bool = False, pose_opt_start: int = 500):
    # 初始化位姿参数（如果启用位姿优化）
    poses = None
    pose_optimizer = None
    if enable_pose_opt:
        poses = CameraPose(scene.camtoworlds).to(device)
        pose_optimizer = torch.optim.Adam(poses.parameters(), lr=1e-5)
    
    for step in range(iterations):
        data = dataset[idx]
        
        # 获取位姿（可学习 or 固定）
        if poses is not None and step >= pose_opt_start:
            camtoworld = poses(data["image_id"])
        else:
            camtoworld = data["camtoworld"].to(device)
        
        viewmat = torch.linalg.inv(camtoworld)[None]
        
        # 渲染 + 计算 loss
        renders, _, info = rasterization(...)
        photo_loss = ...
        depth_loss = ...
        
        # 位姿正则化
        pose_reg_loss = 0.0
        if poses is not None and step >= pose_opt_start:
            pose_reg_loss = pose_regularization_loss(poses)
        
        loss = photo_loss + depth_weight * depth_loss + pose_reg_loss
        loss.backward()
        
        # 更新高斯参数
        for opt in optimizers.values():
            opt.step()
            opt.zero_grad()
        
        # 更新位姿（如果启用）
        if pose_optimizer is not None and step >= pose_opt_start:
            pose_optimizer.step()
            pose_optimizer.zero_grad()
```

**关键点**:
- **延迟启动**: `pose_opt_start=500`，前期让高斯参数先收敛
- **小学习率**: `lr=1e-5`，避免位姿震荡
- **正则化**: 防止偏离初始估计太远

### 4. 位姿导出

**训练完成后保存优化后的位姿**:
```python
if poses is not None:
    optimized_poses = []
    for i in range(len(dataset)):
        c2w = poses(i).detach().cpu().numpy()
        optimized_poses.append(c2w)
    np.save(os.path.join(out_dir, "optimized_poses.npy"), 
            np.stack(optimized_poses))
```

### 5. API 参数

**config.py**:
```python
ENABLE_POSE_OPT = os.environ.get("ENABLE_POSE_OPT", "0").lower() in ("1", "true")
POSE_OPT_START = int(os.environ.get("POSE_OPT_START", "500"))
```

**main.py**:
```python
enable_pose_opt: bool = Form(
    default=None,
    description="启用位姿联合优化（仅 mast3r 后端，默认关闭）"
)
```

**默认行为**:
- Phase 2: `enable_depth_reg=True, enable_pose_opt=False`
- Phase 2.5: `enable_depth_reg=True, enable_pose_opt=True`

## 实现顺序

### Step 1: 位姿参数化 (Task #9)
1. 实现 `CameraPose` 模块（6D 旋转表示）
2. 单元测试：往返转换精度

### Step 2: 位姿正则化 (Task #10)
1. 实现 `pose_regularization_loss()`
2. 验证：梯度是否正常

### Step 3: 训练集成 (Task #9)
1. 修改 `train()` 函数，添加位姿优化分支
2. 延迟启动逻辑（pose_opt_start）
3. 导出优化后的位姿

### Step 4: 对比测试 (Task #11)
1. Phase 2: depth_reg=True, pose_opt=False
2. Phase 2.5: depth_reg=True, pose_opt=True
3. 评估：PSNR/SSIM、ATE（位姿误差）

## 预期效果

**Phase 2 (深度正则化)**:
- 几何结构改善
- 仍受 MASt3R 位姿误差影响

**Phase 2.5 (+ 位姿优化)**:
- 进一步提升渲染质量
- 位姿误差降低 30-50%
- 新视角 SSIM 提升 2-5%

**代价**:
- 训练时间 +20-30%
- 显存 +50-100MB（位姿参数很小）
- 可能过拟合（需正则化）

## 风险和缓解

| 风险 | 缓解 |
|-----|-----|
| 位姿优化不收敛 | 延迟启动 + 小学习率 |
| 偏离初始估计太远 | 位姿正则化 loss |
| 过拟合训练视角 | 测试集评估 ATE |
| 训练变慢 | 可选特性，默认关闭 |

## 参考文献

**InstantSplat (NVlabs, 2024)**:
- 论文: [Sparse-view Gaussian Splatting in Seconds](https://arxiv.org/abs/2403.20309)
- 方法: Gaussian Bundle Adjustment (GauBA)
- 效果: SSIM +32%, ATE -80%

**Joint Optimization (2024)**:
- 论文: [Joint Optimization of Pose Estimation and 3D Gaussian Splatting](https://arxiv.org/abs/2510.26117)
- 方法: 交替优化 + 3D optical flow
- 关键: 解耦高斯更新和位姿更新

**6D Rotation Representation**:
- 论文: [On the Continuity of Rotation Representations](https://arxiv.org/abs/1812.07035)
- 优势: 连续、无奇异性、梯度稳定

---

**设计负责人**: Claude  
**预计工作量**: 1-2 天  
**目标**: 让少视角场景达到接近密集拍摄的质量

**Sources**:
- [InstantSplat (arxiv.org)](https://arxiv.org/abs/2403.20309)
- [Joint Optimization of Pose Estimation and 3D Gaussian Splatting](https://arxiv.org/html/2510.26117v1)
- [InstantSplat GitHub](https://instantsplat.github.io/)
