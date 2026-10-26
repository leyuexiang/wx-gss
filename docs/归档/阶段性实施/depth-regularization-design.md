# Phase 2: 深度正则化设计方案

**日期**: 2026-09-04  
**目标**: 提升少视角（2-15 张）3DGS 重建质量

## 背景

Phase 1 解决了"位姿解不出来"问题，但少视角训练仅靠光度 loss (L1+SSIM) 容易过拟合，产生浮空伪影。需引入 MASt3R 深度图作为几何先验。

## 技术方案

### 1. 深度图存储

**MASt3R 输出** (`pipeline.py` 修改):
- `_run_mast3r()` 保存深度图到 `dataset/depths/`
- 格式: `.npy` (float32, 原图分辨率)
- 包含 MASt3R 512 尺度深度上采样到原图

**数据集加载** (`dataset.py` 修改):
```python
class ColmapScene:
    def __init__(self, dataset_dir, factor):
        # ... 现有代码 ...
        self.depth_dir = os.path.join(dataset_dir, "depths")
        self.has_depth = os.path.isdir(self.depth_dir)

class ColmapDataset:
    def __getitem__(self, idx):
        data = {
            "image": arr,
            "camtoworld": ...,
            "K": ...,
            "image_id": idx,
        }
        if self.scene.has_depth:
            depth_path = os.path.join(self.scene.depth_dir, 
                                     f"{self.scene.names[idx]}.npy")
            if os.path.exists(depth_path):
                depth = np.load(depth_path)
                # 降采样到 factor
                if factor > 1:
                    from scipy.ndimage import zoom
                    depth = zoom(depth, 1.0/factor, order=1)
                data["depth"] = torch.from_numpy(depth)
        return data
```

### 2. 深度渲染

**gsplat 深度输出**:
gsplat 的 `rasterization()` 默认返回 RGB，需启用深度输出。检查 gsplat API:

```python
renders, alphas, info = rasterization(
    means=params["means"],
    # ... 其他参数 ...
    render_mode="RGB+ED",  # Expected Depth
)
rendered_depth = info["depth"]  # [H, W]
```

如果 gsplat 不直接支持深度，手动计算:
```python
# 方案 A: 加权深度（通过 alpha 合成）
# 需要每个高斯到相机的距离
z_gaussians = (viewmat @ torch.cat([
    params["means"], 
    torch.ones(N, 1, device=device)
], dim=1).T).T[:, 2]  # [N]

# 方案 B: 用 rasterization 的 info["means2d"] 反投影
# 实现复杂，优先方案 A
```

### 3. 深度 Loss

**Pearson Correlation Loss** (InstantSplat 使用):
```python
def pearson_depth_loss(pred_depth, gt_depth, mask=None):
    """
    pred_depth: [H, W] 渲染深度
    gt_depth: [H, W] MASt3R 深度
    mask: [H, W] 有效深度掩码 (gt_depth > 0)
    
    Pearson 相关系数惩罚深度的相对结构差异，对尺度不敏感
    """
    if mask is None:
        mask = gt_depth > 1e-3
    
    pred = pred_depth[mask]
    gt = gt_depth[mask]
    
    if len(pred) < 100:  # 有效像素太少
        return 0.0
    
    # 标准化
    pred_mean = pred.mean()
    gt_mean = gt.mean()
    pred_std = pred.std() + 1e-8
    gt_std = gt.std() + 1e-8
    
    pred_norm = (pred - pred_mean) / pred_std
    gt_norm = (gt - gt_mean) / gt_std
    
    # Pearson = E[(x-μx)(y-μy)] / (σx σy)
    # 最大化相关性 = 最小化 1 - correlation
    correlation = (pred_norm * gt_norm).mean()
    loss = 1.0 - correlation
    return loss
```

**L1 深度 Loss** (备选，简单但对尺度敏感):
```python
def l1_depth_loss(pred_depth, gt_depth, mask=None):
    if mask is None:
        mask = gt_depth > 1e-3
    pred = pred_depth[mask]
    gt = gt_depth[mask]
    if len(pred) < 100:
        return 0.0
    # 归一化到相同尺度（避免绝对值差异）
    pred_norm = pred / (pred.mean() + 1e-8)
    gt_norm = gt / (gt.mean() + 1e-8)
    return F.l1_loss(pred_norm, gt_norm)
```

### 4. 训练集成 (`trainer.py`)

**Loss 组合**:
```python
# 训练循环内
l1 = F.l1_loss(pred, image)
ssim_val = ssim_fn(pred[None], image[None])
photo_loss = 0.8 * l1 + 0.2 * (1.0 - ssim_val)

depth_loss_val = 0.0
if "depth" in data and enable_depth_reg:
    gt_depth = data["depth"].to(device)
    rendered_depth = info["depth"]  # 需要 gsplat 支持
    depth_loss_val = pearson_depth_loss(rendered_depth, gt_depth)

# 深度权重调度：前期强正则，后期弱化让光度主导
depth_weight = 0.1 * max(0, 1.0 - step / (iterations * 0.5))
loss = photo_loss + depth_weight * depth_loss_val
```

**超参数**:
- `enable_depth_reg`: 仅 mast3r 后端启用
- 深度权重: 0.1 (前半段) → 0.0 (后半段)
- 退火策略: 让深度约束几何初始化，后期光度微调

### 5. 配置开关

**config.py**:
```python
ENABLE_DEPTH_REG = os.environ.get("ENABLE_DEPTH_REG", "1").lower() in ("1", "true")
DEPTH_REG_WEIGHT = float(os.environ.get("DEPTH_REG_WEIGHT", "0.1"))
```

**API 参数**:
```python
enable_depth_reg: bool = Form(
    default=None,
    description="启用深度正则化（仅 mast3r 后端有效）"
)
```

## 实现顺序

### Step 1: 深度图保存 (Task #5)
1. 修改 `_export_mast3r_dataset()`，保存深度到 `dataset/depths/`
2. 深度上采样到原图分辨率，再按 downsample 降采样
3. 验证: 可视化保存的深度图

### Step 2: 数据集加载 (Task #5)
1. `ColmapScene` 检测 `depths/` 目录
2. `ColmapDataset.__getitem__` 加载深度
3. 验证: 打印 batch 中的 depth shape

### Step 3: 深度渲染 (Task #6)
1. 研究 gsplat rasterization 的深度输出 API
2. 如无直接支持，实现手动深度合成
3. 验证: 渲染深度图可视化

### Step 4: 深度 Loss (Task #6)
1. 实现 `pearson_depth_loss()` 和 `l1_depth_loss()`
2. 集成到训练循环，添加权重调度
3. 单元测试 loss 函数

### Step 5: 对比测试 (Task #7)
1. TestIMG 5 张: Phase 1 vs Phase 2
2. 评估指标: PSNR, SSIM, 视觉检查浮空伪影
3. 显存和速度影响

## 预期效果

**Phase 1 (无深度正则)**:
- 少视角容易过拟合训练视角
- 新视角出现浮空伪影、几何不连续
- SSIM 可能偏高但视觉质量差

**Phase 2 (有深度正则)**:
- 几何结构更连贯
- 浮空伪影减少
- 泛化能力提升（测试视角 PSNR/SSIM 更高）

## 风险和缓解

| 风险 | 缓解 |
|-----|-----|
| gsplat 不支持深度输出 | 手动实现加权深度合成 |
| 深度权重过大导致模糊 | 退火调度 + 超参数扫描 |
| MASt3R 深度质量差影响训练 | Pearson loss 对尺度不敏感，robustness 更好 |
| 显存增加 | 深度图按 downsample 降采样，额外开销 <100MB |
| 训练变慢 | 深度 loss 计算简单，预计 <5% 影响 |

## 参考资料

- [DepthRegularizedGS (CVPRW 2024)](https://github.com/robot0321/DepthRegularizedGS) — 深度正则化 3DGS 专项研究
- [InstantSplat (NVlabs)](https://github.com/NVlabs/InstantSplat) — MASt3R + 3DGS 联合优化
- [PM-Loss (3DV 2026)](https://github.com/aim-uofa/PM-Loss) — 点图表示的深度 loss

---

**设计负责人**: Claude  
**目标完成日期**: 2026-09-06 (预计 2 天)
