# Depth-Anything-V2 集成计划

**制定日期**: 2026-09-04  
**预计总时长**: 3-6 周  
**目标**: 提升深度质量，扩展单图场景

---

## 🎯 整体目标

### 主要目标
1. ✅ **支持单图 3DGS 重建**（扩展功能）
2. ✅ **提升深度正则化质量**（解决 Phase 2 问题）
3. ✅ **可选深度后端**（灵活配置）

### 成功指标
- 单图场景可重建（质量中等）
- Phase 2 Loss 降至 ≤ Phase 1（0.00479）
- 深度细节提升 20-30%
- 推理时间增加 ≤ 20%

---

## 📅 分阶段实施计划

### 🚀 阶段 1: 快速验证（Week 1，3-5天）

**目标**: 验证 DA-V2 的深度质量，快速 MVP

#### 任务 1.1: 环境准备（Day 1，2小时）

**依赖安装**:
```bash
# 在 worker 容器中安装
docker exec 3dgs-worker pip install transformers torch pillow
```

**测试模型加载**:
```python
# test_depth_anything.py
from transformers import pipeline
import time

# 测试三个尺寸
models = {
    "small": "depth-anything/Depth-Anything-V2-Small",    # 25M
    "base": "depth-anything/Depth-Anything-V2-Base",      # 97M
    "large": "depth-anything/Depth-Anything-V2-Large",    # 335M
}

for name, model in models.items():
    print(f"\n=== Testing {name} ===")
    pipe = pipeline("depth-estimation", model=model)
    
    start = time.time()
    result = pipe("test_image.jpg")
    elapsed = time.time() - start
    
    print(f"Time: {elapsed:.2f}s")
    print(f"Shape: {result['depth'].shape}")
```

**验收标准**:
- ✅ 三个模型都能加载
- ✅ Small 模型推理 < 1s
- ✅ Base 模型推理 < 3s

---

#### 任务 1.2: 深度质量对比（Day 1-2，6小时）

**实现对比脚本**:
```python
# scripts/compare_depth_methods.py
"""
对比 MASt3R vs Depth-Anything-V2 深度质量
"""
import numpy as np
import matplotlib.pyplot as plt
from transformers import pipeline

def compare_depths(image_dir, output_dir):
    """对比两种方法的深度估计"""
    
    # 1. MASt3R 深度
    from app.pipeline import run_mast3r
    mast3r_result = run_mast3r(image_dir)
    mast3r_depths = mast3r_result['depths']
    
    # 2. Depth-Anything-V2 深度
    pipe = pipeline("depth-estimation", 
                   model="depth-anything/Depth-Anything-V2-Small")
    
    da_depths = []
    for img_path in mast3r_result['image_paths']:
        depth = pipe(img_path)['depth']
        da_depths.append(np.array(depth))
    
    # 3. 可视化对比
    for i, (mast3r_d, da_d) in enumerate(zip(mast3r_depths, da_depths)):
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # 原图
        img = plt.imread(mast3r_result['image_paths'][i])
        axes[0].imshow(img)
        axes[0].set_title('原图')
        
        # MASt3R 深度
        axes[1].imshow(mast3r_d, cmap='turbo')
        axes[1].set_title('MASt3R 深度')
        
        # DA-V2 深度
        axes[2].imshow(da_d, cmap='turbo')
        axes[2].set_title('Depth-Anything-V2 深度')
        
        plt.savefig(f"{output_dir}/compare_{i}.png")
    
    # 4. 计算指标
    metrics = {
        'edge_sharpness': compare_edge_sharpness(mast3r_depths, da_depths),
        'detail_richness': compare_detail_richness(mast3r_depths, da_depths),
        'consistency': compare_consistency(mast3r_depths, da_depths),
    }
    
    return metrics

def compare_edge_sharpness(depths1, depths2):
    """对比边缘清晰度"""
    from scipy.ndimage import sobel
    
    scores1 = [sobel(d).mean() for d in depths1]
    scores2 = [sobel(d).mean() for d in depths2]
    
    return {
        'mast3r_mean': np.mean(scores1),
        'da_v2_mean': np.mean(scores2),
        'improvement': (np.mean(scores2) - np.mean(scores1)) / np.mean(scores1) * 100
    }

# 运行对比
if __name__ == "__main__":
    metrics = compare_depths(
        image_dir="/data/tasks/compare_phase1/dataset/images",
        output_dir="/tmp/depth_comparison"
    )
    print(json.dumps(metrics, indent=2))
```

**测试数据集**:
- TestIMG (5张)
- 室内场景 (10张)
- 室外场景 (10张)

**验收标准**:
- ✅ 可视化对比图生成
- ✅ 量化指标计算完成
- ✅ DA-V2 边缘清晰度 > MASt3R

---

#### 任务 1.3: 单图重建原型（Day 2-3，8小时）

**实现单图重建**:
```python
# app/monocular_pipeline.py
"""
单目图像 3DGS 重建流程
"""
import torch
import numpy as np
from transformers import pipeline

def reconstruct_from_single_image(
    image_path: str,
    output_dir: str,
    iterations: int = 3000,
    assume_scale: float = 1.0,  # 假设场景尺度
    fov: float = 60.0,           # 假设视场角
):
    """从单张图像重建 3DGS
    
    Args:
        image_path: 输入图片路径
        output_dir: 输出目录
        iterations: 训练迭代次数
        assume_scale: 假设场景深度尺度（米）
        fov: 假设相机视场角（度）
    
    Returns:
        重建统计信息
    """
    
    # 1. 加载图像
    from PIL import Image
    image = Image.open(image_path)
    W, H = image.size
    
    # 2. 估计深度（Depth-Anything-V2）
    print("估计深度图...")
    depth_pipe = pipeline(
        "depth-estimation",
        model="depth-anything/Depth-Anything-V2-Small"
    )
    depth_result = depth_pipe(image_path)
    depth_map = np.array(depth_result['depth'])
    
    # 3. 归一化深度到假设尺度
    depth_normalized = normalize_depth(depth_map, assume_scale)
    
    # 4. 生成伪相机位姿（单位矩阵）
    camtoworld = np.eye(4)
    
    # 5. 生成相机内参（假设）
    focal = W / (2 * np.tan(np.deg2rad(fov) / 2))
    K = np.array([
        [focal, 0, W/2],
        [0, focal, H/2],
        [0, 0, 1]
    ])
    
    # 6. 从深度图生成初始点云
    points, colors = depth_to_pointcloud(
        depth_normalized, 
        image, 
        K, 
        camtoworld
    )
    
    print(f"生成初始点云: {len(points)} 个点")
    
    # 7. 初始化 3DGS 并训练
    from app.trainer import train_from_pointcloud
    
    stats = train_from_pointcloud(
        points=points,
        colors=colors,
        images=[image],
        poses=[camtoworld],
        Ks=[K],
        output_dir=output_dir,
        iterations=iterations,
    )
    
    return stats

def normalize_depth(depth, target_scale=1.0):
    """归一化相对深度到目标尺度"""
    # DA-V2 输出相对深度，需要归一化
    depth_min = depth.min()
    depth_max = depth.max()
    
    # 归一化到 [0, target_scale]
    depth_norm = (depth - depth_min) / (depth_max - depth_min) * target_scale
    
    return depth_norm

def depth_to_pointcloud(depth, image, K, camtoworld):
    """从深度图生成点云"""
    H, W = depth.shape
    
    # 生成像素坐标网格
    u, v = np.meshgrid(np.arange(W), np.arange(H))
    
    # 反投影到 3D
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    
    x = (u - cx) * depth / fx
    y = (v - cy) * depth / fy
    z = depth
    
    # 相机坐标系点云
    points_cam = np.stack([x, y, z], axis=-1).reshape(-1, 3)
    
    # 转换到世界坐标系
    points_world = (camtoworld[:3, :3] @ points_cam.T + camtoworld[:3, 3:4]).T
    
    # 提取颜色
    colors = np.array(image).reshape(-1, 3) / 255.0
    
    # 过滤无效点（深度=0）
    valid = (depth.reshape(-1) > 0)
    points_world = points_world[valid]
    colors = colors[valid]
    
    return points_world, colors
```

**API 接口**:
```python
# app/main.py 添加单图端点
@app.post("/train/monocular", dependencies=[Depends(require_api_key)])
def train_monocular(
    file: UploadFile = File(...),
    iterations: int = Form(default=3000),
    assume_scale: float = Form(default=1.0),
    fov: float = Form(default=60.0),
):
    """单目图像 3DGS 重建（实验性）"""
    
    from app.monocular_pipeline import reconstruct_from_single_image
    
    # 保存上传图片
    task_id = f"mono_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    task_dir = f"/data/tasks/{task_id}"
    os.makedirs(task_dir, exist_ok=True)
    
    image_path = f"{task_dir}/input.jpg"
    with open(image_path, "wb") as f:
        f.write(file.file.read())
    
    # 运行单图重建
    stats = reconstruct_from_single_image(
        image_path=image_path,
        output_dir=f"{task_dir}/output",
        iterations=iterations,
        assume_scale=assume_scale,
        fov=fov,
    )
    
    return {
        "task_id": task_id,
        "stats": stats,
        "warning": "单目重建质量有限，建议使用多视角输入"
    }
```

**验收标准**:
- ✅ 单图可以生成 .ply 文件
- ✅ API 端点可用
- ✅ 质量中等（预期）

---

#### 任务 1.4: Week 1 总结报告（Day 4，2小时）

**生成对比报告**:
- DA-V2 vs MASt3R 深度质量对比
- 单图重建可行性验证
- 性能开销测量
- 是否继续 Phase 2 决策

**决策点**:
- ✅ 如果 DA-V2 质量好 → 进入 Phase 2
- ❌ 如果 DA-V2 质量差 → 停止集成

---

### 🏗️ 阶段 2: 混合深度实现（Week 2-3，10-15天）

**前置条件**: 阶段 1 验证通过

#### 任务 2.1: 深度对齐算法（Week 2，3-5天）

**目标**: 将 DA-V2 相对深度对齐到 MASt3R 度量深度

**实现方案 A: 最小二乘对齐**
```python
# app/depth_alignment.py
def align_depth_scale(
    relative_depth: np.ndarray,  # DA-V2 输出
    metric_depth: np.ndarray,     # MASt3R 输出
    method: str = "least_squares"
) -> np.ndarray:
    """对齐相对深度到度量深度
    
    相对深度: d_rel = s * d_metric + t
    求解: s, t
    """
    
    if method == "least_squares":
        # 1. 展平
        d_rel = relative_depth.flatten()
        d_met = metric_depth.flatten()
        
        # 2. 过滤无效值
        valid = (d_met > 0) & np.isfinite(d_met) & np.isfinite(d_rel)
        d_rel = d_rel[valid]
        d_met = d_met[valid]
        
        # 3. 最小二乘求解 d_rel = s * d_met + t
        A = np.vstack([d_met, np.ones(len(d_met))]).T
        s, t = np.linalg.lstsq(A, d_rel, rcond=None)[0]
        
        # 4. 对齐
        d_aligned = (relative_depth - t) / s
        
        return d_aligned
    
    elif method == "median":
        # 中位数对齐（更鲁棒）
        ratio = np.median(metric_depth[valid] / relative_depth[valid])
        return relative_depth * ratio
    
    elif method == "percentile":
        # 百分位对齐（对异常值鲁棒）
        p10_rel = np.percentile(relative_depth[valid], 10)
        p10_met = np.percentile(metric_depth[valid], 10)
        p90_rel = np.percentile(relative_depth[valid], 90)
        p90_met = np.percentile(metric_depth[valid], 90)
        
        s = (p90_met - p10_met) / (p90_rel - p10_rel)
        t = p10_met - s * p10_rel
        
        return s * relative_depth + t
```

**测试对齐质量**:
```python
def evaluate_alignment(aligned_depth, metric_depth):
    """评估对齐质量"""
    valid = (metric_depth > 0)
    
    # RMSE
    rmse = np.sqrt(((aligned_depth[valid] - metric_depth[valid])**2).mean())
    
    # 相对误差
    rel_error = np.abs(aligned_depth[valid] - metric_depth[valid]) / metric_depth[valid]
    rel_error_mean = rel_error.mean()
    
    # 阈值准确率
    threshold_acc = (rel_error < 0.25).mean()
    
    return {
        'rmse': rmse,
        'rel_error_mean': rel_error_mean,
        'threshold_accuracy_25%': threshold_acc
    }
```

**验收标准**:
- ✅ RMSE < 0.5m (假设场景尺度~10m)
- ✅ 相对误差 < 15%
- ✅ 阈值准确率 > 80%

---

#### 任务 2.2: 混合策略实现（Week 2，2-3天）

**实现置信度计算**:
```python
def compute_depth_confidence(
    mast3r_depth: np.ndarray,
    image: np.ndarray,
    confidence_type: str = "gradient"
) -> np.ndarray:
    """计算深度置信度图
    
    Args:
        mast3r_depth: MASt3R 深度
        image: 原图
        confidence_type: 置信度类型
            - gradient: 基于图像梯度
            - texture: 基于纹理丰富度
            - hybrid: 混合
    
    Returns:
        confidence: [0, 1] 置信度图
    """
    from scipy.ndimage import sobel, generic_filter
    
    if confidence_type == "gradient":
        # 图像梯度越大，纹理越丰富，立体匹配越可靠
        gray = np.mean(image, axis=-1)
        grad_x = sobel(gray, axis=1)
        grad_y = sobel(gray, axis=0)
        gradient = np.sqrt(grad_x**2 + grad_y**2)
        
        # 归一化到 [0, 1]
        confidence = (gradient - gradient.min()) / (gradient.max() - gradient.min())
        
    elif confidence_type == "texture":
        # 纹理丰富度（局部标准差）
        gray = np.mean(image, axis=-1)
        local_std = generic_filter(gray, np.std, size=7)
        
        confidence = (local_std - local_std.min()) / (local_std.max() - local_std.min())
    
    elif confidence_type == "hybrid":
        # 综合多个指标
        conf_grad = compute_depth_confidence(mast3r_depth, image, "gradient")
        conf_text = compute_depth_confidence(mast3r_depth, image, "texture")
        
        confidence = 0.5 * conf_grad + 0.5 * conf_text
    
    # 平滑置信度图
    from scipy.ndimage import gaussian_filter
    confidence = gaussian_filter(confidence, sigma=2.0)
    
    return confidence

def blend_depths(
    mast3r_depth: np.ndarray,
    da_depth_aligned: np.ndarray,
    confidence: np.ndarray,
    blend_mode: str = "weighted"
) -> np.ndarray:
    """混合两种深度估计
    
    Args:
        mast3r_depth: MASt3R 深度（度量）
        da_depth_aligned: DA-V2 深度（已对齐）
        confidence: 置信度图 [0, 1]
        blend_mode: 混合模式
            - weighted: 加权平均
            - threshold: 阈值切换
            - adaptive: 自适应
    
    Returns:
        blended_depth: 混合深度
    """
    
    if blend_mode == "weighted":
        # 加权平均：高置信度用 DA-V2，低置信度用 MASt3R
        w_da = confidence
        w_mast3r = 1 - confidence
        
        blended = w_da * da_depth_aligned + w_mast3r * mast3r_depth
        
    elif blend_mode == "threshold":
        # 阈值切换
        threshold = 0.5
        blended = np.where(confidence > threshold, da_depth_aligned, mast3r_depth)
    
    elif blend_mode == "adaptive":
        # 自适应：根据深度差异动态调整权重
        depth_diff = np.abs(da_depth_aligned - mast3r_depth)
        depth_diff_norm = depth_diff / (mast3r_depth + 1e-6)
        
        # 差异大的区域降低 DA-V2 权重
        w_da = confidence * (1 - np.clip(depth_diff_norm, 0, 1))
        w_mast3r = 1 - w_da
        
        blended = w_da * da_depth_aligned + w_mast3r * mast3r_depth
    
    return blended
```

**验收标准**:
- ✅ 混合深度视觉质量好
- ✅ 边缘保留 DA-V2 细节
- ✅ 平坦区域保持 MASt3R 准确性

---

#### 任务 2.3: 集成到训练流程（Week 3，3-5天）

**修改 pipeline.py**:
```python
# app/pipeline.py
def run_pipeline(
    task_id: str,
    ...,
    depth_backend: str = "mast3r",  # 新参数: mast3r, depth_anything, hybrid
):
    """训练流程"""
    
    # 1. SfM 阶段
    if sfm_backend == "mast3r":
        mast3r_result = run_mast3r_reconstruction(...)
        poses = mast3r_result['poses']
        mast3r_depths = mast3r_result['depths']
    
    # 2. 深度估计阶段（如果需要）
    if depth_backend == "depth_anything":
        # 仅使用 DA-V2
        from app.depth_anything_wrapper import estimate_depths
        final_depths = estimate_depths(images)
        # 对齐到 MASt3R 尺度
        final_depths = align_depths_to_scale(final_depths, mast3r_depths)
        
    elif depth_backend == "hybrid":
        # 混合策略
        from app.depth_hybrid import create_hybrid_depths
        final_depths = create_hybrid_depths(
            images=images,
            mast3r_depths=mast3r_depths,
            blend_mode="adaptive"
        )
    else:
        # 默认使用 MASt3R
        final_depths = mast3r_depths
    
    # 3. 保存深度图
    save_depths(final_depths, ...)
    
    # 4. 训练 3DGS
    stats = train(
        ...,
        depths=final_depths if enable_depth_reg else None
    )
    
    return stats
```

**API 更新**:
```python
# app/main.py
@app.post("/train")
def submit_train(
    ...,
    depth_backend: Literal["mast3r", "depth_anything", "hybrid"] = Form(
        default="mast3r",
        description="深度后端：mast3r（默认）、depth_anything（DA-V2）、hybrid（混合）"
    ),
):
    ...
```

**验收标准**:
- ✅ 三种深度后端都可用
- ✅ API 参数生效
- ✅ 深度图正确保存

---

#### 任务 2.4: Phase 2 重新测试（Week 3，2天）

**运行完整对比测试**:
```bash
# 测试配置矩阵
for depth_backend in mast3r depth_anything hybrid; do
  for depth_weight in 0.01 0.02 0.05; do
    curl -X POST http://localhost:12524/train \
      -F "file=@TestIMG.zip" \
      -F "sfm_backend=mast3r" \
      -F "enable_depth_reg=true" \
      -F "depth_backend=$depth_backend" \
      -F "depth_reg_weight=$depth_weight" \
      -F "task_name=phase2_${depth_backend}_${depth_weight}"
  done
done
```

**对比指标**:
- Loss vs Phase 1 (baseline=0.00479)
- 训练时间
- 显存占用
- 视觉质量

**目标**:
- ✅ 至少一个配置 Loss ≤ Phase 1
- ✅ 深度细节视觉改善明显
- ✅ 时间开销 ≤ +20%

---

### 🔬 阶段 3: 优化和生产化（Week 4-6，15-20天）

#### 任务 3.1: 性能优化（Week 4，5天）

**优化点**:
1. **模型缓存**: 避免重复加载
2. **批处理**: 多图并行推理
3. **模型选择**: 根据场景自动选择 Small/Base/Large
4. **GPU 利用**: 优化显存和计算

```python
# app/depth_anything_wrapper.py
class DepthAnythingV2Pipeline:
    """DA-V2 推理包装器（优化版）"""
    
    def __init__(self, model_size="small", device="cuda"):
        self.model_size = model_size
        self.device = device
        self._pipe = None  # 延迟加载
    
    @property
    def pipe(self):
        """延迟加载+缓存"""
        if self._pipe is None:
            model_name = f"depth-anything/Depth-Anything-V2-{self.model_size.capitalize()}"
            self._pipe = pipeline(
                "depth-estimation",
                model=model_name,
                device=self.device
            )
        return self._pipe
    
    def estimate_batch(self, image_paths, batch_size=4):
        """批处理推理"""
        depths = []
        for i in range(0, len(image_paths), batch_size):
            batch = image_paths[i:i+batch_size]
            batch_depths = [self.pipe(img)['depth'] for img in batch]
            depths.extend(batch_depths)
        return depths
    
    @staticmethod
    def auto_select_model(n_images, image_size):
        """根据场景自动选择模型大小"""
        total_pixels = n_images * image_size[0] * image_size[1]
        
        if total_pixels < 10_000_000:  # < 10MP
            return "small"
        elif total_pixels < 50_000_000:  # < 50MP
            return "base"
        else:
            return "large"

# 全局单例
_da_pipeline = None

def get_depth_anything_pipeline(model_size="small"):
    global _da_pipeline
    if _da_pipeline is None or _da_pipeline.model_size != model_size:
        _da_pipeline = DepthAnythingV2Pipeline(model_size)
    return _da_pipeline
```

**验收标准**:
- ✅ 批处理速度提升 30-50%
- ✅ 显存占用优化
- ✅ 模型切换流畅

---

#### 任务 3.2: 自适应策略（Week 4-5，5天）

**智能选择深度后端**:
```python
def recommend_depth_backend(n_images, scene_type, quality_mode):
    """推荐深度后端
    
    Args:
        n_images: 图片数量
        scene_type: 场景类型 (indoor, outdoor, object)
        quality_mode: 质量模式 (fast, balanced, quality)
    
    Returns:
        backend: 推荐后端
        config: 配置参数
    """
    
    if n_images == 1:
        # 单图：必须用 DA-V2
        return "depth_anything", {"model_size": "base"}
    
    elif n_images <= 3:
        # 极少视角：混合模式
        if quality_mode == "quality":
            return "hybrid", {
                "model_size": "large",
                "blend_mode": "adaptive"
            }
        else:
            return "hybrid", {
                "model_size": "small",
                "blend_mode": "weighted"
            }
    
    elif n_images <= 10:
        # 中等视角：根据质量模式
        if quality_mode == "fast":
            return "mast3r", {}
        elif quality_mode == "quality":
            return "hybrid", {
                "model_size": "base",
                "blend_mode": "adaptive"
            }
        else:
            return "mast3r", {}
    
    else:
        # 密集视角：MASt3R 足够
        return "mast3r", {}
```

**集成到推荐API**:
```python
@app.post("/train/recommend")
def recommend_params(...):
    ...
    
    # 推荐深度后端
    depth_backend, depth_config = recommend_depth_backend(
        n_images=n_images,
        scene_type="auto",  # TODO: 场景分类
        quality_mode="balanced"
    )
    
    recommended["depth_backend"] = depth_backend
    recommended["depth_config"] = depth_config
    
    return recommended
```

---

#### 任务 3.3: 文档和示例（Week 5，3天）

**用户文档**:
1. API 参数说明
2. 深度后端选择指南
3. 最佳实践
4. FAQ

**示例代码**:
```python
# examples/use_depth_anything.py
"""使用 Depth-Anything-V2 提升质量"""

# 1. 混合模式（推荐）
curl -X POST http://localhost:12524/train \
  -F "file=@photos.zip" \
  -F "sfm_backend=mast3r" \
  -F "depth_backend=hybrid" \
  -F "enable_depth_reg=true" \
  -F "depth_reg_weight=0.02"

# 2. 纯 DA-V2 模式
curl -X POST http://localhost:12524/train \
  -F "file=@photos.zip" \
  -F "sfm_backend=mast3r" \
  -F "depth_backend=depth_anything" \
  -F "enable_depth_reg=true" \
  -F "depth_reg_weight=0.02"

# 3. 单图重建
curl -X POST http://localhost:12524/train/monocular \
  -F "file=@single_image.jpg" \
  -F "iterations=3000" \
  -F "assume_scale=1.0"
```

---

#### 任务 3.4: 生产部署（Week 6，3-5天）

**部署检查清单**:
- [ ] 单元测试覆盖率 > 80%
- [ ] 集成测试通过
- [ ] 性能测试达标
- [ ] 文档完整
- [ ] 回滚方案就绪
- [ ] 监控告警配置

**灰度发布**:
```python
# 配置灰度比例
DEPTH_ANYTHING_ROLLOUT_PERCENT = 20  # 20% 用户

def should_use_depth_anything(user_id):
    return hash(user_id) % 100 < DEPTH_ANYTHING_ROLLOUT_PERCENT
```

**监控指标**:
- 深度后端使用分布
- 平均推理时间
- 错误率
- 用户反馈评分

---

## 📊 里程碑和验收

### Milestone 1: Week 1 结束
- [ ] DA-V2 vs MASt3R 对比完成
- [ ] 单图重建原型可用
- [ ] 决策：是否继续 Phase 2

### Milestone 2: Week 3 结束
- [ ] 混合深度实现完成
- [ ] Phase 2 重测完成
- [ ] 至少一个配置优于 Phase 1

### Milestone 3: Week 6 结束
- [ ] 性能优化完成
- [ ] 自适应策略实现
- [ ] 文档完整
- [ ] 生产部署就绪

---

## 📈 成功指标

### 质量指标
- [ ] Phase 2 Loss ≤ 0.00479 (Phase 1 baseline)
- [ ] 深度细节提升 20-30%（主观评估）
- [ ] 边缘清晰度 > MASt3R

### 性能指标
- [ ] 推理时间增加 ≤ 20%
- [ ] 显存增加 ≤ 1GB
- [ ] 批处理吞吐量 > 2 images/sec

### 功能指标
- [ ] 单图重建可用
- [ ] 三种深度后端可切换
- [ ] 自适应推荐准确

---

## 🚨 风险和应对

### 风险 1: 深度对齐不准确
**概率**: 中  
**影响**: 高  
**应对**:
- 多种对齐算法备选
- 可视化验证
- 降级到纯 MASt3R

### 风险 2: 性能开销过大
**概率**: 低  
**影响**: 中  
**应对**:
- 选择 Small 模型
- 批处理优化
- 可选功能（用户自主选择）

### 风险 3: 质量提升不明显
**概率**: 中  
**影响**: 高  
**应对**:
- Week 1 快速验证
- 及时止损
- 作为单图场景补充仍有价值

---

## 💰 资源需求

### 人力
- **开发**: 1 人 × 6 周 = 6 人周
- **测试**: 0.5 人 × 2 周 = 1 人周
- **总计**: 7 人周

### 计算资源
- **开发测试**: 1 × GPU (8GB+)
- **模型存储**: ~2GB (3 个模型)
- **测试数据**: ~10GB

### 依赖
- `transformers` >= 4.35.0
- `torch` >= 2.0.0
- `pillow` >= 10.0.0

---

## 📅 时间表

```
Week 1 (Day 1-5):  环境准备 + 质量对比 + 单图原型 + 决策
                   ├─ Day 1: 环境+模型测试
                   ├─ Day 2: 深度质量对比
                   ├─ Day 3: 单图重建实现
                   ├─ Day 4: Week 1 总结
                   └─ Day 5: Go/No-Go 决策

Week 2 (Day 6-10): 深度对齐 + 混合策略
                   ├─ Day 6-8: 对齐算法实现
                   ├─ Day 9-10: 混合策略实现
                   
Week 3 (Day 11-15): 集成训练 + 重新测试
                    ├─ Day 11-13: Pipeline 集成
                    ├─ Day 14-15: Phase 2 重测
                    
Week 4 (Day 16-20): 性能优化 + 自适应策略
                    ├─ Day 16-18: 性能优化
                    ├─ Day 19-20: 自适应策略
                    
Week 5 (Day 21-25): 文档 + 测试
                    ├─ Day 21-23: 文档编写
                    ├─ Day 24-25: 集成测试
                    
Week 6 (Day 26-30): 生产部署
                    ├─ Day 26-27: 部署准备
                    ├─ Day 28: 灰度发布
                    ├─ Day 29-30: 监控和调整
```

---

## ✅ 验收标准

### 最低要求（MVP）
- [ ] 单图重建功能可用
- [ ] 混合深度模式实现
- [ ] API 参数可用
- [ ] 基本文档完整

### 理想目标
- [ ] Phase 2 Loss 优于 Phase 1
- [ ] 深度细节显著提升
- [ ] 性能开销可接受（≤20%）
- [ ] 完整文档和示例
- [ ] 生产部署就绪

---

## 📞 联系和支持

**项目负责人**: Claude  
**开始日期**: 2026-09-05 (明天)  
**预计完成**: 2026-10-16 (6周后)

---

**下一步**: 明天开始 Week 1 Day 1 - 环境准备！🚀
