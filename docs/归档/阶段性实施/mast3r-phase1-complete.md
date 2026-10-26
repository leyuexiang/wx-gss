# MASt3R Phase 1 集成完成报告

**日期**: 2026-09-04  
**状态**: ✅ 已完成  

## 概述

MASt3R 作为可切换的 SfM 后端已成功集成到 3DGS 训练服务中。用户可通过配置或 API 参数在 COLMAP 和 MASt3R 之间切换。

## 已完成的工作

### 1. 配置层 (`config.py`)

**已添加配置项**:
```python
SFM_BACKEND = os.environ.get("SFM_BACKEND", "colmap")
MAST3R_MIN_IMAGES = 2
MAST3R_MAX_IMAGES = 15
MAST3R_REPO_DIR = os.path.join(DATA_DIR, "poc/mast3r/mast3r")
MAST3R_CKPT = os.path.join(
    DATA_DIR, "poc/mast3r/checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth"
)
```

**配置文件 (`.env`)**:
```bash
SFM_BACKEND=colmap  # 默认使用 COLMAP
```

### 2. Pipeline 层 (`pipeline.py`)

**核心函数**:
- `_run_mast3r()` (277-318行): MASt3R 位姿求解主流程
- `_export_mast3r_dataset()` (211-274行): MASt3R → COLMAP bin 格式适配器
- `_write_colmap_bins()` (185-208行): 手写 COLMAP legacy bin 格式

**智能路由**:
```python
if sfm_backend == "mast3r":
    files = _extract_images(task_dir, MAST3R_MIN_IMAGES, MAST3R_MAX_IMAGES)
    dataset_dir, n_registered, ratio = _run_mast3r(task_dir, files, downsample, status_cb)
else:
    files = _extract_images(task_dir, MIN_IMAGES, MAX_IMAGES)
    dense_dir = _run_colmap(task_dir, len(files), status_cb)
    dataset_dir, n_registered, ratio = _prepare_dataset(...)
```

### 3. API 层 (`main.py`)

**POST /train 接口已支持**:
```python
sfm_backend: Literal["colmap", "mast3r"] = Form(
    default=config.SFM_BACKEND,
    description="SfM 后端选择：colmap（密集拍摄，20-500张）或 mast3r（少视角，2-15张）"
)
```

**请求参数验证**:
- `mast3r`: 图片数量 2-15 张
- `colmap`: 图片数量 20-500 张

**POST /train/recommend 接口**:
- 根据 `sfm_backend` 返回自适应参数推荐

## 集成测试结果

### 测试配置
- **数据集**: TestIMG.zip (5 张实拍照片)
- **后端**: mast3r
- **参数**: iterations=1000, downsample=4, cap_max=100000

### 测试结果
```json
{
  "iterations": 1000,
  "num_gaussians": 145408,
  "final_loss": 0.02205,
  "train_seconds": 16.7,
  "peak_gpu_mem_gb": 2.95,
  "quality_score": 92.0,
  "quality_rating": "优秀"
}
```

**性能指标**:
- ✅ 位姿求解成功 (5/5 张图片)
- ✅ 训练收敛 (loss=0.022)
- ✅ 高斯数量合理 (145K)
- ✅ 显存使用安全 (2.95GB / 8GB)
- ✅ 训练速度快 (59.88 iter/s)
- ✅ 输出文件完整 (point_cloud.ply, point_cloud.splat, train_stats.json)

## API 使用示例

### 少视角场景 (MASt3R)
```bash
curl -X POST http://host:12524/train \
  -H "X-API-Key: your_key" \
  -F "file=@photos.zip" \
  -F "sfm_backend=mast3r" \
  -F "iterations=3000" \
  -F "downsample=2"
```

### 密集拍摄场景 (COLMAP)
```bash
curl -X POST http://host:12524/train \
  -H "X-API-Key: your_key" \
  -F "file=@photos.zip" \
  -F "sfm_backend=colmap" \
  -F "iterations=7000" \
  -F "downsample=4"
```

### 自适应参数推荐
```bash
curl -X POST http://host:12524/train/recommend \
  -F "file=@photos.zip" \
  -F "sfm_backend=mast3r"
```

## 技术细节

### 1. 数据流水线契约
MASt3R 和 COLMAP 产出相同的数据集布局:
```
dataset/
  sparse/0/
    cameras.bin    # PINHOLE 相机内参
    images.bin     # 位姿 (qvec, tvec)
    points3D.bin   # 稀疏点云
  images_{factor}/ # 降采样图片
```

训练器 (`trainer.py`) 无需修改，透明支持两种后端。

### 2. 场景图策略
- **5-20 张图**: `complete` 全连接图 (显存 <4GB)
- **>20 张图**: 需切换到 `swin` 窗口图或 `retrieval` 防 OOM

当前实现使用 `complete`，适合当前的 2-15 张限制。

### 3. 适配器实现
`_export_mast3r_dataset()` 关键步骤:
1. 将 MASt3R 的 512 尺度内参换算到原图分辨率
2. cam2world → world2cam 转换
3. 旋转矩阵投影到 SO(3) (去数值漂移)
4. 反投影深度图生成稀疏点云 (带 RGB)
5. 手写 COLMAP legacy bin 格式

### 4. 显存优化
- MASt3R 位姿求解完成后立即释放模型和缓存
- 训练前显存清空: `torch.cuda.empty_cache()`
- 5 张图: 峰值 2.95GB (训练阶段)

## 已知限制

### 1. 图片数量
- **MASt3R**: 2-15 张 (硬限制，complete 场景图在 15 张时显存约 3-4GB)
- **COLMAP**: 20-500 张

### 2. 相机模型
- MASt3R 假设 **针孔模型** (无畸变)
- COLMAP 支持 **OPENCV 模型** (处理镜头畸变)

→ 手机广角拍摄建议仍用 COLMAP

### 3. 权重大小
- MASt3R 权重: 2.75GB (需预加载到 `/data/poc/mast3r/checkpoints/`)

## 后续工作 (Phase 2)

Phase 1 仅解决"位姿解不出来"的问题。要提升少视角重建质量，还需:

### Phase 2 目标
1. **深度正则化**: 训练时引入 MASt3R 深度图作为几何先验
2. **位姿联合优化**: 参考 InstantSplat 的 GauBA 式位姿微调
3. **质量基准**: 5-10 张图的 PSNR/SSIM 对比 (Phase 1 vs Phase 2)

预计工作量: 1-2 周

## 参考资料

- [评估文档](./mast3r-migration.md) - 技术选型和分期计划
- [Phase 0 验证](../memory/mast3r-phase0.md) - POC 验证记录
- [API 使用手册](../API使用手册.md) - 完整 API 文档

---

**集成负责人**: Claude  
**验证环境**: RTX 4060 8GB, CUDA 12.8, torch 2.8.0  
**Docker 镜像**: 3dgs:latest (commit a69d8e8)
