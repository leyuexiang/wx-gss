# 3DGS 训练服务接口文档

版本：1.2.0 ｜ 更新：2026-09-03

多视角照片 → 稀疏重建（COLMAP 或 MASt3R 两种可选后端）→ gsplat 训练 → 导出 3DGS 模型的异步任务服务。

## 快速开始

### 1. 准备照片

根据拍摄条件选择合适的 SfM 后端：

**密集拍摄（推荐 COLMAP）**：
- 20-500 张照片
- 相邻照片重叠度 ≥ 60%
- 环绕物体连续拍摄
- 纹理清晰、避免运动模糊/强反光

**少视角拍摄（使用 MASt3R）**：
- 2-15 张照片（典型 5-10 张）
- 对重叠度要求较低
- 适合无法密集拍摄的场景

将照片打包为 `.zip` 文件（支持 jpg/jpeg/png/webp/bmp/tif/tiff）。

### 2. 提交训练任务

**COLMAP 示例**（密集拍摄）：
```bash
curl -X POST http://192.168.10.148:12524/train \
  -H "X-API-Key: YOUR_API_KEY" \
  -F "file=@photos.zip" \
  -F "iterations=7000" \
  -F "downsample=4"
```

**MASt3R 示例**（少视角）：
```bash
curl -X POST http://192.168.10.148:12524/train \
  -H "X-API-Key: YOUR_API_KEY" \
  -F "file=@photos.zip" \
  -F "sfm_backend=mast3r" \
  -F "iterations=3000" \
  -F "downsample=2"
```

返回示例：
```json
{"task_id": "9f1c2a3b4d5e6f...", "status": "queued"}
```

### 3. 查询任务进度

每 3-5 秒轮询一次：
```bash
curl -H "X-API-Key: YOUR_API_KEY" \
  http://192.168.10.148:12524/train/9f1c2a3b4d5e6f...
```

返回示例：
```json
{
  "task_id": "9f1c2a3b4d5e6f...",
  "status": "training",
  "progress": 0.62,
  "message": "训练 4200/7000 loss=0.0412 高斯数=812334 显存=5.3GB"
}
```

### 4. 下载结果

当 `status` 变为 `done` 时：
```bash
curl -H "X-API-Key: YOUR_API_KEY" \
  -OJ http://192.168.10.148:12524/train/9f1c2a3b4d5e6f.../result
```

结果文件包含：
- `point_cloud.ply` - Inria 标准 3DGS 格式（可用于 SuperSplat、PlayCanvas、Three.js）
- `point_cloud.splat` - Web 轻量查看器格式
- `train_stats.json` - 训练统计信息

### 5. 快速测试

检查服务健康状态：
```bash
curl http://192.168.10.148:12524/health
```

访问交互式文档：
```
http://192.168.10.148:12524/docs
```

---

## 基本信息

| 项 | 值 |
|---|---|
| Base URL | `http://192.168.10.148:12524`（局域网，端口已绑 0.0.0.0，防火墙无限制） |
| 认证 | 请求头 `X-API-Key: <key>`（`/health` 除外） |
| 交互式文档 | `http://<host>:12524/docs`（Swagger UI）、`/openapi.json` |
| 任务模型 | 异步：提交后轮询状态，完成后下载结果 |
| 并发 | worker concurrency=1，任务串行执行；单任务硬超时 4 小时 |

## SfM 后端选择：colmap / mast3r

`/train` 接口通过 `sfm_backend` 参数选择前端重建方式，两种后端共用同一套鉴权/异步任务/状态轮询/结果下载基础设施，训练侧（gsplat）完全一致。

| | `colmap`（默认） | `mast3r` |
|---|---|---|
| 图片数量 | 20–500 张 | 2–15 张 |
| 适用场景 | 密集环绕拍摄 | 少视角（5–10 张典型） |
| 位姿精度 | 更优，OPENCV 模型处理镜头畸变 | 针孔假设，忽略广角畸变，边缘可能有几何误差 |
| 求解耗时 | 分钟级（特征匹配+增量 BA） | 秒级~分钟级（前馈神经网络） |
| 失败模式 | 重叠不足直接报错（如 `No good initial image pair found`） | 5 张图 COLMAP 完全无法进入训练的场景，MASt3R 可以出结果 |

选型建议：能拍 20 张以上、条件允许环绕连续拍摄，优先用 `colmap`（位姿精度更高）；只有 5–10 张少视角照片时用 `mast3r`。技术选型和 Phase 0 POC 验证记录见 [`docs/mast3r-migration.md`](docs/mast3r-migration.md)（TestIMG 5 张实拍图：位姿求解 13.9s/2.95GB，训练 3000 iter/261,123 高斯/loss 0.0166/140.6s/0.61GB）。

---

## 状态机

```
queued → colmap → training → packaging → done
      ↘ mast3r ↗                    ↘ failed
```

- `status` 为 `colmap / mast3r / training / packaging` 时表示运行中，此时不可删除任务（409）。
- `failed` 时 `error` 字段给出原因；显存不足会自动附带调整建议。

## 接口一览

### 1. GET /health — 健康检查（无需认证）

```bash
curl http://localhost:12524/health
```

```json
{ "status": "ok", "redis": true, "disk_free_gb": 719.0 }
```

`status`：`ok` / `degraded`（redis 不通时）。

---

### 2. POST /train — 提交训练任务

`multipart/form-data`

| 字段 | 类型 | 必填 | 默认 | 约束 |
|---|---|---|---|---|
| `file` | file | ✅ | — | `.zip`，内含多视角照片（jpg/jpeg/png/webp/bmp/tif/tiff），数量要求见下表 |
| `sfm_backend` | string | 否 | `colmap` | `colmap` / `mast3r` |
| `iterations` | int | 否 | 7000 | [500, 30000] |
| `downsample` | int | 否 | 4 | 1 / 2 / 4 / 8（图片降采样倍数，显存不足时调大） |
| `cap_max` | int | 否 | 1000000 | [100000, 1500000]（高斯数量上限） |

图片数量要求随 `sfm_backend` 不同：

| `sfm_backend` | 数量范围 |
|---|---|
| `colmap` | 20–500 张 |
| `mast3r` | 2–15 张 |

COLMAP 密集拍摄示例：

```bash
curl -X POST http://localhost:12524/train \
  -H "X-API-Key: $API_KEY" \
  -F "file=@photos.zip" \
  -F "iterations=7000" -F "downsample=4" -F "cap_max=1000000"
```

MASt3R 少视角示例（图片少、高斯数天然更少，`iterations`/`cap_max` 建议调小，训练更快）：

```bash
curl -X POST http://localhost:12524/train \
  -H "X-API-Key: $API_KEY" \
  -F "file=@photos.zip" -F "sfm_backend=mast3r" \
  -F "iterations=3000" -F "downsample=2" -F "cap_max=500000"
```

响应 `200`：

```json
{ "task_id": "9f1c2a...", "status": "queued" }
```

**COLMAP 图片拍摄要求**（COLMAP 重建成败的关键）：

- **数量 ≥ 20 张**，实拍建议 30 张以上。这不是 3DGS 本身的限制，而是前端 COLMAP（SIFT 特征匹配 + SfM 三角化）的要求：它必须靠相邻视图间的大量匹配特征点来反推相机位姿和稀疏点云，图片太少、视角跨度过大时匹配不上，重建直接失败（典型报错：`No good initial image pair found`、注册图片不足）
- 相邻照片重叠度 ≥ 60%，环绕物体连续拍摄
- 纹理清晰、避免运动模糊 / 强反光 / 纯色无纹理表面
- 重建后注册图片数需 ≥ 50%，否则任务失败并提示重拍

**MASt3R 图片拍摄要求**：

- 数量 2–15 张（上限为显存约束下的保守值，全连接场景图关联对数随张数平方增长）
- 不做去畸变（针孔假设），广角镜头/强畸变照片边缘可能有几何误差，密集场景仍建议用 `colmap`
- 其余拍摄常识（避免运动模糊、纹理清晰）同样适用，但对重叠度的要求远低于 COLMAP

> 市面上有些产品号称"6 张图即可生成 3DGS"，用的就是 MASt3R 同类的学习类方法（DUSt3R/MASt3R、pixelSplat/MVSplat 等大重建模型，或带单目深度/扩散先验的 few-shot 方法）直接预测位姿和几何，绕开了传统特征匹配。代价是未观测区域靠"脑补"，细节和泛化性不如密集拍摄的经典管线——这也是本服务把两种后端都保留、而不是相互替代的原因。

---

### 3. GET /train/{task_id} — 查询任务状态

```bash
curl -H "X-API-Key: $API_KEY" http://localhost:12524/train/9f1c2a...
```

```json
{
  "task_id": "9f1c2a...",
  "status": "training",
  "stage": "training",
  "progress": 0.62,
  "message": "训练 4200/7000 loss=0.0412 高斯数=812334 显存=5.3GB",
  "error": null,
  "params": { "iterations": 7000, "downsample": 4, "cap_max": 1000000, "sfm_backend": "colmap" },
  "created_at": "2026-08-25T16:00:00",
  "updated_at": "2026-08-25T16:12:30"
}
```

| 字段 | 说明 |
|---|---|
| `status` | queued / colmap / mast3r / training / packaging / done / failed |
| `stage` | COLMAP 路径：extract → feature_extractor → matcher → mapper → undistort → colmap_done → training → packaging；MASt3R 路径：extract → mast3r_load → mast3r_align → mast3r_export → mast3r_done → training → packaging |
| `progress` | 0.0–1.0（SfM 段约占 0–0.4，训练段 0.4–0.95） |
| `message` | 当前进度描述，训练期含 loss / 高斯数 / 显存 |
| `error` | 失败原因（无错为 null） |

建议轮询间隔 3–5 秒。

---

### 4. GET /train/{task_id}/result — 下载结果

仅 `status=done` 可下载，否则返回 `409`。

```bash
curl -H "X-API-Key: $API_KEY" -OJ http://localhost:12524/train/9f1c2a.../result
# 保存为 3dgs_9f1c2a....zip
```

**result.zip 内容**：

| 文件 | 格式 | 用途 |
|---|---|---|
| `point_cloud.ply` | Inria 标准 3DGS PLY（binary_little_endian，含 f_dc/f_rest/opacity/scale/rot） | SuperSplat、PlayCanvas、Three.js 编辑器 |
| `point_cloud.splat` | antimatter15 .splat（32 字节/高斯：pos12 + scale12 + rgba4 + rot4） | Web 端轻量查看器 |
| `train_stats.json` | 训练统计 | 质量评估 |

`train_stats.json` 示例（COLMAP，密集场景）：

```json
{
  "iterations": 7000,
  "num_gaussians": 812334,
  "final_loss": 0.03812,
  "train_seconds": 912.5,
  "peak_gpu_mem_gb": 5.4,
  "n_images": 86,
  "downsample": 4
}
```

`train_stats.json` 示例（MASt3R，少视角，对应 TestIMG 5 张图）：

```json
{
  "iterations": 3000,
  "num_gaussians": 261123,
  "final_loss": 0.01657,
  "train_seconds": 140.6,
  "peak_gpu_mem_gb": 0.61,
  "n_images": 5,
  "downsample": 2
}
```

---

### 5. DELETE /train/{task_id} — 删除任务

运行中（colmap/mast3r/training/packaging）返回 `409`；成功返回 `{"deleted": "<task_id>"}`。

```bash
curl -X DELETE -H "X-API-Key: $API_KEY" http://localhost:12524/train/9f1c2a...
```

---

## 错误码

| 码 | 场景 |
|---|---|
| 400 | 参数越界 / 非 zip / task_id 非法 / `sfm_backend` 取值非法 |
| 401 | 缺少或错误的 `X-API-Key` |
| 404 | 任务不存在 / 结果文件缺失 |
| 409 | 任务未完成就下载结果，或运行中删除 |

### 401 排障

**401 表示请求在鉴权阶段被拒，任务根本没有创建**（`/data/tasks/` 下不会有新目录），不是训练失败。排查：

1. 确认请求带了 `X-API-Key` 头（注意是**请求头**，不是 form 字段、不是 query 参数，也不是 `Authorization: Bearer`）。
2. 确认 key 值与宿主机 `/home/3dgs/.env` 里的 `API_KEY=` 完全一致——注意调用方配置里不要带多余空格、引号或换行。
3. 用 curl 自验（把 `$API_KEY` 换成 .env 里的值）：

```bash
curl -X POST http://<host>:12524/train \
  -H "X-API-Key: $API_KEY" \
  -F "file=@photos.zip"
# 返回 {"task_id": "...", "status": "queued"} 即鉴权通过
```

## 常见失败原因（error 字段）

| 提示 | 含义与对策 |
|---|---|
| 图片数量不足 / 过多 | `colmap` 需 20–500 张，`mast3r` 需 2–15 张 |
| 重建失败：重叠度不足或纹理太少 | 重拍，增大重叠、改善光照（COLMAP 路径） |
| 注册图片不足 | 注册率 < 50%，同上（COLMAP 路径） |
| COLMAP 稀疏点过少 | 重建点 < 100，同上 |
| MASt3R 重建点过少 | 图片重叠度可能不足，即便少视角也需要基本的视野重叠 |
| 显存不足 | 增大 `downsample`（4→8）或减小 `cap_max` / `iterations` |

## 附注

- GPU 开关：worker 的 `.env` 中 `COLMAP_USE_GPU=1/0`（默认 1，GPU SIFT 比 CPU 快 4–7 倍），改后 `docker compose up -d` 生效。
- SfM 后端默认值：worker 的 `.env` 中 `SFM_BACKEND=colmap`（未设置时的默认值），单次请求可用 `sfm_backend` 表单字段覆盖，无需重启服务切换。
- 每个任务的中间产物（COLMAP 日志/MASt3R 缓存、数据集、输出）保留在宿主机 `/home/3dgs/data/tasks/<task_id>/`，`colmap/colmap.log` 可用于排查 COLMAP 重建问题。

