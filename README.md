# 3DGS 训练服务

基于 FlexGaussian 的 3D Gaussian Splatting 训练服务，支持多视角照片和视频输入，提供完整的 Web 管理界面。

---

## 🚀 快速开始

### 访问地址

- **前端管理界面**: http://192.168.10.148:3002
- **后端 API**: http://192.168.10.148:12524
- **API 文档**: http://192.168.10.148:12524/docs

### 启动服务

```bash
# 启动后端（Docker）
cd /home/3dgs
cp .env.example .env  # 首次部署时填写实际 API_KEY 和 Redis 地址
docker-compose up -d

# 启动前端
cd /home/3dgs/frontend
npm install
npm run dev -- --host 0.0.0.0 --port 3002
```

---

## ✨ 核心功能

### 1. 多视角照片训练
- 支持 ZIP 压缩包上传（包含多角度拍摄的图像序列）
- 自动 COLMAP/MASt3R 特征提取和相机位姿估计
- 灵活的训练参数配置（迭代次数、分辨率、场景类型）
- 实时训练进度和日志显示

### 2. 视频自动抽帧
- 支持格式：MP4、AVI、MOV、MKV、WebM
- 三种抽帧模式：
  - **FPS 模式**：按帧率抽取（如每秒 5 帧）
  - **时间间隔模式**：每隔 X 秒截取一帧（如 0.3 秒）
  - **总帧数模式**：指定提取总帧数（如 100 帧）
- 智能过滤：去除模糊帧和重复帧
- 时间范围裁剪：指定视频的起止时间

### 3. 模型压缩优化
- 支持多种压缩算法：
  - LightGaussian
  - Mini-Splatting
  - Compact3D
- 可配置压缩级别（低/中/高）
- 质量优先模式
- 压缩前后对比（文件大小、PSNR）

### 4. 去噪与质量优化
- 不透明度过滤：移除低质量高斯点
- 尺寸过滤：清理过小/过大的高斯点
- 梯度裁剪：训练稳定性优化
- 深度正则化：提升几何一致性
- 抗锯齿：改善渲染质量

### 5. Web 管理界面
- 科技风格 UI（深色主题 + 霓虹蓝/紫配色）
- 三大模块：生成、优化、查看
- 实时进度显示和任务管理
- 模型列表和详情查看

---

## 🏗️ 技术架构

### 后端技术栈
- **框架**: FastAPI 0.109.0
- **任务队列**: Celery + Redis
- **容器化**: Docker + docker-compose
- **GPU 加速**: CUDA 12.8.1 + NVIDIA Container Toolkit

### 前端技术栈
- **框架**: Vue 3.4.0
- **UI 组件**: Element Plus 2.5.0
- **构建工具**: Vite 5.0.0
- **状态管理**: Pinia 2.1.7
- **路由**: Vue Router 4.2.5

### 训练引擎
- **3DGS 实现**: FlexGaussian（自适应参数、去噪优化）
- **SfM 后端**:
  - COLMAP（密集拍摄，20-500 张）
  - MASt3R（少视角，2-15 张）
- **视频处理**: FFmpeg

---

## 📁 项目结构

```
/home/3dgs/
├── README.md                  # 本文档
├── API使用手册.md              # 后端 API 详细文档
├── 项目整合说明.md             # 前后端整合方案
├── 视频训练设计分析.md         # 视频抽帧设计与优化建议
│
├── app/                       # 后端服务
│   ├── main.py               # FastAPI 主应用
│   ├── management_api.py     # 前端管理 API
│   ├── pipeline.py           # 训练流程
│   ├── trainer.py            # 训练器
│   └── config.py             # 配置管理
│
├── frontend/                  # 前端管理界面
│   ├── src/
│   │   ├── views/            # 页面组件
│   │   │   ├── Generate.vue  # 模型生成页面
│   │   │   ├── Optimize.vue  # 压缩优化页面
│   │   │   └── Viewer.vue    # 模型查看页面
│   │   ├── api/              # API 封装
│   │   └── router/           # 路由配置
│   ├── 去噪优化指南.md
│   ├── 视频抽帧指南.md
│   └── 后端接口参考.md
│
├── FlexGaussian/             # 独立维护的外部源码仓库，不纳入本仓库
├── data/                     # 数据存储目录
│   ├── tasks/               # 训练任务数据
│   ├── models/              # 生成的模型
│   └── uploads/             # 上传文件
│
├── docs/                     # 文档目录
│   ├── 核心设计/
│   │   ├── 功能设计文档.md
│   │   └── 动态3DGS实施方案.md
│   ├── 升级规划/
│   │   └── 后续升级规划.md
│   └── 归档/
│       ├── 已完成优化/
│       ├── 阶段性实施/
│       └── 实验性功能/
│
├── docker-compose.yml        # Docker 编排配置
├── Dockerfile               # Docker 镜像构建
├── .env.example              # 不含密钥的配置模板
└── .env                      # 本地环境变量配置，不纳入 Git
```

---

## 📖 文档导航

### 使用文档
- [API 使用手册](./API使用手册.md) - 后端 API 完整参考
- [视频抽帧指南](./frontend/视频抽帧指南.md) - 视频输入使用说明
- [去噪优化指南](./frontend/去噪优化指南.md) - 质量优化参数说明

### 设计文档
- [功能设计文档](./docs/核心设计/功能设计文档.md) - 核心功能设计
- [动态 3DGS 实施方案](./docs/核心设计/动态3DGS实施方案.md) - 完整实施方案
- [视频训练设计分析](./视频训练设计分析.md) - 视频功能优化建议

### 升级规划
- [后续升级规划](./docs/升级规划/后续升级规划.md) - 功能路线图
- [文档整理方案](./docs/文档整理方案.md) - 文档管理规范

---

## 🎯 典型使用流程

### 方式 1：照片训练

1. 准备多角度拍摄的照片（建议 50-150 张）
2. 打包为 ZIP 压缩包
3. 访问前端界面，上传 ZIP 文件
4. 配置训练参数（或使用推荐参数）
5. 启动训练，实时查看进度
6. 训练完成后下载或在线查看模型

### 方式 2：视频训练

1. 录制环绕拍摄视频（建议 30-60 秒）
2. 访问前端界面，上传视频文件
3. 配置抽帧参数：
   - 选择抽帧间隔（推荐 0.3-0.5 秒）
   - 开启去模糊和去重
4. 系统自动抽帧并开始训练
5. 查看训练进度和最终模型

### 方式 3：API 调用

```bash
# 提交训练任务
curl -X POST http://192.168.10.148:12524/train \
  -H "X-API-Key: your_key" \
  -F "file=@photos.zip" \
  -F "sfm_backend=colmap" \
  -F "iterations=7000"

# 查询任务状态
curl http://192.168.10.148:12524/task/{task_id}

# 下载结果
curl -O http://192.168.10.148:12524/task/{task_id}/result
```

---

## ⚙️ 配置说明

### 环境变量（.env）

```bash
# API 密钥
API_KEY=your_secret_key

# SfM 后端选择
SFM_BACKEND=colmap  # 或 mast3r

# GPU 配置
COLMAP_USE_GPU=1

# 训练参数
MAX_ITERATIONS=30000
MIN_IMAGES=10
MAX_IMAGES=500
```

### Docker 配置

- **API 端口**: 12524 (映射到容器内 8000)
- **GPU 支持**: 需要 NVIDIA Container Toolkit
- **数据挂载**: `/home/3dgs/data` → `/data`

---

## 🔧 故障排查

### 前端无法连接后端

检查后端服务状态：
```bash
docker ps | grep 3dgs-api
docker logs 3dgs-api
```

### 训练失败

查看详细日志：
```bash
docker logs 3dgs-worker
```

常见问题：
- 图片数量不足（需至少 10 张）
- 图片质量差（模糊、曝光不足）
- GPU 内存不足（尝试降低分辨率）

### 视频抽帧失败

检查 FFmpeg 是否可用：
```bash
docker exec 3dgs-api ffmpeg -version
```

---

## 📊 性能指标

### 训练时间（参考）

| 场景类型 | 图片数量 | 分辨率 | 迭代次数 | 预计时间 |
|---------|---------|--------|---------|---------|
| 小物体 | 50 张 | 1/2 | 7000 | ~15 分钟 |
| 室内场景 | 100 张 | 1/2 | 10000 | ~30 分钟 |
| 室外场景 | 200 张 | 1/4 | 15000 | ~45 分钟 |

*基于 NVIDIA RTX 3090 测试

### 系统要求

- **最低配置**:
  - GPU: 8GB 显存（CUDA 11.8+）
  - RAM: 16GB
  - 磁盘: 50GB 可用空间

- **推荐配置**:
  - GPU: 24GB 显存（RTX 3090/4090）
  - RAM: 32GB
  - 磁盘: 100GB SSD

---

## 🤝 贡献与反馈

如有问题或建议，请通过以下方式联系：

- 查看 [API 使用手册](./API使用手册.md)
- 阅读 [视频训练设计分析](./视频训练设计分析.md) 了解优化建议
- 查看 Docker 容器日志排查问题

---

## 📄 许可证

本项目基于 FlexGaussian 开发。详见 [FlexGaussian/LICENSE.md](./FlexGaussian/LICENSE.md)。

---

**最后更新**: 2026-09-20
