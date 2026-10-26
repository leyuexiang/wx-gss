# 动态 3DGS 实施方案

版本：1.0 | 创建：2026-09-03

---

## 目录

- [1. 方案概述](#1-方案概述)
- [2. 技术选型](#2-技术选型)
- [3. 阶段一：POC 验证（1-2周）](#3-阶段一poc-验证1-2周)
- [4. 阶段二：原型开发（2-3周）](#4-阶段二原型开发2-3周)
- [5. 阶段三：服务集成（3-4周）](#5-阶段三服务集成3-4周)
- [6. 阶段四：优化上线（2-3周）](#6-阶段四优化上线2-3周)
- [7. 技术实现细节](#7-技术实现细节)
- [8. 测试与验证](#8-测试与验证)
- [9. 风险管理](#9-风险管理)

---

## 1. 方案概述

### 1.1 项目目标

在现有静态 3DGS 训练服务基础上，增加**动态场景重建**能力，支持：

- ✅ 视频输入 → 动态 3D 模型
- ✅ 多时间点渲染
- ✅ 动画导出和播放
- ✅ Web 端交互查看

### 1.2 应用场景

| 场景 | 输入 | 输出 | 用途 |
|------|------|------|------|
| **人物动作** | 跳舞/走路视频 | 可交互 3D 动画 | 虚拟偶像、动捕 |
| **产品展示** | 旋转展示视频 | 360° 动态模型 | 电商、广告 |
| **运动捕捉** | 多视角运动视频 | 动态姿态序列 | 体育分析、健身 |
| **自然现象** | 水流/火焰视频 | 动态特效 | 影视、游戏 |

### 1.3 技术路线

根据运动类型选择不同方案：

```
输入视频
    ↓
运动类型判断
    ├─ 刚体运动（旋转/平移）→ Deformation Field（快速）
    ├─ 柔性运动（人物/布料）→ 4DGaussians（高质量）
    └─ 慢速运动（简单场景）→ 多帧插值（最简单）
```

---

## 2. 技术选型

### 2.1 方案对比

| 方案 | 复杂度 | 训练时间 | 质量 | 适用场景 | 推荐度 |
|------|--------|----------|------|----------|--------|
| **4DGaussians** | ⭐⭐⭐⭐⭐ | 30-60 分钟 | ⭐⭐⭐⭐⭐ | 柔性运动 | 🔥 P0 |
| **SC-GS** | ⭐⭐⭐⭐ | 20-40 分钟 | ⭐⭐⭐⭐ | 刚体运动 | 🔸 P1 |
| **Deformation Field** | ⭐⭐⭐ | 15-30 分钟 | ⭐⭐⭐⭐ | 简单动态 | 🔸 P1 |
| **多帧插值** | ⭐⭐ | 10-20 分钟 | ⭐⭐⭐ | 慢速运动 | 🔹 P2 |

### 2.2 最终选型

**主方案**: 4DGaussians（开源、成熟、效果好）  
**备选方案**: Deformation Field（轻量、集成简单）

**理由**：
1. 4DGaussians 已在多个数据集验证，效果最好
2. 开源代码完整，便于集成
3. 社区活跃，问题可以快速解决
4. 支持多种运动类型

---

## 3. 阶段一：POC 验证（1-2周）

### 3.1 目标

验证技术可行性，评估性能和效果。

### 3.2 任务清单

#### Week 1: 环境搭建与基础测试

**Day 1-2: 环境准备**

- [ ] 下载 4DGaussians 代码
- [ ] 安装依赖包
- [ ] 配置 CUDA 环境
- [ ] 下载预训练模型（如有）

**操作步骤**：
```bash
# 1. 克隆仓库
cd /home/3dgs
git clone https://github.com/hustvl/4DGaussians.git
cd 4DGaussians

# 2. 创建虚拟环境（推荐）
conda create -n 4dgs python=3.10
conda activate 4dgs

# 3. 安装依赖
pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt

# 4. 编译 CUDA 扩展
cd submodules/diff-gaussian-rasterization
pip install -e .
cd ../simple-knn
pip install -e .
cd ../..

# 5. 验证安装
python -c "import torch; print(torch.cuda.is_available())"
```

**预期依赖**：
```txt
torch>=2.0.0
torchvision>=0.15.0
plyfile
tqdm
opencv-python
imageio
scikit-image
lpips
```

---

**Day 3-4: 示例数据测试**

- [ ] 下载官方示例数据集
- [ ] 运行训练脚本
- [ ] 记录训练时间和显存占用
- [ ] 评估输出质量

**示例数据**：
- D-NeRF 数据集（合成数据，简单）
- HyperNeRF 数据集（真实数据，复杂）

**训练命令**：
```bash
# 下载示例数据（假设）
wget https://example.com/dnerf_jumpingjacks.zip
unzip dnerf_jumpingjacks.zip -d data/

# 训练
python train.py \
  --source_path ./data/jumpingjacks \
  --model_path ./output/test_4dgs \
  --iterations 30000 \
  --test_iterations 7000 15000 30000 \
  --save_iterations 7000 15000 30000 \
  --checkpoint_iterations 7000 15000 30000

# 渲染
python render.py \
  --model_path ./output/test_4dgs \
  --iteration 30000
```

**性能记录表**：

| 指标 | 预期值 | 实测值 | 备注 |
|------|--------|--------|------|
| 训练时间 | 30-60 分钟 | _____ | RTX 4060 |
| 显存峰值 | 6-8 GB | _____ | 取决于分辨率 |
| 渲染 FPS | 30-60 FPS | _____ | 实时渲染 |
| PSNR | > 30 dB | _____ | 质量评估 |
| 模型大小 | 100-500 MB | _____ | 取决于场景 |

---

**Day 5: 自有数据测试**

- [ ] 准备测试视频（手机拍摄）
- [ ] 数据预处理
- [ ] 训练测试
- [ ] 效果评估

**测试视频要求**：
- 时长：3-5 秒
- 帧率：30 FPS
- 分辨率：1080p 或更低
- 相机：固定机位（推荐）
- 内容：简单物体旋转或人物动作

**数据准备脚本**：
```bash
#!/bin/bash
# prepare_video.sh

VIDEO="$1"
OUTPUT_DIR="$2"
FPS=30

# 1. 提取帧
mkdir -p "$OUTPUT_DIR/images"
ffmpeg -i "$VIDEO" -vf "fps=$FPS" "$OUTPUT_DIR/images/frame_%04d.jpg"

# 2. 运行 COLMAP（位姿估计）
mkdir -p "$OUTPUT_DIR/sparse"
colmap feature_extractor \
  --database_path "$OUTPUT_DIR/database.db" \
  --image_path "$OUTPUT_DIR/images" \
  --ImageReader.single_camera 1

colmap exhaustive_matcher \
  --database_path "$OUTPUT_DIR/database.db"

colmap mapper \
  --database_path "$OUTPUT_DIR/database.db" \
  --image_path "$OUTPUT_DIR/images" \
  --output_path "$OUTPUT_DIR/sparse"

# 3. 转换格式（如需要）
python scripts/colmap_to_4dgs.py \
  --colmap_dir "$OUTPUT_DIR/sparse/0" \
  --output_dir "$OUTPUT_DIR/4dgs_format"

echo "数据准备完成: $OUTPUT_DIR"
```

**使用示例**：
```bash
./prepare_video.sh my_video.mp4 ./data/my_test
```

---

#### Week 2: 深入评估与方案确定

**Day 6-7: 多场景测试**

测试不同类型的运动：

| 场景类型 | 测试内容 | 预期难度 | 结果 |
|----------|----------|----------|------|
| 刚体旋转 | 杯子旋转 | 简单 ⭐ | _____ |
| 人物动作 | 挥手/跳跃 | 中等 ⭐⭐⭐ | _____ |
| 柔性物体 | 布料摆动 | 困难 ⭐⭐⭐⭐ | _____ |
| 多物体 | 多个物体同时运动 | 困难 ⭐⭐⭐⭐ | _____ |

**评估维度**：
- ✅ 几何精度（是否变形）
- ✅ 运动平滑度（是否抖动）
- ✅ 时间一致性（相邻帧是否连贯）
- ✅ 训练稳定性（是否收敛）
- ✅ 渲染质量（视觉效果）

---

**Day 8-9: 备选方案测试**

测试 Deformation Field 方案作为对比：

**实现代码**（简化版）：
```python
# deformation_trainer.py
import torch
import torch.nn as nn

class DeformationNet(nn.Module):
    """时间相关的形变网络"""
    def __init__(self, hidden_dim=128):
        super().__init__()
        # 输入: (x, y, z, t) → 输出: (dx, dy, dz)
        self.net = nn.Sequential(
            nn.Linear(4, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 3)
        )
    
    def forward(self, positions, time):
        """
        positions: (N, 3) 静态位置
        time: (N, 1) or scalar 时间戳 [0, 1]
        返回: (N, 3) 形变后的位置
        """
        if isinstance(time, (int, float)):
            time = torch.full((positions.shape[0], 1), time, 
                            device=positions.device)
        
        # 拼接位置和时间
        input_feat = torch.cat([positions, time], dim=-1)
        deformation = self.net(input_feat)
        return positions + deformation

# 训练循环
def train_with_deformation(static_gaussians, frames, iterations=10000):
    deform_net = DeformationNet().cuda()
    optimizer = torch.optim.Adam(deform_net.parameters(), lr=1e-3)
    
    for step in range(iterations):
        # 随机选择一帧
        frame_idx = torch.randint(0, len(frames), (1,)).item()
        time = frame_idx / len(frames)
        
        # 应用形变
        deformed_pos = deform_net(static_gaussians.positions, time)
        
        # 渲染（使用现有 gsplat）
        rendered = render(deformed_pos, static_gaussians.colors, ...)
        
        # 损失
        loss = F.mse_loss(rendered, frames[frame_idx])
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if step % 100 == 0:
            print(f"Step {step}, Loss: {loss.item():.4f}")
    
    return deform_net
```

**对比测试**：

| 方案 | 训练时间 | 显存占用 | 效果评分 | 推荐 |
|------|----------|----------|----------|------|
| 4DGaussians | _____ | _____ | _____ /10 | _____ |
| Deformation Field | _____ | _____ | _____ /10 | _____ |

---

**Day 10: 方案确定与报告**

- [ ] 汇总测试结果
- [ ] 编写 POC 报告
- [ ] 确定技术方案
- [ ] 规划下一阶段

**POC 报告模板**：

```markdown
# 动态 3DGS POC 验证报告

## 1. 测试环境
- GPU: RTX 4060 8GB
- CUDA: 12.8
- 测试数据: 5 个场景，共 XX 帧

## 2. 性能数据
| 方案 | 平均训练时间 | 显存峰值 | PSNR | 主观评分 |
|------|-------------|----------|------|----------|
| 4DGaussians | XX 分钟 | XX GB | XX dB | X/10 |
| Deformation | XX 分钟 | XX GB | XX dB | X/10 |

## 3. 效果评估
### 优点
- ...

### 缺点
- ...

### 典型问题
- ...

## 4. 方案推荐
推荐使用: ______
理由: ______

## 5. 下一步行动
- ...
```

---

### 3.3 交付物

- ✅ 4DGaussians 环境（已安装）
- ✅ 测试数据集（3-5 个）
- ✅ 训练/渲染脚本（可运行）
- ✅ POC 验证报告（文档）
- ✅ 性能基准数据（表格）

---

## 4. 阶段二：原型开发（2-3周）

### 4.1 目标

开发独立的动态训练工具，验证完整流程。

### 4.2 功能设计

#### 核心功能

```
动态训练工具
├── 视频预处理
│   ├── 帧提取（ffmpeg）
│   ├── 分辨率调整
│   └── 帧率转换
├── 位姿估计
│   ├── COLMAP 自动运行
│   └── 位姿优化
├── 动态训练
│   ├── 4DGaussians 训练
│   ├── 进度监控
│   └── 中间结果保存
├── 渲染导出
│   ├── 指定时间点渲染
│   ├── 动画序列渲染
│   └── 视频合成
└── 质量评估
    ├── PSNR/SSIM 计算
    └── 可视化对比
```

---

### 4.3 实现计划

#### Week 1: 数据处理模块

**任务 1: 视频预处理器**

文件：`app/video_processor.py`

```python
"""视频预处理：提取帧、格式转换"""
import subprocess
import os
from pathlib import Path

class VideoProcessor:
    def __init__(self, video_path, output_dir, fps=30, max_resolution=1080):
        self.video_path = video_path
        self.output_dir = Path(output_dir)
        self.fps = fps
        self.max_resolution = max_resolution
    
    def extract_frames(self):
        """提取视频帧"""
        frames_dir = self.output_dir / "images"
        frames_dir.mkdir(parents=True, exist_ok=True)
        
        # 构建 ffmpeg 命令
        cmd = [
            "ffmpeg", "-i", str(self.video_path),
            "-vf", f"fps={self.fps},scale='min({self.max_resolution},iw)':-2",
            "-q:v", "2",  # 高质量
            str(frames_dir / "frame_%04d.jpg")
        ]
        
        subprocess.run(cmd, check=True, capture_output=True)
        
        # 返回帧列表
        frames = sorted(frames_dir.glob("frame_*.jpg"))
        return frames
    
    def get_video_info(self):
        """获取视频信息"""
        cmd = [
            "ffprobe", "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=width,height,r_frame_rate,duration",
            "-of", "json",
            str(self.video_path)
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        import json
        info = json.loads(result.stdout)
        
        stream = info["streams"][0]
        fps_parts = stream["r_frame_rate"].split("/")
        fps = float(fps_parts[0]) / float(fps_parts[1])
        
        return {
            "width": stream["width"],
            "height": stream["height"],
            "fps": fps,
            "duration": float(stream.get("duration", 0))
        }
```

**使用示例**：
```python
processor = VideoProcessor("input.mp4", "./data/scene1", fps=30)
info = processor.get_video_info()
print(f"视频信息: {info}")

frames = processor.extract_frames()
print(f"提取了 {len(frames)} 帧")
```

---

**任务 2: COLMAP 自动化**

文件：`app/colmap_runner.py`

```python
"""COLMAP 位姿估计自动化"""
import subprocess
from pathlib import Path

class ColmapRunner:
    def __init__(self, image_dir, output_dir, use_gpu=True):
        self.image_dir = Path(image_dir)
        self.output_dir = Path(output_dir)
        self.use_gpu = use_gpu
        self.db_path = self.output_dir / "database.db"
        self.sparse_dir = self.output_dir / "sparse"
    
    def run_sfm(self, single_camera=True):
        """运行完整 SfM 流程"""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.sparse_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. 特征提取
        print("1/3 特征提取...")
        self._feature_extractor(single_camera)
        
        # 2. 特征匹配
        print("2/3 特征匹配...")
        self._feature_matcher()
        
        # 3. 稀疏重建
        print("3/3 稀疏重建...")
        self._mapper()
        
        return self.sparse_dir / "0"
    
    def _feature_extractor(self, single_camera):
        cmd = [
            "colmap", "feature_extractor",
            "--database_path", str(self.db_path),
            "--image_path", str(self.image_dir),
            "--ImageReader.single_camera", "1" if single_camera else "0",
            "--ImageReader.camera_model", "OPENCV",
            "--SiftExtraction.use_gpu", "1" if self.use_gpu else "0"
        ]
        subprocess.run(cmd, check=True)
    
    def _feature_matcher(self):
        # 对于视频序列，使用 sequential_matcher 更高效
        cmd = [
            "colmap", "sequential_matcher",
            "--database_path", str(self.db_path),
            "--SequentialMatching.overlap", "10",
            "--SiftMatching.use_gpu", "1" if self.use_gpu else "0"
        ]
        subprocess.run(cmd, check=True)
    
    def _mapper(self):
        cmd = [
            "colmap", "mapper",
            "--database_path", str(self.db_path),
            "--image_path", str(self.image_dir),
            "--output_path", str(self.sparse_dir)
        ]
        subprocess.run(cmd, check=True)
```

---

**任务 3: 数据格式转换**

文件：`app/format_converter.py`

```python
"""COLMAP → 4DGaussians 格式转换"""
import numpy as np
import json
from pathlib import Path

def colmap_to_4dgs(colmap_dir, output_dir, fps=30):
    """
    将 COLMAP 输出转换为 4DGaussians 格式
    
    4DGaussians 格式:
    {
        "frames": [
            {
                "file_path": "images/frame_0001.jpg",
                "time": 0.0,
                "transform_matrix": [[...], [...], [...], [...]]
            },
            ...
        ],
        "camera": {
            "w": 1920,
            "h": 1080,
            "fl_x": 1234.5,
            "fl_y": 1234.5,
            "cx": 960.0,
            "cy": 540.0
        }
    }
    """
    import pycolmap
    
    reconstruction = pycolmap.Reconstruction(colmap_dir)
    
    # 读取相机参数（假设单相机）
    camera = list(reconstruction.cameras.values())[0]
    
    # 读取所有图片的位姿
    frames = []
    sorted_images = sorted(reconstruction.images.items(), 
                          key=lambda x: x[1].name)
    
    for idx, (img_id, image) in enumerate(sorted_images):
        # COLMAP: world-to-camera
        # 4DGS 需要: camera-to-world
        R = image.rotmat()
        t = image.tvec
        
        # 转换为 4x4 变换矩阵
        transform = np.eye(4)
        transform[:3, :3] = R.T  # 转置得到 camera-to-world
        transform[:3, 3] = -R.T @ t
        
        frames.append({
            "file_path": f"images/{image.name}",
            "time": idx / fps,
            "transform_matrix": transform.tolist()
        })
    
    # 构建输出
    output_data = {
        "frames": frames,
        "camera": {
            "w": camera.width,
            "h": camera.height,
            "fl_x": camera.focal_length_x,
            "fl_y": camera.focal_length_y,
            "cx": camera.principal_point_x,
            "cy": camera.principal_point_y
        }
    }
    
    # 保存
    output_path = Path(output_dir) / "transforms.json"
    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)
    
    return output_path
```

---

#### Week 2: 训练模块

**任务 4: 4DGS 训练封装**

文件：`app/dynamic_trainer.py`

```python
"""4DGaussians 训练封装"""
import subprocess
import json
from pathlib import Path

class DynamicTrainer:
    def __init__(self, data_dir, output_dir, config=None):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.config = config or self._default_config()
    
    def _default_config(self):
        return {
            "iterations": 30000,
            "position_lr": 0.00016,
            "feature_lr": 0.0025,
            "opacity_lr": 0.05,
            "scaling_lr": 0.005,
            "rotation_lr": 0.001,
            "time_smoothness_weight": 0.01,
            "densify_from_iter": 500,
            "densify_until_iter": 15000,
            "densification_interval": 100
        }
    
    def train(self, progress_callback=None):
        """训练动态高斯"""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 构建训练命令
        cmd = [
            "python", "train.py",
            "--source_path", str(self.data_dir),
            "--model_path", str(self.output_dir),
            "--iterations", str(self.config["iterations"]),
            "--position_lr_init", str(self.config["position_lr"]),
            "--feature_lr", str(self.config["feature_lr"]),
            "--opacity_lr", str(self.config["opacity_lr"]),
            "--scaling_lr", str(self.config["scaling_lr"]),
            "--rotation_lr", str(self.config["rotation_lr"])
        ]
        
        # 运行训练（实时输出）
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )
        
        for line in process.stdout:
            print(line.rstrip())
            
            # 解析进度
            if progress_callback and "Iteration" in line:
                # 示例: "Iteration 1000/30000, Loss: 0.123"
                parts = line.split()
                if len(parts) >= 2:
                    iter_str = parts[1].split("/")
                    if len(iter_str) == 2:
                        current = int(iter_str[0])
                        total = int(iter_str[1])
                        progress_callback(current, total)
        
        process.wait()
        if process.returncode != 0:
            raise RuntimeError("训练失败")
        
        return self.output_dir
```

---

**任务 5: 渲染模块**

文件：`app/dynamic_renderer.py`

```python
"""动态场景渲染"""
import subprocess
from pathlib import Path

class DynamicRenderer:
    def __init__(self, model_path):
        self.model_path = Path(model_path)
    
    def render_frame(self, time, camera_pose=None, output_path=None):
        """渲染指定时间点的单帧"""
        cmd = [
            "python", "render.py",
            "--model_path", str(self.model_path),
            "--time", str(time)
        ]
        
        if camera_pose:
            # 保存相机位姿到临时文件
            import json
            pose_file = self.model_path / "temp_camera.json"
            with open(pose_file, "w") as f:
                json.dump(camera_pose, f)
            cmd.extend(["--camera_path", str(pose_file)])
        
        if output_path:
            cmd.extend(["--output", str(output_path)])
        
        subprocess.run(cmd, check=True)
        return output_path
    
    def render_video(self, duration, fps=30, output_video="output.mp4"):
        """渲染完整动画视频"""
        frames_dir = self.model_path / "rendered_frames"
        frames_dir.mkdir(exist_ok=True)
        
        num_frames = int(duration * fps)
        
        # 渲染所有帧
        for i in range(num_frames):
            time = i / fps
            output_path = frames_dir / f"frame_{i:04d}.png"
            self.render_frame(time, output_path=output_path)
            print(f"渲染进度: {i+1}/{num_frames}")
        
        # 合成视频
        import subprocess
        cmd = [
            "ffmpeg", "-y",
            "-framerate", str(fps),
            "-i", str(frames_dir / "frame_%04d.png"),
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            output_video
        ]
        subprocess.run(cmd, check=True)
        
        return output_video
```

---

#### Week 3: CLI 工具与测试

**任务 6: 命令行工具**

文件：`tools/train_dynamic.py`

```python
#!/usr/bin/env python3
"""动态 3DGS 训练命令行工具"""
import argparse
from pathlib import Path
import sys

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.video_processor import VideoProcessor
from app.colmap_runner import ColmapRunner
from app.format_converter import colmap_to_4dgs
from app.dynamic_trainer import DynamicTrainer
from app.dynamic_renderer import DynamicRenderer

def main():
    parser = argparse.ArgumentParser(description="动态 3DGS 训练工具")
    parser.add_argument("input", help="输入视频文件或图片目录")
    parser.add_argument("output", help="输出目录")
    parser.add_argument("--fps", type=int, default=30, help="帧率")
    parser.add_argument("--iterations", type=int, default=30000, help="训练迭代次数")
    parser.add_argument("--skip-colmap", action="store_true", help="跳过 COLMAP（已有位姿）")
    parser.add_argument("--render-video", action="store_true", help="训练后渲染视频")
    
    args = parser.parse_args()
    
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. 视频预处理
    if Path(args.input).is_file():
        print("==> 步骤 1/5: 提取视频帧...")
        processor = VideoProcessor(args.input, output_dir, fps=args.fps)
        frames = processor.extract_frames()
        print(f"    提取了 {len(frames)} 帧")
        images_dir = output_dir / "images"
    else:
        print("==> 步骤 1/5: 使用现有图片...")
        images_dir = Path(args.input)
    
    # 2. COLMAP 位姿估计
    if not args.skip_colmap:
        print("==> 步骤 2/5: 运行 COLMAP...")
        colmap = ColmapRunner(images_dir, output_dir / "colmap")
        sparse_dir = colmap.run_sfm()
        print(f"    COLMAP 完成: {sparse_dir}")
    else:
        print("==> 步骤 2/5: 跳过 COLMAP")
        sparse_dir = output_dir / "colmap" / "sparse" / "0"
    
    # 3. 格式转换
    print("==> 步骤 3/5: 转换数据格式...")
    transforms_path = colmap_to_4dgs(sparse_dir, output_dir, fps=args.fps)
    print(f"    转换完成: {transforms_path}")
    
    # 4. 训练
    print("==> 步骤 4/5: 开始训练...")
    trainer = DynamicTrainer(
        output_dir,
        output_dir / "model",
        config={"iterations": args.iterations}
    )
    
    def progress_cb(current, total):
        percent = current / total * 100
        print(f"    训练进度: {current}/{total} ({percent:.1f}%)")
    
    model_path = trainer.train(progress_callback=progress_cb)
    print(f"    训练完成: {model_path}")
    
    # 5. 渲染（可选）
    if args.render_video:
        print("==> 步骤 5/5: 渲染动画视频...")
        renderer = DynamicRenderer(model_path)
        video_path = renderer.render_video(
            duration=len(frames) / args.fps,
            fps=args.fps,
            output_video=str(output_dir / "output.mp4")
        )
        print(f"    视频已保存: {video_path}")
    else:
        print("==> 步骤 5/5: 跳过渲染")
    
    print("\n✓ 全部完成！")
    print(f"  模型: {model_path}")
    if args.render_video:
        print(f"  视频: {output_dir / 'output.mp4'}")

if __name__ == "__main__":
    main()
```

**使用示例**：
```bash
# 完整流程（从视频到模型）
python tools/train_dynamic.py input.mp4 output/scene1 --render-video

# 仅训练（已有 COLMAP 结果）
python tools/train_dynamic.py images/ output/scene2 --skip-colmap

# 快速测试（少量迭代）
python tools/train_dynamic.py input.mp4 output/test --iterations 5000
```

---

### 4.4 交付物

- ✅ 视频处理模块（`video_processor.py`）
- ✅ COLMAP 自动化（`colmap_runner.py`）
- ✅ 格式转换工具（`format_converter.py`）
- ✅ 训练封装（`dynamic_trainer.py`）
- ✅ 渲染模块（`dynamic_renderer.py`）
- ✅ CLI 工具（`train_dynamic.py`）
- ✅ 测试用例（3+ 个场景）
- ✅ 使用文档（README）

---

## 5. 阶段三：服务集成（3-4周）

### 5.1 目标

将动态训练功能集成到现有 3DGS API 服务中。

### 5.2 架构设计

#### API 端点设计

```
现有端点:
POST   /train                    # 静态训练
GET    /train/{task_id}          # 查询状态
GET    /train/{task_id}/result   # 下载结果
DELETE /train/{task_id}          # 删除任务

新增端点:
POST   /train/dynamic            # 动态训练（新）⭐
GET    /train/dynamic/{task_id}/render  # 渲染指定时间点
POST   /train/dynamic/{task_id}/video   # 生成动画视频
GET    /train/dynamic/{task_id}/player  # Web 播放器
```

---

### 5.3 实现计划

#### Week 1: API 端点开发

**任务 1: 动态训练端点**

文件：`app/main.py`（追加）

```python
from fastapi import UploadFile, File, Form
from typing import Literal

@app.post("/train/dynamic", dependencies=[Depends(require_api_key)])
async def submit_dynamic_train(
    file: UploadFile = File(...),  # 视频文件或帧序列 zip
    fps: int = Form(30),
    iterations: int = Form(30000),
    input_type: Literal["video", "frames"] = Form("video"),
    skip_colmap: bool = Form(False),
    auto_render: bool = Form(False)
):
    """
    提交动态场景训练任务
    
    参数:
    - file: 视频文件(.mp4, .mov) 或帧序列压缩包(.zip)
    - fps: 帧率（仅用于 video 输入）
    - iterations: 训练迭代次数（建议 30000）
    - input_type: 输入类型（video 或 frames）
    - skip_colmap: 跳过 COLMAP（frames 输入且已有位姿时）
    - auto_render: 训练完成后自动渲染视频
    
    返回:
    - task_id: 任务 ID
    - status: queued
    - estimated_time: 预估时间（分钟）
    """
    # 参数校验
    if not file.filename.lower().endswith(('.mp4', '.mov', '.avi', '.zip')):
        raise HTTPException(
            status_code=400,
            detail="仅支持视频文件（mp4/mov/avi）或帧序列压缩包（zip）"
        )
    
    if not (5000 <= iterations <= 50000):
        raise HTTPException(
            status_code=400,
            detail="iterations 必须在 [5000, 50000] 范围内"
        )
    
    if not (10 <= fps <= 60):
        raise HTTPException(
            status_code=400,
            detail="fps 必须在 [10, 60] 范围内"
        )
    
    # 创建任务
    task_id = uuid.uuid4().hex
    tdir = task_dir(task_id)
    os.makedirs(tdir)
    
    # 保存上传文件
    if input_type == "video":
        save_path = os.path.join(tdir, "input_video.mp4")
    else:
        save_path = os.path.join(tdir, "input_frames.zip")
    
    with open(save_path, "wb") as f:
        shutil.copyfileobj(file.file, f, length=4 * 1024 * 1024)
    
    # 预估时间（基于经验公式）
    estimated_minutes = iterations / 1000 * 1.5  # 粗略估计
    
    # 创建状态
    now = datetime.now().isoformat(timespec="seconds")
    status = {
        "task_id": task_id,
        "status": "queued",
        "stage": "queued",
        "progress": 0.0,
        "message": "",
        "error": None,
        "task_type": "dynamic",  # 标记为动态任务
        "params": {
            "fps": fps,
            "iterations": iterations,
            "input_type": input_type,
            "skip_colmap": skip_colmap,
            "auto_render": auto_render
        },
        "created_at": now,
        "updated_at": now
    }
    
    with open(os.path.join(tdir, "status.json"), "w") as f:
        json.dump(status, f, indent=2, ensure_ascii=False)
    
    # 提交 Celery 任务
    celery_app.send_task(
        "app.tasks.run_dynamic_training",
        args=[task_id],
        kwargs={
            "fps": fps,
            "iterations": iterations,
            "input_type": input_type,
            "skip_colmap": skip_colmap,
            "auto_render": auto_render
        },
        task_id=task_id,
        queue=config.QUEUE_NAME
    )
    
    return {
        "task_id": task_id,
        "status": "queued",
        "estimated_time_minutes": round(estimated_minutes, 1)
    }
```

---

**任务 2: 渲染端点**

```python
@app.get("/train/dynamic/{task_id}/render")
async def render_dynamic_frame(
    task_id: str,
    time: float = Query(..., ge=0.0, description="时间点（秒）"),
    camera_index: int = Query(0, description="相机索引")
):
    """
    渲染动态场景的指定时间点
    
    参数:
    - time: 时间点（秒）
    - camera_index: 相机索引（0 表示第一个相机视角）
    
    返回:
    - 渲染的图片（PNG）
    """
    status = read_status(task_id)
    if status["status"] != "done":
        raise HTTPException(
            status_code=409,
            detail=f"task is {status['status']}, not done"
        )
    
    if status.get("task_type") != "dynamic":
        raise HTTPException(
            status_code=400,
            detail="task is not a dynamic training task"
        )
    
    # 渲染
    model_dir = os.path.join(task_dir(task_id), "model")
    output_path = os.path.join(task_dir(task_id), f"render_t{time:.2f}.png")
    
    # 调用渲染函数（需要实现）
    from app.dynamic_renderer import DynamicRenderer
    renderer = DynamicRenderer(model_dir)
    renderer.render_frame(time, output_path=output_path)
    
    return FileResponse(
        output_path,
        media_type="image/png",
        filename=f"render_t{time:.2f}.png"
    )


@app.post("/train/dynamic/{task_id}/video")
async def generate_dynamic_video(
    task_id: str,
    fps: int = Form(30),
    duration: float = Form(None)  # None 表示使用训练时的完整时长
):
    """
    生成动态场景的完整动画视频
    
    参数:
    - fps: 输出视频帧率
    - duration: 视频时长（秒），None 表示完整时长
    
    返回:
    - video_url: 视频下载地址
    - estimated_time: 预估生成时间（秒）
    """
    status = read_status(task_id)
    if status["status"] != "done":
        raise HTTPException(status_code=409, detail="task not done")
    
    # 提交视频渲染任务（异步）
    video_task_id = f"{task_id}_video"
    celery_app.send_task(
        "app.tasks.render_dynamic_video",
        args=[task_id, fps, duration],
        task_id=video_task_id
    )
    
    return {
        "video_task_id": video_task_id,
        "status": "rendering",
        "estimated_time_seconds": duration * 2  # 粗略估计
    }
```

---

**任务 3: Web 播放器端点**

```python
@app.get("/train/dynamic/{task_id}/player")
async def dynamic_player(task_id: str):
    """
    动态场景 Web 播放器
    
    返回 HTML 页面，可交互查看动态 3D 模型
    """
    status = read_status(task_id)
    if status["status"] != "done":
        raise HTTPException(status_code=409, detail="task not done")
    
    # 渲染播放器页面
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>动态 3DGS 播放器 - {task_id}</title>
        <style>
            body {{ margin: 0; overflow: hidden; }}
            #canvas {{ width: 100%; height: 100vh; }}
            #controls {{
                position: absolute;
                bottom: 20px;
                left: 50%;
                transform: translateX(-50%);
                background: rgba(0,0,0,0.7);
                padding: 15px;
                border-radius: 10px;
                color: white;
            }}
            #timeSlider {{ width: 300px; }}
        </style>
    </head>
    <body>
        <canvas id="canvas"></canvas>
        <div id="controls">
            <button id="playBtn">播放</button>
            <input type="range" id="timeSlider" min="0" max="100" value="0">
            <span id="timeDisplay">0.00s</span>
        </div>
        
        <script type="module">
            // TODO: 集成 4DGS 播放器
            // 使用 Three.js + 自定义着色器
            console.log("Task ID: {task_id}");
        </script>
    </body>
    </html>
    """
    
    return HTMLResponse(content=html_content)
```

---

#### Week 2: Celery 任务实现

**任务 4: 动态训练 Celery 任务**

文件：`app/tasks.py`（追加）

```python
@celery_app.task(name="app.tasks.run_dynamic_training", bind=True)
def run_dynamic_training(
    self,
    task_id: str,
    fps: int = 30,
    iterations: int = 30000,
    input_type: str = "video",
    skip_colmap: bool = False,
    auto_render: bool = False
):
    """动态场景训练任务"""
    from app.video_processor import VideoProcessor
    from app.colmap_runner import ColmapRunner
    from app.format_converter import colmap_to_4dgs
    from app.dynamic_trainer import DynamicTrainer
    
    writer = StatusWriter(task_id)
    task_dir_path = os.path.join(TASKS_DIR, task_id)
    
    try:
        # 1. 预处理
        writer.update("processing", "extract", 0.05, "提取视频帧")
        
        if input_type == "video":
            video_path = os.path.join(task_dir_path, "input_video.mp4")
            processor = VideoProcessor(video_path, task_dir_path, fps=fps)
            frames = processor.extract_frames()
            images_dir = os.path.join(task_dir_path, "images")
        else:
            # 解压帧序列
            zip_path = os.path.join(task_dir_path, "input_frames.zip")
            images_dir = os.path.join(task_dir_path, "images")
            # ... 解压逻辑 ...
        
        writer.update("processing", "extract", 0.10, f"已提取 {len(frames)} 帧")
        
        # 2. COLMAP
        if not skip_colmap:
            writer.update("colmap", "feature_extractor", 0.15, "COLMAP 特征提取")
            colmap_runner = ColmapRunner(images_dir, 
                                        os.path.join(task_dir_path, "colmap"))
            sparse_dir = colmap_runner.run_sfm()
            writer.update("colmap", "colmap_done", 0.30, "COLMAP 完成")
        else:
            sparse_dir = os.path.join(task_dir_path, "colmap", "sparse", "0")
        
        # 3. 格式转换
        writer.update("processing", "convert", 0.32, "转换数据格式")
        transforms_path = colmap_to_4dgs(sparse_dir, task_dir_path, fps=fps)
        
        # 4. 训练
        writer.update("training", "training", 0.35, "开始动态训练")
        
        trainer = DynamicTrainer(
            task_dir_path,
            os.path.join(task_dir_path, "model"),
            config={"iterations": iterations}
        )
        
        def training_progress(current, total):
            progress = 0.35 + 0.55 * (current / total)
            writer.update("training", "training", progress,
                         f"训练中 {current}/{total}")
        
        model_path = trainer.train(progress_callback=training_progress)
        
        # 5. 导出
        writer.update("packaging", "packaging", 0.92, "打包结果")
        # TODO: 导出模型文件
        
        # 6. 自动渲染（可选）
        if auto_render:
            writer.update("rendering", "rendering", 0.95, "渲染动画视频")
            from app.dynamic_renderer import DynamicRenderer
            renderer = DynamicRenderer(model_path)
            video_path = renderer.render_video(
                duration=len(frames) / fps,
                fps=fps,
                output_video=os.path.join(task_dir_path, "animation.mp4")
            )
        
        writer.update("done", "done", 1.0, "训练完成")
        
    except Exception as e:
        msg = str(e)
        writer.update("failed", "failed", 0.0, error=msg)
        raise


@celery_app.task(name="app.tasks.render_dynamic_video")
def render_dynamic_video(task_id: str, fps: int, duration: float):
    """渲染动态场景视频（异步任务）"""
    # 实现视频渲染逻辑
    pass
```

---

#### Week 3-4: 前端与文档

**任务 5: Swagger UI 更新**

在 `app/main.py` 的 `FastAPI()` 初始化中添加文档：

```python
app = FastAPI(
    title="3DGS Training Service",
    version="2.0.0",  # 升级版本号
    description="""
    ## 功能列表
    
    ### 静态 3DGS
    - POST /train - 静态场景训练
    
    ### 动态 3DGS ⭐ 新功能
    - POST /train/dynamic - 动态场景训练（视频输入）
    - GET /train/dynamic/{task_id}/render - 渲染指定时间点
    - POST /train/dynamic/{task_id}/video - 生成完整动画视频
    - GET /train/dynamic/{task_id}/player - Web 播放器
    
    ### 其他
    - GET /health - 健康检查
    - GET /docs - API 文档（本页面）
    """,
    docs_url="/docs",
    redoc_url="/redoc"
)
```

---

**任务 6: 更新文档**

文件：`API使用手册.md`（追加动态训练章节）

```markdown
## 动态 3DGS 训练

### 1. 提交动态训练任务

**端点**: `POST /train/dynamic`

**请求示例**（视频输入）:
\```bash
curl -X POST http://host:12524/train/dynamic \
  -H "X-API-Key: $API_KEY" \
  -F "file=@dancing_person.mp4" \
  -F "fps=30" \
  -F "iterations=30000" \
  -F "input_type=video" \
  -F "auto_render=true"
\```

**返回**:
\```json
{
  "task_id": "abc123...",
  "status": "queued",
  "estimated_time_minutes": 45.0
}
\```

### 2. 查询训练状态

使用与静态训练相同的端点:
\```bash
curl -H "X-API-Key: $API_KEY" \
  http://host:12524/train/abc123...
\```

### 3. 渲染指定时间点

\```bash
curl -H "X-API-Key: $API_KEY" \
  "http://host:12524/train/dynamic/abc123.../render?time=1.5" \
  -o frame_1.5s.png
\```

### 4. 查看 Web 播放器

在浏览器中访问:
\```
http://host:12524/train/dynamic/abc123.../player
\```
```

---

### 5.4 交付物

- ✅ API 端点实现（4 个新端点）
- ✅ Celery 任务（动态训练、视频渲染）
- ✅ Web 播放器（基础版）
- ✅ API 文档更新
- ✅ 集成测试用例
- ✅ 部署脚本更新

---

## 6. 阶段四：优化上线（2-3周）

### 6.1 性能优化

#### 优化项 1: 显存优化

**问题**: 动态训练显存占用比静态训练高 50-100%

**解决方案**:
1. **梯度累积**: 减小 batch size，累积梯度
2. **混合精度**: 使用 FP16 训练
3. **动态加载**: 不同时间点的帧按需加载

**代码示例**:
```python
# 启用混合精度训练
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

for step in range(iterations):
    optimizer.zero_grad()
    
    with autocast():  # 自动混合精度
        output = model(input)
        loss = criterion(output, target)
    
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

---

#### 优化项 2: 训练速度

**问题**: 30000 次迭代需要 30-60 分钟

**解决方案**:
1. **自适应迭代**: 根据收敛情况提前停止
2. **分段训练**: 粗糙阶段 + 精细阶段
3. **并行处理**: 多个时间点并行训练

**实现**:
```python
def adaptive_training(model, data, max_iterations=30000, patience=1000):
    """自适应训练：收敛后自动停止"""
    best_loss = float('inf')
    no_improve_count = 0
    
    for step in range(max_iterations):
        loss = train_one_step(model, data)
        
        if loss < best_loss * 0.99:  # 有明显改善
            best_loss = loss
            no_improve_count = 0
        else:
            no_improve_count += 1
        
        if no_improve_count >= patience:
            print(f"提前停止于 {step} 次迭代（已收敛）")
            break
    
    return model
```

---

#### 优化项 3: 渲染速度

**问题**: 渲染 300 帧视频需要 10-15 分钟

**解决方案**:
1. **GPU 批量渲染**: 一次渲染多帧
2. **低分辨率预览**: 提供快速预览选项
3. **增量渲染**: 仅渲染变化的部分

---

### 6.2 质量提升

#### 提升项 1: 时间平滑

**问题**: 相邻帧之间可能出现抖动

**解决方案**: 增加时间平滑约束

```python
def temporal_smoothness_loss(positions_t1, positions_t2, weight=0.01):
    """时间平滑损失：相邻时刻的位置变化应该平滑"""
    velocity = positions_t2 - positions_t1
    acceleration = velocity[1:] - velocity[:-1]
    return weight * torch.mean(acceleration ** 2)

# 在训练中使用
loss_smooth = temporal_smoothness_loss(pos_t, pos_t_next)
total_loss = loss_render + loss_smooth
```

---

#### 提升项 2: 运动模糊处理

**问题**: 快速运动时会有拖影

**解决方案**: 在训练时模拟运动模糊

```python
def motion_blur_augmentation(image, velocity, num_samples=5):
    """运动模糊增强"""
    blurred = torch.zeros_like(image)
    for i in range(num_samples):
        t = i / (num_samples - 1)
        offset = velocity * t
        shifted = shift_image(image, offset)
        blurred += shifted
    return blurred / num_samples
```

---

### 6.3 用户体验

#### 改进项 1: 实时预览

**功能**: 训练过程中预览当前效果

**实现**: WebSocket 推送预览帧

```python
@app.websocket("/train/dynamic/{task_id}/preview")
async def preview_stream(websocket: WebSocket, task_id: str):
    await websocket.accept()
    
    preview_dir = os.path.join(task_dir(task_id), "previews")
    last_preview = None
    
    while True:
        # 检查新预览图
        previews = sorted(Path(preview_dir).glob("preview_*.jpg"))
        if previews and previews[-1] != last_preview:
            last_preview = previews[-1]
            with open(last_preview, "rb") as f:
                await websocket.send_bytes(f.read())
        
        await asyncio.sleep(2)
```

---

#### 改进项 2: 交互式播放器

**功能**: Web 端播放动态 3D 模型

**技术栈**:
- Three.js（渲染引擎）
- React（UI 框架）
- 自定义着色器（4DGS 渲染）

**功能列表**:
- ✅ 播放/暂停
- ✅ 时间轴拖动
- ✅ 相机旋转/缩放
- ✅ 速度调节
- ✅ 截图保存

---

### 6.4 监控与告警

#### 监控指标

```python
# 训练监控
metrics = {
    "task_id": task_id,
    "status": "training",
    "metrics": {
        "current_iteration": 15000,
        "total_iterations": 30000,
        "current_loss": 0.0234,
        "fps": 2.5,  # 训练速度（iter/s）
        "gpu_memory_used_gb": 6.8,
        "estimated_remaining_minutes": 20
    }
}
```

#### 告警规则

| 条件 | 告警级别 | 处理 |
|------|----------|------|
| GPU 显存 > 7.5GB | 警告 | 降低 batch size |
| 训练超过 2 小时 | 警告 | 检查是否卡住 |
| Loss 不下降（1000 iter） | 错误 | 停止训练 |
| GPU 温度 > 85°C | 警告 | 降低训练速度 |

---

### 6.5 交付物

- ✅ 性能优化代码
- ✅ 质量提升算法
- ✅ 实时预览功能
- ✅ 交互式播放器（完整版）
- ✅ 监控面板
- ✅ 压力测试报告
- ✅ 上线检查清单

---

## 7. 技术实现细节

### 7.1 数据流架构

```
视频输入 (input.mp4)
    ↓ [VideoProcessor]
帧序列 (frame_0001.jpg, frame_0002.jpg, ...)
    ↓ [ColmapRunner]
相机位姿 (cameras.bin, images.bin)
    ↓ [FormatConverter]
4DGS 格式 (transforms.json)
    ↓ [DynamicTrainer]
动态模型 (model.pth, gaussians.ply)
    ↓ [DynamicRenderer]
渲染结果 (animation.mp4, frame_0001.png, ...)
```

---

### 7.2 存储结构

```
/home/3dgs/data/tasks/{task_id}/
├── input_video.mp4          # 原始视频
├── images/                   # 提取的帧
│   ├── frame_0001.jpg
│   ├── frame_0002.jpg
│   └── ...
├── colmap/                   # COLMAP 输出
│   ├── database.db
│   ├── sparse/0/
│   │   ├── cameras.bin
│   │   ├── images.bin
│   │   └── points3D.bin
│   └── colmap.log
├── transforms.json           # 4DGS 格式数据
├── model/                    # 训练的模型
│   ├── iteration_30000/
│   │   ├── point_cloud.ply
│   │   └── deformation_field.pth
│   └── checkpoints/
├── previews/                 # 训练预览图
│   ├── preview_5000.jpg
│   ├── preview_10000.jpg
│   └── ...
├── rendered/                 # 渲染结果
│   ├── frames/
│   │   ├── frame_0001.png
│   │   └── ...
│   └── animation.mp4
├── status.json              # 任务状态
└── train_stats.json         # 训练统计
```

---

### 7.3 配置管理

文件：`app/config.py`（追加）

```python
# 动态训练配置
DYNAMIC_MIN_FRAMES = 30         # 最少帧数
DYNAMIC_MAX_FRAMES = 600        # 最多帧数（20秒 @ 30fps）
DYNAMIC_MIN_FPS = 10
DYNAMIC_MAX_FPS = 60
DYNAMIC_DEFAULT_ITERATIONS = 30000
DYNAMIC_MAX_ITERATIONS = 50000

# 视频处理
VIDEO_MAX_RESOLUTION = 1080     # 最大分辨率
VIDEO_SUPPORTED_FORMATS = [".mp4", ".mov", ".avi"]

# 渲染配置
RENDER_DEFAULT_FPS = 30
RENDER_MAX_FPS = 60
RENDER_TIMEOUT_SECONDS = 3600   # 1小时超时
```

---

## 8. 测试与验证

### 8.1 单元测试

```python
# tests/test_video_processor.py
import unittest
from app.video_processor import VideoProcessor

class TestVideoProcessor(unittest.TestCase):
    def test_extract_frames(self):
        processor = VideoProcessor("test_video.mp4", "./test_output")
        frames = processor.extract_frames()
        self.assertGreater(len(frames), 0)
        self.assertTrue(frames[0].exists())
    
    def test_video_info(self):
        processor = VideoProcessor("test_video.mp4", "./test_output")
        info = processor.get_video_info()
        self.assertIn("width", info)
        self.assertIn("height", info)
        self.assertIn("fps", info)
```

---

### 8.2 集成测试

**测试场景**:

| 场景 | 输入 | 预期输出 | 通过 |
|------|------|----------|------|
| 简单旋转 | 杯子旋转视频（3秒） | 模型 + 动画 | [ ] |
| 人物动作 | 跳舞视频（5秒） | 模型 + 动画 | [ ] |
| 多物体 | 多物体运动（4秒） | 模型 + 动画 | [ ] |
| 长视频 | 10秒视频 | 训练成功或提示过长 | [ ] |
| 无效视频 | 损坏的视频文件 | 错误提示 | [ ] |

---

### 8.3 性能测试

**测试环境**: RTX 4060 8GB

| 场景 | 帧数 | 分辨率 | 训练时间 | 显存峰值 | 渲染时间 |
|------|------|--------|----------|----------|----------|
| 简单（杯子） | 90 | 1080p | _____ | _____ | _____ |
| 中等（人物） | 150 | 1080p | _____ | _____ | _____ |
| 复杂（多物体） | 120 | 1080p | _____ | _____ | _____ |

**目标**:
- 训练时间 < 60 分钟（150 帧，30000 次迭代）
- 显存峰值 < 7.5 GB
- 渲染速度 > 10 FPS

---

### 8.4 质量测试

**评估标准**:

| 维度 | 指标 | 目标 |
|------|------|------|
| 几何精度 | 边缘清晰度 | 主观评分 > 7/10 |
| 时间一致性 | 相邻帧 SSIM | > 0.95 |
| 运动平滑度 | 抖动检测 | < 5% 帧抖动 |
| 渲染质量 | PSNR | > 28 dB |

---

## 9. 风险管理

### 9.1 技术风险

| 风险 | 概率 | 影响 | 缓解措施 |
|------|------|------|----------|
| 4DGaussians 集成困难 | 中 | 高 | 使用 Deformation 作为备选 |
| 显存不足 | 中 | 中 | 降低分辨率、梯度累积 |
| 训练不稳定 | 低 | 中 | 调整学习率、增加正则化 |
| COLMAP 失败 | 中 | 高 | 提供手动位姿输入选项 |

---

### 9.2 进度风险

| 风险 | 概率 | 影响 | 缓解措施 |
|------|------|------|----------|
| POC 验证失败 | 低 | 高 | 预留 1 周 buffer 时间 |
| 集成延期 | 中 | 中 | 分阶段交付，核心先上线 |
| 性能不达标 | 中 | 中 | 降低质量要求或增加硬件 |

---

### 9.3 应急预案

**Plan B**: 如果 4DGaussians 无法集成

1. 使用简化的 Deformation Field 方案
2. 先支持刚体运动（旋转/平移）
3. 后续再迭代升级

**Plan C**: 如果动态训练效果不佳

1. 仅提供多时间点静态 3DGS
2. 使用插值生成动画
3. 标记为 Beta 功能

---

## 10. 里程碑与交付

### 10.1 总体时间线

```
Week 1-2:  POC 验证
  ├─ Day 1-5:   环境搭建、示例测试
  └─ Day 6-10:  自有数据测试、方案确定

Week 3-5:  原型开发
  ├─ Week 3:    数据处理模块
  ├─ Week 4:    训练模块
  └─ Week 5:    CLI 工具与测试

Week 6-9:  服务集成
  ├─ Week 6:    API 端点开发
  ├─ Week 7:    Celery 任务实现
  └─ Week 8-9:  前端与文档

Week 10-12: 优化上线
  ├─ Week 10:   性能优化
  ├─ Week 11:   质量提升与监控
  └─ Week 12:   测试与上线
```

---

### 10.2 检查点

**POC 检查点**（Week 2 结束）:
- [ ] 4DGaussians 环境运行正常
- [ ] 至少 3 个测试场景通过
- [ ] 性能数据符合预期
- [ ] POC 报告完成

**原型检查点**（Week 5 结束）:
- [ ] CLI 工具可运行
- [ ] 完整流程打通（视频 → 模型 → 渲染）
- [ ] 代码质量达标（测试覆盖率 > 70%）

**集成检查点**（Week 9 结束）:
- [ ] API 端点全部实现
- [ ] Swagger UI 更新
- [ ] 文档完整
- [ ] 集成测试通过

**上线检查点**（Week 12 结束）:
- [ ] 性能达标
- [ ] 用户验收测试通过
- [ ] 监控告警配置完成
- [ ] 上线部署文档齐全

---

## 11. 资源需求

### 11.1 人力

| 角色 | 时间投入 | 职责 |
|------|----------|------|
| 算法工程师 | 全职 6 周 | POC、训练算法、质量优化 |
| 后端工程师 | 全职 6 周 | API、Celery、数据处理 |
| 前端工程师 | 半职 4 周 | Web 播放器、UI |
| 测试工程师 | 半职 4 周 | 测试、验证、文档 |

**总计**: 约 4 人月

---

### 11.2 硬件

| 资源 | 当前 | 需求 | 缺口 |
|------|------|------|------|
| GPU | RTX 4060 8GB × 1 | RTX 4090 24GB × 1 | 需升级 |
| 存储 | 500GB SSD | 1TB SSD | 需扩容 |
| 内存 | 16GB | 32GB | 需扩容 |

**预算**: 约 2-3 万元（GPU 升级为主）

---

### 11.3 软件

| 软件 | 用途 | 费用 |
|------|------|------|
| 4DGaussians | 开源训练框架 | 免费 |
| COLMAP | 位姿估计 | 免费 |
| FFmpeg | 视频处理 | 免费 |
| Three.js | Web 渲染 | 免费 |

**总计**: 0 元（全开源）

---

## 12. 附录

### 12.1 参考资料

**论文**:
- 4D Gaussian Splatting for Real-Time Dynamic Scene Rendering
- SC-GS: Sparse-Controlled Gaussian Splatting
- Deformable 3D Gaussians for High-Fidelity Monocular Dynamic Scene Reconstruction

**开源项目**:
- 4DGaussians: https://github.com/hustvl/4DGaussians
- SC-GS: https://github.com/yihua7/SC-GS
- Deformable-3D-Gaussians: https://github.com/ingra14m/Deformable-3D-Gaussians

---

### 12.2 术语表

| 术语 | 说明 |
|------|------|
| 4DGS | 4D Gaussian Splatting，动态高斯点云 |
| Deformation Field | 形变场，用于建模物体运动 |
| SfM | Structure from Motion，从运动中恢复结构 |
| COLMAP | 开源 SfM 软件 |
| Temporal Smoothness | 时间平滑性，相邻帧的连贯性 |

---

### 12.3 FAQ

**Q: 动态训练比静态训练慢多少？**
A: 通常慢 2-3 倍。静态训练 10-15 分钟，动态训练 30-60 分钟。

**Q: 可以训练多长的视频？**
A: 建议 3-10 秒。更长的视频需要更多显存和训练时间。

**Q: 支持移动相机吗？**
A: 支持，但需要 COLMAP 估计每帧的相机位姿。固定相机效果更好。

**Q: 能否导出到 Unity/Unreal？**
A: 可以导出 PLY 格式，但需要自定义插件来播放动态效果。

---

## 13. 下一步行动

### 立即行动（本周）

1. [ ] Review 本方案，确认优先级
2. [ ] 准备 POC 测试数据（3-5 个视频）
3. [ ] 搭建 4DGaussians 环境
4. [ ] 运行第一个示例

### 短期行动（本月）

1. [ ] 完成 POC 验证
2. [ ] 开始原型开发
3. [ ] 准备技术分享（团队同步）

### 中期行动（下季度）

1. [ ] 完成服务集成
2. [ ] 上线 Beta 版本
3. [ ] 收集用户反馈

---

**文档维护者**: 3DGS 开发团队  
**创建日期**: 2026-09-03  
**最后更新**: 2026-09-03  
**下次审查**: 2026-09-17

---

*本实施方案为动态文档，随项目进展持续更新。*
