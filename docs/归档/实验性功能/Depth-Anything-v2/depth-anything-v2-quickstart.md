# Depth-Anything-V2 集成 - 快速启动指南

**开始日期**: 2026-09-05  
**第一阶段**: Week 1 快速验证（3-5天）

---

## 🎯 明天的任务清单

### Day 1: 环境准备和模型测试（2-3小时）

#### 任务 1: 安装依赖（30分钟）

```bash
# 1. 进入 worker 容器
docker exec -it 3dgs-worker bash

# 2. 安装依赖
pip install transformers==4.35.0 -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install accelerate -i https://pypi.tuna.tsinghua.edu.cn/simple

# 3. 验证安装
python3 << EOF
from transformers import pipeline
print("✅ transformers 安装成功")
EOF
```

#### 任务 2: 测试模型加载（1小时）

创建测试脚本：
```python
# /tmp/test_depth_anything.py
"""测试 Depth-Anything-V2 三个模型"""
import time
import torch
from transformers import pipeline
from PIL import Image
import numpy as np

def test_model(model_size="small"):
    """测试单个模型"""
    print(f"\n{'='*60}")
    print(f"测试 Depth-Anything-V2-{model_size.capitalize()}")
    print(f"{'='*60}")
    
    model_name = f"depth-anything/Depth-Anything-V2-{model_size.capitalize()}"
    
    # 1. 加载模型
    print(f"加载模型: {model_name}")
    start = time.time()
    pipe = pipeline(
        task="depth-estimation",
        model=model_name,
        device=0 if torch.cuda.is_available() else -1
    )
    load_time = time.time() - start
    print(f"✅ 模型加载耗时: {load_time:.2f}s")
    
    # 2. 创建测试图片
    test_img = Image.new('RGB', (640, 480), color='red')
    test_img.save('/tmp/test.jpg')
    
    # 3. 推理测试
    print("运行推理...")
    start = time.time()
    result = pipe('/tmp/test.jpg')
    infer_time = time.time() - start
    
    depth = np.array(result['depth'])
    print(f"✅ 推理耗时: {infer_time:.2f}s")
    print(f"✅ 深度图尺寸: {depth.shape}")
    print(f"✅ 深度范围: [{depth.min():.2f}, {depth.max():.2f}]")
    
    # 4. 显存占用
    if torch.cuda.is_available():
        mem_allocated = torch.cuda.max_memory_allocated() / 1024**3
        print(f"✅ 显存占用: {mem_allocated:.2f} GB")
        torch.cuda.reset_peak_memory_stats()
    
    return {
        'model_size': model_size,
        'load_time': load_time,
        'infer_time': infer_time,
        'memory_gb': mem_allocated if torch.cuda.is_available() else 0,
        'depth_shape': depth.shape,
    }

if __name__ == "__main__":
    print("开始测试 Depth-Anything-V2 模型")
    print(f"CUDA 可用: {torch.cuda.is_available()}")
    
    results = []
    for size in ['small', 'base', 'large']:
        try:
            result = test_model(size)
            results.append(result)
        except Exception as e:
            print(f"❌ {size} 模型测试失败: {e}")
    
    # 汇总结果
    print(f"\n{'='*60}")
    print("测试结果汇总")
    print(f"{'='*60}")
    print(f"{'模型':<10} {'加载(s)':<10} {'推理(s)':<10} {'显存(GB)':<10}")
    print(f"{'-'*60}")
    for r in results:
        print(f"{r['model_size']:<10} {r['load_time']:<10.2f} {r['infer_time']:<10.2f} {r['memory_gb']:<10.2f}")
    
    print(f"\n{'='*60}")
    print("推荐配置:")
    if results:
        fastest = min(results, key=lambda x: x['infer_time'])
        print(f"✅ 速度最快: {fastest['model_size']} ({fastest['infer_time']:.2f}s)")
        
        if any(r['memory_gb'] < 2.0 for r in results):
            print(f"✅ 显存友好: small ({results[0]['memory_gb']:.2f}GB)")
    print(f"{'='*60}")
```

运行测试：
```bash
docker exec 3dgs-worker python3 /tmp/test_depth_anything.py
```

**验收标准**:
- [ ] Small 模型加载成功
- [ ] 推理时间 < 1s
- [ ] 显存占用 < 2GB

---

#### 任务 3: 真实图片测试（30分钟）

```python
# /tmp/test_real_image.py
"""使用真实图片测试深度估计"""
from transformers import pipeline
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

# 1. 加载模型
pipe = pipeline(
    "depth-estimation",
    model="depth-anything/Depth-Anything-V2-Small"
)

# 2. 使用 TestIMG 的一张图
image_path = "/data/smoke/TestIMG/images/frame_00001.jpg"
image = Image.open(image_path)

# 3. 估计深度
print("估计深度...")
result = pipe(image_path)
depth = np.array(result['depth'])

# 4. 可视化
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].imshow(image)
axes[0].set_title('原图')
axes[0].axis('off')

im = axes[1].imshow(depth, cmap='turbo')
axes[1].set_title('Depth-Anything-V2 深度图')
axes[1].axis('off')
plt.colorbar(im, ax=axes[1])

plt.tight_layout()
plt.savefig('/tmp/depth_test.png', dpi=150, bbox_inches='tight')
print("✅ 保存到: /tmp/depth_test.png")

# 5. 统计信息
print(f"\n深度统计:")
print(f"  尺寸: {depth.shape}")
print(f"  范围: [{depth.min():.2f}, {depth.max():.2f}]")
print(f"  均值: {depth.mean():.2f}")
print(f"  标准差: {depth.std():.2f}")
```

运行：
```bash
docker exec 3dgs-worker python3 /tmp/test_real_image.py
# 查看结果
xdg-open /tmp/depth_test.png  # 或 docker cp 到本地查看
```

---

### Day 1 完成后检查清单

- [ ] transformers 安装成功
- [ ] Small 模型可以运行
- [ ] 推理速度可接受（< 1s）
- [ ] 真实图片深度估计正常
- [ ] 可视化结果看起来合理

**如果全部通过 → 进入 Day 2**  
**如果有问题 → 排查解决**

---

## 📋 Week 1 完整计划

### Day 2: 深度质量对比（6小时）

**目标**: 对比 DA-V2 vs MASt3R 深度质量

**关键产出**:
- 对比可视化图（10张）
- 量化指标表格
- 决策：DA-V2 是否更好

**验收**:
- [ ] 边缘清晰度对比完成
- [ ] 细节丰富度对比完成
- [ ] 结论明确

---

### Day 3: 单图重建原型（8小时）

**目标**: 实现单张图片 3DGS 重建

**关键产出**:
- `monocular_pipeline.py` 实现
- API 端点 `/train/monocular`
- 测试重建 1-2 个场景

**验收**:
- [ ] 单图可以生成 .ply 文件
- [ ] API 可用
- [ ] 重建质量中等（预期）

---

### Day 4: Week 1 总结和决策（2小时）

**产出**:
- Week 1 总结报告
- Go/No-Go 决策
- Week 2 调整计划（如果 Go）

**决策标准**:
- ✅ **Go**: DA-V2 深度质量 > MASt3R
- ❌ **No-Go**: DA-V2 深度质量 ≤ MASt3R

---

## 🔧 开发环境配置

### 推荐 IDE 设置

```python
# .vscode/settings.json (如果用 VSCode)
{
    "python.defaultInterpreterPath": "/usr/bin/python3",
    "python.linting.enabled": true,
    "python.linting.pylintEnabled": true,
    "python.formatting.provider": "black",
    "editor.formatOnSave": true
}
```

### 调试配置

```python
# 在代码中添加调试点
import pdb; pdb.set_trace()

# 或使用 IPython
from IPython import embed; embed()
```

---

## 📊 进度跟踪

### Week 1 进度表

| 日期 | 任务 | 预计 | 实际 | 状态 |
|------|------|------|------|------|
| Day 1 | 环境准备 | 3h | - | ⏳ 待开始 |
| Day 2 | 质量对比 | 6h | - | ⏳ 待开始 |
| Day 3 | 单图原型 | 8h | - | ⏳ 待开始 |
| Day 4 | 总结决策 | 2h | - | ⏳ 待开始 |

每天结束后更新状态。

---

## 🚨 常见问题

### Q1: 模型下载慢怎么办？
```bash
# 使用镜像站
export HF_ENDPOINT=https://hf-mirror.com

# 或手动下载到本地
git lfs install
git clone https://huggingface.co/depth-anything/Depth-Anything-V2-Small
```

### Q2: CUDA out of memory？
```python
# 使用 CPU 推理
pipe = pipeline("depth-estimation", model="...", device=-1)

# 或降低图片分辨率
image = image.resize((W//2, H//2))
```

### Q3: 推理太慢？
```python
# 使用 FP16
pipe = pipeline("depth-estimation", model="...", torch_dtype=torch.float16)

# 或使用 Small 模型
model = "depth-anything/Depth-Anything-V2-Small"
```

---

## 📞 每日同步

### 每天结束时汇报

**汇报内容**:
1. 今天完成了什么
2. 遇到了什么问题
3. 明天计划做什么
4. 需要什么支持

**汇报格式**:
```markdown
## Day X 工作总结

### 完成
- [ ] 任务 1
- [ ] 任务 2

### 问题
- 问题描述
- 解决方案

### 明天计划
- 任务 A
- 任务 B

### 需要支持
- （如果有）
```

---

## ✅ 准备就绪检查

开始 Day 1 之前确认：

- [ ] Docker 服务运行正常
- [ ] GPU 可用（`nvidia-smi`）
- [ ] 网络连接正常（能访问 HuggingFace）
- [ ] 磁盘空间充足（> 10GB）
- [ ] TestIMG 数据可用

**全部就绪 → 明天开始 Day 1！** 🚀

---

## 📚 参考资料速查

### 官方文档
- [Depth-Anything-V2 论文](https://arxiv.org/abs/2406.09414)
- [Transformers 文档](https://huggingface.co/docs/transformers)
- [Pipeline API](https://huggingface.co/docs/transformers/main_classes/pipelines)

### 示例代码
```python
# 最简单的使用
from transformers import pipeline

pipe = pipeline("depth-estimation", 
               model="depth-anything/Depth-Anything-V2-Small")
depth = pipe("image.jpg")['depth']
```

---

**准备好了吗？明天就开始！** 💪
