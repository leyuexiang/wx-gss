# MASt3R / VGGT 前端改造评估（COLMAP 替代方案）

版本：1.0 ｜ 日期：2026-08-26 ｜ 状态：评估完成，Phase 0 进行中

> **商业属性已确认：内部工具**。MASt3R 的 CC BY-NC-SA 4.0 非商业许可证不构成阻碍，方案 A（MASt3R）可行。
> 若未来服务转为对外商业产品，须改用 VGGT-1B-Commercial 权重或联系 Naver 授权。

## 结论（TL;DR）

技术可行，但**不做"替换"，做可切换的并行后端**：`SFM_BACKEND=colmap|mast3r`。
COLMAP 保留为密集拍摄（≥20 张）场景的默认后端（位姿精度、畸变建模更优）；
MASt3R 兜底少视角场景（COLMAP 特征匹配解不出来的情况，典型如 5–10 张图）。

要管理预期：换前端只解决"位姿解不出来"；少视角要训出能看的 3DGS，
训练侧还需加深度正则（Phase 2），否则 L1+SSIM 纯光度损失会过拟合出浮空伪影。

## 背景与动机

现有管线：`照片 → COLMAP（SIFT 匹配 + 增量 SfM）→ gsplat 训练`。
COLMAP 是纯几何方法，要求相邻视图大量特征匹配，导致：

- 图片数下限 MIN_IMAGES=20，实拍建议 30+；
- 重叠不足时直接失败（smoke 合成数据集 36 张仅注册 2 张，报错 `No good initial image pair found`）；
- 用户少量图片（如 TestIMG.zip 仅 5 张）完全无法进入训练。

市面上"6 张图即可"的产品走的是学习类路线（DUSt3R/MASt3R、VGGT、pixelSplat 等），
用神经网络先验直接预测位姿与几何，绕开传统特征匹配。

## 现有管线的接口边界（改造面）

训练侧（`app/trainer.py` + `app/dataset.py`）只依赖一个目录契约：

```
dataset/
  sparse/0/{cameras.bin, images.bin, points3D.bin}   # PINHOLE 相机
  images_{factor}/*.jpg                               # 降采样图
```

`ColmapScene`（dataset.py）只取四样：`camtoworlds`、`Ks`、`points`、`points_rgb`。

**只要 MASt3R 后端用 pycolmap 构造 `Reconstruction` 写出同布局 bin 文件，
训练侧零改动。** 改造集中在 `pipeline.py`：

- 新增 `_run_mast3r()`（与 `_run_colmap()` 同构）；
- 适配器：MASt3R 位姿/内参/点云 → COLMAP bin（约 100 行）；
- MASt3R 假设针孔模型、无需 undistort，`images_{factor}` 直接由原图降采样，比 COLMAP 流程还少一步。

## 候选技术对比

| 模型 | 许可证 | 方式 | 备注 |
|---|---|---|---|
| DUSt3R / MASt3R（naver） | CC BY-NC-SA 4.0 | 逐对匹配 + 稀疏全局对齐 | 内部工具可用 ✅ |
| InstantSplat（NVlabs） | 依赖 MASt3R 权重 | MASt3R 先验 + GauBA 联合优化 | 少视角 3DGS 的参照实现；3 视角 SSIM 0.3755→0.7624（vs COLMAP+3DGS） |
| VGGT（Meta/Oxford，CVPR2025 Best Paper） | 代码可商用（禁军事）；原版权重非商用，VGGT-1B-Commercial 填表申请后可商用 | 前馈单次推理，直接出位姿+内参+深度+点图 | 商用场景的首选；工程更简单 |

## 硬件与环境约束（RTX 4060 8GB）

| 约束 | 评估 |
|---|---|
| 显存 | 单对推理 512px ≈ 4GB ✅；50 张图全局对齐 ≈ 16GB，8GB 会 OOM ⚠️。缓解：限制场景图规模、retriever 选关键帧、分块对齐。少视角（<20 张）无压力 |
| 权重 | MASt3R ~2.4GB，HF 被墙 → hf-mirror.com，打进镜像或挂 volume |
| 依赖 | github 被墙 → gitee/gitclone.com 镜像（同 COLMAP 编译时的办法）；RoPE CUDA kernel 可选 |
| 与训练共存 | 分阶段执行，前端跑完释放显存再训练 ✅ |
| 速度 | 位姿求解从分钟级降到秒级（单对 0.5–2s） |

## 精度预期

- **少视角（2–10 张）**：MASt3R 完胜，COLMAP 根本解不出来。
- **密集拍摄（30+ 张）**：COLMAP 增量 BA 位姿精度仍更优，且 OPENCV 模型处理镜头畸变；
  MASt3R 针孔假设会忽略手机广角畸变，边缘有几何误差。→ 并行后端而非替换。
- **少视角训练陷阱**：仅换前端不够，需 Phase 2 的深度正则 + 可选 GauBA 式位姿联合优化。

## 分期落地计划

### Phase 0 — 验证（1–2 天）✅ 已完成（2026-08-26）

不改主服务，独立 POC 容器验证三件事：

1. 8GB 显存够不够（目标场景 <20 张图）；
2. 重建质量是否达业务预期（TestIMG.zip 5 张图 + 真实业务照片）；
3. 权重/依赖能否在国内网络环境装齐。

**Phase 0 不过，后续不谈。**

#### Phase 0 结果（实测，RTX 4060 8GB，torch 2.8.0+cu128）

| 验证项 | 结果 | 结论 |
|---|---|---|
| TestIMG.zip（5 张实拍） | 位姿求解 **11s**，显存峰值 **2.61GB**，5 相机全部注册，点云 58 万点，几何结构清晰（mean_conf 2.20） | ✅ 通过 |
| synth 36 张（COLMAP 仅注册 2/36 的失败数据集） | swin 窗口图 **166s**，显存峰值 **3.15GB**，36 相机全部注册、轨迹平滑 | ⚠️ 全部注册但 mean_conf 仅 0.107，几何退化为近平面——该数据集本身病态，MASt3R 至少能出结果（COLMAP 直接失败） |
| 显存边界 | 36 张 **complete 全连接图（1260 对）OOM（>6.1GB）**；swin 窗口图（~3.15GB）正常 | 集成时图规模 >~20 张必须用 swin/retrieval 场景图 |
| 网络环境 | 代码：gitcode.com 镜像（`GitHub_Trending/ma/mast3r`、`gh_mirrors/cr/croco` 等，gitclone/github 代理均不通）；权重：naver 官方 CDN **走代理 9.6MB/s**（直连会断）；pip：清华 tuna | ✅ 全部可解，已踩平 |
| 速度对比 | MASt3R 位姿求解秒级~分钟级 vs COLMAP 分钟级 | ✅ |

POC 资产（保留在宿主机，POC 容器 `mast3r-poc` 仍在运行）：

- 代码：`/home/3dgs/data/poc/mast3r/mast3r`（含 dust3r/croco 子模块，均已按官方固定 commit 检出）
- 权重：`/home/3dgs/data/poc/mast3r/checkpoints/`（2.75GB，已校验完整性）
- 脚本：`poc_sfm.py`（位姿求解+导出 poses.json/points.ply/summary.json）、`viz_poc.py`（轨迹投影图）
- 结果：`out_testimg/`、`out_synth_swin/`、`viz_testimg.png`、`viz_synth.png`

**遗留验证项**：需一批真实业务照片（10–30 张常规拍摄）复测质量，确认非病态数据集上的效果。

### Phase 1 — 可切换前端（3–5 天）

- `config.py` 加 `SFM_BACKEND=colmap|mast3r`（沿用 .env 模式）；
- `pipeline.py` 新增 `_run_mast3r()` + COLMAP bin 适配器；
- `MIN_IMAGES` 按后端区分（mast3r 可放宽到 3–5）；
- COLMAP 保持默认；注册率校验保留。

### Phase 2 — 少视角训练增强（1–2 周，决定效果上限）

- 训练引入 MASt3R/VGGT 深度图正则（depth Pearson loss）；
- 视效果决定是否做位姿联合优化（GauBA 式）；
- 基准：同一批 5–8 张图测试集，对比 Phase 1 vs Phase 2 的 PSNR/SSIM。

## 风险清单

| 风险 | 等级 | 缓解 |
|---|---|---|
| 未来转商用时许可证不合规 | 🟡 中 | 已记录：转 VGGT-Commercial 或 Naver 授权 |
| 8GB 显存在多图全局对齐 OOM | 🟡 中 | 限制关键帧数/分块；少视角场景图少 |
| 针孔假设 vs 手机广角畸变 | 🟡 中 | 密集场景仍用 COLMAP；必要时离线预去畸变 |
| 少视角训练伪影 | 🟡 中 | Phase 0 验证；Phase 2 深度正则兜底 |
| 双后端维护成本 | 🟢 低 | 契约统一（COLMAP bin 布局），测试覆盖切换逻辑 |

## 参考资料

- [naver/mast3r (GitHub)](https://github.com/naver/mast3r) ｜ [CHECKPOINTS_NOTICE](https://github.com/naver/mast3r/blob/main/CHECKPOINTS_NOTICE)
- [NVlabs/InstantSplat (GitHub)](https://github.com/NVlabs/InstantSplat) ｜ [InstantSplat 论文](https://arxiv.org/pdf/2403.20309.pdf)
- [facebookresearch/VGGT (GitHub)](https://github.com/facebookresearch/VGGT) ｜ [VGGT 论文](https://arxiv.org/abs/2503.11651)
