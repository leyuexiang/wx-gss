import os

REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379/3")
API_KEY = os.environ.get("API_KEY", "")
DATA_DIR = os.environ.get("DATA_DIR", "/data")
TASKS_DIR = os.path.join(DATA_DIR, "tasks")

# 训练默认参数（8GB 显存，vllm 常驻 1.5GB，预算 ~6GB）
DEFAULT_ITERATIONS = 7000
MAX_ITERATIONS = 30000
DEFAULT_DOWNSAMPLE = 4          # 图片降采样倍数
DEFAULT_CAP_MAX = 1_000_000     # 高斯数量硬上限
MAX_IMAGES = 500
MIN_IMAGES = 20
QUEUE_NAME = "3dgs"

# COLMAP GPU SIFT 开关：镜像内为 CUDA 版 COLMAP，需 worker 具备 graphics 能力；
# 置 0 可回退 CPU SIFT
COLMAP_USE_GPU = os.environ.get("COLMAP_USE_GPU", "1").lower() in ("1", "true", "yes")

# COLMAP 特征匹配器最大匹配数：0 表示无限制，推荐 32768（COLMAP 默认值）
COLMAP_MAX_NUM_MATCHES = int(os.environ.get("COLMAP_MAX_NUM_MATCHES", "32768"))

# SfM 后端：colmap（默认，密集拍摄 ≥20 张，位姿精度更优）｜ mast3r（少视角，2-15 张）
# 详见 docs/mast3r-migration.md。/train 请求可用 sfm_backend 覆盖此默认值
SFM_BACKEND = os.environ.get("SFM_BACKEND", "colmap")

MAST3R_MIN_IMAGES = 2
# complete 场景图显存随图片数平方增长：Phase 0 实测 5 张 2.95GB，36 张 complete 会 OOM（>6.1GB）。
# 15 是外推的保守值，落地前需用 10/15 张实拍图实测显存峰值校正（见 mast3r-migration.md 验证记录）
MAST3R_MAX_IMAGES = 15
MAST3R_REPO_DIR = os.path.join(DATA_DIR, "poc/mast3r/mast3r")
MAST3R_CKPT = os.path.join(
    DATA_DIR, "poc/mast3r/checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth"
)
