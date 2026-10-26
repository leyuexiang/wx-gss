import json
import os
import shutil
import uuid
import zipfile
from datetime import datetime
from typing import Literal

from fastapi import Depends, FastAPI, File, Form, HTTPException, Security, UploadFile
from fastapi.responses import FileResponse
from fastapi.security.api_key import APIKeyHeader

from app import config
from app.celery_app import celery_app
from app.pipeline import IMG_EXTS

app = FastAPI(
    title="3DGS Training Service",
    version="1.2.0",
    description="多视角照片 → 3D Gaussian Splatting 模型训练服务",
    docs_url="/docs",
    redoc_url="/redoc",
)

# 添加 CORS 支持（前端需要跨域访问）
from fastapi.middleware.cors import CORSMiddleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 集成管理 API
from app.management_api import router as management_router
app.include_router(management_router)

# 集成视频预检 API
from app.video_precheck_api import router_precheck
app.include_router(router_precheck)

# 集成存储管理 API
from app.storage_api import router as storage_router
app.include_router(storage_router)

_api_key_header = APIKeyHeader(
    name="X-API-Key",
    auto_error=False,
    description="API 密钥，在请求头中传递：X-API-Key: your_key_here"
)


def require_api_key(key: str = Security(_api_key_header)):
    if not config.API_KEY or key != config.API_KEY:
        raise HTTPException(status_code=401, detail="invalid or missing X-API-Key")
    return key


def task_dir(task_id: str) -> str:
    # 防路径穿越
    if not task_id or any(c not in "0123456789abcdef" for c in task_id):
        raise HTTPException(status_code=400, detail="invalid task_id")
    return os.path.join(config.TASKS_DIR, task_id)


def read_status(task_id: str) -> dict:
    path = os.path.join(task_dir(task_id), "status.json")
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="task not found")
    with open(path) as f:
        return json.load(f)


@app.get("/health")
def health():
    """健康检查（无需认证）

    返回：
    - status: ok（正常）或 degraded（Redis 不可用）
    - redis: Redis 连通性
    - disk_free_gb: 可用磁盘空间（GB）
    """
    try:
        celery_app.backend.client.ping()
        redis_ok = True
    except Exception:
        redis_ok = False
    stat = shutil.disk_usage(config.DATA_DIR)
    return {
        "status": "ok" if redis_ok else "degraded",
        "redis": redis_ok,
        "disk_free_gb": round(stat.free / 2**30, 1),
    }


@app.post("/train/recommend")
def recommend_params(
    file: UploadFile = File(..., description="多视角照片压缩包（.zip 格式）"),
    sfm_backend: Literal["colmap", "mast3r"] = Form(
        default="mast3r",
        description="SfM 后端选择"
    )
):
    """获取训练参数推荐（不实际训练）

    上传照片包，系统会分析图片数量并返回推荐的训练参数。

    参数:
    - file: .zip 压缩包，包含多视角照片
    - sfm_backend: 使用的 SfM 后端

    返回:
    - n_images: 图片数量
    - sfm_backend: 使用的后端
    - recommended_params: 推荐的参数配置
    - usage: 使用说明
    """
    import tempfile
    import zipfile
    import os

    # 临时保存文件
    with tempfile.NamedTemporaryFile(delete=False, suffix=".zip") as tmp:
        shutil.copyfileobj(file.file, tmp, length=4 * 1024 * 1024)
        tmp_path = tmp.name

    try:
        # 统计图片数量
        n_images = 0
        with zipfile.ZipFile(tmp_path) as zf:
            for info in zf.infolist():
                if info.is_dir():
                    continue
                name = os.path.basename(info.filename)
                if name and os.path.splitext(name)[1].lower() in IMG_EXTS:
                    n_images += 1

        # 获取推荐参数
        from app.adaptive_params import get_adaptive_params
        recommended = get_adaptive_params(n_images, sfm_backend, gpu_memory_gb=8)

        # 构建使用示例
        usage_cmd = f"""使用推荐参数训练:
curl -X POST http://host:12524/train \\
  -H "X-API-Key: your_key" \\
  -F "file=@photos.zip" \\
  -F "use_adaptive=true"

或手动指定推荐参数:
curl -X POST http://host:12524/train \\
  -H "X-API-Key: your_key" \\
  -F "file=@photos.zip" \\
  -F "sfm_backend={sfm_backend}" \\
  -F "iterations={recommended['iterations']}" \\
  -F "downsample={recommended['downsample']}" \\
  -F "cap_max={recommended['cap_max']}" \\
  -F "auto_cleanup={str(recommended['auto_cleanup']).lower()}" \\
  -F "cleanup_threshold={recommended['cleanup_threshold']}"
"""

        return {
            "n_images": n_images,
            "sfm_backend": sfm_backend,
            "recommended_params": recommended,
            "usage": usage_cmd
        }
    finally:
        os.unlink(tmp_path)


@app.post("/train", dependencies=[Depends(require_api_key)])
def submit_train(
    file: UploadFile = File(..., description="多视角照片压缩包（.zip 格式）"),
    sfm_backend: Literal["colmap", "mast3r"] = Form(
        default=config.SFM_BACKEND,
        description="SfM 后端选择：colmap（密集拍摄，20-500张）或 mast3r（少视角，2-15张）"
    ),
    use_adaptive: bool = Form(
        default=False,
        description="使用自适应参数推荐（自动根据图片数量选择最优参数）"
    ),
    iterations: int = Form(
        default=None,
        ge=500,
        le=config.MAX_ITERATIONS,
        description="训练迭代次数（500-30000），None 时使用推荐值"
    ),
    downsample: int = Form(
        default=None,
        description="图片降采样倍数（1/2/4/8），None 时使用推荐值"
    ),
    cap_max: int = Form(
        default=None,
        ge=100_000,
        le=1_500_000,
        description="高斯数量上限（100000-1500000），None 时使用推荐值"
    ),
    auto_cleanup: bool = Form(
        default=None,
        description="自动清理杂点和离群高斯，None 时使用推荐值"
    ),
    cleanup_threshold: float = Form(
        default=None,
        ge=0.01,
        le=0.3,
        description="清理阈值，None 时使用推荐值"
    ),
    enable_depth_reg: bool = Form(
        default=False,
        description="⚠️ 实验性：深度正则化（当前参数可能降低质量，不推荐使用）"
    ),
    depth_reg_weight: float = Form(
        default=0.02,
        ge=0.0,
        le=1.0,
        description="深度正则化权重（实验性功能，建议 0.01-0.05）"
    ),
    enable_pose_opt: bool = Form(
        default=False,
        description="⚠️ 实验性：位姿优化（速度慢70%且质量未验证，不推荐使用）"
    ),
):
    """提交训练任务

    上传多视角照片压缩包，启动 3DGS 训练流程。

    参数：
    - file: .zip 压缩包，包含多视角照片（jpg/png/webp/bmp/tif）
    - sfm_backend: SfM 后端选择
      - colmap: 密集拍摄（20-500张），位姿精度更高
      - mast3r: 少视角（2-15张），适合手机拍摄
    - iterations: 训练迭代次数（500-30000），默认 7000
    - downsample: 图片降采样倍数（1/2/4/8），默认 4
      - 1: 原始分辨率（需大显存）
      - 2: 1/4 像素（推荐，质量与显存平衡）
      - 4: 1/16 像素（默认，8GB 显存安全）
      - 8: 1/64 像素（极限省显存）
    - cap_max: 高斯数量上限（100000-1500000），默认 1000000

    返回：
    - task_id: 任务ID（用于查询状态和下载结果）
    - status: queued（已加入队列）

    示例（密集拍摄）：
    ```bash
    curl -X POST http://host:12524/train \\
      -H "X-API-Key: your_key" \\
      -F "file=@photos.zip" \\
      -F "sfm_backend=colmap" \\
      -F "iterations=7000" \\
      -F "downsample=4"
    ```

    示例（少视角）：
    ```bash
    curl -X POST http://host:12524/train \\
      -H "X-API-Key: your_key" \\
      -F "file=@photos.zip" \\
      -F "sfm_backend=mast3r" \\
      -F "iterations=3000" \\
      -F "downsample=2"
    ```
    """
    if not file.filename.lower().endswith(".zip"):
        raise HTTPException(status_code=400, detail="only .zip archives are accepted")

    task_id = uuid.uuid4().hex
    tdir = task_dir(task_id)
    os.makedirs(tdir)

    zip_path = os.path.join(tdir, "input.zip")
    with open(zip_path, "wb") as f:
        shutil.copyfileobj(file.file, f, length=4 * 1024 * 1024)

    # 早期快速失败提示：真正的强校验在 pipeline.py 里（解压后按实际张数校验）
    n_images = 0
    try:
        with zipfile.ZipFile(zip_path) as zf:
            n_images = sum(
                1 for info in zf.infolist()
                if not info.is_dir()
                and os.path.splitext(info.filename)[1].lower() in IMG_EXTS
            )
    except zipfile.BadZipFile:
        raise HTTPException(status_code=400, detail="invalid zip archive")
    min_images, max_images = (
        (config.MAST3R_MIN_IMAGES, config.MAST3R_MAX_IMAGES)
        if sfm_backend == "mast3r"
        else (config.MIN_IMAGES, config.MAX_IMAGES)
    )
    if not (min_images <= n_images <= max_images):
        raise HTTPException(
            status_code=400,
            detail=f"图片数量为 {n_images} 张，sfm_backend={sfm_backend} 要求 {min_images}-{max_images} 张",
        )

    # 自适应参数推荐
    if use_adaptive or any(p is None for p in [iterations, downsample, cap_max, auto_cleanup, cleanup_threshold]):
        from app.adaptive_params import get_adaptive_params, apply_adaptive_params

        recommended = get_adaptive_params(n_images, sfm_backend, gpu_memory_gb=8)

        # 合并用户参数和推荐参数
        final_params = apply_adaptive_params(recommended, {
            "iterations": iterations,
            "downsample": downsample,
            "cap_max": cap_max,
            "auto_cleanup": auto_cleanup,
            "cleanup_threshold": cleanup_threshold
        })

        iterations = final_params["iterations"]
        downsample = final_params["downsample"]
        cap_max = final_params["cap_max"]
        auto_cleanup = final_params["auto_cleanup"]
        cleanup_threshold = final_params["cleanup_threshold"]

        print(f"应用自适应参数: iterations={iterations}, downsample={downsample}, "
              f"cap_max={cap_max}, reason={recommended['reason']}")
    else:
        # 使用用户提供的参数或默认值
        iterations = iterations if iterations is not None else config.DEFAULT_ITERATIONS
        downsample = downsample if downsample is not None else config.DEFAULT_DOWNSAMPLE
        cap_max = cap_max if cap_max is not None else config.DEFAULT_CAP_MAX
        auto_cleanup = auto_cleanup if auto_cleanup is not None else True
        cleanup_threshold = cleanup_threshold if cleanup_threshold is not None else 0.1

    # 参数验证
    if not (500 <= iterations <= config.MAX_ITERATIONS):
        raise HTTPException(status_code=400, detail=f"iterations must be in [500, {config.MAX_ITERATIONS}]")
    if downsample not in (1, 2, 4, 8):
        raise HTTPException(status_code=400, detail="downsample must be one of 1/2/4/8")
    if not (100_000 <= cap_max <= 1_500_000):
        raise HTTPException(status_code=400, detail="cap_max must be in [100000, 1500000]")
    if not (0.01 <= cleanup_threshold <= 0.3):
        raise HTTPException(status_code=400, detail="cleanup_threshold must be in [0.01, 0.3]")
    if sfm_backend not in ("colmap", "mast3r"):
        raise HTTPException(status_code=400, detail="sfm_backend must be one of colmap/mast3r")

    now = datetime.now().isoformat(timespec="seconds")
    status = {
        "task_id": task_id,
        "status": "queued",
        "stage": "queued",
        "progress": 0.0,
        "message": "",
        "error": None,
        "params": {
            "iterations": iterations,
            "downsample": downsample,
            "cap_max": cap_max,
            "sfm_backend": sfm_backend,
        },
        "created_at": now,
        "updated_at": now,
    }
    with open(os.path.join(tdir, "status.json"), "w") as f:
        json.dump(status, f, indent=2, ensure_ascii=False)

    celery_app.send_task(
        "app.tasks.run_training",
        args=[task_id],
        kwargs={
            "iterations": iterations,
            "downsample": downsample,
            "cap_max": cap_max,
            "sfm_backend": sfm_backend,
            "auto_cleanup": auto_cleanup,
            "cleanup_threshold": cleanup_threshold,
            "enable_depth_reg": enable_depth_reg,
            "depth_reg_weight": depth_reg_weight,
            "enable_pose_opt": enable_pose_opt,
        },
        task_id=task_id,
        queue=config.QUEUE_NAME,
    )
    return {"task_id": task_id, "status": "queued"}


@app.get("/train/{task_id}", dependencies=[Depends(require_api_key)])
def get_status(task_id: str):
    """查询任务状态

    参数：
    - task_id: 任务ID（由 /train 接口返回）

    返回：
    - status: queued/colmap/mast3r/training/packaging/done/failed
    - progress: 进度（0.0-1.0）
    - message: 当前状态描述
    - error: 错误信息（失败时）
    """
    return read_status(task_id)


@app.get("/train/{task_id}/result", dependencies=[Depends(require_api_key)])
def get_result(task_id: str):
    """下载训练结果

    参数：
    - task_id: 任务ID

    返回：
    - result.zip 文件，包含：
      - point_cloud.ply（标准格式）
      - point_cloud.splat（Web格式）
      - train_stats.json（训练统计）
    """
    status = read_status(task_id)
    if status["status"] != "done":
        raise HTTPException(status_code=409, detail=f"task is {status['status']}, not done")
    path = os.path.join(task_dir(task_id), "output", "result.zip")
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="result file missing")
    return FileResponse(path, media_type="application/zip", filename=f"3dgs_{task_id}.zip")


@app.delete("/train/{task_id}", dependencies=[Depends(require_api_key)])
def delete_task(task_id: str):
    """删除任务

    参数：
    - task_id: 任务ID

    注意：运行中的任务无法删除（返回 409）
    """
    status = read_status(task_id)
    if status["status"] in ("colmap", "mast3r", "training", "packaging"):
        raise HTTPException(status_code=409, detail="task is running, cannot delete")
    shutil.rmtree(task_dir(task_id), ignore_errors=True)
    return {"deleted": task_id}
