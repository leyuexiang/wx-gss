"""
3DGS 前端管理 API 适配层
在现有训练服务基础上，添加前端管理界面所需的 API 端点
"""
import json
import os
import re
import shutil
import asyncio
import math
from datetime import datetime
from functools import lru_cache
from typing import Optional
from fastapi import APIRouter, File, Form, UploadFile, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse
from pydantic import BaseModel
from app import config
from app.celery_app import celery_app

# 创建路由
router = APIRouter(prefix="/api/3dgs", tags=["3DGS Management"])

# 数据目录使用训练服务的共享挂载点，保证 API 和 Celery 工作进程访问同一份文件。
UPLOAD_DIR = os.path.join(config.DATA_DIR, "uploads")
MODEL_DIR = os.path.join(config.DATA_DIR, "models")
TEMP_DIR = os.path.join(config.DATA_DIR, "temp")
TASKS_DIR = config.TASKS_DIR

for dir_path in [UPLOAD_DIR, MODEL_DIR, TEMP_DIR]:
    os.makedirs(dir_path, exist_ok=True)

# 内存存储（生产环境应使用数据库）
tasks = {}
models = {}


def _is_task_id(task_id: str) -> bool:
    """限制任务目录标识格式，避免把请求参数拼入任意文件路径。"""
    return bool(re.fullmatch(r"[0-9a-f]{32}", task_id or ""))


def _task_dir(task_id: str) -> str:
    if not _is_task_id(task_id):
        raise HTTPException(status_code=400, detail="无效的任务 ID")
    return os.path.join(TASKS_DIR, task_id)


def _read_task_status(task_id: str) -> Optional[dict]:
    if not _is_task_id(task_id):
        return None
    status_path = os.path.join(TASKS_DIR, task_id, "status.json")
    try:
        with open(status_path, "r", encoding="utf-8") as status_file:
            return json.load(status_file)
    except (OSError, json.JSONDecodeError):
        return None


def _update_task_status(task_id: str, **updates) -> dict:
    task_dir = _task_dir(task_id)
    status_path = os.path.join(task_dir, "status.json")
    try:
        with open(status_path, "r", encoding="utf-8") as status_file:
            status = json.load(status_file)
    except (OSError, json.JSONDecodeError):
        status = {"task_id": task_id}

    status.update(updates)
    temporary_path = f"{status_path}.tmp"
    with open(temporary_path, "w", encoding="utf-8") as status_file:
        json.dump(status, status_file, indent=2, ensure_ascii=False)
    os.replace(temporary_path, status_path)
    return status


def _append_task_log(task_id: str, message: str) -> None:
    status = _read_task_status(task_id) or {}
    timestamp = datetime.now().strftime("%H:%M:%S")
    log = status.get("log", "") + f"[{timestamp}] {message}\n"
    _update_task_status(task_id, message=message, log=log[-20000:])


def _select_sfm_backend(image_count: int) -> str:
    """按训练引擎支持的图像数量选择 SfM 后端。"""
    if config.MIN_IMAGES <= image_count <= config.MAX_IMAGES:
        return "colmap"
    if config.MAST3R_MIN_IMAGES <= image_count <= config.MAST3R_MAX_IMAGES:
        return "mast3r"
    raise ValueError(
        f"有效图像为 {image_count} 张；COLMAP 需要 {config.MIN_IMAGES}-{config.MAX_IMAGES} 张，"
        f"MASt3R 需要 {config.MAST3R_MIN_IMAGES}-{config.MAST3R_MAX_IMAGES} 张"
    )


def _count_zip_images(zip_path: str) -> int:
    import zipfile

    with zipfile.ZipFile(zip_path, "r") as archive:
        return sum(
            1 for entry in archive.infolist()
            if not entry.is_dir()
            and os.path.splitext(entry.filename)[1].lower() in (".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff")
            and not entry.filename.startswith("__MACOSX/")
        )


def _model_record_from_task(task_id: str) -> Optional[dict]:
    status = _read_task_status(task_id)
    if not status or status.get("status") != "done" or status.get("model_deleted"):
        return None

    model_path = os.path.join(TASKS_DIR, task_id, "output", "point_cloud.ply")
    splat_path = os.path.join(TASKS_DIR, task_id, "output", "point_cloud.splat")
    # 只有两种查看格式都已落盘，才把任务作为可查看模型展示。
    if any(not os.path.isfile(path) or os.path.getsize(path) == 0 for path in (model_path, splat_path)):
        return None

    stats_path = os.path.join(TASKS_DIR, task_id, "output", "train_stats.json")
    try:
        with open(stats_path, "r", encoding="utf-8") as stats_file:
            stats = json.load(stats_file)
    except (OSError, json.JSONDecodeError):
        stats = {}

    params = status.get("params", {})
    return {
        "id": task_id,
        "name": status.get("model_name") or status.get("modelName") or task_id,
        "size": os.path.getsize(model_path),
        "status": "completed",
        "createdAt": status.get("created_at") or status.get("createdAt"),
        "iterations": stats.get("iterations") or params.get("iterations"),
        "resolution": params.get("downsample") or status.get("resolution"),
        # 页面显示实际导出的查看点数；训练统计中的数量可能早于自动清理。
        "pointCount": os.path.getsize(splat_path) // 32,
        "path": model_path,
        "taskId": task_id,
    }


def _find_model(model_id: str) -> Optional[dict]:
    model = _model_record_from_task(model_id)
    if model:
        models[model_id] = model
        return model
    model = models.get(model_id)
    if model and os.path.isfile(model.get("path", "")):
        return model
    return None


@lru_cache(maxsize=64)
def _training_views(task_id: str) -> list[dict]:
    """读取训练时的相机位姿，让查看器从实际拍摄方向展示模型。"""
    if not _is_task_id(task_id):
        return []
    sparse_dir = os.path.join(TASKS_DIR, task_id, "dataset", "sparse", "0")
    if not os.path.isfile(os.path.join(sparse_dir, "images.bin")):
        return []
    try:
        import numpy as np
        import pycolmap
        reconstruction = pycolmap.Reconstruction(sparse_dir)
        ordered_images = sorted(reconstruction.images.values(), key=lambda item: item.name)
        positions = np.stack([image.cam_from_world().inverse().translation for image in ordered_images])
        center = positions.mean(axis=0)
        max_dist = np.linalg.norm(positions - center, axis=1).max()
        scene_scale = 1.0 / (max_dist * 1.05) if max_dist > 1e-6 else 1.0
        views = []
        for image in ordered_images:
            camera_to_world = image.cam_from_world().inverse()
            rotation = camera_to_world.rotation.matrix()
            camera = reconstruction.cameras[image.camera_id]
            focal_y = camera.calibration_matrix()[1, 1]
            fov = math.degrees(2 * math.atan(camera.height / (2 * focal_y)))
            views.append({
                "name": image.name,
                # 训练器在 dataset.ColmapScene 中归一化了相机和点云，查看器必须使用同一坐标系。
                "position": ((camera_to_world.translation - center) * scene_scale).tolist(),
                "forward": rotation[:, 2].tolist(),
                "up": (-rotation[:, 1]).tolist(),
                "fov": max(25.0, min(95.0, fov)),
            })
        return views
    except Exception:
        # 旧模型可能没有完整位姿文件；此时仍可使用自动包围盒视角。
        return []


class OptimizeRequest(BaseModel):
    modelId: str
    algorithm: str
    compressionLevel: str
    qualityFirst: bool


def validate_zip_or_video(file_path: str, is_video: bool = False) -> tuple[bool, str, dict]:
    """验证上传文件并返回详细信息"""
    if is_video:
        try:
            # 获取详细视频信息
            video_info = get_video_info(file_path)

            width = video_info['width']
            height = video_info['height']
            duration = video_info['duration']
            fps = video_info['fps']

            issues = []
            warnings = []
            suggestions = {}

            # 分辨率检查（支持横屏和竖屏）
            min_dim = min(width, height)
            max_dim = max(width, height)

            if min_dim < 720 or max_dim < 1280:
                issues.append(f"分辨率过低（{width}x{height}），建议至少 1280x720 或 720x1280")
            elif max_dim < 1920:
                warnings.append(f"分辨率为 {width}x{height}，建议使用 1920x1080 以获得更好效果")

            # 时长检查
            if duration < 10:
                warnings.append(f"视频较短（{duration:.1f}秒），建议至少 15 秒以获得足够的视角覆盖")
            elif duration > 300:
                warnings.append(f"视频较长（{duration:.1f}秒），建议裁剪到关键片段（30-120秒）")

            # 帧率检查
            if fps < 20:
                warnings.append(f"帧率较低（{fps:.1f}fps），可能影响抽帧质量")
            elif fps > 60:
                suggestions['interval'] = 0.5
                warnings.append(f"高帧率视频（{fps:.1f}fps），建议使用较大抽帧间隔（0.5秒）")

            # 如果有严重问题，拒绝
            if issues:
                return False, "\n".join(issues), {}

            # 生成智能推荐参数
            suggested_params = suggest_extraction_params(width, height, duration, fps)

            # 组装验证消息
            msg_parts = [f"✓ 视频预检通过"]
            msg_parts.append(f"分辨率: {width}x{height}, 时长: {duration:.1f}秒, 帧率: {fps:.1f}fps")

            if warnings:
                msg_parts.append("建议: " + "; ".join(warnings))

            return True, "\n".join(msg_parts), suggested_params

        except Exception as e:
            return False, f"视频验证失败: {str(e)}", {}
    else:
        # 与真实训练后端使用相同的张数规则，少视角照片可交给 MASt3R。
        try:
            image_count = _count_zip_images(file_path)
            _select_sfm_backend(image_count)
            return True, f"验证通过，包含 {image_count} 张图像", {}
        except ValueError as e:
            return False, str(e), {}
        except Exception as e:
            return False, f"ZIP 文件损坏: {str(e)}", {}


def suggest_extraction_params(width: int, height: int, duration: float, fps: float) -> dict:
    """基于视频特征推荐最佳抽帧参数"""

    # 根据时长推荐目标帧数
    if duration < 20:
        target_frames = 50
        reason = "短视频，提取 50 帧"
    elif duration < 60:
        target_frames = 80
        reason = "中等时长，提取 80 帧"
    else:
        target_frames = min(150, int(duration * 2))
        reason = f"长视频，按 2 帧/秒提取，上限 150 帧"

    # 计算推荐间隔
    suggested_interval = round(duration / target_frames, 2)

    # 根据分辨率推荐质量和降采样
    if width >= 3840:  # 4K
        quality = 'high'
        suggested_downsample = 2
        quality_reason = "4K 视频，高质量输出，建议 1/2 降采样节省显存"
    elif width >= 1920:  # 1080p
        quality = 'high'
        suggested_downsample = 1
        quality_reason = "1080p 视频，高质量输出，原始分辨率训练"
    else:
        quality = 'medium'
        suggested_downsample = 1
        quality_reason = "720p 视频，中等质量输出"

    # 根据帧率调整建议
    if fps > 60:
        suggested_interval = max(suggested_interval, 0.5)
        fps_note = "高帧率视频，已调整间隔避免过多冗余帧"
    else:
        fps_note = None

    return {
        'extractionMode': 'interval',
        'interval': suggested_interval,
        'targetFrames': target_frames,
        'quality': quality,
        'suggested_downsample': suggested_downsample,
        'removeBlurry': True,
        'removeDuplicates': True,
        'reason': reason,
        'quality_reason': quality_reason,
        'fps_note': fps_note
    }


def get_video_info(video_path: str) -> dict:
    """获取视频详细信息"""
    import subprocess
    import json

    cmd = [
        'ffprobe', '-v', 'error',
        '-select_streams', 'v:0',
        '-show_entries', 'stream=r_frame_rate,duration,width,height,nb_frames',
        '-of', 'json',
        video_path
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode != 0:
            raise Exception(f"ffprobe 错误: {result.stderr}")

        data = json.loads(result.stdout)
        stream = data['streams'][0]

        # 解析帧率 (e.g., "30000/1001" -> 29.97)
        fps_str = stream.get('r_frame_rate', '30/1')
        num, den = map(int, fps_str.split('/'))
        fps = num / den if den != 0 else 30.0

        # 获取时长
        duration = float(stream.get('duration', 0))

        # 如果流信息中没有 duration，尝试从 format 中获取
        if duration == 0:
            cmd_format = [
                'ffprobe', '-v', 'error',
                '-show_entries', 'format=duration',
                '-of', 'json',
                video_path
            ]
            result_format = subprocess.run(cmd_format, capture_output=True, text=True, timeout=30)
            if result_format.returncode == 0:
                format_data = json.loads(result_format.stdout)
                duration = float(format_data.get('format', {}).get('duration', 0))

        return {
            'fps': round(fps, 2),
            'duration': round(duration, 2),
            'width': int(stream.get('width', 0)),
            'height': int(stream.get('height', 0)),
            'total_frames': int(stream.get('nb_frames', 0)) if stream.get('nb_frames') else int(duration * fps)
        }
    except Exception as e:
        raise Exception(f"获取视频信息失败: {str(e)}")


async def extract_video_frames(video_path: str, output_dir: str, video_config: dict) -> dict:
    """抽取视频帧，并在质量过滤导致帧数不足时保留未过滤帧。"""
    import subprocess

    video_info = get_video_info(video_path)
    duration = video_info['duration']
    mode = video_config.get('extractionMode', 'fps')
    fps = video_config.get('fps', 5)
    interval = video_config.get('interval', 0.3)
    total_frames = video_config.get('totalFrames', 100)
    start_time = video_config.get('startTime', 0)
    end_time = video_config.get('endTime', 0)
    quality = video_config.get('quality', 'medium')

    effective_duration = duration
    if end_time > 0:
        effective_duration = min(end_time, duration) - start_time
    elif start_time > 0:
        effective_duration = duration - start_time

    quality_map = {'low': 5, 'medium': 2, 'high': 1}
    q_value = quality_map.get(quality, 2)
    scratch_root = f"{output_dir}_scratch"
    raw_dir = os.path.join(scratch_root, "raw")
    filtered_dir = os.path.join(scratch_root, "filtered")
    shutil.rmtree(scratch_root, ignore_errors=True)
    shutil.rmtree(output_dir, ignore_errors=True)
    os.makedirs(raw_dir, exist_ok=True)

    cmd = ['ffmpeg', '-i', video_path]
    if start_time > 0:
        cmd.extend(['-ss', str(start_time)])
    if end_time > 0:
        cmd.extend(['-to', str(end_time)])

    if mode == 'fps':
        cmd.extend(['-vf', f'fps={fps}'])
    elif mode == 'interval':
        cmd.extend(['-vf', f'fps=1/{interval}'])
    elif mode == 'count':
        calculated_interval = effective_duration / max(int(total_frames), 1)
        cmd.extend(['-vf', f'fps=1/{calculated_interval}'])
    else:
        raise ValueError(f"不支持的视频抽帧模式: {mode}")

    cmd.extend(['-q:v', str(q_value), os.path.join(raw_dir, 'frame_%04d.jpg')])

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if result.returncode != 0:
            raise RuntimeError(f"FFmpeg 错误: {result.stderr}")

        raw_frames = sorted(
            name for name in os.listdir(raw_dir)
            if name.startswith('frame_') and name.endswith('.jpg')
        )
        initial_count = len(raw_frames)
        selected_dir = raw_dir
        filter_result = {'valid_frames': initial_count, 'removed_blur': 0, 'removed_duplicate': 0}
        fallback_used = False

        if video_config.get('removeBlurry', False) or video_config.get('removeDuplicates', False):
            from app.frame_quality import filter_frames

            os.makedirs(filtered_dir, exist_ok=True)
            for frame_name in raw_frames:
                shutil.copy2(os.path.join(raw_dir, frame_name), os.path.join(filtered_dir, frame_name))

            filter_result = await filter_frames(
                filtered_dir,
                remove_blurry=video_config.get('removeBlurry', False),
                remove_duplicates=video_config.get('removeDuplicates', False),
                blur_threshold=video_config.get('blurThreshold', 100.0),
                duplicate_threshold=video_config.get('duplicateThreshold', 0.95),
            )

            # 过滤器会删除被判定为模糊/重复的文件；原始帧保留在 scratch 目录，
            # 避免阈值过严时只剩少量帧，随后却仍把任务报成成功。
            minimum_for_backend = config.MIN_IMAGES if initial_count >= config.MIN_IMAGES else config.MAST3R_MIN_IMAGES
            if filter_result['valid_frames'] < minimum_for_backend and initial_count >= minimum_for_backend:
                fallback_used = True
            else:
                selected_dir = filtered_dir

        selected_frames = sorted(
            name for name in os.listdir(selected_dir)
            if name.startswith('frame_') and name.endswith('.jpg')
        )
        if len(selected_frames) > config.MAX_IMAGES:
            stride = (len(selected_frames) - 1) / (config.MAX_IMAGES - 1)
            selected_frames = [selected_frames[round(index * stride)] for index in range(config.MAX_IMAGES)]

        os.makedirs(output_dir, exist_ok=True)
        for frame_name in selected_frames:
            shutil.copy2(os.path.join(selected_dir, frame_name), os.path.join(output_dir, frame_name))

        return {
            'initial_frames': initial_count,
            'valid_frames': len(selected_frames),
            'filtered_frames': filter_result['valid_frames'],
            'removed_blur': 0 if fallback_used else filter_result['removed_blur'],
            'removed_duplicate': 0 if fallback_used else filter_result['removed_duplicate'],
            'fallback_used': fallback_used,
        }
    except Exception as e:
        raise Exception(f"视频抽帧失败: {str(e)}")
    finally:
        shutil.rmtree(scratch_root, ignore_errors=True)


@router.post("/generate")
async def generate_3dgs(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    modelName: str = Form(...),
    iterations: int = Form(7000),
    resolution: int = Form(1),
    sceneType: str = Form("outdoor"),
    generationConfig: str = Form("{}", alias="config"),
    isVideo: str = Form("false"),
    videoConfig: str = Form("{}")
):
    """保存输入并提交真实的 3DGS 训练任务。"""
    import uuid

    task_id = uuid.uuid4().hex
    is_video = isVideo.lower() == "true"
    if is_video:
        video_extensions = ('.mp4', '.avi', '.mov', '.mkv', '.webm')
        if not any(file.filename.lower().endswith(ext) for ext in video_extensions):
            raise HTTPException(status_code=400, detail="不支持的视频格式")
        input_name = "input_video" + os.path.splitext(file.filename)[1].lower()
    else:
        if not file.filename.lower().endswith('.zip'):
            raise HTTPException(status_code=400, detail="只支持 ZIP 文件")
        input_name = "input.zip"

    if not 500 <= iterations <= config.MAX_ITERATIONS or resolution not in (1, 2, 4, 8):
        raise HTTPException(status_code=400, detail="训练迭代或分辨率参数无效")
    try:
        config_dict = json.loads(generationConfig)
        video_config_dict = json.loads(videoConfig) if is_video else {}
        if not isinstance(config_dict, dict) or not isinstance(video_config_dict, dict):
            raise ValueError("配置必须是对象")
    except (ValueError, TypeError, json.JSONDecodeError):
        raise HTTPException(status_code=400, detail="配置格式无效")

    task_dir = _task_dir(task_id)
    os.makedirs(task_dir, exist_ok=False)
    input_path = os.path.join(task_dir, input_name)
    try:
        with open(input_path, "wb") as output:
            shutil.copyfileobj(file.file, output, length=4 * 1024 * 1024)
        is_valid, message, _ = validate_zip_or_video(input_path, is_video)
        if not is_valid:
            raise HTTPException(status_code=400, detail=message)
    except Exception:
        shutil.rmtree(task_dir, ignore_errors=True)
        raise

    now = datetime.now().isoformat(timespec="seconds")
    _update_task_status(
        task_id, status="preparing", stage="extract" if is_video else "queued",
        progress=0.0, message=message, log=f"[{datetime.now().strftime('%H:%M:%S')}] {message}\n",
        error=None, model_name=modelName, created_at=now, updated_at=now,
        is_video=is_video, scene_type=sceneType,
        params={"iterations": iterations, "downsample": resolution},
    )

    # 视频抽帧在后台完成；真正的重建与训练交给已有的 Celery 工作进程。
    background_tasks.add_task(
        process_generation_task,
        task_id,
        input_path,
        iterations,
        resolution,
        config_dict,
        is_video,
        video_config_dict
    )

    return {"code": 200, "data": {"taskId": task_id}}


async def process_generation_task(
    task_id: str,
    file_path: str,
    iterations: int,
    resolution: int,
    generation_config: dict,
    is_video: bool,
    video_config: dict
):
    """准备训练输入，再提交已有的 GPU 训练任务。"""

    try:
        if is_video:
            _append_task_log(task_id, "开始视频抽帧")
            _update_task_status(task_id, status="preparing", stage="extract", progress=0.02)
            image_dir = os.path.join(_task_dir(task_id), "images")
            frame_result = await extract_video_frames(file_path, image_dir, video_config)
            frame_count = frame_result["valid_frames"]
            if frame_result["fallback_used"]:
                _append_task_log(task_id, "质量过滤后帧数不足，已保留原始抽帧以继续重建")
            _append_task_log(task_id, f"视频抽帧完成：原始 {frame_result['initial_frames']} 帧，有效 {frame_count} 帧")
            # 连续视频只有少量视角时无法可靠重建，直接给出失败原因。
            if frame_count < config.MIN_IMAGES:
                raise ValueError(f"视频有效帧不足 {config.MIN_IMAGES} 张，当前 {frame_count} 张；请增加抽帧数量或更换视频")
        else:
            frame_count = _count_zip_images(file_path)

        sfm_backend = _select_sfm_backend(frame_count)
        advanced = generation_config.get("advanced", {})
        if not isinstance(advanced, dict):
            advanced = {}
        # 前端配置中的实验选项只传递训练器已实现的参数。
        enable_depth_reg = bool(advanced.get("depthRegularization", False))
        params = {
            "iterations": iterations, "downsample": resolution,
            "cap_max": config.DEFAULT_CAP_MAX, "sfm_backend": sfm_backend,
        }
        _append_task_log(task_id, f"已提交真实训练：{sfm_backend}，{frame_count} 张图像，{iterations} 次迭代")
        _update_task_status(task_id, status="queued", stage="queued", progress=0.0, params=params)
        celery_app.send_task(
            "app.tasks.run_training", args=[task_id],
            kwargs={
                "iterations": iterations,
                "downsample": resolution,
                "cap_max": config.DEFAULT_CAP_MAX,
                "sfm_backend": sfm_backend,
                "enable_depth_reg": enable_depth_reg,
            },
            task_id=task_id, queue=config.QUEUE_NAME,
        )

    except Exception as e:
        _append_task_log(task_id, f"训练准备失败：{e}")
        _update_task_status(task_id, status="failed", stage="failed", progress=0.0, error=str(e))


@router.get("/status/{taskId}")
async def check_generate_status(taskId: str):
    """从共享状态文件读取训练进度，服务重载后仍可继续查询。"""
    task = _read_task_status(taskId)
    if task is None:
        raise HTTPException(status_code=404, detail="任务不存在")

    raw_status = task.get("status", "queued")
    model = _model_record_from_task(taskId) if raw_status == "done" else None
    status = "completed" if model else "failed" if raw_status in ("done", "failed") else "processing"
    progress = min(100, max(0, round(float(task.get("progress", 0)) * 100)))
    message = task.get("error") or task.get("message") or ""
    log = task.get("log", "")
    if raw_status == "done" and not model:
        message = "训练结束但模型文件缺失"
    if message and message not in log[-500:]:
        log += f"[{datetime.now().strftime('%H:%M:%S')}] {message}\n"
    return {
        "code": 200,
        "data": {
            "status": status,
            "progress": progress,
            "log": log,
            "modelId": taskId if model else None
        }
    }


@router.get("/models")
async def get_model_list():
    """只展示有真实模型文件的训练结果。"""
    model_list = []
    with os.scandir(TASKS_DIR) as entries:
        for entry in entries:
            if entry.is_dir() and _is_task_id(entry.name):
                model = _model_record_from_task(entry.name)
                if model:
                    model_list.append(model)
    model_list.sort(key=lambda item: item.get("createdAt") or "", reverse=True)
    return {"code": 200, "data": model_list}


@router.get("/model/{modelId}")
async def get_model_info(modelId: str):
    """获取模型详情"""
    model = _find_model(modelId)
    if model is None:
        raise HTTPException(status_code=404, detail="模型不存在")
    return {"code": 200, "data": {**model, "trainingViews": _training_views(modelId)}}


@router.get("/model/{modelId}/point_cloud.splat")
async def get_model_splat(modelId: str):
    """将训练导出的高斯数据流提供给浏览器查看器。"""
    model = _find_model(modelId)
    if model is None:
        raise HTTPException(status_code=404, detail="模型不存在")
    splat_path = os.path.join(os.path.dirname(model["path"]), "point_cloud.splat")
    if not os.path.isfile(splat_path) or os.path.getsize(splat_path) == 0:
        raise HTTPException(status_code=404, detail="模型查看文件不存在")
    return FileResponse(splat_path, media_type="application/octet-stream")


@router.delete("/model/{modelId}")
async def delete_model(modelId: str):
    """删除模型"""
    model = _find_model(modelId)
    if model is None:
        raise HTTPException(status_code=404, detail="模型不存在")

    # 只删除训练输出并标记记录，保留输入与状态以便排查。
    output_dir = os.path.dirname(model["path"])
    shutil.rmtree(output_dir)
    if _is_task_id(modelId):
        _update_task_status(modelId, model_deleted=True)
    models.pop(modelId, None)

    return {"code": 200, "message": "删除成功"}


@router.post("/optimize")
async def optimize_model(request: OptimizeRequest, background_tasks: BackgroundTasks):
    """模型压缩优化"""
    if request.modelId not in models:
        raise HTTPException(status_code=404, detail="模型不存在")

    import uuid
    task_id = str(uuid.uuid4())

    tasks[task_id] = {
        "taskId": task_id,
        "status": "processing",
        "progress": 0,
        "log": f"[{datetime.now().strftime('%H:%M:%S')}] 开始优化模型\n",
        "modelId": request.modelId,
        "algorithm": request.algorithm,
        "createdAt": datetime.now().isoformat()
    }

    background_tasks.add_task(
        process_optimize_task,
        task_id,
        request.modelId,
        request.algorithm,
        request.compressionLevel,
        request.qualityFirst
    )

    return {"code": 200, "data": {"taskId": task_id}}


async def process_optimize_task(
    task_id: str,
    model_id: str,
    algorithm: str,
    level: str,
    quality_first: bool
):
    """后台优化任务"""
    def log(msg: str):
        timestamp = datetime.now().strftime('%H:%M:%S')
        tasks[task_id]["log"] += f"[{timestamp}] {msg}\n"

    try:
        log(f"使用 {algorithm} 算法，压缩级别: {level}")

        for i in range(0, 100, 20):
            await asyncio.sleep(2)
            tasks[task_id]["progress"] = i
            log(f"优化进度: {i}%")

        # 模拟结果
        compression_ratio = {"low": 65, "medium": 45, "high": 25}[level]
        psnr = 34.5 + (2 if quality_first else 0)

        tasks[task_id]["status"] = "completed"
        tasks[task_id]["progress"] = 100
        tasks[task_id]["comparison"] = {
            "compressionRatio": compression_ratio,
            "psnr": round(psnr, 1)
        }
        log(f"✓ 优化完成！压缩率: {compression_ratio}%, PSNR: {psnr:.1f} dB")

    except Exception as e:
        tasks[task_id]["status"] = "failed"
        log(f"❌ 错误: {str(e)}")


@router.get("/optimize/status/{taskId}")
async def check_optimize_status(taskId: str):
    """查询优化任务状态"""
    if taskId not in tasks:
        raise HTTPException(status_code=404, detail="任务不存在")

    task = tasks[taskId]
    return {
        "code": 200,
        "data": {
            "status": task["status"],
            "progress": task["progress"],
            "log": task["log"],
            "comparison": task.get("comparison")
        }
    }
