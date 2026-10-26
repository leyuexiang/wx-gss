"""
存储管理 API - 磁盘空间监控、模型归档、批量删除
"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, List
import os

from app.storage_manager import (
    check_disk_space,
    get_storage_stats,
    auto_cleanup_old_tasks,
    archive_old_models,
    cleanup_temp_files
)
from app.task_state import get_task_state_manager

router = APIRouter(prefix="/storage", tags=["storage"])


class CleanupRequest(BaseModel):
    task_id: str
    keep_output: bool = True


class AutoCleanupRequest(BaseModel):
    max_age_days: int = 30
    min_free_gb: float = 10.0


class ArchiveRequest(BaseModel):
    max_age_days: int = 90


@router.get("/disk-space")
async def get_disk_space(path: str = "/home/3dgs/data"):
    """
    获取磁盘空间信息

    Args:
        path: 要检查的路径

    Returns:
        磁盘空间信息
    """
    result = check_disk_space(path)

    if 'error' in result:
        raise HTTPException(status_code=500, detail=result['error'])

    return {
        "code": 200,
        "message": "磁盘空间查询成功",
        "data": result
    }


@router.get("/stats")
async def get_stats():
    """
    获取存储统计信息

    Returns:
        存储统计（各目录大小、文件数量等）
    """
    stats = get_storage_stats()

    return {
        "code": 200,
        "message": "存储统计查询成功",
        "data": stats
    }


@router.post("/cleanup-task")
async def cleanup_task(request: CleanupRequest):
    """
    清理指定任务的临时文件

    Args:
        request: 清理请求

    Returns:
        清理结果
    """
    from app.config import TASKS_DIR

    task_dir = os.path.join(TASKS_DIR, request.task_id)

    if not os.path.exists(task_dir):
        raise HTTPException(status_code=404, detail="任务目录不存在")

    result = cleanup_temp_files(task_dir, keep_output=request.keep_output)

    if 'error' in result:
        raise HTTPException(status_code=500, detail=result['error'])

    return {
        "code": 200,
        "message": f"清理完成，释放 {result['total_size_freed_mb']} MB",
        "data": result
    }


@router.post("/auto-cleanup")
async def auto_cleanup(request: AutoCleanupRequest):
    """
    自动清理旧任务以释放空间

    Args:
        request: 自动清理请求

    Returns:
        清理结果
    """
    from app.config import DATA_DIR

    result = auto_cleanup_old_tasks(
        data_dir=DATA_DIR,
        max_age_days=request.max_age_days,
        min_free_gb=request.min_free_gb
    )

    if 'error' in result:
        raise HTTPException(status_code=500, detail=result['error'])

    return {
        "code": 200,
        "message": f"自动清理完成，清理了 {result.get('cleaned_tasks', 0)} 个任务",
        "data": result
    }


@router.post("/archive-models")
async def archive_models(request: ArchiveRequest):
    """
    归档旧模型

    Args:
        request: 归档请求

    Returns:
        归档结果
    """
    from app.config import DATA_DIR

    models_dir = os.path.join(DATA_DIR, 'models')
    archive_dir = os.path.join(DATA_DIR, 'archive')

    result = archive_old_models(
        models_dir=models_dir,
        archive_dir=archive_dir,
        max_age_days=request.max_age_days
    )

    if 'error' in result:
        raise HTTPException(status_code=500, detail=result['error'])

    return {
        "code": 200,
        "message": f"归档完成，归档了 {result['archived_count']} 个模型",
        "data": result
    }


@router.delete("/batch-delete-models")
async def batch_delete_models(model_ids: List[str]):
    """
    批量删除模型

    Args:
        model_ids: 模型 ID 列表

    Returns:
        删除结果
    """
    from app.config import DATA_DIR
    import shutil

    models_dir = os.path.join(DATA_DIR, 'models')
    deleted = []
    failed = []

    for model_id in model_ids:
        model_path = os.path.join(models_dir, model_id)

        if not os.path.exists(model_path):
            failed.append({
                'model_id': model_id,
                'reason': '模型不存在'
            })
            continue

        try:
            shutil.rmtree(model_path)
            deleted.append(model_id)
        except Exception as e:
            failed.append({
                'model_id': model_id,
                'reason': str(e)
            })

    return {
        "code": 200,
        "message": f"批量删除完成，成功 {len(deleted)} 个，失败 {len(failed)} 个",
        "data": {
            "deleted": deleted,
            "failed": failed
        }
    }


@router.get("/task-states/stats")
async def get_task_states_stats():
    """
    获取任务状态统计

    Returns:
        任务状态统计信息
    """
    manager = get_task_state_manager()
    stats = manager.get_stats()

    return {
        "code": 200,
        "message": "任务状态统计查询成功",
        "data": stats
    }


@router.post("/task-states/cleanup")
async def cleanup_task_states(max_age_days: int = 30):
    """
    清理旧任务状态记录

    Args:
        max_age_days: 最大保留天数

    Returns:
        清理结果
    """
    manager = get_task_state_manager()
    cleaned_count = manager.cleanup_old_tasks(max_age_days)

    return {
        "code": 200,
        "message": f"清理完成，删除了 {cleaned_count} 条旧记录",
        "data": {
            "cleaned_count": cleaned_count,
            "max_age_days": max_age_days
        }
    }


@router.post("/task-states/backup")
async def backup_task_states():
    """
    备份任务状态文件

    Returns:
        备份结果
    """
    manager = get_task_state_manager()

    try:
        manager.backup_state()
        return {
            "code": 200,
            "message": "任务状态已备份"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
