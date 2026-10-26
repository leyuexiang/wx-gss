"""
存储管理模块 - 临时文件清理、磁盘空间监控、模型归档
"""
import os
import shutil
import time
from pathlib import Path
from typing import Dict, List, Tuple
import json


def cleanup_temp_files(task_dir: str, keep_output: bool = True) -> Dict:
    """
    清理任务目录中的临时文件，保留必要的输出

    Args:
        task_dir: 任务目录路径
        keep_output: 是否保留输出文件（PLY、SPLAT、日志等）

    Returns:
        清理统计信息
    """
    if not os.path.exists(task_dir):
        return {'error': '任务目录不存在'}

    removed_files = []
    removed_dirs = []
    total_size_freed = 0

    # 定义要清理的目录和文件模式
    temp_patterns = {
        'dirs': [
            'images',           # 抽帧后的图片
            'sparse',           # COLMAP 稀疏重建
            'dense',            # COLMAP 稠密重建
            'database.db',      # COLMAP 数据库
            'depths',           # MASt3R 深度图
            'dataset',          # 训练数据集副本
            '__pycache__',      # Python 缓存
        ],
        'files': [
            'input.zip',        # 原始上传文件
            'input.mp4',        # 原始视频文件
            'input.avi',
            'input.mov',
            'input.mkv',
            'input.webm',
            '*.tmp',            # 临时文件
            '*.log.bak',        # 备份日志
        ]
    }

    # 保留的输出文件（如果 keep_output=True）
    keep_patterns = [
        'output/point_cloud.ply',
        'output/point_cloud.splat',
        'output/train_stats.json',
        'output/optimized_poses.npy',
        'training_config.json',
        'colmap.log',
        'training.log',
    ] if keep_output else []

    # 清理目录
    for dir_name in temp_patterns['dirs']:
        dir_path = os.path.join(task_dir, dir_name)
        if os.path.isdir(dir_path):
            try:
                # 计算目录大小
                dir_size = sum(
                    os.path.getsize(os.path.join(dirpath, filename))
                    for dirpath, dirnames, filenames in os.walk(dir_path)
                    for filename in filenames
                )
                shutil.rmtree(dir_path)
                removed_dirs.append(dir_name)
                total_size_freed += dir_size
            except Exception as e:
                print(f"清理目录失败 {dir_path}: {e}")

    # 清理文件
    for file_pattern in temp_patterns['files']:
        if '*' in file_pattern:
            # 通配符匹配
            import glob
            for file_path in glob.glob(os.path.join(task_dir, file_pattern)):
                if os.path.isfile(file_path):
                    try:
                        file_size = os.path.getsize(file_path)
                        os.remove(file_path)
                        removed_files.append(os.path.basename(file_path))
                        total_size_freed += file_size
                    except Exception as e:
                        print(f"清理文件失败 {file_path}: {e}")
        else:
            file_path = os.path.join(task_dir, file_pattern)
            if os.path.isfile(file_path):
                try:
                    file_size = os.path.getsize(file_path)
                    os.remove(file_path)
                    removed_files.append(file_pattern)
                    total_size_freed += file_size
                except Exception as e:
                    print(f"清理文件失败 {file_path}: {e}")

    return {
        'removed_dirs': removed_dirs,
        'removed_files': removed_files,
        'total_size_freed_mb': round(total_size_freed / (1024 * 1024), 2),
        'kept_patterns': keep_patterns if keep_output else []
    }


def check_disk_space(path: str = '/home/3dgs/data') -> Dict:
    """
    检查磁盘空间使用情况

    Args:
        path: 要检查的路径

    Returns:
        磁盘空间信息
    """
    try:
        stat = os.statvfs(path)

        # 计算空间（字节）
        total = stat.f_blocks * stat.f_frsize
        free = stat.f_bfree * stat.f_frsize
        available = stat.f_bavail * stat.f_frsize
        used = total - free

        # 计算百分比
        usage_percent = (used / total) * 100 if total > 0 else 0

        # 转换为 GB
        total_gb = total / (1024 ** 3)
        used_gb = used / (1024 ** 3)
        free_gb = free / (1024 ** 3)
        available_gb = available / (1024 ** 3)

        # 判断警告级别
        if usage_percent >= 95:
            warning_level = 'critical'
            message = '⛔ 磁盘空间严重不足！'
        elif usage_percent >= 85:
            warning_level = 'warning'
            message = '⚠️ 磁盘空间不足'
        elif usage_percent >= 70:
            warning_level = 'info'
            message = '💡 磁盘使用率较高'
        else:
            warning_level = 'ok'
            message = '✓ 磁盘空间充足'

        return {
            'path': path,
            'total_gb': round(total_gb, 2),
            'used_gb': round(used_gb, 2),
            'free_gb': round(free_gb, 2),
            'available_gb': round(available_gb, 2),
            'usage_percent': round(usage_percent, 1),
            'warning_level': warning_level,
            'message': message
        }

    except Exception as e:
        return {
            'error': str(e),
            'warning_level': 'unknown',
            'message': '无法获取磁盘空间信息'
        }


def auto_cleanup_old_tasks(data_dir: str = '/home/3dgs/data',
                           max_age_days: int = 30,
                           min_free_gb: float = 10.0) -> Dict:
    """
    自动清理旧任务以释放空间

    Args:
        data_dir: 数据目录
        max_age_days: 任务最大保留天数
        min_free_gb: 最小剩余空间（GB）

    Returns:
        清理结果
    """
    tasks_dir = os.path.join(data_dir, 'tasks')

    if not os.path.exists(tasks_dir):
        return {'error': '任务目录不存在'}

    # 检查磁盘空间
    disk_info = check_disk_space(data_dir)

    # 如果空间充足且没有超龄任务，不执行清理
    if disk_info.get('available_gb', 0) > min_free_gb:
        return {
            'action': 'skip',
            'reason': f"磁盘空间充足 ({disk_info.get('available_gb')} GB 可用)",
            'cleaned_tasks': 0
        }

    # 获取所有任务目录及其修改时间
    task_list = []
    for task_id in os.listdir(tasks_dir):
        task_path = os.path.join(tasks_dir, task_id)
        if os.path.isdir(task_path):
            mtime = os.path.getmtime(task_path)
            age_days = (time.time() - mtime) / (24 * 3600)

            # 计算任务目录大小
            task_size = sum(
                os.path.getsize(os.path.join(dirpath, filename))
                for dirpath, dirnames, filenames in os.walk(task_path)
                for filename in filenames
            )

            task_list.append({
                'task_id': task_id,
                'path': task_path,
                'age_days': age_days,
                'size_mb': task_size / (1024 * 1024),
                'mtime': mtime
            })

    # 按年龄排序（最旧的优先）
    task_list.sort(key=lambda x: x['mtime'])

    cleaned_tasks = []
    total_freed_mb = 0

    for task in task_list:
        # 检查是否超龄
        if task['age_days'] > max_age_days:
            try:
                shutil.rmtree(task['path'])
                cleaned_tasks.append({
                    'task_id': task['task_id'],
                    'age_days': round(task['age_days'], 1),
                    'size_mb': round(task['size_mb'], 2)
                })
                total_freed_mb += task['size_mb']

                # 重新检查空间
                disk_info = check_disk_space(data_dir)
                if disk_info.get('available_gb', 0) > min_free_gb:
                    break

            except Exception as e:
                print(f"清理任务失败 {task['task_id']}: {e}")

    return {
        'action': 'cleaned',
        'cleaned_tasks': len(cleaned_tasks),
        'total_freed_mb': round(total_freed_mb, 2),
        'details': cleaned_tasks,
        'current_free_gb': disk_info.get('available_gb', 0)
    }


def get_directory_size(path: str) -> int:
    """
    递归计算目录大小（字节）

    Args:
        path: 目录路径

    Returns:
        大小（字节）
    """
    total_size = 0
    try:
        for dirpath, dirnames, filenames in os.walk(path):
            for filename in filenames:
                file_path = os.path.join(dirpath, filename)
                if os.path.exists(file_path):
                    total_size += os.path.getsize(file_path)
    except Exception as e:
        print(f"计算目录大小失败 {path}: {e}")

    return total_size


def get_storage_stats(data_dir: str = '/home/3dgs/data') -> Dict:
    """
    获取存储统计信息

    Args:
        data_dir: 数据目录

    Returns:
        存储统计
    """
    stats = {
        'data_dir': data_dir,
        'subdirs': {}
    }

    # 统计各子目录大小
    subdirs = ['uploads', 'tasks', 'models', 'optimized']

    for subdir in subdirs:
        subdir_path = os.path.join(data_dir, subdir)
        if os.path.exists(subdir_path):
            size_bytes = get_directory_size(subdir_path)
            size_mb = size_bytes / (1024 * 1024)
            size_gb = size_bytes / (1024 ** 3)

            # 统计文件数量
            file_count = sum(
                len(files) for _, _, files in os.walk(subdir_path)
            )

            stats['subdirs'][subdir] = {
                'size_bytes': size_bytes,
                'size_mb': round(size_mb, 2),
                'size_gb': round(size_gb, 3),
                'file_count': file_count
            }

    # 计算总大小
    total_size_bytes = sum(s['size_bytes'] for s in stats['subdirs'].values())
    stats['total_size_mb'] = round(total_size_bytes / (1024 * 1024), 2)
    stats['total_size_gb'] = round(total_size_bytes / (1024 ** 3), 3)

    # 磁盘空间信息
    stats['disk_space'] = check_disk_space(data_dir)

    return stats


def archive_old_models(models_dir: str = '/home/3dgs/data/models',
                       archive_dir: str = '/home/3dgs/data/archive',
                       max_age_days: int = 90) -> Dict:
    """
    归档旧模型到归档目录

    Args:
        models_dir: 模型目录
        archive_dir: 归档目录
        max_age_days: 归档阈值（天）

    Returns:
        归档结果
    """
    if not os.path.exists(models_dir):
        return {'error': '模型目录不存在'}

    os.makedirs(archive_dir, exist_ok=True)

    archived_models = []
    total_size_mb = 0

    for model_id in os.listdir(models_dir):
        model_path = os.path.join(models_dir, model_id)

        if not os.path.isdir(model_path):
            continue

        # 检查年龄
        mtime = os.path.getmtime(model_path)
        age_days = (time.time() - mtime) / (24 * 3600)

        if age_days > max_age_days:
            # 计算大小
            size_bytes = get_directory_size(model_path)
            size_mb = size_bytes / (1024 * 1024)

            # 移动到归档目录
            archive_path = os.path.join(archive_dir, model_id)
            try:
                shutil.move(model_path, archive_path)
                archived_models.append({
                    'model_id': model_id,
                    'age_days': round(age_days, 1),
                    'size_mb': round(size_mb, 2)
                })
                total_size_mb += size_mb
            except Exception as e:
                print(f"归档模型失败 {model_id}: {e}")

    return {
        'archived_count': len(archived_models),
        'total_size_mb': round(total_size_mb, 2),
        'details': archived_models,
        'archive_dir': archive_dir
    }
