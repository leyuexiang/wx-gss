"""
任务状态持久化管理器 - 使用 JSON 文件存储任务状态
"""
import json
import os
import time
from typing import Dict, List, Optional
from pathlib import Path
import threading


class TaskStateManager:
    """
    任务状态管理器 - 持久化任务状态到 JSON 文件
    """

    def __init__(self, state_file: str = '/home/3dgs/data/task_states.json'):
        """
        初始化任务状态管理器

        Args:
            state_file: 状态文件路径
        """
        self.state_file = state_file
        self.lock = threading.Lock()
        self._ensure_file_exists()

    def _ensure_file_exists(self):
        """确保状态文件存在"""
        if not os.path.exists(self.state_file):
            os.makedirs(os.path.dirname(self.state_file), exist_ok=True)
            self._write_state({'tasks': {}, 'metadata': {'version': '1.0'}})

    def _read_state(self) -> Dict:
        """读取状态文件"""
        try:
            with open(self.state_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"读取状态文件失败: {e}")
            return {'tasks': {}, 'metadata': {'version': '1.0'}}

    def _write_state(self, state: Dict):
        """写入状态文件"""
        try:
            with open(self.state_file, 'w', encoding='utf-8') as f:
                json.dump(state, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"写入状态文件失败: {e}")

    def save_task(self, task_id: str, task_data: Dict):
        """
        保存任务状态

        Args:
            task_id: 任务 ID
            task_data: 任务数据
        """
        with self.lock:
            state = self._read_state()

            # 添加时间戳
            task_data['updated_at'] = time.time()
            if 'created_at' not in task_data:
                task_data['created_at'] = time.time()

            state['tasks'][task_id] = task_data
            self._write_state(state)

    def get_task(self, task_id: str) -> Optional[Dict]:
        """
        获取任务状态

        Args:
            task_id: 任务 ID

        Returns:
            任务数据或 None
        """
        with self.lock:
            state = self._read_state()
            return state['tasks'].get(task_id)

    def update_task(self, task_id: str, updates: Dict):
        """
        更新任务状态（部分更新）

        Args:
            task_id: 任务 ID
            updates: 要更新的字段
        """
        with self.lock:
            state = self._read_state()

            if task_id in state['tasks']:
                state['tasks'][task_id].update(updates)
                state['tasks'][task_id]['updated_at'] = time.time()
                self._write_state(state)
            else:
                print(f"任务不存在: {task_id}")

    def delete_task(self, task_id: str):
        """
        删除任务状态

        Args:
            task_id: 任务 ID
        """
        with self.lock:
            state = self._read_state()

            if task_id in state['tasks']:
                del state['tasks'][task_id]
                self._write_state(state)

    def list_tasks(self, status: Optional[str] = None, limit: int = 100) -> List[Dict]:
        """
        列出任务

        Args:
            status: 过滤状态（processing/completed/failed）
            limit: 返回数量限制

        Returns:
            任务列表
        """
        with self.lock:
            state = self._read_state()
            tasks = []

            for task_id, task_data in state['tasks'].items():
                if status is None or task_data.get('status') == status:
                    tasks.append({
                        'taskId': task_id,
                        **task_data
                    })

            # 按更新时间倒序排序
            tasks.sort(key=lambda x: x.get('updated_at', 0), reverse=True)

            return tasks[:limit]

    def cleanup_old_tasks(self, max_age_days: int = 30) -> int:
        """
        清理旧任务记录

        Args:
            max_age_days: 最大保留天数

        Returns:
            清理的任务数量
        """
        with self.lock:
            state = self._read_state()
            current_time = time.time()
            max_age_seconds = max_age_days * 24 * 3600

            tasks_to_delete = []

            for task_id, task_data in state['tasks'].items():
                updated_at = task_data.get('updated_at', task_data.get('created_at', 0))
                age_seconds = current_time - updated_at

                if age_seconds > max_age_seconds:
                    tasks_to_delete.append(task_id)

            # 删除旧任务
            for task_id in tasks_to_delete:
                del state['tasks'][task_id]

            if tasks_to_delete:
                self._write_state(state)

            return len(tasks_to_delete)

    def get_stats(self) -> Dict:
        """
        获取任务统计信息

        Returns:
            统计信息
        """
        with self.lock:
            state = self._read_state()
            tasks = state['tasks']

            stats = {
                'total_tasks': len(tasks),
                'by_status': {},
                'oldest_task_age_days': 0,
                'newest_task_age_days': 0
            }

            # 按状态统计
            for task_data in tasks.values():
                status = task_data.get('status', 'unknown')
                stats['by_status'][status] = stats['by_status'].get(status, 0) + 1

            # 计算年龄
            if tasks:
                current_time = time.time()
                created_times = [
                    t.get('created_at', current_time) for t in tasks.values()
                ]
                oldest = min(created_times)
                newest = max(created_times)

                stats['oldest_task_age_days'] = round((current_time - oldest) / (24 * 3600), 1)
                stats['newest_task_age_days'] = round((current_time - newest) / (24 * 3600), 1)

            return stats

    def restore_tasks(self) -> Dict[str, Dict]:
        """
        恢复所有任务（用于服务启动时）

        Returns:
            任务字典 {task_id: task_data}
        """
        with self.lock:
            state = self._read_state()
            return state['tasks'].copy()

    def backup_state(self, backup_path: Optional[str] = None):
        """
        备份状态文件

        Args:
            backup_path: 备份文件路径（默认在原文件旁边）
        """
        with self.lock:
            if backup_path is None:
                timestamp = time.strftime('%Y%m%d_%H%M%S')
                backup_path = f"{self.state_file}.{timestamp}.bak"

            try:
                state = self._read_state()
                with open(backup_path, 'w', encoding='utf-8') as f:
                    json.dump(state, f, indent=2, ensure_ascii=False)
                print(f"状态文件已备份到: {backup_path}")
            except Exception as e:
                print(f"备份状态文件失败: {e}")


# 全局实例
_task_state_manager = None


def get_task_state_manager() -> TaskStateManager:
    """
    获取全局任务状态管理器实例（单例模式）

    Returns:
        TaskStateManager 实例
    """
    global _task_state_manager
    if _task_state_manager is None:
        _task_state_manager = TaskStateManager()
    return _task_state_manager


def save_task_state(task_id: str, task_data: Dict):
    """
    快捷函数：保存任务状态

    Args:
        task_id: 任务 ID
        task_data: 任务数据
    """
    manager = get_task_state_manager()
    manager.save_task(task_id, task_data)


def get_task_state(task_id: str) -> Optional[Dict]:
    """
    快捷函数：获取任务状态

    Args:
        task_id: 任务 ID

    Returns:
        任务数据或 None
    """
    manager = get_task_state_manager()
    return manager.get_task(task_id)


def update_task_state(task_id: str, updates: Dict):
    """
    快捷函数：更新任务状态

    Args:
        task_id: 任务 ID
        updates: 要更新的字段
    """
    manager = get_task_state_manager()
    manager.update_task(task_id, updates)


def restore_all_tasks() -> Dict[str, Dict]:
    """
    快捷函数：恢复所有任务

    Returns:
        任务字典
    """
    manager = get_task_state_manager()
    return manager.restore_tasks()
