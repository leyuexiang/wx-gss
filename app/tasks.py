import json
import os
from datetime import datetime

from app.celery_app import celery_app
from app.config import TASKS_DIR


class StatusWriter:
    """每步更新 status.json，供 API 轮询。"""

    def __init__(self, task_id: str):
        self.path = os.path.join(TASKS_DIR, task_id, "status.json")

    def update(self, status: str, stage: str, progress: float, message: str = "", error=None):
        try:
            with open(self.path) as f:
                data = json.load(f)
        except Exception:
            data = {}
        data.update({
            "status": status,
            "stage": stage,
            "progress": round(progress, 3),
            "message": message,
            "error": error,
            "updated_at": datetime.now().isoformat(timespec="seconds"),
        })
        with open(self.path, "w") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)


@celery_app.task(name="app.tasks.run_training", bind=True)
def run_training(self, task_id: str, iterations: int = 7000, downsample: int = 4, cap_max: int = 1_000_000,
                  sfm_backend: str = "colmap", auto_cleanup: bool = True,
                  cleanup_threshold: float = 0.1, enable_depth_reg: bool = None,
                  depth_reg_weight: float = 0.1, enable_pose_opt: bool = False):
    from app.pipeline import run_pipeline  # 延迟导入：torch/gsplat 只在 worker 加载

    writer = StatusWriter(task_id)
    try:
        run_pipeline(
            task_id=task_id,
            iterations=iterations,
            downsample=downsample,
            cap_max=cap_max,
            status_cb=writer.update,
            sfm_backend=sfm_backend,
            auto_cleanup=auto_cleanup,
            cleanup_threshold=cleanup_threshold,
            enable_depth_reg=enable_depth_reg,
            depth_reg_weight=depth_reg_weight,
            enable_pose_opt=enable_pose_opt,
        )
        # 导出文件缺失时必须报失败，避免界面显示完成却没有可查看的模型。
        output_dir = os.path.join(TASKS_DIR, task_id, "output")
        for name in ("point_cloud.ply", "point_cloud.splat"):
            path = os.path.join(output_dir, name)
            if not os.path.isfile(path) or os.path.getsize(path) == 0:
                raise RuntimeError(f"训练导出文件缺失或为空：{name}")
        writer.update("done", "done", 1.0, "训练完成")
    except Exception as e:
        msg = str(e)
        if "out of memory" in msg.lower() or "CUDA" in msg:
            msg += "（显存不足：请增大 downsample 或减小 cap_max / iterations）"
        writer.update("failed", "failed", 0.0, error=msg)
        raise
