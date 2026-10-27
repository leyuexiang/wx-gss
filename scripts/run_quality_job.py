"""在独立任务中准备或执行真实视频质量验证，避免将未发布代码交给旧工作进程。"""

import argparse
import asyncio
import json
import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


async def prepare(args):
    from fastapi import BackgroundTasks, UploadFile
    from app import management_api as api

    jobs = []
    config = {"sceneType": "object", "qualityMode": "high", "objectQuick": False,
              "objectTargetViews": args.views, "objectSquareCrop": False,
              "colmap": {"matcher": "exhaustive", "featureExtractor": "sift-gpu"},
              "advanced": {"maxGaussians": 500000, "maxImageSize": 2000, "seed": 42,
                           "sizeFiltering": False, "smoothnessWeight": 0,
                           "depthRegularization": False, "validationEvery": 8}}
    background = BackgroundTasks()
    with open(args.video, "rb") as source, patch.object(api.celery_app, "send_task",
                                                       side_effect=lambda *a, **kw: jobs.append(kw)):
        response = await api.generate_3dgs(
            background_tasks=background, modelName="手机视频-60视角-清晰度验证",
            file=UploadFile(filename=Path(args.video).name, file=source), images=None,
            iterations=args.iterations, resolution=args.downsample, sceneType="object",
            generationConfig=json.dumps(config), isVideo="true", videoConfig="{}")
        await background()
    task_id = response["data"]["taskId"]
    if not jobs:
        raise RuntimeError(api._read_task_status(task_id).get("error", "准备任务失败"))
    path = Path(api.TASKS_DIR) / task_id / "quality_job.json"
    path.write_text(json.dumps(jobs[0]["kwargs"], ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"task_id": task_id, "params": jobs[0]["kwargs"]}, ensure_ascii=False), flush=True)
    return task_id


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video")
    parser.add_argument("--resume")
    parser.add_argument("--views", type=int, default=60)
    parser.add_argument("--iterations", type=int, default=7000)
    parser.add_argument("--downsample", type=int, choices=[1, 2, 4], default=1)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    if not args.video and not args.resume:
        parser.error("需要视频路径或已有的验证任务编号")
    task_id = args.resume or asyncio.run(prepare(args))
    if args.prepare_only:
        return
    from app.config import TASKS_DIR
    from app.pipeline import run_pipeline
    from app.tasks import StatusWriter
    writer = StatusWriter(task_id)
    settings = json.loads((Path(TASKS_DIR) / task_id / "quality_job.json").read_text(encoding="utf-8"))

    def progress(status, stage, ratio, message):
        writer.update(status, stage, ratio, message)
        print(f"{status} {ratio:.1%}: {message}", flush=True)

    try:
        stats = run_pipeline(task_id, status_cb=progress, **settings)
        writer.update("done", "done", 1, "质量验证任务完成")
        print(json.dumps(stats, ensure_ascii=False), flush=True)
    except Exception as error:
        writer.update("failed", "failed", 0, error=str(error))
        raise


if __name__ == "__main__":
    main()
