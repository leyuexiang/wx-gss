from celery import Celery

from app.config import QUEUE_NAME, REDIS_URL

celery_app = Celery("3dgs", broker=REDIS_URL, backend=REDIS_URL)

celery_app.conf.update(
    task_default_queue=QUEUE_NAME,
    task_routes={"app.tasks.run_training": {"queue": QUEUE_NAME}},
    task_acks_late=True,
    task_time_limit=4 * 3600,
    task_soft_time_limit=4 * 3600 - 300,
    worker_prefetch_multiplier=1,
    worker_max_tasks_per_child=5,   # 定期重建 worker 进程，释放显存碎片
    result_expires=24 * 3600,
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    timezone="Asia/Shanghai",
)

celery_app.autodiscover_tasks(["app"])
