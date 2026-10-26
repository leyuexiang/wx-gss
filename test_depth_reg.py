#!/usr/bin/env python3
import sys
sys.path.insert(0, "/app")
from app.pipeline import run_pipeline

def cb(stage, substage, progress, msg):
    if int(progress * 100) % 10 == 0 or progress > 0.95:
        pct = int(progress * 100)
        print(f"[{pct}%] {msg}")

print("=" * 60)
print("测试深度正则化集成")
print("=" * 60)

stats = run_pipeline(
    task_id="test_depth_quick",
    iterations=1000,
    downsample=4,
    cap_max=100_000,
    status_cb=cb,
    sfm_backend="mast3r",
    auto_cleanup=False,
    cleanup_threshold=0.1,
    enable_depth_reg=True,
    depth_reg_weight=0.1
)

print("\n" + "=" * 60)
print("测试成功!")
print("=" * 60)
loss = stats["final_loss"]
gs = stats["num_gaussians"]
time = stats["train_seconds"]
mem = stats["peak_gpu_mem_gb"]
print(f"Loss: {loss}")
print(f"Gaussians: {gs}")
print(f"Time: {time}s")
print(f"Memory: {mem}GB")
