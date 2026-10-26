#!/usr/bin/env python3
"""Phase 1 vs Phase 2 vs Phase 2.5 对比测试"""
import os
import sys
import shutil
import json
import time

sys.path.insert(0, "/app")
from app.pipeline import run_pipeline

# 测试配置
TEST_CONFIGS = [
    {
        "name": "Phase 1: 仅 MASt3R",
        "task_id": "compare_phase1",
        "enable_depth_reg": False,
        "enable_pose_opt": False,
    },
    {
        "name": "Phase 2: MASt3R + 深度正则化",
        "task_id": "compare_phase2",
        "enable_depth_reg": True,
        "enable_pose_opt": False,
    },
    {
        "name": "Phase 2.5: MASt3R + 深度 + 位姿优化",
        "task_id": "compare_phase2_5",
        "enable_depth_reg": True,
        "enable_pose_opt": True,
    },
]

COMMON_PARAMS = {
    "iterations": 3000,
    "downsample": 4,
    "cap_max": 200_000,
    "sfm_backend": "mast3r",
    "auto_cleanup": False,
    "cleanup_threshold": 0.1,
    "depth_reg_weight": 0.1,
}

def cb(stage, substage, progress, msg):
    """简化回调，减少输出"""
    if int(progress * 100) % 20 == 0 or progress > 0.95:
        print(f"  [{int(progress*100):3d}%] {msg}")

def run_test(config):
    """运行单个测试"""
    task_id = config["task_id"]
    task_dir = f"/data/tasks/{task_id}"

    # 清理并准备
    if os.path.exists(task_dir):
        shutil.rmtree(task_dir)
    os.makedirs(task_dir)
    shutil.copy2("/data/smoke/TestIMG.zip", f"{task_dir}/input.zip")

    print(f"\n{'='*60}")
    print(config["name"])
    print(f"{'='*60}")

    start_time = time.time()
    try:
        stats = run_pipeline(
            task_id=task_id,
            status_cb=cb,
            enable_depth_reg=config["enable_depth_reg"],
            enable_pose_opt=config["enable_pose_opt"],
            **COMMON_PARAMS
        )
        elapsed = time.time() - start_time

        print(f"\n✅ {config['name']} 完成")
        print(f"  Loss: {stats['final_loss']}")
        print(f"  Gaussians: {stats['num_gaussians']}")
        print(f"  Time: {stats['train_seconds']}s")
        print(f"  Memory: {stats['peak_gpu_mem_gb']}GB")
        print(f"  Total elapsed: {elapsed:.1f}s")

        return {
            "config": config,
            "stats": stats,
            "elapsed": elapsed,
            "success": True,
        }
    except Exception as e:
        print(f"\n❌ {config['name']} 失败: {e}")
        import traceback
        traceback.print_exc()
        return {
            "config": config,
            "error": str(e),
            "success": False,
        }

def main():
    print("="*60)
    print("Phase 对比测试")
    print(f"数据集: TestIMG.zip (5张)")
    print(f"参数: iterations={COMMON_PARAMS['iterations']}, "
          f"downsample={COMMON_PARAMS['downsample']}")
    print("="*60)

    results = []
    for config in TEST_CONFIGS:
        result = run_test(config)
        results.append(result)

        # 短暂休息，让 GPU 冷却
        if result != results[-1]:
            print("\n等待 GPU 冷却...")
            time.sleep(10)

    # 保存结果
    output_file = "/data/tasks/phase_comparison_results.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # 生成对比报告
    print("\n" + "="*60)
    print("对比报告")
    print("="*60)

    successful = [r for r in results if r["success"]]
    if len(successful) < 2:
        print("❌ 测试失败过多，无法生成对比报告")
        return

    # 表格对比
    print(f"\n{'配置':<30} {'Loss':<10} {'Gaussians':<12} {'时间(s)':<10} {'显存(GB)':<10}")
    print("-"*72)
    for r in successful:
        name = r["config"]["name"].split(":")[0]
        stats = r["stats"]
        print(f"{name:<30} {stats['final_loss']:<10.5f} "
              f"{stats['num_gaussians']:<12} "
              f"{stats['train_seconds']:<10.1f} "
              f"{stats['peak_gpu_mem_gb']:<10.2f}")

    # 相对提升
    if len(successful) >= 2:
        baseline = successful[0]["stats"]
        print(f"\n相对 Phase 1 的提升:")
        for r in successful[1:]:
            name = r["config"]["name"].split(":")[0]
            stats = r["stats"]
            loss_improve = (baseline['final_loss'] - stats['final_loss']) / baseline['final_loss'] * 100
            time_overhead = (stats['train_seconds'] - baseline['train_seconds']) / baseline['train_seconds'] * 100
            mem_overhead = stats['peak_gpu_mem_gb'] - baseline['peak_gpu_mem_gb']

            print(f"\n{name}:")
            print(f"  Loss 改善: {loss_improve:+.1f}%")
            print(f"  时间开销: {time_overhead:+.1f}%")
            print(f"  显存开销: {mem_overhead:+.2f}GB")

    print(f"\n详细结果已保存到: {output_file}")
    print("="*60)

if __name__ == "__main__":
    main()
