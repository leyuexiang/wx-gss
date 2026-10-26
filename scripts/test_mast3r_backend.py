#!/usr/bin/env python3
"""测试 MASt3R 后端集成"""
import os
import sys
import tempfile
import shutil
import zipfile

sys.path.insert(0, '/app')

from app.pipeline import run_pipeline

def test_mast3r_backend():
    """用 TestIMG.zip（5张）测试 mast3r 后端"""
    print("=" * 60)
    print("测试 MASt3R 后端集成")
    print("=" * 60)

    # 准备测试任务目录
    task_id = "test_mast3r_integration"
    task_dir = f"/data/tasks/{task_id}"

    # 清理旧测试
    if os.path.exists(task_dir):
        shutil.rmtree(task_dir)
    os.makedirs(task_dir)

    # 复制 TestIMG.zip
    src_zip = "/data/smoke/TestIMG.zip"
    dst_zip = os.path.join(task_dir, "input.zip")
    if not os.path.exists(src_zip):
        print(f"❌ 测试数据不存在: {src_zip}")
        return False
    shutil.copy2(src_zip, dst_zip)
    print(f"✓ 测试数据已复制: {src_zip}")

    # 测试回调
    def status_callback(stage, substage, progress, msg):
        print(f"[{progress:.0%}] {stage}/{substage}: {msg}")

    try:
        print("\n开始运行 MASt3R 后端...")
        stats = run_pipeline(
            task_id=task_id,
            iterations=1000,  # 少量迭代快速测试
            downsample=4,
            cap_max=100_000,
            status_cb=status_callback,
            sfm_backend="mast3r",
            auto_cleanup=False,
        )

        print("\n" + "=" * 60)
        print("✅ MASt3R 后端测试通过")
        print("=" * 60)
        print(f"最终 PSNR: {stats.get('psnr', 'N/A')}")
        print(f"最终 SSIM: {stats.get('ssim', 'N/A')}")
        print(f"高斯数量: {stats.get('num_gaussians', 'N/A')}")
        print(f"质量评分: {stats.get('quality_score', 'N/A')}")
        print(f"质量等级: {stats.get('quality_rating', 'N/A')}")

        # 检查输出文件
        result_zip = os.path.join(task_dir, "output", "result.zip")
        if os.path.exists(result_zip):
            print(f"\n✓ 结果文件已生成: {result_zip}")
            print(f"  大小: {os.path.getsize(result_zip) / 1024 / 1024:.2f} MB")

        return True

    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_mast3r_backend()
    sys.exit(0 if success else 1)
