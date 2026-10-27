"""只读检查历史任务，并在独立目录渲染固定视角；不会重训或覆盖历史模型。"""

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task_dir")
    parser.add_argument("output_dir")
    parser.add_argument("--factor", type=int, choices=(1, 2, 4, 8), default=None)
    parser.add_argument("--roi", type=int, nargs=4)
    parser.add_argument("--no-render", action="store_true")
    args = parser.parse_args()
    root, output = Path(args.task_dir).resolve(), Path(args.output_dir).resolve()
    if output == root or root in output.parents:
        raise ValueError("诊断输出必须位于原任务目录之外")
    # 独立且新建的目录防止把前一次实验结果混入本次评估。
    output.mkdir(parents=True, exist_ok=False)
    from PIL import Image
    import torch
    import gsplat

    stats = json.loads((root / "output/train_stats.json").read_text(encoding="utf-8"))
    status = json.loads((root / "status.json").read_text(encoding="utf-8"))
    factor = args.factor or stats["downsample"]
    dimensions = {}
    for directory in [root / "images", root / "dataset" / f"images_{stats['downsample']}"]:
        entries = []
        for path in sorted(directory.iterdir()):
            if path.is_file():
                with Image.open(path) as image:
                    entries.append({"name": path.name, "width": image.width, "height": image.height,
                                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
        dimensions[str(directory.relative_to(root))] = entries
    manifest = {
        "task": root.name, "status": status, "stats": stats, "dimensions": dimensions,
        "versions": {"torch": torch.__version__, "gsplat": gsplat.__version__,
                     "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None},
        "source_hashes": {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                          for path in (Path(__file__).resolve().parents[1] / "app").glob("*.py")},
        "missing": [name for name in ["training_config.json", "effective_config.json", "frame_manifest.json"]
                    if not (root / name).exists()],
        "evaluation_factor": factor,
    }
    (output / "baseline.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    if not args.no_render:
        from app.dataset import ColmapDataset, ColmapScene
        from app.evaluation import evaluate_views, load_ply_params
        dataset_dir = root / "dataset"
        if not (dataset_dir / f"images_{factor}").is_dir():
            # 使用相机对应的去畸变图像；MASt3R 数据直接取整理后的原始输入。
            source_dir = root / "colmap/dense/images"
            if not source_dir.is_dir():
                source_dir = root / "images"
            dataset_dir = output / "dataset"
            shutil.copytree(root / "dataset/sparse", dataset_dir / "sparse")
            image_dir = dataset_dir / f"images_{factor}"
            image_dir.mkdir()
            for path in sorted(source_dir.iterdir()):
                if not path.is_file():
                    continue
                if factor == 1:
                    shutil.copy2(path, image_dir / path.name)
                else:
                    with Image.open(path) as image:
                        image.resize((image.width // factor, image.height // factor), Image.Resampling.LANCZOS).save(
                            image_dir / path.name, quality=95)
        dataset = ColmapDataset(ColmapScene(str(dataset_dir), factor))
        params = load_ply_params(str(root / "output/point_cloud.ply"))
        metrics = evaluate_views(params, dataset, str(output / "renders"), degrees=[0, 2, 3], roi=args.roi)
        print(json.dumps(metrics["averages"], ensure_ascii=False))
    print(str(output))


if __name__ == "__main__":
    main()
