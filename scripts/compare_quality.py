"""复用固定相机和点云，在独立目录对照训练分辨率或增密策略。"""

import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task_dir")
    parser.add_argument("output_dir")
    parser.add_argument("--iterations", type=int, default=5000)
    parser.add_argument("--factors", type=int, nargs="+", default=[4, 2])
    parser.add_argument("--strategy", choices=["mcmc", "default"], default="mcmc")
    parser.add_argument("--cap-max", type=int, default=500000)
    parser.add_argument("--position-weight", type=float, default=0.0)
    parser.add_argument("--scale-weight", type=float, default=0.0)
    args = parser.parse_args()
    from PIL import Image
    from app.trainer import train
    root, output = Path(args.task_dir).resolve(), Path(args.output_dir).resolve()
    if output == root or root in output.parents:
        raise ValueError("对照输出必须位于原任务目录之外")
    output.mkdir(parents=True, exist_ok=False)
    source_dir = root / "colmap/dense/images"
    if not source_dir.is_dir():
        source_dir = root / "images"
    results = []
    for factor in args.factors:
        if factor not in (1, 2, 4, 8):
            raise ValueError("降采样只支持 1、2、4、8")
        run_dir = output / f"factor_{factor}"
        dataset = run_dir / "dataset"
        shutil.copytree(root / "dataset/sparse", dataset / "sparse")
        image_dir = dataset / f"images_{factor}"
        image_dir.mkdir()
        for path in sorted(source_dir.iterdir()):
            if path.suffix.lower() not in (".jpg", ".jpeg", ".png"):
                continue
            if factor == 1:
                shutil.copy2(path, image_dir / path.name)
            else:
                with Image.open(path) as image:
                    image.resize((image.width // factor, image.height // factor), Image.Resampling.LANCZOS).save(
                        image_dir / path.name, quality=95)
        model_dir = run_dir / "output"
        model_dir.mkdir()
        stats = train(str(dataset), str(model_dir), args.iterations, factor, args.cap_max,
                      lambda step, total, message: print(f"factor={factor} {message}", flush=True),
                      auto_cleanup=False, denoise_config={"regularization": {
                          "position_variance_weight": args.position_weight, "scale_variance_weight": args.scale_weight}},
                      training_options={"seed": 42, "strategy": args.strategy})
        # 在原始尺寸评估所有结果，防止低分辨率结果因较容易拟合而获得虚高分数。
        native_dir = dataset / "images_1"
        if factor != 1:
            shutil.copytree(source_dir, native_dir)
        from app.dataset import ColmapScene, ColmapDataset
        from app.evaluation import evaluate_views, load_ply_params
        metrics = evaluate_views(load_ply_params(str(model_dir / "point_cloud.ply")),
                                 ColmapDataset(ColmapScene(str(dataset), 1)), str(run_dir / "native_evaluation"))
        results.append({"factor": factor, "stats": stats, "native_metrics": metrics["averages"]})
        (output / "comparison.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(results[-1], ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
