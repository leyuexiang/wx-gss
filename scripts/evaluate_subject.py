"""使用固定参考图矩形评估主体，标注文件必须来自输入图像并在对照前冻结。"""

import argparse
import json
import hashlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_dir")
    parser.add_argument("ply_path")
    parser.add_argument("annotations")
    parser.add_argument("output_dir")
    parser.add_argument("--factor", type=int, default=1)
    parser.add_argument("--rigid-normalization", action="store_true", help="与新实验保持相同刚性相机约定，不能混用于历史模型")
    parser.add_argument("--split", default="heldout_rgb_supervision")
    parser.add_argument("--include-background", action="store_true",
                        help="同一次渲染输出全图、冻结主体矩形及矩形外背景误差")
    args = parser.parse_args()
    from app.dataset import ColmapScene, ColmapDataset
    from app.evaluation import load_ply_params, evaluate_views
    annotations = json.loads(Path(args.annotations).read_text(encoding="utf-8"))
    dataset = ColmapDataset(ColmapScene(args.dataset_dir, args.factor,
                                      rigid_normalization=args.rigid_normalization))
    by_name = {name: index for index, name in enumerate(dataset.scene.names)}
    unknown = set(annotations["frames"]) - set(by_name)
    if unknown:
        raise ValueError(f"数据集缺少标注视角：{sorted(unknown)}")
    expected_hashes = annotations.get("reference_sha256")
    if expected_hashes is not None:
        # 相同文件名不保证参考图相同，核对字节后才允许复用冻结的主体标注。
        for name in annotations["frames"]:
            actual = hashlib.sha256((Path(dataset.scene.image_dir) / name).read_bytes()).hexdigest()
            if expected_hashes.get(name) != actual:
                raise ValueError(f"主体标注对应的参考图已改变：{name}")
    indices = [by_name[name] for name in annotations["frames"]]
    # 输出新目录，避免把不同模型的图和指标混在一起。
    Path(args.output_dir).mkdir(parents=True, exist_ok=False)
    regions = {"additional_roi": annotations["frames"]} if args.include_background else {"roi": annotations["frames"]}
    result = evaluate_views(load_ply_params(args.ply_path), dataset, args.output_dir,
                            indices=indices, split=args.split, **regions)
    result["annotation_sha256"] = hashlib.sha256(Path(args.annotations).read_bytes()).hexdigest()
    result["annotation_kind"] = annotations.get("annotation_kind", "reference_rectangle")
    (Path(args.output_dir) / "metrics.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    reported = {"full_frame": result["averages"], "subject": result["subject_averages"],
                "background": result["background_averages"]} if args.include_background else result["averages"]
    print(json.dumps(reported, ensure_ascii=False))


if __name__ == "__main__":
    main()
