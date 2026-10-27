"""训练图单独重建，验证图只定位到固定地图；核查几何、颜色和训练相机未改变。"""

import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np


def snapshot_map(reconstruction, train_names):
    """只记录需要冻结的属性，不把新增的验证观测轨迹当作几何修改。"""
    return {
        "points": {str(index): {"xyz": point.xyz.tolist(), "color": point.color.tolist()}
                   for index, point in reconstruction.points3D.items()},
        "cameras": {str(index): camera.params.tolist() for index, camera in reconstruction.cameras.items()},
        "poses": {image.name: image.cam_from_world().matrix().tolist()
                  for image in reconstruction.images.values() if image.name in train_names},
    }


def verify_frozen_map(before, after):
    """拒绝新增点、颜色变化或相机变化，避免把受验证图影响的结果标记为独立重建。"""
    for field in ("points", "cameras", "poses"):
        if set(before[field]) != set(after[field]):
            raise ValueError(f"定位验证图时改变了 {field} 的成员")
    for index, point in before["points"].items():
        if not np.allclose(point["xyz"], after["points"][index]["xyz"], rtol=0, atol=1e-12):
            raise ValueError("定位验证图时改变了初始点云位置")
        if point["color"] != after["points"][index]["color"]:
            raise ValueError("定位验证图时改变了初始点云颜色")
    for field in ("cameras", "poses"):
        for index, value in before[field].items():
            if not np.allclose(value, after[field][index], rtol=0, atol=1e-12):
                raise ValueError(f"定位验证图时改变了训练 {field}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task_dir")
    parser.add_argument("output_dir")
    parser.add_argument("--database", required=True)
    args = parser.parse_args()
    import pycolmap

    source, root = Path(args.task_dir).resolve(), Path(args.output_dir).resolve()
    if root == source or source in root.parents:
        raise ValueError("独立重建必须保存到源任务之外")
    root.mkdir(parents=True, exist_ok=False)
    effective = json.loads((source / "output/effective_config.json").read_text())
    names = [entry["name"] for entry in effective["image_dimensions"]]
    validation_names = [names[index] for index in effective["validation_indices"]]
    train_names = set(names) - set(validation_names)
    (root / "train_names.txt").write_text("\n".join(sorted(train_names)) + "\n", encoding="utf-8")
    shutil.copy2(args.database, root / "database.db")
    sparse = root / "training_sparse"
    sparse.mkdir()

    def run(command, name):
        with open(root / f"{name}.log", "w", encoding="utf-8") as log:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)

    run(["colmap", "mapper", "--database_path", str(root / "database.db"),
         "--image_path", str(source / "images"), "--image_list_path", str(root / "train_names.txt"),
         "--output_path", str(sparse), "--Mapper.init_min_tri_angle", "4"], "training_mapper")
    candidates = [path for path in sparse.iterdir() if (path / "images.bin").is_file()]
    if not candidates:
        raise ValueError("训练图单独重建失败")
    selected = max(candidates, key=lambda path: len(pycolmap.Reconstruction(str(path)).images))
    training = pycolmap.Reconstruction(str(selected))
    registered_names = {image.name for image in training.images.values()}
    if registered_names != train_names:
        raise ValueError(f"训练集合不完整或混入验证图：缺少 {sorted(train_names - registered_names)}")
    image_names = {index: image.name for index, image in training.images.items()}
    for point in training.points3D.values():
        if any(image_names[element.image_id] not in train_names for element in point.track.elements):
            raise ValueError("初始点云包含验证图观测")
    before = snapshot_map(training, train_names)
    positions = np.stack([image.cam_from_world().inverse().translation for image in training.images.values()])
    center = positions.mean(axis=0)
    radius = np.linalg.norm(positions - center, axis=1).max()
    if not np.isfinite(radius) or radius <= 1e-6:
        raise ValueError("训练相机轨迹缺少有效空间跨度")
    normalization = {"center": center.tolist(), "scale": float(1 / (radius * 1.05)),
                     "source": "training_cameras_only", "training_names": sorted(train_names)}
    localized_dir = root / "localized_sparse"
    localized_dir.mkdir()
    # 共享相机已由训练集合校准；定位后仍对全部内参、训练位姿和点云做严格不变检查。
    run(["colmap", "image_registrator", "--database_path", str(root / "database.db"),
         "--input_path", str(selected), "--output_path", str(localized_dir)], "localization")
    localized = pycolmap.Reconstruction(str(localized_dir))
    verify_frozen_map(before, snapshot_map(localized, train_names))
    localized_names = {image.name for image in localized.images.values()}
    if localized_names != set(names):
        raise ValueError(f"验证图未全部定位：{sorted(set(names) - localized_names)}")
    dense = root / "undistorted"
    run(["colmap", "image_undistorter", "--image_path", str(source / "images"),
         "--input_path", str(localized_dir), "--output_path", str(dense),
         "--output_type", "COLMAP", "--max_image_size", "2000"], "undistortion")
    dataset = root / "dataset"
    (dataset / "sparse").mkdir(parents=True)
    shutil.copytree(dense / "sparse", dataset / "sparse/0")
    shutil.copytree(dense / "images", dataset / "images_1")
    (dataset / "scene_normalization.json").write_text(json.dumps(normalization, indent=2), encoding="utf-8")
    config = json.loads((source / "training_config.json").read_text())
    config["training"].update({"validation_names": validation_names,
        "validation_scope": "训练图单独建立几何和颜色；验证图仅通过特征定位至固定地图，不参与几何优化"})
    (root / "training_config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    runtime = {"iterations": 7000, "downsample": 1, "cap_max": 500000,
               "requested_config": config, "source_task": source.name}
    (root / "effective_config.json").write_text(json.dumps(runtime, ensure_ascii=False, indent=2), encoding="utf-8")
    fingerprint = hashlib.sha256(json.dumps(before, sort_keys=True).encode()).hexdigest()
    report = {"training_names": sorted(train_names), "validation_names": validation_names,
              "point_count": len(training.points3D), "geometry_and_colors_unchanged": True,
              "training_camera_poses_and_intrinsics_unchanged": True,
              "training_map_fingerprint": fingerprint,
              "normalization": "training_cameras_only",
              "scope": "验证图允许通过特征估计自身相机位姿，但不更新训练地图和相机内参"}
    (root / "protocol.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("point_count", "geometry_and_colors_unchanged", "training_camera_poses_and_intrinsics_unchanged")}))


if __name__ == "__main__":
    main()
