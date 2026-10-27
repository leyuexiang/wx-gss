"""用精确球体投影的合成图和掩码验证显卡训练链路，不作为茶罐质量结论。"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir")
    parser.add_argument("--steps", type=int, default=20)
    args = parser.parse_args()
    from app.pipeline import _rotmat_to_quat, _write_colmap_bins
    from app.trainer import train

    root = Path(args.output_dir)
    root.mkdir(parents=True, exist_ok=False)
    sparse, images, masks = root / "dataset/sparse/0", root / "dataset/images_1", root / "masks"
    for path in (sparse, images, masks):
        path.mkdir(parents=True)
    rng, size, focal = np.random.default_rng(42), 96, 85.
    points = rng.normal(size=(512, 3))
    points = points / np.linalg.norm(points, axis=1, keepdims=True) * .7
    colors = np.clip((points / 1.4 + .5) * 255, 0, 255).astype(np.uint8)
    rows, entries = [], {}
    pixels_y, pixels_x = np.mgrid[:size, :size]
    directions = np.stack(((pixels_x + .5 - size / 2) / focal,
                           (pixels_y + .5 - size / 2) / focal, np.ones_like(pixels_x)), axis=-1)
    directions /= np.linalg.norm(directions, axis=-1, keepdims=True)
    for index in range(8):
        angle = index * np.pi / 4
        eye = np.array([3 * np.cos(angle), .4, 3 * np.sin(angle)])
        forward = -eye / np.linalg.norm(eye)
        right = np.cross(forward, [0., -1., 0.]); right /= np.linalg.norm(right)
        down = np.cross(forward, right)
        rotation = np.stack((right, down, forward))
        rays = directions @ rotation
        b = rays @ eye
        discriminant = b * b - (eye @ eye - .7**2)
        valid = discriminant >= 0
        distance = -b - np.sqrt(np.maximum(discriminant, 0))
        surface = eye + distance[:, :, None] * rays
        color = np.clip((surface / 1.4 + .5) * 255, 0, 255).astype(np.uint8)
        image = np.full((size, size, 3), 185, dtype=np.uint8)
        image[valid] = color[valid]
        name, mask_name = f"frame_{index + 1:04d}.png", f"mask_{index + 1:04d}.png"
        Image.fromarray(image).save(images / name)
        Image.fromarray(valid.astype(np.uint8) * 255).save(masks / mask_name)
        rows.append((index + 1, _rotmat_to_quat(rotation), -rotation @ eye, 1, name))
        entries[name] = {"mask": mask_name, "reviewed": False,
                         "image_sha256": hashlib.sha256((images / name).read_bytes()).hexdigest(),
                         "mask_sha256": hashlib.sha256((masks / mask_name).read_bytes()).hexdigest()}
    _write_colmap_bins(str(sparse), [(1, size, size, [focal, focal, size / 2, size / 2])], rows, (points, colors))
    (masks / "mask_manifest.json").write_text(json.dumps({"schema_version": 1,
        "kind": "synthetic-sphere-fixture", "frames": entries}, indent=2), encoding="utf-8")
    options = {"strategy": "default", "seed": 42, "validation_every": 4,
               "image_cache_mb": 8, "sh_degree": 3}
    for label, extra in (("baseline", {}), ("weighted", {"mask_dir": str(masks),
            "mask_foreground_weight": 1.5, "mask_background_weight": .25, "allow_unreviewed_masks": True})):
        output = root / label
        output.mkdir()
        stats = train(str(root / "dataset"), str(output), args.steps, 1, 10000, lambda *a: None,
                      cleanup_threshold=.005, denoise_config={"distance_percentile": 100},
                      training_options={**options, **extra})
        assert (output / "point_cloud.ply").stat().st_size > 0
        assert stats["n_validation_images"] == 2
        print(json.dumps({"run": label, "steps": args.steps, "points": stats["num_gaussians"]}), flush=True)


if __name__ == "__main__":
    main()
