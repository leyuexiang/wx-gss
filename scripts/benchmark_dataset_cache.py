"""固定随机访问序列，对照图像解码次数与缓存占用；不把共享服务器耗时当作纯算法排名。"""

import argparse
import json
import sys
import time
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_dir")
    parser.add_argument("output_file")
    parser.add_argument("--samples", type=int, default=200)
    args = parser.parse_args()
    from app.dataset import ColmapScene, ColmapDataset
    scene = ColmapScene(args.dataset_dir, 1)
    indices = np.random.default_rng(42).integers(0, scene.n_images, args.samples)
    report = {}
    for name, cache_mb, use_bytes in [("float64mb", 64, False), ("uint8_256mb", 256, True)]:
        dataset = ColmapDataset(scene, cache_bytes=cache_mb * 1024**2, uint8_images=use_bytes)
        started = time.perf_counter()
        with patch("app.dataset.Image.open", wraps=Image.open) as decoder:
            for index in indices:
                dataset[int(index)]
        report[name] = {"samples": args.samples, "image_decodes": decoder.call_count,
                        "cache_used_bytes": dataset._cached_bytes,
                        "elapsed_seconds_reference_only": time.perf_counter() - started}
        del dataset
    Path(args.output_file).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
