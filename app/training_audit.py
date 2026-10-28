"""可选训练复现审计：保存真实初始化和抽样，不改变随机数或训练调度。"""

import hashlib
import json
from pathlib import Path

import numpy as np


def array_digest(array):
    """同时记录类型、形状和字节，识别点顺序或相机排列变化。"""
    array = np.ascontiguousarray(array)
    return {"shape": list(array.shape), "dtype": str(array.dtype),
            "sha256": hashlib.sha256(array.tobytes()).hexdigest()}


class TrainingAudit:
    def __init__(self, output_dir, params, scene, iterations):
        self.root = Path(output_dir) / "reproduction_audit"
        self.root.mkdir(exist_ok=False)
        # 仅在显式审计实验中进行一次显卡回读；普通训练不承担快照开销。
        arrays = {"param_" + key: value.detach().cpu().numpy().copy()
                  for key, value in params.items()}
        arrays.update(cameras=scene.camtoworlds, intrinsics=scene.Ks)
        np.savez(self.root / "initialization.npz", **arrays)
        self.initialization = {key: array_digest(value) for key, value in arrays.items()}
        self.images = {name: hashlib.sha256((Path(scene.image_dir) / name).read_bytes()).hexdigest()
                       for name in scene.names}
        self.samples = np.empty(iterations, dtype=np.int64)
        self.completed = 0
        self.save()

    def record(self, index):
        self.samples[self.completed] = index
        self.completed += 1

    def save(self):
        # 只写已经实际使用的样本；阶段保存失败也不会伪造后续抽样记录。
        used = self.samples[:self.completed]
        np.save(self.root / "sampled_indices.npy", used)
        report = {"initialization": self.initialization, "image_sha256": self.images,
                  "completed_samples": self.completed, "sampling": array_digest(used),
                  "scope": "诊断输入与抽样一致性；不保证显卡算子逐位确定，不含优化器续训状态"}
        (self.root / "audit.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
