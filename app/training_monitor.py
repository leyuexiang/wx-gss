"""保存训练阶段与固定验证指标；只记录最佳阶段，不自动替换交付模型。"""

import hashlib
import json
import math
import os
import random
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch


def write_json_atomic(path, value):
    """中断时保留上一份完整记录，禁止产生半截指标文件。"""
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    os.replace(temporary, path)


@contextmanager
def preserve_random_state():
    """阶段评估不能消耗下一步训练的随机序列，包括后续可能增加的随机评估代码。"""
    python_state, numpy_state = random.getstate(), np.random.get_state()
    cpu_state = torch.get_rng_state()
    cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None
    try:
        yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.set_rng_state(cpu_state)
        if cuda_state is not None:
            torch.cuda.set_rng_state_all(cuda_state)


def best_stage(records, selection_metric):
    """按预先声明的验证指标选阶段；同分保留更早的模型，不按训练损失选优。"""
    if selection_metric not in ("subject_psnr", "full_psnr"):
        raise ValueError("阶段选择指标只支持主体或全图 PSNR")
    if not records:
        return None
    if any(not math.isfinite(record[selection_metric]) for record in records):
        raise ValueError("阶段指标包含非有限数")
    return max(records, key=lambda record: record[selection_metric])


class StageMonitor:
    def __init__(self, output_dir, dataset, indices, interval, degree, antialiased, roi_path=None):
        if isinstance(interval, bool) or not isinstance(interval, int) or interval < 1:
            raise ValueError("阶段保存间隔必须为正整数")
        if not indices:
            raise ValueError("阶段评估必须配置固定验证视角")
        self.root = Path(output_dir)
        self.dataset, self.indices = dataset, list(indices)
        self.interval, self.degree, self.antialiased = interval, degree, antialiased
        self.regions = None
        self.annotation_sha256 = None
        if roi_path:
            from app.evaluation import resolve_roi

            source = Path(roi_path)
            annotation = json.loads(source.read_text(encoding="utf-8"))
            names = {dataset.scene.names[index] for index in indices}
            regions, fingerprints = annotation["frames"], annotation["reference_sha256"]
            if set(regions) != names or set(fingerprints) != names:
                raise ValueError("阶段主体标注必须完整对应固定验证集")
            dimensions = {entry["name"]: entry for entry in dataset.scene.dimensions}
            for name in sorted(names):
                image_path = Path(dataset.scene.image_dir) / name
                if hashlib.sha256(image_path.read_bytes()).hexdigest() != fingerprints[name]:
                    raise ValueError(f"阶段评估参考图与冻结标注不一致：{name}")
                size = dimensions[name]
                resolve_roi(regions, name, size["width"], size["height"])
            self.regions = regions
            self.annotation_sha256 = hashlib.sha256(source.read_bytes()).hexdigest()
            write_json_atomic(self.root / "monitor_reference_roi.json", annotation)
        self.selection_metric = "subject_psnr" if self.regions is not None else "full_psnr"
        self.records = []
        self.seconds = 0.0

    def due(self, completed_steps, total_steps):
        return completed_steps % self.interval == 0 or completed_steps == total_steps

    def record(self, params, completed_steps, learning_rate, active_degree):
        from app.evaluation import evaluate_views
        from app.export import export_ply

        started = time.perf_counter()
        destination = self.root / "stages" / f"step_{completed_steps:06d}"
        destination.mkdir(parents=True, exist_ok=False)
        model = destination / "point_cloud_raw.ply"
        with preserve_random_state():
            # 仅写一份未清理的标准模型，避免每阶段重复写原始、清理和查看文件。
            export_ply(params, str(model) + ".tmp")
            os.replace(str(model) + ".tmp", model)
            # 全图与主体复用一次渲染；阶段监控不写大量图片，选定阶段后再出对照图。
            metrics = evaluate_views(params, self.dataset, str(destination), indices=self.indices,
                                     degrees=[self.degree], split="validation_for_stage_selection",
                                     antialiased=self.antialiased, additional_roi=self.regions,
                                     save_images=False)
        full = metrics["averages"][str(self.degree)]
        record = {"completed_steps": completed_steps, "point_count": len(params["means"]),
                  "full_psnr": full["psnr"], "full_ssim": full["ssim"],
                  "position_lr": learning_rate, "active_training_sh_degree": active_degree,
                  "evaluation_sh_degree": self.degree,
                  "model": str(model.relative_to(self.root)),
                  "metrics": str((destination / "metrics.json").relative_to(self.root))}
        if self.regions is not None:
            subject = metrics["subject_averages"][str(self.degree)]
            record.update(subject_psnr=subject["psnr"], subject_ssim=subject["ssim"])
        record["monitor_seconds"] = round(time.perf_counter() - started, 4)
        self.seconds += record["monitor_seconds"]
        self.records.append(record)
        chosen = best_stage(self.records, self.selection_metric)
        history = {"selection_metric": self.selection_metric,
                   "selection_scope": "用于选择阶段的验证集，不是独立测试集",
                   "model_kind": "raw_ply_without_optimizer_resume_state",
                   "annotation_sha256": self.annotation_sha256,
                   "interval": self.interval, "records": self.records, "best_stage": chosen}
        write_json_atomic(self.root / "stage_history.json", history)
        write_json_atomic(self.root / "best_stage.json", chosen)
        return record
