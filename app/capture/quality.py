"""通用采集质量门禁，不依赖具体前景颜色或分割模型。"""

import json
import math
import os
from pathlib import Path
from typing import Any, Dict, Iterable, Optional


def _finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def evaluate_foreground_support(mask, support, min_coverage=.95):
    """检查掩码是否删掉外部前景证据；证据可来自人工或先验，不等同于真值。"""
    import numpy as np
    if (mask.shape != support.shape or mask.ndim != 2
            or not np.all((mask == 0) | (mask == 255))
            or not np.all((support == 0) | (support == 255))):
        raise ValueError("前景证据与掩码必须是同尺寸二值图")
    if not _finite(min_coverage) or not 0 < min_coverage <= 1:
        raise ValueError("前景覆盖门槛无效")
    reference = support > 0
    pixels = int(reference.sum())
    if pixels == 0:
        raise ValueError("没有可用的前景证据")
    coverage = int(((mask > 0) & reference).sum()) / pixels
    return {"passed": coverage >= min_coverage, "support_pixels": pixels,
            "coverage": coverage, "min_coverage": min_coverage}


def evaluate_mask_ratios(ratios: Iterable[float], min_ratio: float = .01,
                         max_ratio: float = .95, max_jump: float = .20) -> Dict[str, Any]:
    """检查前景面积范围和相邻帧跳变，返回报告而不是静默放行。"""
    values = list(ratios)
    if not values or any(not _finite(value) for value in values):
        raise ValueError("前景面积比例必须是非空有限数字序列")
    if not 0 < min_ratio < max_ratio <= 1 or not 0 < max_jump <= 1:
        raise ValueError("前景面积门槛无效")
    jumps = [abs(current - previous) for previous, current in zip(values, values[1:])]
    checks = {
        "non_empty": len(values) > 0,
        "ratio_range": min(values) >= min_ratio and max(values) <= max_ratio,
        "temporal_stability": not jumps or max(jumps) <= max_jump,
    }
    return {
        "passed": all(checks.values()), "checks": checks, "count": len(values),
        "min_ratio": min(values), "max_ratio": max(values), "mean_ratio": sum(values) / len(values),
        "max_adjacent_jump": max(jumps) if jumps else 0.,
        "thresholds": {"min_ratio": min_ratio, "max_ratio": max_ratio, "max_jump": max_jump},
    }


def evaluate_reconstruction(report: Dict[str, Any], min_registration_ratio: float = .90,
                            max_reprojection_error: float = 1.0,
                            max_angular_gap_degrees: Optional[float] = 30.) -> Dict[str, Any]:
    """统一判断重建是否允许进入训练，缺失诊断字段时明确失败。"""
    required = ("input_images", "registered_images", "mean_reprojection_error")
    missing = [key for key in required if key not in report]
    if missing:
        raise ValueError(f"重建报告缺少字段：{', '.join(missing)}")
    total, registered = report["input_images"], report["registered_images"]
    error = report["mean_reprojection_error"]
    if type(total) is not int or type(registered) is not int or total < 1 or not 0 <= registered <= total:
        raise ValueError("重建图片数量无效")
    if error is not None and (not _finite(error) or error < 0):
        raise ValueError("重投影误差必须是有限数字或明确为空")
    if (not _finite(min_registration_ratio) or not 0 < min_registration_ratio <= 1
            or not _finite(max_reprojection_error) or max_reprojection_error <= 0
            or (max_angular_gap_degrees is not None and
                (not _finite(max_angular_gap_degrees) or not 0 < max_angular_gap_degrees <= 360))):
        raise ValueError("重建质量门槛无效")
    ratio = registered / total
    checks = {
        "registration_ratio": ratio >= min_registration_ratio,
        "reprojection_error": error is not None and error <= max_reprojection_error,
    }
    angular_gap = report.get("max_angular_gap_degrees")
    if max_angular_gap_degrees is not None:
        checks["angular_coverage"] = (angular_gap is not None and _finite(angular_gap)
                                       and 0 <= angular_gap <= max_angular_gap_degrees)
    return {
        "passed": all(checks.values()), "checks": checks,
        "registration_ratio": ratio, "mean_reprojection_error": error,
        "max_angular_gap_degrees": angular_gap,
        "thresholds": {"min_registration_ratio": min_registration_ratio,
                       "max_reprojection_error": max_reprojection_error,
                       "max_angular_gap_degrees": max_angular_gap_degrees},
    }


def write_protocol(path: str, payload: Dict[str, Any]) -> None:
    """原子写入任务协议，避免中断后留下半份质量记录。"""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    os.replace(temporary, destination)
