"""训练前冻结实际输入和源码摘要，旧素材缺失的拍摄信息保持未知。"""

import hashlib
import json
from pathlib import Path

from .quality import write_protocol


def file_digest(path):
    """流式读取，避免视频或高分辨率图像整份驻留内存。"""
    result = hashlib.sha256()
    with open(path, "rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def freeze_training_protocol(task, options, iterations, source_root):
    """先验证掩码与实际图一致，再创建不可覆盖的本轮协议。"""
    from app.dataset import ColmapScene
    from app.subject_masks import validate_mask_set

    task, root = Path(task), Path(source_root)
    destination = task / "protocol.json"
    if destination.exists():
        raise ValueError("训练协议已存在，请使用独立任务保留实验")
    selection = json.loads((task / "selection.json").read_text(encoding="utf-8"))
    mask_manifest = Path(options["mask_dir"]) / "mask_manifest.json"
    scene = ColmapScene(str(task / "dataset"), factor=1,
                        rigid_normalization=options.get("rigid_normalization", False))
    _, report = validate_mask_set(scene, options["mask_dir"],
                                 allow_unreviewed=options.get("allow_unreviewed_masks", False))
    # 只审计本链路直接使用的训练、几何和适配器源码，不扫描数据或个人文件。
    sources = ["app/trainer.py", "app/dataset.py", "app/pipeline.py", "app/subject_masks.py",
               "app/evaluation.py", "app/foreground/__init__.py", "app/capture/provenance.py",
               "scripts/run_greenscreen_trial.py"]
    sources += [str(path.relative_to(root)).replace("\\", "/")
                for path in sorted((root / "app/foreground/priors").glob("*.py"))]
    masks = json.loads(mask_manifest.read_text(encoding="utf-8"))
    payload = {"schema_version": 1, "capture_mode": "unknown",
        "capture_mode_scope": "未运行自动运动识别；前景模式不代表已验证的转台约束",
        "foreground_provider": masks.get("provider", "legacy_unspecified"),
        "foreground_kind": masks.get("kind", "unspecified"),
        "reference_kind": "foreground_black_background" if options.get("foreground_only") else "original_full_frame",
        "iterations": iterations, "training_options": options,
        "actual_image_dimensions": scene.dimensions, "frames": report["frames"],
        "registered_names": scene.names, "selection_sha256": file_digest(task / "selection.json"),
        "source_video_sha256": selection.get("video_sha256"),
        "mask_manifest_sha256": report["manifest_sha256"],
        "sources": {name: file_digest(root / name) for name in sources},
        "validation_scope": "仅留出颜色监督；几何可能使用全部注册图，不是独立几何测试"}
    write_protocol(str(destination), payload)
    return payload
