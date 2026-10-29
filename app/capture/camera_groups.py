"""重建前校验组内共享内参和组间隔离，防止命令成功但分组悄然失效。"""

import math
import sqlite3
from contextlib import closing
from pathlib import Path

import numpy as np
from PIL import Image


def validate_feature_groups(groups, images_dir, camera_model):
    """每张原图只能属于一组；裁切主点允许在图外，不能擅自改成裁切中心。"""
    root = Path(images_dir).resolve()
    counts = {"SIMPLE_RADIAL": 4, "PINHOLE": 4, "OPENCV": 8}
    if camera_model not in counts:
        raise ValueError("相机分组不支持此内参模型")
    prepared, seen = [], set()
    for group in groups:
        params = group["camera_params"]
        if (not isinstance(params, (list, tuple)) or len(params) != counts[camera_model]
                or any(isinstance(value, bool) or not isinstance(value, (int, float))
                       or not math.isfinite(value) for value in params)):
            raise ValueError("相机组内参长度或数值无效")
        focal_count = 1 if camera_model == "SIMPLE_RADIAL" else 2
        if any(value <= 0 for value in params[:focal_count]):
            raise ValueError("相机组焦距必须为正数")
        names = Path(group["image_list"]).read_text(encoding="utf-8-sig").splitlines()
        if not names or any(not name.strip() or name != name.strip() for name in names):
            raise ValueError("相机组图像清单不能为空或含空白名称")
        size = None
        for name in names:
            path = (root / name).resolve()
            if not path.is_relative_to(root) or not path.is_file():
                raise ValueError(f"相机组图像不存在或越界：{name}")
            if name in seen:
                raise ValueError(f"相机组存在重复图像：{name}")
            seen.add(name)
            with Image.open(path) as image:
                if size is not None and image.size != size:
                    raise ValueError("共享内参的相机组图像尺寸必须一致")
                size = image.size
        prepared.append({"names": names, "size": size, "camera_params": list(params)})
    # 与准备入口支持的图像类型保持一致，不漏掉未分组的补拍照片。
    extensions = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
    inventory = {p.relative_to(root).as_posix() for p in root.rglob("*")
                 if p.is_file() and p.suffix.lower() in extensions}
    if seen != inventory:
        raise ValueError(f"相机组必须完整覆盖原图，缺少：{sorted(inventory - seen)}")
    return prepared


def audit_feature_groups(database_path, groups):
    """从工具实际写入的数据库核验分组，而不是相信命令行参数已生效。"""
    # 数据库事务上下文不会关闭连接；显式关闭才能及时释放 Windows 文件句柄。
    with closing(sqlite3.connect(f"{Path(database_path).resolve().as_uri()}?mode=ro", uri=True)) as connection:
        images = dict(connection.execute("SELECT name, camera_id FROM images"))
        cameras = {row[0]: row[1:] for row in connection.execute(
            "SELECT camera_id, width, height, params FROM cameras")}
    expected = {name for group in groups for name in group["names"]}
    if set(images) != expected:
        raise ValueError("特征数据库与相机组图像集合不一致")
    seen_ids, records = set(), []
    for group in groups:
        ids = {images[name] for name in group["names"]}
        if len(ids) != 1 or ids & seen_ids:
            raise ValueError("相机组内未共享内参或跨组错误共用相机")
        camera_id = next(iter(ids))
        seen_ids.add(camera_id)
        width, height, blob = cameras[camera_id]
        params = np.frombuffer(blob, dtype="<f8")
        if ((width, height) != tuple(group["size"]) or len(params) != len(group["camera_params"])
                or not np.allclose(params, group["camera_params"], rtol=0, atol=1e-9)):
            raise ValueError("数据库实际相机尺寸或初始内参与声明不一致")
        records.append({"camera_id": camera_id, "names": group["names"],
                        "size": [width, height], "initial_camera_params": params.tolist()})
    return {"passed": True, "scope": "特征提取完成后，几何优化之前的内参与分组",
            "group_count": len(records), "image_count": len(images), "groups": records}
