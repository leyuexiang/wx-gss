"""COLMAP / MASt3R 稀疏重建 → gsplat 训练 → 导出打包 的完整流水线。"""

import os
import json
import hashlib
import shutil
import subprocess
import sys
import zipfile

import numpy as np

from app.config import (
    COLMAP_MAX_NUM_MATCHES,
    COLMAP_USE_GPU,
    MAST3R_CKPT,
    MAST3R_MAX_IMAGES,
    MAST3R_MIN_IMAGES,
    MAST3R_REPO_DIR,
    MAX_IMAGES,
    MIN_IMAGES,
    TASKS_DIR,
)

IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


def _source_hash(name):
    """记录实际运行源码版本，文件句柄及时关闭。"""
    with open(os.path.join(os.path.dirname(__file__), name), "rb") as source:
        return hashlib.sha256(source.read()).hexdigest()


def _run(cmd: list, log_path: str, stage: str):
    with open(log_path, "ab") as log:
        log.write(("\n$ " + " ".join(cmd) + "\n").encode())
        proc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT)
    if proc.returncode != 0:
        tail = ""
        try:
            with open(log_path, "rb") as f:
                f.seek(max(0, os.path.getsize(log_path) - 3000))
                tail = f.read().decode(errors="replace")[-1500:]
        except Exception:
            pass
        raise RuntimeError(f"COLMAP 阶段 [{stage}] 失败（exit {proc.returncode}）: {tail}")


def _extract_images(task_dir: str, min_images: int, max_images: int) -> list:
    zip_path = os.path.join(task_dir, "input.zip")
    images_dir = os.path.join(task_dir, "images")
    os.makedirs(images_dir, exist_ok=True)
    # 视频任务已在共享目录抽帧，无需再压缩、解压一遍；照片任务仍从 ZIP 读取。
    if os.path.isfile(zip_path):
        with zipfile.ZipFile(zip_path) as zf:
            for info in zf.infolist():
                if info.is_dir():
                    continue
                name = os.path.basename(info.filename)
                if not name or os.path.splitext(name)[1].lower() not in IMG_EXTS:
                    continue
                with zf.open(info) as src, open(os.path.join(images_dir, name), "wb") as dst:
                    shutil.copyfileobj(src, dst)
    files = sorted(name for name in os.listdir(images_dir) if os.path.splitext(name)[1].lower() in IMG_EXTS)
    if len(files) < min_images:
        raise RuntimeError(f"图片数量不足：{len(files)} 张（至少需要 {min_images} 张多视角照片）")
    if len(files) > max_images:
        raise RuntimeError(f"图片数量过多：{len(files)} 张（上限 {max_images} 张）")
    return files


def _run_colmap(task_dir: str, n_images: int, status_cb, options=None):
    images_dir = os.path.join(task_dir, "images")
    colmap_dir = os.path.join(task_dir, "colmap")
    os.makedirs(colmap_dir, exist_ok=True)
    db = os.path.join(colmap_dir, "database.db")
    sparse = os.path.join(colmap_dir, "sparse")
    log_path = os.path.join(colmap_dir, "colmap.log")

    options = options or {}
    use_gpu = "1" if options.get("use_gpu", COLMAP_USE_GPU) else "0"

    status_cb("colmap", "feature_extractor", 0.05, "COLMAP 特征提取中")
    _run([
        "colmap", "feature_extractor",
        "--database_path", db,
        "--image_path", images_dir,
        "--ImageReader.single_camera", "1",
        "--ImageReader.camera_model", options.get("camera_model", "OPENCV"),
        "--SiftExtraction.use_gpu", use_gpu,
    ], log_path, "feature_extractor")

    status_cb("colmap", "matcher", 0.15, "COLMAP 特征匹配中")
    matcher = options.get("matcher", "auto")
    if matcher == "exhaustive" or (matcher == "auto" and n_images <= 300):
        _run([
            "colmap", "exhaustive_matcher",
            "--database_path", db,
            "--SiftMatching.use_gpu", use_gpu,
            "--SiftMatching.max_num_matches", str(COLMAP_MAX_NUM_MATCHES),
        ], log_path, "matcher")
    else:
        _run([
            "colmap", "sequential_matcher",
            "--database_path", db,
            "--SequentialMatching.overlap", "20",
            # 未配置词汇树时不启动依赖外部模型的回环检索。
            "--SequentialMatching.loop_detection", "0",
            "--SiftMatching.use_gpu", use_gpu,
            "--SiftMatching.max_num_matches", str(COLMAP_MAX_NUM_MATCHES),
        ], log_path, "matcher")

    status_cb("colmap", "mapper", 0.25, "COLMAP 稀疏重建中")
    os.makedirs(sparse, exist_ok=True)
    _run([
        "colmap", "mapper",
        "--database_path", db,
        "--image_path", images_dir,
        "--output_path", sparse,
        "--Mapper.init_min_tri_angle", str(options.get("init_min_tri_angle", 16)),
    ], log_path, "mapper")

    # 编号 0 不保证是最大分量，不能因第一个小分量只注册两张就丢弃后续完整重建。
    import pycolmap
    candidates = []
    for name in os.listdir(sparse):
        path = os.path.join(sparse, name)
        if os.path.isfile(os.path.join(path, "images.bin")):
            model = pycolmap.Reconstruction(path)
            candidates.append((len(model.images), len(model.points3D), path))
    if not candidates:
        raise RuntimeError("COLMAP 重建失败：图片之间重叠度不足或纹理太少，请重新拍摄")
    registered, _, model_dir = max(candidates)
    status_cb("colmap", "component", 0.32, f"选择最大重建分量：{registered}/{n_images} 张")

    status_cb("colmap", "undistort", 0.35, "COLMAP 图像去畸变中")
    dense = os.path.join(colmap_dir, "dense")
    _run([
        "colmap", "image_undistorter",
        "--image_path", images_dir,
        "--input_path", model_dir,
        "--output_path", dense,
        "--output_type", "COLMAP",
        "--max_image_size", str(options.get("max_image_size", 2000)),
    ], log_path, "undistort")
    return dense


def _prepare_dataset(task_dir: str, dense_dir: str, downsample: int, n_images: int, status_cb):
    """整理成 dataset/{sparse/0, images_{factor}} 布局并校验注册率。"""
    import pycolmap
    from PIL import Image

    dataset_dir = os.path.join(task_dir, "dataset")
    os.makedirs(os.path.join(dataset_dir, "sparse", "0"), exist_ok=True)
    for f in ("cameras.bin", "images.bin", "points3D.bin"):
        shutil.copy2(os.path.join(dense_dir, "sparse", f),
                     os.path.join(dataset_dir, "sparse", "0", f))

    rec = pycolmap.Reconstruction(os.path.join(dataset_dir, "sparse", "0"))
    n_registered = len(rec.images)
    ratio = n_registered / n_images
    # 记录重建诊断，成功注册不等于几何精度可靠。
    errors = [point.error for point in rec.points3D.values() if np.isfinite(point.error)]
    with open(os.path.join(task_dir, "reconstruction_report.json"), "w", encoding="utf-8") as output:
        json.dump({"backend": "colmap", "input_images": n_images, "registered_images": n_registered,
                   "registration_ratio": ratio, "mean_reprojection_error": float(np.mean(errors)) if errors else None,
                   "warning": "注册率低于 90%，需要检查缺失视角" if ratio < 0.9 else None},
                  output, ensure_ascii=False, indent=2)
    if n_registered < MIN_IMAGES or ratio < 0.5:
        raise RuntimeError(
            f"COLMAP 注册图片不足：{n_registered}/{n_images}（{ratio:.0%}）。"
            "请保证相邻照片有 60% 以上重叠且纹理清晰"
        )

    src_dir = os.path.join(dense_dir, "images")
    dst_dir = os.path.join(dataset_dir, f"images_{downsample}")
    os.makedirs(dst_dir, exist_ok=True)
    for i, name in enumerate(sorted(os.listdir(src_dir))):
        source_path, destination_path = os.path.join(src_dir, name), os.path.join(dst_dir, name)
        if downsample == 1:
            # 原尺寸无需再次解码和默认 JPEG 压缩，保持去畸变结果的原始字节。
            shutil.copy2(source_path, destination_path)
        else:
            with Image.open(source_path) as img:
                img.resize((img.width // downsample, img.height // downsample), Image.LANCZOS).save(
                    destination_path, quality=95)
        if i % 20 == 0:
            status_cb("colmap", "downsample", 0.36 + 0.04 * i / max(n_registered, 1),
                      f"整理数据集 {i}/{n_registered}")
    return dataset_dir, n_registered, ratio


def _so3(R):
    """投影到最近的旋转矩阵（去数值漂移/尺度）。"""
    U, _, Vt = np.linalg.svd(R)
    Rn = U @ Vt
    if np.linalg.det(Rn) < 0:
        U[:, -1] *= -1
        Rn = U @ Vt
    return Rn


def _rotmat_to_quat(R):
    """旋转矩阵 → 四元数 (w,x,y,z)。"""
    t = np.trace(R)
    if t > 0:
        s = np.sqrt(t + 1.0) * 2
        return np.array([0.25 * s, (R[2, 1] - R[1, 2]) / s,
                         (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s])
    i = int(np.argmax(np.diag(R)))
    j, k = (i + 1) % 3, (i + 2) % 3
    s = np.sqrt(1.0 + R[i, i] - R[j, j] - R[k, k]) * 2
    q = np.zeros(4)
    q[0] = (R[k, j] - R[j, k]) / s
    q[i + 1] = 0.25 * s
    q[j + 1] = (R[j, i] + R[i, j]) / s
    q[k + 1] = (R[k, i] + R[i, k]) / s
    return q / np.linalg.norm(q)


def _write_colmap_bins(sparse_dir, cameras, images, points):
    """手写 COLMAP legacy bin（PINHOLE，空 track），避开 pycolmap 4.x 写入 API 变更。

    cameras: [(cam_id, W, H, [fx,fy,cx,cy])]
    images:  [(img_id, qvec_wxyz, tvec, cam_id, name)]
    points:  (N,3) xyz, (N,3) uint8 rgb
    """
    import struct
    with open(os.path.join(sparse_dir, "cameras.bin"), "wb") as f:
        f.write(struct.pack("<Q", len(cameras)))
        for cid, w, h, params in cameras:
            f.write(struct.pack("<IiQQ", cid, 1, w, h))          # model 1 = PINHOLE
            f.write(struct.pack("<4d", *params))
    with open(os.path.join(sparse_dir, "images.bin"), "wb") as f:
        f.write(struct.pack("<Q", len(images)))
        for iid, q, t, cid, name in images:
            f.write(struct.pack("<I4d3dI", iid, *q, *t, cid))
            f.write(name.encode() + b"\x00")
            f.write(struct.pack("<Q", 0))                        # num_points2D
    xyz, rgb = points
    with open(os.path.join(sparse_dir, "points3D.bin"), "wb") as f:
        f.write(struct.pack("<Q", len(xyz)))
        for pid, (p, c) in enumerate(zip(xyz, rgb), start=1):
            f.write(struct.pack("<Q3d3BdQ", pid, *p, *c, 0.0, 0))


def mast3r_crop_transform(width, height, cropped_width, cropped_height, size=512):
    """还原 MASt3R 长边缩放后的居中裁切，不能用裁切尺寸直接拉伸内参。"""
    resized_width = int(round(width * size / max(width, height)))
    resized_height = int(round(height * size / max(width, height)))
    return (width / resized_width, height / resized_height,
            resized_width // 2 - cropped_width // 2,
            resized_height // 2 - cropped_height // 2,
            resized_width, resized_height)


def _export_mast3r_dataset(scene, loaded, files, dataset_dir, downsample,
                            points_stride=2, max_points=300_000):
    """MASt3R scene → dataset/{sparse/0, images_{factor}, depths}（训练器认的布局）。

    内参/深度在 512 尺度，换算回原图分辨率写 PINHOLE 相机；不做 undistort
    （MASt3R 假设针孔模型），images_{factor} 直接由原图降采样而来。
    深度图保存到 depths/ 用于训练时的深度正则化。
    """
    from PIL import Image, ImageOps

    sparse_dir = os.path.join(dataset_dir, "sparse", "0")
    img_out = os.path.join(dataset_dir, f"images_{downsample}")
    depth_out = os.path.join(dataset_dir, "depths")
    os.makedirs(sparse_dir, exist_ok=True)
    os.makedirs(img_out, exist_ok=True)
    os.makedirs(depth_out, exist_ok=True)

    cam2w = scene.get_im_poses().detach().cpu().numpy()
    Ks512 = scene.intrinsics.detach().cpu().numpy()
    _, dense_dmaps, confs = scene.get_dense_pts3d()
    depths = [d.detach().cpu().numpy().reshape(c.shape)
              for d, c in zip(dense_dmaps, confs)]

    cameras, images, pts_all, col_all = [], [], [], []
    for i, f in enumerate(files):
        name = os.path.basename(f)
        with Image.open(f) as im:
            im = ImageOps.exif_transpose(im)
            W0, H0 = im.size
            if downsample > 1:
                im = im.resize((W0 // downsample, H0 // downsample), Image.LANCZOS)
            # 该路径必须应用方向和缩放，显式保留高质量，避免默认 JPEG 编码再次丢失细节。
            im.convert("RGB").save(os.path.join(img_out, name), quality=95)

        H512, W512 = depths[i].shape
        sx, sy, crop_x, crop_y, resized_width, resized_height = mast3r_crop_transform(W0, H0, W512, H512)
        K = Ks512[i].copy()
        K[0, 0] *= sx; K[0, 2] = (K[0, 2] + crop_x) * sx
        K[1, 1] *= sy; K[1, 2] = (K[1, 2] + crop_y) * sy
        cameras.append((i + 1, W0, H0, [K[0, 0], K[1, 1], K[0, 2], K[1, 2]]))

        c2w = np.eye(4)
        c2w[:3, :3] = _so3(cam2w[i][:3, :3])
        c2w[:3, 3] = cam2w[i][:3, 3]
        w2c = np.linalg.inv(c2w)
        images.append((i + 1, _rotmat_to_quat(w2c[:3, :3]), w2c[:3, 3], i + 1, name))

        # 保存深度图：上采样到原图分辨率，再降采样到训练尺度
        depth512 = depths[i]
        # 上采样到原图
        # 裁切外区域没有深度依据，用零标记无效；直接缩放到训练尺寸避免先生成原图大数组。
        depth_canvas = np.zeros((resized_height, resized_width), dtype=np.float32)
        depth_canvas[crop_y:crop_y + H512, crop_x:crop_x + W512] = depth512
        depth_train = np.asarray(Image.fromarray(depth_canvas).resize(
            (W0 // downsample, H0 // downsample), Image.Resampling.NEAREST))
        # 保存为 .npy（与降采样后图片对应）
        depth_path = os.path.join(depth_out, os.path.splitext(name)[0] + ".npy")
        np.save(depth_path, depth_train.astype(np.float32))

        # 反投影 512 尺度深度图 → 世界坐标点（带真实颜色）
        s = points_stride
        ys, xs = np.mgrid[0:H512:s, 0:W512:s]
        d = depths[i][::s, ::s]
        valid = np.isfinite(d) & (d > 1e-3)
        fx, fy, cx, cy = Ks512[i][0, 0], Ks512[i][1, 1], Ks512[i][0, 2], Ks512[i][1, 2]
        x = (xs[valid] - cx) * d[valid] / fx
        y = (ys[valid] - cy) * d[valid] / fy
        pts_w = np.stack([x, y, d[valid]], axis=1) @ c2w[:3, :3].T + c2w[:3, 3]
        img512 = loaded[i]["img"][0].permute(1, 2, 0).numpy() * 0.5 + 0.5
        col = (img512[::s, ::s][valid].reshape(-1, 3) * 255).clip(0, 255).astype(np.uint8)
        pts_all.append(pts_w)
        col_all.append(col)

    xyz = np.concatenate(pts_all)
    rgb = np.concatenate(col_all)
    if len(xyz) > max_points:
        idx = np.random.default_rng(0).choice(len(xyz), max_points, replace=False)
        xyz, rgb = xyz[idx], rgb[idx]

    _write_colmap_bins(sparse_dir, cameras, images, (xyz, rgb))
    return len(xyz)


def _run_mast3r(task_dir: str, files: list, downsample: int, status_cb):
    """MASt3R 少视角位姿求解 → COLMAP bin 适配器，产出与 _prepare_dataset 同布局的数据集。"""
    import torch

    images_dir = os.path.join(task_dir, "images")
    dataset_dir = os.path.join(task_dir, "dataset")
    cache_dir = os.path.join(task_dir, "mast3r_cache")
    file_paths = [os.path.join(images_dir, f) for f in files]

    if MAST3R_REPO_DIR not in sys.path:
        sys.path.insert(0, MAST3R_REPO_DIR)
        sys.path.insert(0, os.path.join(MAST3R_REPO_DIR, "dust3r"))

    from dust3r.image_pairs import make_pairs
    from dust3r.utils.image import load_images
    from mast3r.cloud_opt.sparse_ga import sparse_global_alignment
    from mast3r.model import AsymmetricMASt3R

    status_cb("mast3r", "mast3r_load", 0.05, "MASt3R 模型加载中")
    device = "cuda"
    model = AsymmetricMASt3R.from_pretrained(MAST3R_CKPT).to(device).eval()

    try:
        status_cb("mast3r", "mast3r_align", 0.1, f"MASt3R 位姿求解中（{len(files)} 张）")
        loaded = load_images(file_paths, size=512, square_ok=True, verbose=False)
        pairs = make_pairs(loaded, scene_graph="complete", symmetrize=True)
        scene = sparse_global_alignment(
            file_paths, pairs, cache_dir, model,
            subsample=8, device=device,
            lr1=0.07, niter1=500, lr2=0.014, niter2=200,
            opt_depth=False, shared_intrinsics=True, matching_conf_thr=5.0,
        )

        status_cb("mast3r", "mast3r_export", 0.3, "导出稀疏重建结果")
        n_points = _export_mast3r_dataset(scene, loaded, file_paths, dataset_dir, downsample)
    finally:
        del model
        torch.cuda.empty_cache()

    if n_points < 100:
        raise RuntimeError(f"MASt3R 重建点过少（{n_points}），图片重叠度可能不足")
    with open(os.path.join(task_dir, "reconstruction_report.json"), "w", encoding="utf-8") as output:
        json.dump({"backend": "mast3r", "input_images": len(files), "pose_count": len(files),
                   "initial_points": n_points, "registration_ratio": None,
                   "note": "输出全部位姿不等于位姿全部准确"}, output, ensure_ascii=False, indent=2)
    return dataset_dir, len(files), 1.0


def run_pipeline(task_id: str, iterations: int, downsample: int, cap_max: int, status_cb,
                  sfm_backend: str = "colmap", auto_cleanup: bool = True,
                  cleanup_threshold: float = 0.1, enable_depth_reg: bool = None,
                  depth_reg_weight: float = 0.1, enable_pose_opt: bool = False,
                  pose_opt_start: int = 500, pose_opt_lr: float = 1e-5):
    task_dir = os.path.join(TASKS_DIR, task_id)
    # 请求配置在重建前读取，确保特征提取、尺寸和训练参数使用同一份记录。
    from app.training_config import save_training_config
    config_path = os.path.join(task_dir, "training_config.json")
    full_config = {}
    if os.path.isfile(config_path):
        with open(config_path, encoding="utf-8") as source:
            full_config = json.load(source)
    training_options = full_config.get("training", {})
    denoise_config = full_config.get("denoise")
    if denoise_config is not None:
        auto_cleanup = denoise_config.get("enable", auto_cleanup)
        cleanup_threshold = denoise_config.get("opacity_threshold", cleanup_threshold)
    runtime = {"iterations": iterations, "downsample": downsample, "cap_max": cap_max,
               "sfm_backend": sfm_backend, "requested_config": full_config,
               "source_hashes": {name: _source_hash(name)
                                 for name in ("trainer.py", "pipeline.py", "dataset.py", "export.py",
                                              "evaluation.py", "training_monitor.py")}}
    save_training_config(task_dir, runtime, "effective_config.json")

    status_cb("queued", "extract", 0.02, "解压图片包")
    if sfm_backend == "mast3r":
        files = _extract_images(task_dir, MAST3R_MIN_IMAGES, MAST3R_MAX_IMAGES)
        dataset_dir, n_registered, ratio = _run_mast3r(task_dir, files, downsample, status_cb)
        status_cb("mast3r", "mast3r_done", 0.4,
                  f"MASt3R 完成：{n_registered}/{len(files)} 张图片位姿求解成功")
    else:
        files = _extract_images(task_dir, MIN_IMAGES, MAX_IMAGES)
        dense_dir = _run_colmap(task_dir, len(files), status_cb, full_config.get("colmap"))
        dataset_dir, n_registered, ratio = _prepare_dataset(task_dir, dense_dir, downsample, len(files), status_cb)
        status_cb("colmap", "colmap_done", 0.4,
                  f"COLMAP 完成：{n_registered}/{len(files)} 张图片注册成功（{ratio:.0%}）")

    from app.trainer import train
    out_dir = os.path.join(task_dir, "output")
    os.makedirs(out_dir, exist_ok=True)

    def train_progress(step, total, msg):
        status_cb("training", "training", 0.4 + 0.55 * step / total, msg)

    # 深度开关以明确请求为准，不能仅因保存了深度权重就自动启用。
    if enable_depth_reg is None:
        enable_depth_reg = full_config.get("depth", {}).get("enable", False)
    depth_reg_weight = full_config.get("depth", {}).get("weight", depth_reg_weight)

    stats = train(
        dataset_dir=dataset_dir,
        out_dir=out_dir,
        iterations=iterations,
        downsample=downsample,
        cap_max=cap_max,
        progress_cb=train_progress,
        auto_cleanup=auto_cleanup,
        cleanup_threshold=cleanup_threshold,
        enable_depth_reg=enable_depth_reg,
        depth_reg_weight=depth_reg_weight,
        enable_pose_opt=enable_pose_opt,
        pose_opt_start=pose_opt_start,
        pose_opt_lr=pose_opt_lr,
        denoise_config=denoise_config,
        training_options=training_options,
    )

    # 质量评分（新增）
    status_cb("packaging", "quality_scoring", 0.95, "评估质量")
    from app.quality_scorer import calculate_quality_score
    quality_result = calculate_quality_score(stats, {
        "iterations": iterations,
        "downsample": downsample,
        "cap_max": cap_max,
        "auto_cleanup": auto_cleanup,
        "cleanup_threshold": cleanup_threshold
    })

    # 合并质量评分到统计中
    stats["quality_score"] = quality_result["score"]
    stats["quality_rating"] = quality_result["rating"]
    stats["quality_factors"] = quality_result["factors"]
    stats["recommendations"] = quality_result["recommendations"]

    # 重新保存统计文件（包含质量评分）
    with open(os.path.join(out_dir, "train_stats.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    status_cb("packaging", "packaging", 0.97, "打包结果")
    result_zip = os.path.join(out_dir, "result.zip")
    with zipfile.ZipFile(result_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        for name in ("point_cloud.ply", "point_cloud.splat", "train_stats.json", "effective_config.json", "cleanup_report.json",
                     "stage_history.json", "best_stage.json", "monitor_reference_roi.json", "mask_validation.json"):
            p = os.path.join(out_dir, name)
            if os.path.exists(p):
                zf.write(p, name)
        # 结果包只附带最佳阶段，全部阶段继续保存在任务目录，避免下载包重复膨胀。
        selected = stats.get("monitoring", {}).get("best_stage")
        if selected:
            stage_path = os.path.realpath(os.path.join(out_dir, selected["model"]))
            if os.path.commonpath([os.path.realpath(out_dir), stage_path]) != os.path.realpath(out_dir):
                raise ValueError("最佳阶段模型超出输出目录")
            zf.write(stage_path, "best_stage/point_cloud_raw.ply")
    return stats
