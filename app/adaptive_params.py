"""自适应参数推荐系统

根据图片数量、SfM 后端和硬件条件，自动推荐最优训练参数。
"""


def get_adaptive_params(n_images: int, sfm_backend: str = "mast3r",
                       gpu_memory_gb: float = 8.0):
    """
    根据输入自动推荐最优参数

    参数:
    - n_images: 图片数量
    - sfm_backend: SfM 后端（colmap 或 mast3r）
    - gpu_memory_gb: GPU 显存大小（GB）

    返回:
    {
        "iterations": 5000,
        "downsample": 2,
        "cap_max": 800000,
        "auto_cleanup": True,
        "cleanup_threshold": 0.1,
        "estimated_time_minutes": 4.5,
        "estimated_gpu_memory_gb": 5.2,
        "quality_level": "high",
        "reason": "中等视角（6-10张），推荐高质量"
    }
    """

    if sfm_backend == "mast3r":
        # 少视角场景（2-15 张）
        if n_images <= 5:
            # 极少视角：保守配置
            return {
                "iterations": 3000,
                "downsample": 2 if gpu_memory_gb >= 4 else 4,
                "cap_max": 500000,
                "auto_cleanup": True,
                "cleanup_threshold": 0.12,  # 稍高阈值，避免过拟合杂点
                "estimated_time_minutes": 2.5,
                "estimated_gpu_memory_gb": 2.9,
                "quality_level": "medium",
                "reason": "少视角（≤5张），平衡速度与质量"
            }
        elif n_images <= 10:
            # 中等视角：高质量配置
            return {
                "iterations": 5000,
                "downsample": 1 if gpu_memory_gb >= 6 else 2,
                "cap_max": 800000,
                "auto_cleanup": True,
                "cleanup_threshold": 0.10,
                "estimated_time_minutes": 4.8,
                "estimated_gpu_memory_gb": 4.5 if gpu_memory_gb >= 6 else 3.2,
                "quality_level": "high",
                "reason": "中等视角（6-10张），推荐高质量配置"
            }
        else:  # 11-15 张
            # 较多视角：极致质量
            return {
                "iterations": 7000,
                "downsample": 1 if gpu_memory_gb >= 8 else 2,
                "cap_max": 1000000,
                "auto_cleanup": True,
                "cleanup_threshold": 0.08,  # 稍低阈值，数据质量好
                "estimated_time_minutes": 7.0,
                "estimated_gpu_memory_gb": 6.5 if gpu_memory_gb >= 8 else 4.8,
                "quality_level": "ultra",
                "reason": "较多视角（11-15张），可达极致质量"
            }
    else:  # colmap
        # 密集拍摄场景（20-500 张）
        if n_images <= 30:
            # 少量密集拍摄
            return {
                "iterations": 5000,
                "downsample": 2,
                "cap_max": 800000,
                "auto_cleanup": True,
                "cleanup_threshold": 0.10,
                "estimated_time_minutes": 8.5,
                "estimated_gpu_memory_gb": 5.8,
                "quality_level": "high",
                "reason": "密集拍摄（20-30张），高质量配置"
            }
        elif n_images <= 100:
            # 中等密集
            return {
                "iterations": 7000,
                "downsample": 2,
                "cap_max": 1200000,
                "auto_cleanup": True,
                "cleanup_threshold": 0.08,
                "estimated_time_minutes": 15.0,
                "estimated_gpu_memory_gb": 7.2,
                "quality_level": "ultra",
                "reason": "大量图片（30-100张），极致质量配置"
            }
        else:  # 100+ 张
            # 超大数据集
            if gpu_memory_gb < 16:
                # 显存受限
                return {
                    "iterations": 7000,
                    "downsample": 4,  # 降低分辨率以节省显存
                    "cap_max": 1000000,
                    "auto_cleanup": True,
                    "cleanup_threshold": 0.10,
                    "estimated_time_minutes": 20.0,
                    "estimated_gpu_memory_gb": 7.5,
                    "quality_level": "high",
                    "reason": "超大数据集（100+张），受限于显存"
                }
            else:
                # 专业级配置
                return {
                    "iterations": 10000,
                    "downsample": 2,
                    "cap_max": 1500000,
                    "auto_cleanup": True,
                    "cleanup_threshold": 0.05,
                    "estimated_time_minutes": 30.0,
                    "estimated_gpu_memory_gb": 12.0,
                    "quality_level": "ultra",
                    "reason": "超大数据集（100+张），专业级配置"
                }


def apply_adaptive_params(recommended: dict, user_overrides: dict = None):
    """
    应用自适应参数，用户参数优先

    参数:
    - recommended: 推荐的参数字典
    - user_overrides: 用户提供的参数（可能为 None）

    返回: 合并后的参数
    """
    result = recommended.copy()

    # 用户参数覆盖推荐参数
    if user_overrides:
        for key, value in user_overrides.items():
            if value is not None:
                result[key] = value
                # 标记哪些参数被用户覆盖
                if "overridden_params" not in result:
                    result["overridden_params"] = []
                result["overridden_params"].append(key)

    return result


def get_quality_presets():
    """
    获取预设的质量配置

    返回预设配置字典，用户可以直接选择
    """
    return {
        "fast": {
            "name": "快速预览",
            "iterations": 2000,
            "downsample": 4,
            "cap_max": 300000,
            "auto_cleanup": True,
            "cleanup_threshold": 0.12,
            "use_case": "快速测试，验证拍摄质量",
            "time_estimate": "1-2 分钟"
        },
        "balanced": {
            "name": "平衡模式",
            "iterations": 3000,
            "downsample": 2,
            "cap_max": 500000,
            "auto_cleanup": True,
            "cleanup_threshold": 0.10,
            "use_case": "日常使用，质量与速度平衡",
            "time_estimate": "2-3 分钟"
        },
        "high_quality": {
            "name": "高质量",
            "iterations": 5000,
            "downsample": 1,
            "cap_max": 800000,
            "auto_cleanup": True,
            "cleanup_threshold": 0.10,
            "use_case": "专业展示，高清晰度",
            "time_estimate": "4-5 分钟"
        },
        "ultra": {
            "name": "极致质量",
            "iterations": 7000,
            "downsample": 1,
            "cap_max": 1000000,
            "auto_cleanup": True,
            "cleanup_threshold": 0.08,
            "use_case": "存档、商业展示",
            "time_estimate": "6-8 分钟"
        }
    }
