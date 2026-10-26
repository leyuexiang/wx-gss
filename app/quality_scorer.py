"""质量评分系统

自动评估训练结果质量，给出评分和改进建议。
"""


def calculate_quality_score(stats: dict, params: dict = None):
    """
    计算训练质量评分（0-100）

    参数:
    - stats: train_stats.json 的内容
    - params: 训练参数（可选）

    返回:
    {
        "score": 92,
        "rating": "优秀",
        "factors": {
            "loss": {"score": 95, "value": 0.0155, "weight": 0.4, "description": "..."},
            ...
        },
        "recommendations": ["建议1", "建议2", ...]
    }
    """
    if params is None:
        params = {}

    score = 100.0
    factors = {}
    recommendations = []

    # 因子 1: Loss（权重 40%）
    loss = stats.get("final_loss", 1.0)
    loss_score = 100

    if loss > 0.10:
        loss_score = 0
        recommendations.append("Loss 过高（>0.10），建议增加训练迭代次数或改善拍摄质量")
    elif loss > 0.05:
        loss_score = 50
        recommendations.append("Loss 较高（>0.05），可增加 iterations 到 5000-7000")
    elif loss > 0.02:
        loss_score = 80
    elif loss > 0.01:
        loss_score = 95
    else:
        loss_score = 100

    factors["loss"] = {
        "score": loss_score,
        "value": round(loss, 5),
        "weight": 0.4,
        "description": "训练收敛程度（越低越好）",
        "status": "excellent" if loss_score >= 95 else "good" if loss_score >= 80 else "warning" if loss_score >= 50 else "critical"
    }
    score -= (100 - loss_score) * 0.4

    # 因子 2: 高斯数量（权重 30%）
    n_gaussians = stats.get("num_gaussians", 0)
    gaussians_score = 100

    if n_gaussians < 50000:
        gaussians_score = 20
        recommendations.append("高斯点过少（<50K），可能重建失败，请检查图片质量和数量")
    elif n_gaussians < 100000:
        gaussians_score = 60
        recommendations.append("高斯点较少（<100K），可增加 cap_max 或检查图片覆盖度")
    elif n_gaussians > 1500000:
        gaussians_score = 70
        recommendations.append("高斯点过多（>1.5M），建议减小 cap_max 或启用自动清理")
    elif n_gaussians > 1200000:
        gaussians_score = 85
        recommendations.append("高斯点较多（>1.2M），可考虑减小 cap_max")
    else:
        gaussians_score = 100

    factors["gaussians"] = {
        "score": gaussians_score,
        "value": n_gaussians,
        "weight": 0.3,
        "description": "高斯点数量合理性（100K-1.2M 为理想）",
        "status": "excellent" if gaussians_score >= 95 else "good" if gaussians_score >= 80 else "warning" if gaussians_score >= 50 else "critical"
    }
    score -= (100 - gaussians_score) * 0.3

    # 因子 3: 显存效率（权重 20%）
    gpu_mem = stats.get("peak_gpu_mem_gb", 0)
    mem_score = 100

    if gpu_mem > 7.8:
        mem_score = 40
        recommendations.append("显存占用接近上限（>7.8GB），建议增大 downsample 或减小 cap_max")
    elif gpu_mem > 7.5:
        mem_score = 60
        recommendations.append("显存占用较高（>7.5GB），考虑优化参数避免 OOM")
    elif gpu_mem > 7.0:
        mem_score = 80
    else:
        mem_score = 100

    factors["memory"] = {
        "score": mem_score,
        "value": round(gpu_mem, 2),
        "weight": 0.2,
        "description": "显存使用效率（< 7GB 为安全）",
        "status": "excellent" if mem_score >= 95 else "good" if mem_score >= 80 else "warning" if mem_score >= 50 else "critical"
    }
    score -= (100 - mem_score) * 0.2

    # 因子 4: 训练效率（权重 10%）
    train_time = stats.get("train_seconds", 1)
    iterations = stats.get("iterations", 1)
    speed = iterations / max(train_time, 1)  # iter/s

    speed_score = 100
    if speed < 10:
        speed_score = 70
        # 不添加建议，训练速度通常不是主要问题
    elif speed < 20:
        speed_score = 85
    else:
        speed_score = 100

    factors["efficiency"] = {
        "score": speed_score,
        "value": round(speed, 2),
        "weight": 0.1,
        "description": "训练速度（iter/s）",
        "status": "excellent" if speed_score >= 95 else "good" if speed_score >= 80 else "warning"
    }
    score -= (100 - speed_score) * 0.1

    # 综合评分
    score = max(0, min(100, round(score, 1)))

    # 评级
    if score >= 90:
        rating = "优秀"
        rating_emoji = "🏆"
    elif score >= 80:
        rating = "良好"
        rating_emoji = "✅"
    elif score >= 70:
        rating = "一般"
        rating_emoji = "⚠️"
    elif score >= 60:
        rating = "及格"
        rating_emoji = "⚠️"
    else:
        rating = "较差"
        rating_emoji = "❌"

    # 无建议时的默认提示
    if not recommendations:
        if score >= 90:
            recommendations.append("质量已经很好，无需调整")
        else:
            recommendations.append("训练成功完成")

    # 根据参数给出额外建议
    if params.get("downsample", 4) > 1 and score >= 80:
        recommendations.append("如需更高清晰度，可尝试 downsample=1")

    if params.get("iterations", 7000) < 5000 and loss > 0.02:
        recommendations.append("如需更好收敛，可尝试增加 iterations 到 5000-7000")

    return {
        "score": score,
        "rating": rating,
        "rating_emoji": rating_emoji,
        "factors": factors,
        "recommendations": recommendations
    }


def get_quality_summary(quality_result: dict) -> str:
    """
    生成质量摘要文本

    参数:
    - quality_result: calculate_quality_score 的返回值

    返回: 摘要字符串
    """
    score = quality_result["score"]
    rating = quality_result["rating"]
    emoji = quality_result.get("rating_emoji", "")

    summary = f"{emoji} 质量评分: {score}/100 ({rating})\n"

    # 各因子得分
    factors = quality_result["factors"]
    summary += "\n各项指标:\n"
    for name, factor in factors.items():
        status_emoji = {
            "excellent": "✅",
            "good": "✅",
            "warning": "⚠️",
            "critical": "❌"
        }.get(factor.get("status", "good"), "•")

        summary += f"  {status_emoji} {factor['description']}: {factor['score']}/100\n"

    # 建议
    if quality_result["recommendations"]:
        summary += "\n改进建议:\n"
        for i, rec in enumerate(quality_result["recommendations"], 1):
            summary += f"  {i}. {rec}\n"

    return summary.strip()
