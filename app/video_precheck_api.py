"""
视频预检 API - 返回视频分析和智能参数推荐
"""
from fastapi import APIRouter, UploadFile, File, HTTPException
import os
import shutil
import tempfile

router_precheck = APIRouter(prefix="/api/3dgs", tags=["Video Pre-check"])


@router_precheck.post("/video/precheck")
async def video_precheck(file: UploadFile = File(...)):
    """
    视频预检 - 上传视频文件，返回分析结果和智能推荐参数

    不会保存视频文件，仅用于预检分析
    """
    # 检查文件格式
    video_extensions = ('.mp4', '.avi', '.mov', '.mkv', '.webm')
    if not any(file.filename.lower().endswith(ext) for ext in video_extensions):
        raise HTTPException(status_code=400, detail="不支持的视频格式")

    # 临时保存文件进行分析
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        # 导入管理 API 中的函数
        from app.management_api import get_video_info, suggest_extraction_params

        # 获取视频信息
        video_info = get_video_info(tmp_path)

        width = video_info['width']
        height = video_info['height']
        duration = video_info['duration']
        fps = video_info['fps']

        # 质量检查
        issues = []
        warnings = []

        # 检查分辨率（支持横屏和竖屏）
        min_dim = min(width, height)
        max_dim = max(width, height)

        if min_dim < 720 or max_dim < 1280:
            issues.append(f"分辨率过低（{width}x{height}），建议至少 1280x720 或 720x1280")
        elif max_dim < 1920:
            warnings.append(f"分辨率为 {width}x{height}，建议使用 1920x1080 以获得更好效果")

        if duration < 10:
            warnings.append(f"视频较短（{duration:.1f}秒），建议至少 15 秒")
        elif duration > 300:
            warnings.append(f"视频较长（{duration:.1f}秒），建议裁剪到关键片段（30-120秒）")

        if fps < 20:
            warnings.append(f"帧率较低（{fps:.1f}fps），可能影响抽帧质量")

        # 如果有严重问题
        if issues:
            return {
                "code": 400,
                "message": "\n".join(issues),
                "data": {
                    "valid": False,
                    "videoInfo": video_info,
                    "issues": issues
                }
            }

        # 获取智能推荐参数
        suggested_params = suggest_extraction_params(width, height, duration, fps)

        return {
            "code": 200,
            "message": "视频预检通过",
            "data": {
                "valid": True,
                "videoInfo": video_info,
                "warnings": warnings,
                "suggestedParams": suggested_params
            }
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"视频分析失败: {str(e)}")

    finally:
        # 清理临时文件
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
