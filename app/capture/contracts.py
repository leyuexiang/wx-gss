"""通用采集流程的数据契约。

契约只描述跨场景稳定的信息，白罐、红罐等颜色先验仍由外部适配器提供。
"""

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional
import math


@dataclass
class CaptureManifest:
    """记录输入素材和实际训练尺寸，避免用原始分辨率代替有效细节。"""

    schema_version: int = 1
    source_kind: str = "video"
    capture_mode: str = "unknown"
    frame_count: int = 0
    selected_count: int = 0
    fps: Optional[float] = None
    original_size: List[int] = field(default_factory=list)
    training_size: List[int] = field(default_factory=list)
    provider: str = "none"
    source_sha256: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        """在任务排队前拒绝不完整或自相矛盾的素材描述。"""
        if self.schema_version != 1:
            raise ValueError("采集协议版本不受支持")
        if self.source_kind not in {"video", "images", "mixed"}:
            raise ValueError("source_kind 必须是 video、images 或 mixed")
        if self.capture_mode not in {"unknown", "static_scene", "turntable_object",
                                     "foreground_scene", "mixed_motion"}:
            raise ValueError("capture_mode 不是受支持的拍摄模式")
        if (type(self.frame_count) is not int or type(self.selected_count) is not int
                or self.frame_count < 1 or self.selected_count < 1 or self.selected_count > self.frame_count):
            raise ValueError("采集帧数关系无效")
        if self.fps is not None and (isinstance(self.fps, bool) or not isinstance(self.fps, (int, float))
                                    or not math.isfinite(self.fps) or self.fps <= 0):
            raise ValueError("帧率必须为正数")
        for name, size in (("original_size", self.original_size), ("training_size", self.training_size)):
            if len(size) != 2 or any(type(value) is not int or value <= 0 for value in size):
                raise ValueError(f"{name} 必须是正整数宽高")

    def to_dict(self) -> Dict[str, Any]:
        """输出可写入 protocol.json 的普通字典。"""
        self.validate()
        return asdict(self)


@dataclass
class ForegroundReport:
    """前景适配器的统一结果摘要，不保存具体分割算法内部状态。"""

    provider: str
    frame_count: int
    reviewed: bool
    min_ratio: float
    max_ratio: float
    mean_ratio: float
    mask_sha256: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PoseReport:
    """位姿和稀疏几何的统一质量摘要。"""

    backend: str
    input_images: int
    registered_images: int
    mean_reprojection_error: Optional[float]
    max_angular_gap_degrees: Optional[float] = None
    points3d: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
