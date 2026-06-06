"""
統一資料模型
"""

import time
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class SegmentType(str, Enum):
    """片段類型"""

    USABLE = "usable"  # 可用
    CUT = "cut"  # 要剪
    SILENCE = "silence"  # 靜音
    FILLER = "filler"  # 填充詞
    REPEAT = "repeat"  # 重複
    HESITATION = "hesitation"  # 猶豫
    NERVOUS = "nervous"  # 口頭禪
    CORRECTION = "correction"  # 自我修正
    UNCERTAIN = "uncertain"  # 信心不足


class Priority(str, Enum):
    """優先級"""

    HIGH = "high"
    MID = "mid"
    LOW = "low"


@dataclass
class TranscriptionSegment:
    """Whisper 轉錄片段"""

    id: int
    start: float
    end: float
    text: str
    confidence: float = 1.0
    avg_logprob: float = 0.0
    no_speech_prob: float = 0.0
    words: List[Dict] = field(default_factory=list)

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)


@dataclass
class AnalysisSegment:
    """分析後的可剪輯片段"""

    id: int
    start: float
    end: float
    text: str
    confidence: float = 1.0

    type: SegmentType = SegmentType.USABLE
    priority: Priority = Priority.LOW
    reason: str = ""
    rules_hit: List[str] = field(default_factory=list)
    llm_verified: bool = False
    llm_reason: str = ""

    manual_override: Optional[bool] = None  # None=未設定, True=手動保留, False=手動剪

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)

    @property
    def effective_type(self) -> SegmentType:
        """考慮手動覆寫後的最終類型"""
        if self.manual_override is True:
            return SegmentType.USABLE
        if self.manual_override is False:
            return SegmentType.CUT
        return self.type

    @property
    def is_kept(self) -> bool:
        return self.effective_type == SegmentType.USABLE

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "start": self.start,
            "end": self.end,
            "text": self.text,
            "confidence": self.confidence,
            "type": self.type.value,
            "priority": self.priority.value,
            "reason": self.reason,
            "rules_hit": list(self.rules_hit),
            "llm_verified": self.llm_verified,
            "llm_reason": self.llm_reason,
            "manual_override": self.manual_override,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AnalysisSegment":
        return cls(
            id=data["id"],
            start=data["start"],
            end=data["end"],
            text=data.get("text", ""),
            confidence=data.get("confidence", 1.0),
            type=SegmentType(data.get("type", SegmentType.USABLE.value)),
            priority=Priority(data.get("priority", Priority.LOW.value)),
            reason=data.get("reason", ""),
            rules_hit=list(data.get("rules_hit", [])),
            llm_verified=data.get("llm_verified", False),
            llm_reason=data.get("llm_reason", ""),
            manual_override=data.get("manual_override"),
        )


@dataclass
class VideoInfo:
    """視頻資訊"""

    path: str
    duration: float
    fps: float
    width: int
    height: int
    codec: str = ""
    audio_codec: str = ""
    file_size: int = 0


@dataclass
class AnalysisResult:
    """完整分析結果"""

    video_path: str
    video_info: Optional[VideoInfo] = None
    segments: List[AnalysisSegment] = field(default_factory=list)
    transcriptions: List[TranscriptionSegment] = field(default_factory=list)
    fps: float = 30.0
    total_duration: float = 0.0
    used_duration: float = 0.0
    cut_duration: float = 0.0
    analysis_time: float = 0.0
    timestamp: float = field(default_factory=time.time)
    cache_hit: bool = False
    llm_enabled: bool = False
    energy_profile: List[Dict[str, float]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "video_path": self.video_path,
            "video_info": asdict(self.video_info) if self.video_info else None,
            "segments": [s.to_dict() for s in self.segments],
            "fps": self.fps,
            "total_duration": self.total_duration,
            "used_duration": self.used_duration,
            "cut_duration": self.cut_duration,
            "analysis_time": self.analysis_time,
            "timestamp": self.timestamp,
            "cache_hit": self.cache_hit,
            "llm_enabled": self.llm_enabled,
            "energy_profile": self.energy_profile,
        }


@dataclass
class QueueTask:
    """佇列任務"""

    id: str
    video_path: str
    status: str = "pending"  # pending, running, done, error
    progress: float = 0.0
    stage: str = ""
    message: str = ""
    result: Optional[AnalysisResult] = None
    error: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    finished_at: Optional[float] = None
