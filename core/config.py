"""
統一設定管理
支援從 JSON 載入設定,提供型別安全的存取介面
"""
import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("smart_aroll.config")


DEFAULT_CONFIG = {
    "whisper": {
        "model_size": "large-v3",
        "language": "zh",
        "device": "cuda",
        "compute_type": "float16",
        "beam_size": 1,
        "vad_filter": True,
        "vad_threshold": 0.5,
        "vad_min_silence_ms": 500,
        "vad_speech_pad_ms": 200,
        "condition_on_previous_text": False,
    },
    "llm": {
        "enabled": False,
        "model_path": "",
        "n_ctx": 2048,
        "n_threads": 4,
        "n_gpu_layers": 0,
        "max_tokens": 256,
        "temperature": 0.05,
        "use_flash_attention": False,
    },
    "analysis": {
        "confidence_threshold": 0.2,
        "pause_gap_sec": 2.0,
        "min_segment_duration": 0.3,
        "sigmoid_k": 4.0,
        "sigmoid_threshold": -0.5,
    },
    "rules": {
        "filler_patterns": [
            r"\b(嗯+|啊+|呃+|喔+|哦+)\b",
            r"\b(就是|然後|那個|這個|對吧|是不是)\b",
            r"\b(基本上|其實|反正|總之|然後呢|對不對)\b",
        ],
        "repeat_patterns": [
            r"\b(\w{2,})\b[\s，。]*\1",
        ],
        "cue_patterns": {
            "clap_cue": r"\b拍手\b",
            "cut_cue": r"\b(cut|剪這裡|重來|NG)\b",
        },
        "throat_patterns": [
            r"\b(嗯哼|嗯嗯|啊啊)\b",
        ],
        "correction_patterns": [
            r"\b(不是|不對|我是說|應該是)\b",
        ],
    },
    "export": {
        "frame_rate": None,  # null = 自動偵測
        "dv_track_index": 1,
        "result_dir": "results",
        "log_dir": "logs",
        "cache_dir": str(Path.home() / ".cache" / "smart_aroll"),
        "enable_cache": False,
    },
    "ffmpeg": {
        "path": "",
        "ffprobe_path": "",
        "prefer_nvenc": True,
        "nvenc_preset": "p4",
        "crf": 23,
    },
    "tts": {
        "enabled": False,
        "engine": "edge-tts",
        "voice_zh": "zh-TW-HsiaoChenNeural",
        "rate": "+0%",
        "pitch": "+0Hz",
    },
    "enhance": {
        "enabled": False,
        "highpass_freq": 80,
        "lowpass_freq": 12000,
        "normalize": True,
        "target_lufs": -16.0,
        "denoise": False,
    },
    "server": {
        "host": "127.0.0.1",
        "port": 8888,
    },
    "supported_formats": {
        "video": [".mp4", ".mov", ".mkv", ".avi", ".mxf", ".webm", ".flv", ".wmv"],
        "audio": [".mp3", ".wav", ".flac", ".aac", ".ogg", ".m4a", ".wma", ".opus"],
    },
}


def deep_merge(base: dict, override: dict) -> dict:
    """深度合併字典"""
    result = base.copy()
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = deep_merge(result[k], v)
        else:
            result[k] = v
    return result


@dataclass
class Config:
    """統一設定"""
    raw: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def load(cls, path: str = None) -> "Config":
        """載入設定,自動合併預設值"""
        if path is None:
            path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config.json")

        if not os.path.isfile(path):
            logger.warning(f"設定檔不存在: {path},使用預設值")
            return cls(raw=DEFAULT_CONFIG)

        try:
            with open(path, "r", encoding="utf-8") as f:
                user_config = json.load(f)
            merged = deep_merge(DEFAULT_CONFIG, user_config)
            logger.info(f"已載入設定: {path}")
            return cls(raw=merged)
        except Exception as e:
            logger.error(f"設定檔載入失敗: {e},使用預設值")
            return cls(raw=DEFAULT_CONFIG)

    def get(self, section: str, key: str = None, default: Any = None) -> Any:
        """取得設定值,可用 section.key 或 section['key']"""
        if key is None:
            return self.raw.get(section, default)
        return self.raw.get(section, {}).get(key, default)

    def section(self, name: str) -> Dict[str, Any]:
        return self.raw.get(name, {})

    @property
    def whisper(self) -> Dict[str, Any]:
        return self.section("whisper")

    @property
    def llm(self) -> Dict[str, Any]:
        return self.section("llm")

    @property
    def analysis(self) -> Dict[str, Any]:
        return self.section("analysis")

    @property
    def rules(self) -> Dict[str, Any]:
        return self.section("rules")

    @property
    def export(self) -> Dict[str, Any]:
        return self.section("export")

    @property
    def ffmpeg(self) -> Dict[str, Any]:
        return self.section("ffmpeg")

    @property
    def tts(self) -> Dict[str, Any]:
        return self.section("tts")

    @property
    def enhance(self) -> Dict[str, Any]:
        return self.section("enhance")

    @property
    def server(self) -> Dict[str, Any]:
        return self.section("server")


# 全域單例
_global_config: Optional[Config] = None


def get_config(path: str = None) -> Config:
    global _global_config
    if _global_config is None:
        _global_config = Config.load(path)
    return _global_config


def reload_config(path: str = None) -> Config:
    global _global_config
    _global_config = Config.load(path)
    return _global_config
