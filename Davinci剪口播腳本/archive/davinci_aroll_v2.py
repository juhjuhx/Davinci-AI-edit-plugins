"""
DaVinci Resolve 智能剪口播插件 v3.0 — GPU 加速完全本地版
修復所有已知問題，支援佇列處理、多模型、多音訊格式
"""

import argparse
import dataclasses
import hashlib
import json
import logging
import math
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from dataclasses import dataclass, field
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

# ════════════════════════════════════════════════════════
#  結構化日誌
# ════════════════════════════════════════════════════════

def setup_logging(log_dir: Optional[str] = None) -> logging.Logger:
    """設置結構化日誌"""
    logger = logging.getLogger("smart_aroll")
    logger.setLevel(logging.DEBUG)
    
    formatter = logging.Formatter(
        "[%(asctime)s][%(levelname)s][%(name)s] %(message)s",
        datefmt="%H:%M:%S"
    )
    
    # 控制台輸出
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.INFO)
    console.setFormatter(formatter)
    logger.addHandler(console)
    
    # 檔案輸出
    if log_dir:
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)
        file_handler = RotatingFileHandler(
            log_path / "smart_aroll.log",
            maxBytes=10 * 1024 * 1024,  # 10MB
            backupCount=5,
            encoding="utf-8"
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger

logger = setup_logging()

def _log(module: str, msg: str, level: str = "info") -> None:
    """輕量日誌工具"""
    log_func = getattr(logger, level, logger.info)
    log_func(f"[{module}] {msg}")


# ════════════════════════════════════════════════════════
#  DLL 路徑修復（延遲執行，非 import 時）
# ════════════════════════════════════════════════════════

_dll_fixed = False

def _fix_dll_path():
    """將 CUDA DLL 目錄加入 DLL 搜尋路徑"""
    global _dll_fixed
    if _dll_fixed:
        return
    
    import ctypes
    
    # 取得 conda 環境路徑
    conda_prefix = os.environ.get("CONDA_PREFIX", "")
    if not conda_prefix:
        # 嘗試從 Python 路徑推斷
        python_path = sys.executable
        if "conda" in python_path.lower() or "envs" in python_path.lower():
            conda_prefix = os.path.dirname(os.path.dirname(python_path))
    
    # 可能的 DLL 目錄
    dll_dirs = [
        r"D:\666\llama.cpp",
        r"D:\666\ffmpeg\bin",
    ]
    
    # 添加 conda 環境目錄
    if conda_prefix:
        dll_dirs.extend([
            os.path.join(conda_prefix, "lib", "site-packages", "llama_cpp", "lib"),
            os.path.join(conda_prefix, "Library", "bin"),
            os.path.join(conda_prefix, "DLLs"),
            os.path.join(conda_prefix, "bin"),
        ])
    
    # NVIDIA CUDA 常見安裝路徑
    cuda_versions = ["v12.6", "v12.5", "v12.4", "v12.3", "v12.2", "v12.1", "v12.0", "v11.8"]
    for ver in cuda_versions:
        dll_dirs.append(rf"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\{ver}\bin")
    
    # cuDNN 路徑
    cudnn_versions = ["v9.5", "v9.4", "v9.3", "v9.2", "v9.1", "v9.0", "v8.9", "v8.8", "v8.7"]
    for ver in cudnn_versions:
        dll_dirs.append(rf"C:\Program Files\NVIDIA\CUDNN\{ver}\bin")
    
    # NVIDIA 驅動目錄
    dll_dirs.append(r"C:\Windows\System32")
    
    # CUDA DLL 名稱
    cuda_dlls = [
        "cudart64_12.dll", "cublas64_12.dll", "cublasLt64_12.dll",
        "cudnn64_9.dll", "cudnn64_8.dll",
        "cudart64_11.dll", "cublas64_11.dll", "cublasLt64_11.dll",
        "cudnn_ops64_9.dll", "cudnn_ops64_8.dll",
        "cudnn_cnn64_9.dll", "cudnn_cnn64_8.dll",
    ]
    
    loaded_dlls = set()
    
    for d in dll_dirs:
        if not os.path.isdir(d):
            continue
        
        try:
            os.add_dll_directory(d)
        except OSError:
            pass
        
        if d not in os.environ.get("PATH", ""):
            os.environ["PATH"] = d + ";" + os.environ.get("PATH", "")
        
        # 嘗試載入 CUDA DLL
        for dll_name in cuda_dlls:
            if dll_name in loaded_dlls:
                continue
            dll_path = os.path.join(d, dll_name)
            if os.path.isfile(dll_path):
                try:
                    ctypes.CDLL(dll_path)
                    loaded_dlls.add(dll_name)
                    logger.debug(f"已載入 DLL: {dll_path}")
                except OSError:
                    pass
    
    _dll_fixed = True
    if loaded_dlls:
        logger.debug(f"DLL 路徑修復完成，已載入 {len(loaded_dlls)} 個 DLL: {', '.join(loaded_dlls)}")
    else:
        logger.warning("未找到任何 CUDA DLL，可能需要安裝 CUDA 或使用 CPU 模式")


# ════════════════════════════════════════════════════════
#  設定管理（支援設定檔）
# ════════════════════════════════════════════════════════

@dataclass
class AnalysisConfig:
    """分析設定（可序列化，線程安全）"""
    # Whisper 設定
    whisper_model_size: str = "large-v3"
    whisper_language: str = "zh"
    use_vad_filter: bool = True
    vad_threshold: float = 0.5
    vad_min_silence_ms: int = 500
    vad_speech_pad_ms: int = 200
    whisper_beam_size: int = 1  # 優化：使用 beam_size=1 提升速度
    whisper_condition_on_previous: bool = False  # 優化：禁用以提升速度
    
    # LLM 設定
    enable_llm: bool = False  # 新增：禁用 LLM
    llm_model_path: str = r"D:\666\Davinci剪口播腳本\models\Gemma-4-E2B-Uncensored-HauhauCS-Aggressive-Q2_K_P.gguf"
    llm_context_size: int = 2048
    llm_max_tokens: int = 256
    llm_batch_size: int = 8
    llm_temperature: float = 0.05
    llm_n_gpu_layers: int = -1
    llm_n_threads: Optional[int] = None
    llm_use_flash_attention: bool = True
    
    # GPU 設定
    gpu_auto_detect: bool = True
    force_backend: str = "cpu"
    
    # 分析設定
    confidence_threshold: float = 0.55
    pause_gap_sec: float = 1.2
    
    # 快取設定
    cache_dir: str = str(Path.home() / ".cache" / "davinci_aroll")
    enable_cache: bool = True
    cache_version: str = "3.0"  # 快取版本控制
    
    # DaVinci 設定
    dv_track_index: int = 1
    dv_frame_rate: Optional[float] = None
    
    # 日誌設定
    log_dir: Optional[str] = None
    
    # 支援的音訊格式
    supported_audio_formats: List[str] = field(default_factory=lambda: [
        ".mp3", ".wav", ".flac", ".aac", ".ogg", ".m4a", ".wma", ".opus"
    ])
    
    # 支援的影片格式
    supported_video_formats: List[str] = field(default_factory=lambda: [
        ".mp4", ".mov", ".mkv", ".avi", ".mxf", ".webm", ".flv", ".wmv"
    ])
    
    # 提示詞模板
    cue_patterns: Dict[str, str] = field(default_factory=lambda: {
        # 高優先級：立即標記為不可用
        "clap_cue": r"\b拍手\b",
        "cut_cue": r"\b(cut|剪這裡|重來|NG)\b",
        "hesitation": r"\b(額|呃|嗯|啊|喔|哦)\b",  # 猶豫詞
        "nervous_tic": r"\b(就是|然後|那個|這個|對吧|是不是)\b",  # 緊張口癖
        "throat_clearing": r"\b(嗯哼|嗯嗯|啊啊)\b",  # 清喉嚨
        
        # 中優先級：標記為需剪輯
        "filler": r"\b(嗯+|啊+|呃+|就是說|然後就|對對對|那個那個|所以說)\b",
        "repetition": r"\b(\w{2,})\b[\s，。]*\1",
        "filler_extended": r"\b(基本上|其實|反正|總之|然後呢|對不對)\b",  # 擴展填充詞
        "self_correction": r"\b(不是|不對|我是說|應該是)\b",  # 自我修正
        
        # 低優先級：只有超長停頓才標記
        "long_pause_marker": r"\b(停頓|停一下)\b",  # 明確停頓
    })
    
    def to_dict(self) -> Dict[str, Any]:
        """轉換為字典"""
        return dataclasses.asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AnalysisConfig":
        """從字典建立"""
        # 過濾掉不在 dataclass 中的欄位
        valid_fields = {f.name for f in dataclasses.fields(cls)}
        filtered = {k: v for k, v in data.items() if k in valid_fields}
        return cls(**filtered)
    
    def save(self, path: str):
        """儲存到檔案"""
        Path(path).write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8"
        )
    
    @classmethod
    def load(cls, path: str) -> "AnalysisConfig":
        """從檔案載入"""
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.from_dict(data)


# 全域設定（只讀取，不修改）
DEFAULT_CONFIG = AnalysisConfig()


# ════════════════════════════════════════════════════════
#  GPU 後端自動偵測（快取結果）
# ════════════════════════════════════════════════════════

@dataclass
class BackendInfo:
    name: str
    whisper_device: str
    llm_n_gpu_layers: int
    compute_type: str
    vram_gb: float = 0.0
    details: str = ""


_backend_cache: Optional[BackendInfo] = None
_backend_lock = threading.Lock()


def detect_best_backend(force_cpu: bool = False) -> BackendInfo:
    """偵測最佳後端（快取結果）"""
    global _backend_cache
    
    if _backend_cache is not None:
        return _backend_cache
    
    with _backend_lock:
        # 雙重檢查
        if _backend_cache is not None:
            return _backend_cache
        
        if force_cpu:
            cpu_cores = os.cpu_count() or 4
            _backend_cache = BackendInfo(
                name=f"CPU ({cpu_cores} 核心)",
                whisper_device="cpu",
                llm_n_gpu_layers=0,
                compute_type="int8",
                details="CPU 推理"
            )
            return _backend_cache
        
        sys_os = platform.system()
        machine = platform.machine()
        
        # Apple Silicon
        if sys_os == "Darwin" and machine in ("arm64", "aarch64"):
            _backend_cache = BackendInfo(
                name="Apple Silicon (Metal)",
                whisper_device="cpu",
                llm_n_gpu_layers=-1,
                compute_type="int8",
                vram_gb=16.0,
                details="Metal 加速"
            )
            return _backend_cache
        
        # NVIDIA GPU
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=4, check=True
            )
            line = result.stdout.strip().split("\n")[0]
            parts = line.split(",")
            name = parts[0].strip()
            vram = int(parts[1].strip()) / 1024
            
            if vram >= 8:
                compute = "float16"
            elif vram >= 4:
                compute = "float32"
            else:
                compute = "int8"
            
            _backend_cache = BackendInfo(
                name=f"NVIDIA CUDA ({name})",
                whisper_device="cuda",
                llm_n_gpu_layers=-1,
                compute_type=compute,
                vram_gb=vram,
                details=f"VRAM {vram:.1f}GB"
            )
            return _backend_cache
        except (subprocess.CalledProcessError, FileNotFoundError, OSError) as e:
            logger.debug(f"nvidia-smi 偵測失敗: {e}")
        
        # CPU fallback
        cpu_cores = os.cpu_count() or 4
        _backend_cache = BackendInfo(
            name=f"CPU ({cpu_cores} 核心)",
            whisper_device="cpu",
            llm_n_gpu_layers=0,
            compute_type="int8",
            details="CPU 推理"
        )
        return _backend_cache


# ════════════════════════════════════════════════════════
#  信心度歸一化
# ════════════════════════════════════════════════════════

def _normalize_confidence(avg_logprob: float) -> float:
    """將 avg_logprob 歸一化到 0-1 範圍（適合中文語音）"""
    # avg_logpb 範圍：-1.0 ~ 0.0
    # 使用 sigmoid 函數，調整參數使值分佈更合理
    # 目標：-0.5 → ~0.5, -0.8 → ~0.3, -0.2 → ~0.8
    k = 4.0  # 斜率（降低以獲得更平緩的曲線）
    threshold = -0.5  # 中心點
    return 1.0 / (1.0 + math.exp(-k * (avg_logprob - threshold)))


# ════════════════════════════════════════════════════════
#  核心資料結構
# ════════════════════════════════════════════════════════

@dataclass
class Segment:
    start: float
    end: float
    text: str = ""
    confidence: float = 1.0
    flags: List[str] = field(default_factory=list)
    usable: bool = True
    cut_cue: Optional[str] = None
    suggestion: str = ""
    llm_type: Optional[str] = None

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)

    def has_flag(self, *flags: str) -> bool:
        return bool(set(self.flags) & set(flags))

    def add_flag(self, flag: str, note: str = "") -> None:
        if flag not in self.flags:
            self.flags.append(flag)
        if note and not self.suggestion:
            self.suggestion = note

    def to_dict(self) -> Dict[str, Any]:
        """轉換為字典（包含所有欄位）"""
        return dataclasses.asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Segment":
        """從字典建立"""
        valid_fields = {f.name for f in dataclasses.fields(cls)}
        filtered = {k: v for k, v in data.items() if k in valid_fields}
        return cls(**filtered)

    def __repr__(self) -> str:
        status = "✅" if self.usable else "❌"
        flags_str = ",".join(self.flags) or "無"
        return f"{status} [{self.start:.2f}→{self.end:.2f}] ({self.duration:.1f}s) [{flags_str}] '{self.text[:30]}'"


# ════════════════════════════════════════════════════════
#  FFmpeg 路徑（快取）
# ════════════════════════════════════════════════════════

_ffmpeg_path: Optional[str] = None
_ffmpeg_lock = threading.Lock()


def get_ffmpeg_path() -> str:
    """取得 FFmpeg 路徑（快取結果）"""
    global _ffmpeg_path
    
    if _ffmpeg_path is not None:
        return _ffmpeg_path
    
    with _ffmpeg_lock:
        if _ffmpeg_path is not None:
            return _ffmpeg_path
        
        # 方法1: shutil.which
        path = shutil.which("ffmpeg")
        if path:
            _ffmpeg_path = path
            return _ffmpeg_path
        
        # 方法2: 常見安裝目錄
        common_paths = [
            r"D:\666\ffmpeg\bin\ffmpeg.exe",
            r"C:\ffmpeg\bin\ffmpeg.exe",
            r"C:\Program Files\ffmpeg\bin\ffmpeg.exe",
        ]
        for p in common_paths:
            if os.path.isfile(p):
                _ffmpeg_path = p
                return _ffmpeg_path
        
        # 方法3: 從 PATH 環境變數搜尋
        for path_dir in os.environ.get("PATH", "").split(";"):
            candidate = os.path.join(path_dir.strip(), "ffmpeg.exe")
            if os.path.isfile(candidate):
                _ffmpeg_path = candidate
                return _ffmpeg_path
        
        raise EnvironmentError(
            "找不到 ffmpeg！請確認 D:\\666\\ffmpeg\\bin\\ffmpeg.exe 存在"
        )


# ════════════════════════════════════════════════════════
#  音訊處理
# ════════════════════════════════════════════════════════

SUPPORTED_MEDIA_EXTENSIONS = {
    # 影片
    ".mp4", ".mov", ".mkv", ".avi", ".mxf", ".webm", ".flv", ".wmv",
    # 音訊
    ".mp3", ".wav", ".flac", ".aac", ".ogg", ".m4a", ".wma", ".opus"
}


def is_media_file(file_path: str) -> bool:
    """檢查是否為支援的媒體檔案"""
    return Path(file_path).suffix.lower() in SUPPORTED_MEDIA_EXTENSIONS


def is_audio_file(file_path: str) -> bool:
    """檢查是否為音訊檔案"""
    audio_exts = {".mp3", ".wav", ".flac", ".aac", ".ogg", ".m4a", ".wma", ".opus"}
    return Path(file_path).suffix.lower() in audio_exts


def extract_audio(video_path: str, output_dir: str) -> str:
    """提取音訊（支援影片和音訊檔案）"""
    ffmpeg_path = get_ffmpeg_path()
    
    # 如果是音訊檔案，直接轉換格式
    if is_audio_file(video_path):
        out = os.path.join(output_dir, "audio_16k_mono.wav")
        cmd = [
            ffmpeg_path, "-i", video_path,
            "-ac", "1", "-ar", "16000",
            "-acodec", "pcm_s16le",
            "-threads", "0",
            "-y", out
        ]
    else:
        out = os.path.join(output_dir, "audio_16k_mono.wav")
        cmd = [
            ffmpeg_path, "-i", video_path,
            "-vn", "-ac", "1", "-ar", "16000",
            "-acodec", "pcm_s16le",
            "-threads", "0",
            "-y", out
        ]
    
    _log("音訊", f"處理 '{Path(video_path).name}'...")
    
    try:
        result = subprocess.run(
            cmd, capture_output=True, timeout=300,
            encoding="utf-8", errors="replace"
        )
        if result.returncode != 0:
            _log("音訊", f"ffmpeg 錯誤碼: {result.returncode}", "error")
            if result.stderr:
                _log("音訊", f"錯誤: {result.stderr[:500]}", "error")
            raise RuntimeError(f"ffmpeg 失敗: {result.returncode}")
        _log("音訊", "處理完成")
    except subprocess.TimeoutExpired:
        _log("音訊", "ffmpeg 超時", "error")
        raise
    
    return out


def get_file_hash(file_path: str) -> str:
    """計算檔案 hash（使用檔案大小+修改時間+首尾 1MB）"""
    stat = Path(file_path).stat()
    h = hashlib.md5()
    h.update(f"{stat.st_size}:{stat.st_mtime_ns}".encode())
    
    with open(file_path, "rb") as f:
        # 首 1MB
        h.update(f.read(1024 * 1024))
        # 尾 1MB
        f.seek(max(0, stat.st_size - 1024 * 1024))
        h.update(f.read(1024 * 1024))
    
    return h.hexdigest()


def detect_video_fps(video_path: str) -> float:
    """偵測影片幀率"""
    ffmpeg_path = get_ffmpeg_path()
    
    # 多種方法嘗試偵測幀率
    methods = [
        # 方法1: 使用 ffprobe 偵測 r_frame_rate
        [ffmpeg_path.replace("ffmpeg", "ffprobe"), "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=r_frame_rate", "-of", "default=noprint_wrappers=1:nokey=1", video_path],
        # 方法2: 使用 ffprobe 偵測 avg_frame_rate
        [ffmpeg_path.replace("ffmpeg", "ffprobe"), "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=avg_frame_rate", "-of", "default=noprint_wrappers=1:nokey=1", video_path],
        # 方法3: 使用 ffmpeg 偵測
        [ffmpeg_path, "-i", video_path, "-vn", "-f", "null", "-"],
    ]
    
    for cmd in methods:
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                fps_str = result.stdout.strip()
                if "/" in fps_str:
                    num, den = fps_str.split("/")
                    fps = float(num) / float(den)
                    if 10 < fps < 120:  # 合理範圍
                        _log("FFmpeg", f"偵測幀率: {fps:.2f} fps (from {fps_str})")
                        return fps
                elif fps_str.replace(".", "").isdigit():
                    fps = float(fps_str)
                    if 10 < fps < 120:
                        _log("FFmpeg", f"偵測幀率: {fps:.2f} fps")
                        return fps
        except Exception:
            continue
    
    # 預設幀率
    _log("FFmpeg", "無法偵測幀率，使用預設 30.0 fps")
    return 30.0


# ════════════════════════════════════════════════════════
#  延遲載入 Whisper（單例模式）
# ════════════════════════════════════════════════════════

_whisper_instance = None
_whisper_lock = threading.Lock()


class LazyWhisper:
    def __init__(self, config: AnalysisConfig, backend: BackendInfo):
        self._config = config
        self._backend = backend
        self._model = None

    def _load(self) -> None:
        if self._model is not None:
            return
        
        with _whisper_lock:
            if self._model is not None:
                return
            
            _fix_dll_path()
            
            try:
                from faster_whisper import WhisperModel
            except ImportError as e:
                raise ImportError(f"無法載入 faster-whisper: {e}\n請確認已安裝: pip install faster-whisper")
            
            _log("Whisper", f"載入模型 '{self._config.whisper_model_size}' ...")
            t0 = time.time()
            
            try:
                self._model = WhisperModel(
                    self._config.whisper_model_size,
                    device=self._backend.whisper_device,
                    compute_type=self._backend.compute_type,
                    num_workers=2,
                    cpu_threads=min(8, os.cpu_count() or 4),
                )
            except Exception as e:
                error_msg = str(e)
                if "Could not find module" in error_msg or "DLL" in error_msg:
                    raise RuntimeError(
                        f"無法載入 Whisper 模型: {error_msg}\n"
                        f"解決方法:\n"
                        f"1. 確認 CUDA 已安裝: nvidia-smi\n"
                        f"2. 確認 faster-whisper 已安裝: pip show faster-whisper\n"
                        f"3. 嘗試使用 CPU 模式: 在 config.json 設定 gpu_auto_detect=false"
                    )
                raise
            
            _log("Whisper", f"載入完成（{time.time()-t0:.1f}s）")

    def transcribe(self, audio_path: str, progress_cb: Optional[Callable[[float], None]] = None) -> List[Segment]:
        """轉錄音訊"""
        self._load()
        
        # VAD 參數
        vad_params = None
        if self._config.use_vad_filter:
            vad_params = {
                "threshold": self._config.vad_threshold,
                "min_silence_duration_ms": self._config.vad_min_silence_ms,
                "speech_pad_ms": self._config.vad_speech_pad_ms,
            }
        
        segments_gen, info = self._model.transcribe(
            audio_path,
            language=self._config.whisper_language,
            beam_size=self._config.whisper_beam_size,
            vad_filter=self._config.use_vad_filter,
            vad_parameters=vad_params,
            condition_on_previous_text=self._config.whisper_condition_on_previous,
            word_timestamps=False,
        )
        
        segments = []
        audio_dur = getattr(info, "duration", 1.0) or 1.0
        
        for seg in segments_gen:
            avg_logprob = getattr(seg, "avg_logprob", -1.0)
            confidence = _normalize_confidence(avg_logprob)
            _log("Whisper", f"片段: avg_logprob={avg_logprob:.3f}, confidence={confidence:.3f}", "debug")
            segments.append(Segment(
                start=seg.start, end=seg.end,
                text=seg.text.strip(), confidence=confidence
            ))
            if progress_cb and audio_dur > 0:
                progress_cb(min(seg.end / audio_dur, 1.0))
        
        _log("Whisper", f"轉錄完成：{len(segments)} 片段（{audio_dur:.1f}s 音訊）")
        
        # 轉錄後卸載模型以釋放 GPU 記憶體
        self._unload()
        
        return segments

    def _unload(self):
        """卸載模型以釋放記憶體"""
        if self._model is not None:
            del self._model
            self._model = None
            if 'cuda' in self._backend.whisper_device:
                try:
                    import torch
                    torch.cuda.empty_cache()
                except ImportError:
                    pass
            _log("Whisper", "模型已卸載")


def get_whisper(config: AnalysisConfig, backend: BackendInfo) -> LazyWhisper:
    """取得 Whisper 單例"""
    global _whisper_instance
    if _whisper_instance is None:
        _whisper_instance = LazyWhisper(config, backend)
    return _whisper_instance


# ════════════════════════════════════════════════════════
#  延遲載入 LLM（單例模式）
# ════════════════════════════════════════════════════════

_llm_instance = None
_llm_lock = threading.Lock()


class LazyLLM:
    def __init__(self, config: AnalysisConfig, backend: BackendInfo):
        self._config = config
        self._backend = backend
        self._llm = None
        self._use_api = False
        
        # 先修復 DLL 路徑
        _fix_dll_path()
        
        try:
            from llama_cpp import Llama
            self._has_llama = True
        except ImportError as e:
            self._has_llama = False
            self._use_api = True
            _log("LLM", f"未偵測到 llama-cpp-python，切換為 LM Studio API 模式: {e}")
        except RuntimeError as e:
            error_msg = str(e)
            if "0xc000001d" in error_msg or "illegal instruction" in error_msg.lower():
                # GPU 不支援，嘗試 CPU 模式
                logger.warning("GPU 不支援 llama-cpp-python，嘗試 CPU 模式...")
                self._has_llama = True
                # 強制使用 CPU
                self._backend = BackendInfo(
                    name="CPU (fallback)",
                    whisper_device="cpu",
                    llm_n_gpu_layers=0,
                    compute_type="int8"
                )
            else:
                raise

    def _load(self) -> None:
        if self._use_api or self._llm is not None:
            return
        
        with _llm_lock:
            if self._llm is not None:
                return
            
            if not Path(self._config.llm_model_path).is_file():
                raise FileNotFoundError(f"找不到 LLM 模型：{self._config.llm_model_path}")
            
            _fix_dll_path()
            
            try:
                from llama_cpp import Llama
            except ImportError as e:
                raise ImportError(f"無法載入 llama-cpp-python: {e}\n請確認已安裝且 CUDA DLL 在 PATH 中")
            
            n_layers = self._config.llm_n_gpu_layers if not self._config.gpu_auto_detect else self._backend.llm_n_gpu_layers
            threads = self._config.llm_n_threads or os.cpu_count() or 4
            
            _log("LLM", f"載入模型 '{Path(self._config.llm_model_path).name}' | n_gpu_layers={n_layers}")
            t0 = time.time()
            
            try:
                self._llm = Llama(
                    model_path=self._config.llm_model_path,
                    n_gpu_layers=n_layers,
                    n_ctx=self._config.llm_context_size,
                    n_threads=threads,
                    n_batch=512,
                    use_mmap=True,
                    verbose=False,
                    flash_attn=False,  # GTX 1050 Ti 不支援 flash attention
                )
            except Exception as e:
                error_msg = str(e)
                if "Could not find module" in error_msg or "DLL" in error_msg:
                    raise RuntimeError(
                        f"無法載入 llama.dll: {error_msg}\n"
                        f"解決方法:\n"
                        f"1. 確認 CUDA 已安裝: nvidia-smi\n"
                        f"2. 確認 llama-cpp-python 已安裝: pip show llama-cpp-python\n"
                        f"3. 嘗試重新安裝: pip install llama-cpp-python --force-reinstall --no-cache-dir"
                    )
                elif "0xc000001d" in error_msg or "illegal instruction" in error_msg.lower():
                    # GTX 1050 Ti 不支援某些指令，嘗試 CPU 模式
                    logger.warning("GPU 不支援此模型，嘗試 CPU 模式...")
                    self._llm = Llama(
                        model_path=self._config.llm_model_path,
                        n_gpu_layers=0,  # 使用 CPU
                        n_ctx=self._config.llm_context_size,
                        n_threads=threads,
                        n_batch=512,
                        use_mmap=True,
                        verbose=False,
                        flash_attn=False,
                    )
                else:
                    raise
            
            _log("LLM", f"載入完成（{time.time()-t0:.1f}s）")

    def chat(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """與 LLM 對話"""
        if system_prompt is None:
            system_prompt = "你是專業影視剪輯師。分析口播逐字稿中的問題。只輸出純 JSON 陣列。"
        
        if self._use_api:
            url = "http://localhost:1234/v1/chat/completions"
            payload = {
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
                "max_tokens": self._config.llm_max_tokens,
                "temperature": self._config.llm_temperature,
            }
            try:
                import urllib.request
                data_bytes = json.dumps(payload).encode("utf-8")
                req = urllib.request.Request(
                    url, data=data_bytes,
                    headers={"Content-Type": "application/json"},
                    method="POST"
                )
                with urllib.request.urlopen(req, timeout=30) as response:
                    res = json.loads(response.read().decode("utf-8"))
                    return res["choices"][0]["message"]["content"].strip()
            except Exception as e:
                raise RuntimeError(f"無法連接到 LM Studio 本地 API ({url})。錯誤: {e}")
        else:
            self._load()
            resp = self._llm.create_chat_completion(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=self._config.llm_max_tokens,
                temperature=self._config.llm_temperature,
            )
            return resp["choices"][0]["message"]["content"].strip()

    def unload(self) -> None:
        """卸載模型"""
        if self._llm:
            del self._llm
            self._llm = None
            if 'cuda' in self._backend.whisper_device:
                try:
                    import torch
                    torch.cuda.empty_cache()
                except ImportError:
                    pass
            _log("LLM", "模型已卸載")


def get_llm(config: AnalysisConfig, backend: BackendInfo) -> LazyLLM:
    """取得 LLM 單例"""
    global _llm_instance
    if _llm_instance is None:
        _llm_instance = LazyLLM(config, backend)
    return _llm_instance


# ════════════════════════════════════════════════════════
#  分析引擎
# ════════════════════════════════════════════════════════

class ArollAnalyzer:
    _DISQUALIFY = frozenset({
        "cut_cue", "clap_cue", "hesitation", "nervous_tic", "throat_clearing",
        "filler", "repetition", "filler_extended", "self_correction",
        "cut_cue_detected", "deletion_zone", "low_quality", "low_confidence"
    })

    def __init__(self, config: Optional[AnalysisConfig] = None, progress_cb: Optional[Callable[[str, float], None]] = None):
        self._config = config or DEFAULT_CONFIG
        self._progress = progress_cb or (lambda s, p: None)
        
        # 偵測後端
        if self._config.gpu_auto_detect:
            self.backend = detect_best_backend()
        else:
            self.backend = BackendInfo(
                name=self._config.force_backend,
                whisper_device="cpu",
                llm_n_gpu_layers=0,
                compute_type="int8"
            )
        
        # 初始化模型
        self._whisper = get_whisper(self._config, self.backend)
        self._llm = get_llm(self._config, self.backend)
        
        # 編譯正則表達式
        self._cue_re = {name: re.compile(pat, re.UNICODE) for name, pat in self._config.cue_patterns.items()}
        
        # 建立快取目錄
        Path(self._config.cache_dir).mkdir(parents=True, exist_ok=True)
        
        _log("插件", f"初始化完成，後端：{self.backend.name}")

    def analyze(self, video_path: str) -> List[Segment]:
        """分析影片"""
        # 清理路徑（移除不可見字符）
        video_path = video_path.strip()
        video_path = video_path.replace('\u200b', '')  # 移除零寬空格
        video_path = video_path.replace('\u200e', '')  # 移除 LTR 標記
        video_path = video_path.replace('\u200f', '')  # 移除 RTL 標記
        video_path = video_path.replace('\u202a', '')  # 移除 LTR 嵌入
        video_path = video_path.replace('\u202b', '')  # 移除 RTL 嵌入
        video_path = video_path.replace('\u202c', '')  # 移除 POP 格式
        video_path = video_path.replace('\u202d', '')  # 移除 LTR 覆蓋
        video_path = video_path.replace('\u202e', '')  # 移除 RTL 覆蓋
        
        # 除錯日誌
        _log("插件", f"分析路徑: {repr(video_path)}")
        
        # 檢查檔案是否存在
        if not Path(video_path).is_file():
            raise FileNotFoundError(f"找不到檔案：{video_path}")
        
        # 檢查是否為支援的格式
        if not is_media_file(video_path):
            raise ValueError(f"不支援的檔案格式：{Path(video_path).suffix}")
        
        # 檢查快取
        if self._config.enable_cache:
            file_hash = get_file_hash(video_path)
            cache_file = Path(self._config.cache_dir) / f"{file_hash}_{self._config.cache_version}.json"
            if cache_file.exists():
                try:
                    _log("插件", "命中快取，跳過分析")
                    # 偵測幀率
                    fps = detect_video_fps(video_path)
                    return self._load_cache(cache_file), fps
                except (json.JSONDecodeError, TypeError, KeyError) as e:
                    logger.warning(f"快取損壞，重新分析: {e}")
                    cache_file.unlink(missing_ok=True)
        else:
            cache_file = None
        
        # 偵測幀率
        fps = detect_video_fps(video_path)
        _log("插件", f"偵測幀率: {fps:.2f} fps")
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # 音訊提取
            self._progress("音訊提取", 0.0)
            audio = extract_audio(video_path, tmpdir)
            self._progress("音訊提取", 1.0)
            
            # 語音轉錄
            self._progress("語音轉錄", 0.0)
            segs = self._whisper.transcribe(audio, progress_cb=lambda p: self._progress("語音轉錄", p))
            self._progress("語音轉錄", 1.0)
        
        # 信心度過濾
        self._progress("信心度過濾", 0.0)
        segs = self._detect_pauses(segs)
        segs = self._filter_confidence(segs)
        self._progress("信心度過濾", 1.0)
        
        # 規則引擎
        self._progress("規則引擎", 0.0)
        segs = self._apply_cue_rules(segs)
        self._progress("規則引擎", 1.0)
        
        # LLM 分析（可選）
        if self._config.enable_llm and self._llm._has_llama:
            self._progress("LLM 分析", 0.0)
            segs = self._analyze_llm(segs)
            self._progress("LLM 分析", 1.0)
        else:
            _log("插件", "使用規則引擎模式（LLM 已禁用）")
        
        # 最終決策
        segs = self._final_decision(segs)
        
        # 儲存快取
        if self._config.enable_cache and cache_file:
            try:
                self._save_cache(segs, cache_file)
            except OSError as e:
                logger.warning(f"無法儲存快取: {e}")
        
        _log("插件", f"分析完成：{len(segs)} 片段")
        return segs, fps

    def release(self):
        """釋放資源"""
        # 注意：使用單例模式後，這裡不實際卸載模型
        # 模型會在轉錄完成後自動卸載
        pass

    def _detect_pauses(self, segs: List[Segment]) -> List[Segment]:
        """偵測停頓"""
        for i in range(1, len(segs)):
            gap = segs[i].start - segs[i-1].end
            if gap > self._config.pause_gap_sec:
                segs[i].add_flag("long_pause", f"前方停頓 {gap:.1f}s")
        return segs

    def _filter_confidence(self, segs: List[Segment]) -> List[Segment]:
        """過濾低信心度片段"""
        for s in segs:
            if s.confidence < self._config.confidence_threshold:
                s.add_flag("low_confidence", f"信心度 {s.confidence:.2f}")
        return segs

    def _apply_cue_rules(self, segs: List[Segment]) -> List[Segment]:
        """應用規則引擎（分優先級）"""
        # 高優先級規則：立即標記為不可用
        high_priority = {"clap_cue", "cut_cue", "hesitation", "nervous_tic", "throat_clearing"}
        
        for i, seg in enumerate(segs):
            for name, pat in self._cue_re.items():
                m = pat.search(seg.text)
                if not m:
                    continue
                
                if name in high_priority:
                    # 高優先級：立即標記為不可用
                    seg.usable = False
                    seg.add_flag(name, f"「{m.group()}」")
                    if name in ("clap_cue", "cut_cue"):
                        seg.cut_cue = name
                        self._mark_deletion_zone(segs, i)
                elif name in ("filler", "repetition", "filler_extended", "self_correction"):
                    # 中優先級：標記為需剪輯
                    seg.add_flag(name, f"「{m.group()}」")
                else:
                    # 低優先級：只記錄
                    seg.add_flag(name, f"「{m.group()}」")
                break  # 每個片段只匹配一個規則
        return segs

    def _mark_deletion_zone(self, segs: List[Segment], cue_idx: int):
        """標記刪除區域"""
        j = cue_idx - 1
        while j >= 0 and segs[j].cut_cue is None:
            segs[j].usable = False
            segs[j].add_flag("deletion_zone", "因暗號觸發")
            j -= 1

    def _analyze_llm(self, segs: List[Segment]) -> List[Segment]:
        """LLM 分析（只分析有問題的片段）"""
        # 預過濾：只分析有 flag 或低信心度的片段
        problem_indices = [i for i, s in enumerate(segs) if s.flags or s.confidence < 0.7]
        
        if not problem_indices:
            _log("LLM", "無問題片段，跳過 LLM 分析")
            return segs
        
        # 合併相鄰片段
        merged_indices = self._merge_adjacent_indices(problem_indices, max_gap=2)
        
        n_batch = self._config.llm_batch_size
        n_total_batches = (len(merged_indices) + n_batch - 1) // n_batch
        
        for batch_idx, i in enumerate(range(0, len(merged_indices), n_batch)):
            batch_indices = merged_indices[i:i+n_batch]
            batch = [segs[idx] for idx in batch_indices]
            
            context = "\n".join(f"[{s.start:.2f}s~{s.end:.2f}s] {s.text}" for s in batch)
            prompt = (
                "分析以下口播逐字稿，找出需要剪輯的問題片段。\n"
                "問題類型（type 只能是以下之一）：\n"
                "  filler       = 口癖/填充詞\n"
                "  pause_restart= 卡詞後重啟語句\n"
                "  repetition   = 重複表達同一意思\n"
                "  low_quality  = 語意不通或廢話過多\n"
                "輸出：純 JSON 陣列，每項含 type/start/end/description。無問題時輸出 []\n"
                "逐字稿：\n" + context
            )
            
            try:
                raw = self._llm.chat(prompt)
                raw = re.sub(r"```(?:json)?|```", "", raw).strip()
                match = re.search(r"\[.*\]", raw, re.DOTALL)
                issues = json.loads(match.group(0)) if match else []
                self._apply_llm_issues(batch, issues)
                _log("LLM", f"批次 {batch_idx+1}/{n_total_batches}：{len(issues)} 個問題")
            except json.JSONDecodeError as e:
                logger.warning(f"LLM JSON 解析錯誤: {e}")
            except Exception as e:
                logger.warning(f"LLM 批次 {batch_idx+1} 錯誤：{e}")
            
            self._progress("LLM 分析", (batch_idx + 1) / n_total_batches)
        
        return segs

    def _merge_adjacent_indices(self, indices: List[int], max_gap: int = 2) -> List[int]:
        """合併相鄰索引"""
        if not indices:
            return []
        
        merged = [indices[0]]
        for idx in indices[1:]:
            if idx - merged[-1] <= max_gap:
                merged[-1] = idx  # 更新為最後一個
            else:
                merged.append(idx)
        return merged

    def _apply_llm_issues(self, batch: List[Segment], issues: List[Dict]):
        """應用 LLM 分析結果"""
        type_map = {
            "filler": "filler",
            "pause_restart": "long_pause",
            "repetition": "repetition",
            "low_quality": "low_quality"
        }
        for issue in issues:
            flag = type_map.get(issue.get("type", ""))
            if not flag:
                continue
            s_start = float(issue.get("start", 0))
            s_end = float(issue.get("end", 0))
            desc = issue.get("description", "")
            for seg in batch:
                if seg.start <= s_end and seg.end >= s_start:
                    seg.add_flag(flag, desc)

    def _final_decision(self, segs: List[Segment]) -> List[Segment]:
        """最終決策"""
        for seg in segs:
            if seg.usable and self._DISQUALIFY & set(seg.flags):
                seg.usable = False
        return segs

    def _save_cache(self, segs: List[Segment], path: Path):
        """儲存快取（原子寫入）"""
        data = [s.to_dict() for s in segs]
        tmp_path = path.with_suffix(".tmp")
        try:
            tmp_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
            tmp_path.replace(path)
        except OSError:
            tmp_path.unlink(missing_ok=True)
            raise

    def _load_cache(self, path: Path) -> List[Segment]:
        """載入快取"""
        data = json.loads(path.read_text(encoding="utf-8"))
        return [Segment.from_dict(d) for d in data]


# ════════════════════════════════════════════════════════
#  匯出工具
# ════════════════════════════════════════════════════════

def export_json(segs: List[Segment], path: str):
    """匯出 JSON（包含所有欄位）"""
    data = [s.to_dict() for s in segs]
    Path(path).write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def export_edl(segs: List[Segment], path: str, fps: float = 24.0, source_name: str = "AX"):
    """匯出 EDL（包含所有片段，正確的時間碼）"""
    _log("EDL", f"匯出 EDL: fps={fps}, 片段數={len(segs)}")
    
    def timecode(sec: float) -> str:
        """計算時間碼（自適應幀率）"""
        total_frames = int(round(sec * fps))
        h = total_frames // int(3600 * fps)
        m = (total_frames % int(3600 * fps)) // int(60 * fps)
        s = (total_frames % int(60 * fps)) // int(fps)
        f = total_frames % int(fps)
        return f"{h:02d}:{m:02d}:{s:02d}:{f:02d}"
    
    lines = [
        "TITLE: Smart A-Roll Edit",
        "FCM: NON-DROP FRAME",
        ""
    ]
    
    # 計算累積時間碼（所有片段都累積）
    record_tc = 0.0
    
    for idx, seg in enumerate(segs, 1):
        # 標記類型
        if seg.usable:
            comment = f"USABLE - {seg.text[:60]}"
        else:
            flag_str = ", ".join(seg.flags[:3]) if seg.flags else "unknown"
            comment = f"CUT [{flag_str}] - {seg.text[:60]}"
        
        # 來源時間碼（原始影片）
        src_in = seg.start
        src_out = seg.end
        
        # 目標時間碼（累積時間）
        rec_in = record_tc
        rec_out = record_tc + seg.duration
        
        lines.append(f"{idx:03d}  {source_name}       AA/V  C")
        lines.append(f"  {timecode(src_in)} {timecode(src_out)} {timecode(rec_in)} {timecode(rec_out)}")
        lines.append(f"* COMMENT: {comment}")
        lines.append("")
        
        _log("EDL", f"片段 {idx}: src={timecode(src_in)}-{timecode(src_out)}, rec={timecode(rec_in)}-{timecode(rec_out)}", "debug")
        
        # 所有片段都累積時間碼
        record_tc = rec_out
    
    # 原子寫入
    tmp_path = Path(path).with_suffix(".tmp")
    try:
        tmp_path.write_text("\n".join(lines), encoding="utf-8")
        tmp_path.replace(Path(path))
    except OSError:
        tmp_path.unlink(missing_ok=True)
        raise


# ════════════════════════════════════════════════════════
#  DaVinci Resolve 整合
# ════════════════════════════════════════════════════════

class ResolveIntegration:
    """DaVinci Resolve 整合"""
    
    _CLIP_COLOR = {"unusable": "Red", "usable": "Green"}
    _MARKER_COLORS = {
        "cut_cue_detected": "Red",
        "deletion_zone": "Orange",
        "filler": "Orange",
        "long_pause": "Yellow",
        "repetition": "Yellow",
        "low_quality": "Blue",
        "low_confidence": "Purple"
    }
    
    @classmethod
    def apply(cls, segs: List[Segment], project, track_index: int = 1) -> int:
        """套用分析結果到 DaVinci Resolve"""
        timeline = project.GetCurrentTimeline()
        if not timeline:
            return 0
        
        fps_str = timeline.GetSetting("timelineFrameRate")
        fps = float(fps_str) if fps_str else 24.0
        
        clips = timeline.GetItemListInTrack("video", track_index)
        if not clips:
            return 0
        
        applied = 0
        for seg in segs:
            seg_sf = int(seg.start * fps)
            seg_ef = int(seg.end * fps)
            
            for clip in clips:
                if clip.GetStart() > seg_ef or clip.GetEnd() < seg_sf:
                    continue
                
                color_key = "unusable" if not seg.usable else "usable"
                clip.SetClipColor(cls._CLIP_COLOR[color_key])
                
                if not seg.usable:
                    marker_color = next(
                        (cls._MARKER_COLORS[f] for f in seg.flags if f in cls._MARKER_COLORS),
                        "Yellow"
                    )
                    note = " | ".join(filter(None, [
                        ", ".join(seg.flags[:3]),
                        seg.suggestion,
                        seg.text[:60]
                    ]))
                    clip.AddMarker(clip.GetStart(), marker_color, "AI 建議刪除", note, max(1, seg_ef - seg_sf))
                
                applied += 1
        
        return applied


# ════════════════════════════════════════════════════════
#  佇列管理器
# ════════════════════════════════════════════════════════

class AnalysisQueue:
    """分析佇列管理器"""
    
    def __init__(self, config: Optional[AnalysisConfig] = None):
        self._config = config or DEFAULT_CONFIG
        self._queue: List[Dict[str, Any]] = []
        self._lock = threading.Lock()
        self._current_task: Optional[str] = None
        self._analyzer: Optional[ArollAnalyzer] = None
    
    def add_task(self, video_path: str, config: Optional[AnalysisConfig] = None) -> str:
        """添加任務到佇列"""
        task_id = str(uuid.uuid4())
        task_config = config or self._config
        
        with self._lock:
            self._queue.append({
                "task_id": task_id,
                "video_path": video_path,
                "config": task_config,
                "status": "pending",
                "progress": 0,
                "message": "等待中",
                "result": None,
                "error": None
            })
        
        return task_id
    
    def get_task(self, task_id: str) -> Optional[Dict[str, Any]]:
        """取得任務狀態"""
        with self._lock:
            for task in self._queue:
                if task["task_id"] == task_id:
                    return task.copy()
        return None
    
    def get_all_tasks(self) -> List[Dict[str, Any]]:
        """取得所有任務"""
        with self._lock:
            return [t.copy() for t in self._queue]
    
    def process_next(self) -> Optional[Dict[str, Any]]:
        """處理下一個任務"""
        with self._lock:
            # 找到下一個待處理的任務
            for task in self._queue:
                if task["status"] == "pending":
                    task["status"] = "processing"
                    self._current_task = task["task_id"]
                    break
            else:
                return None
        
        # 處理任務
        try:
            def progress_cb(stage: str, pct: float):
                with self._lock:
                    for task in self._queue:
                        if task["task_id"] == self._current_task:
                            task["progress"] = int(pct * 100)
                            task["message"] = f"{stage} ({int(pct*100)}%)"
                            break
            
            analyzer = ArollAnalyzer(config=task["config"], progress_cb=progress_cb)
            segments, fps = analyzer.analyze(task["video_path"])
            
            with self._lock:
                for t in self._queue:
                    if t["task_id"] == self._current_task:
                        t["status"] = "completed"
                        t["result"] = [s.to_dict() for s in segments]
                        t["fps"] = fps
                        t["progress"] = 100
                        t["message"] = "分析完成"
                        break
            
            return task
        
        except Exception as e:
            with self._lock:
                for t in self._queue:
                    if t["task_id"] == self._current_task:
                        t["status"] = "failed"
                        t["error"] = str(e)
                        t["message"] = f"失敗: {e}"
                        break
            
            return task
        
        finally:
            self._current_task = None
    
    def clear_completed(self):
        """清除已完成的任務"""
        with self._lock:
            self._queue = [t for t in self._queue if t["status"] not in ("completed", "failed")]


# ════════════════════════════════════════════════════════
#  CLI 執行入口
# ════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="智能剪口播分析引擎 CLI")
    parser.add_argument("--video", required=True, help="影片/音訊路徑")
    parser.add_argument("--output-dir", required=True, help="輸出目錄")
    parser.add_argument("--config", help="設定檔路徑 (JSON)")
    parser.add_argument("--model", help="LLM GGUF 模型路徑")
    parser.add_argument("--whisper", help="Whisper 模型大小")
    parser.add_argument("--lang", help="語言")
    parser.add_argument("--pause-gap", type=float, help="停頓閾值")
    parser.add_argument("--confidence", type=float, help="信心度門檻")
    parser.add_argument("--batch-size", type=int, help="LLM 批次大小")
    parser.add_argument("--no-vad", action="store_true", help="停用 VAD")
    parser.add_argument("--no-cache", action="store_true", help="停用快取")
    parser.add_argument("--edl", action="store_true", help="輸出 EDL")
    parser.add_argument("--json", action="store_true", help="輸出 JSON")
    parser.add_argument("--fps", type=float, default=24.0, help="幀率")
    args = parser.parse_args()
    
    # 載入設定
    if args.config:
        config = AnalysisConfig.load(args.config)
    else:
        config = AnalysisConfig()
    
    # 覆蓋設定
    if args.model:
        config.llm_model_path = args.model
    if args.whisper:
        config.whisper_model_size = args.whisper
    if args.lang:
        config.whisper_language = args.lang
    if args.pause_gap is not None:
        config.pause_gap_sec = args.pause_gap
    if args.confidence is not None:
        config.confidence_threshold = args.confidence
    if args.batch_size is not None:
        config.llm_batch_size = args.batch_size
    if args.no_vad:
        config.use_vad_filter = False
    if args.no_cache:
        config.enable_cache = False
    
    # 設定日誌
    if config.log_dir:
        setup_logging(config.log_dir)
    
    video_path = args.video
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    base = output_dir / Path(video_path).stem
    
    def progress_cb(stage: str, pct: float) -> None:
        print(f"\r[{stage}] {int(pct*100)}%", end="", flush=True)
    
    _log("CLI", f"開始分析：{Path(video_path).name}")
    t_start = time.time()
    
    try:
        analyzer = ArollAnalyzer(config=config, progress_cb=progress_cb)
        segments, fps = analyzer.analyze(video_path)
        
        # 匯出 JSON
        if args.json:
            json_path = str(base) + "_segments.json"
            export_json(segments, json_path)
            _log("CLI", f"JSON 輸出：{json_path}")
        
        # 匯出 EDL
        edl_path = None
        if args.edl:
            edl_path = str(base) + "_edit.edl"
            export_edl(segments, edl_path, fps=fps)
            _log("CLI", f"EDL 輸出：{edl_path}")
        
        usable = sum(1 for s in segments if s.usable)
        unusable = len(segments) - usable
        elapsed = time.time() - t_start
        
        result = {
            "status": "completed",
            "elapsed": round(elapsed, 1),
            "video_path": video_path,
            "total_segments": len(segments),
            "usable": usable,
            "unusable": unusable,
            "segments": [s.to_dict() for s in segments],
        }
        
        (output_dir / "result.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2),
            encoding="utf-8"
        )
        
        print()  # 換行
        _log("CLI", f"分析完成：{len(segments)} 片段（{usable} 可用，{unusable} 需剪輯）")
        _log("CLI", f"耗時：{elapsed:.1f} 秒")
        
        print(json.dumps({
            "status": "completed",
            "usable": usable,
            "unusable": unusable,
            "elapsed": round(elapsed, 1)
        }))
        
    except Exception as e:
        error_result = {"status": "failed", "error": str(e)}
        (output_dir / "result.json").write_text(
            json.dumps(error_result, ensure_ascii=False),
            encoding="utf-8"
        )
        print(json.dumps(error_result), file=sys.stderr)
        sys.exit(1)
