"""
AI 視頻編輯整合模組
整合 ai-video-editor 項目的功能
"""

import os
import sys
import json
import tempfile
import subprocess
from pathlib import Path
from typing import List, Tuple, Optional, Callable
from dataclasses import dataclass
import logging

logger = logging.getLogger("smart_aroll")

# 添加 ai-video-editor 路徑
AI_VIDEO_EDITOR_PATH = r"D:\666\ai-video-editor"
sys.path.insert(0, AI_VIDEO_EDITOR_PATH)


@dataclass
class SilenceSegment:
    """靜音片段"""
    start: float
    end: float
    duration: float


@dataclass
class VideoSegment:
    """影片片段"""
    start: float
    end: float
    text: str = ""
    confidence: float = 1.0
    is_silence: bool = False


class AIVideoEditor:
    """AI 視頻編輯器"""
    
    def __init__(self, ffmpeg_path: str = None):
        self.ffmpeg_path = ffmpeg_path or self._find_ffmpeg()
        self._librosa = None
        self._np = None
    
    def _find_ffmpeg(self) -> str:
        """查找 ffmpeg 路徑"""
        import shutil
        path = shutil.which("ffmpeg")
        if path:
            return path
        
        # 常見路徑
        common_paths = [
            r"D:\666\ffmpeg\bin\ffmpeg.exe",
            r"C:\ffmpeg\bin\ffmpeg.exe",
            r"C:\Program Files\ffmpeg\bin\ffmpeg.exe",
        ]
        for p in common_paths:
            if os.path.isfile(p):
                return p
        
        raise FileNotFoundError("找不到 ffmpeg")
    
    def _load_librosa(self):
        """延遲載入 librosa"""
        if self._librosa is None:
            try:
                import librosa
                import numpy as np
                self._librosa = librosa
                self._np = np
            except ImportError as e:
                raise ImportError(f"無法載入 librosa: {e}\n請執行: pip install librosa")
    
    def detect_silence(
        self,
        video_path: str,
        threshold: float = 0.006,
        step: float = 0.5,
        progress_cb: Optional[Callable[[float], None]] = None
    ) -> List[SilenceSegment]:
        """偵測靜音片段"""
        self._load_librosa()
        
        logger.info(f"開始分析靜音: {video_path}")
        
        # 載入音訊
        y, sr = self._librosa.load(video_path, sr=None)
        duration = self._librosa.get_duration(y=y, sr=sr)
        
        silence_segments = []
        current_silence = None
        
        for t in self._np.arange(0, duration, step):
            chunk = y[int(t*sr):int((t+step)*sr)]
            rms = self._np.sqrt(self._np.mean(chunk ** 2))
            
            if rms <= threshold:
                if current_silence is None:
                    current_silence = t
            else:
                if current_silence is not None:
                    silence_segments.append(SilenceSegment(
                        start=round(current_silence, 1),
                        end=round(t, 1),
                        duration=round(t - current_silence, 1)
                    ))
                    current_silence = None
            
            if progress_cb:
                progress_cb(min(t / duration, 1.0))
        
        # 處理最後一個靜音片段
        if current_silence is not None:
            silence_segments.append(SilenceSegment(
                start=round(current_silence, 1),
                end=round(duration, 1),
                duration=round(duration - current_silence, 1)
            ))
        
        logger.info(f"偵測到 {len(silence_segments)} 個靜音片段")
        return silence_segments
    
    def detect_speech_segments(
        self,
        video_path: str,
        threshold: float = 0.006,
        step: float = 0.5,
        progress_cb: Optional[Callable[[float], None]] = None
    ) -> List[VideoSegment]:
        """偵測有語音的片段"""
        self._load_librosa()
        
        logger.info(f"開始分析語音片段: {video_path}")
        
        # 載入音訊
        y, sr = self._librosa.load(video_path, sr=None)
        duration = self._librosa.get_duration(y=y, sr=sr)
        
        speech_segments = []
        current_speech = None
        
        for t in self._np.arange(0, duration, step):
            chunk = y[int(t*sr):int((t+step)*sr)]
            rms = self._np.sqrt(self._np.mean(chunk ** 2))
            
            if rms > threshold:
                if current_speech is None:
                    current_speech = t
            else:
                if current_speech is not None:
                    speech_segments.append(VideoSegment(
                        start=round(current_speech, 1),
                        end=round(t, 1)
                    ))
                    current_speech = None
            
            if progress_cb:
                progress_cb(min(t / duration, 1.0))
        
        # 處理最後一個語音片段
        if current_speech is not None:
            speech_segments.append(VideoSegment(
                start=round(current_speech, 1),
                end=round(duration, 1)
            ))
        
        logger.info(f"偵測到 {len(speech_segments)} 個語音片段")
        return speech_segments
    
    def remove_silence(
        self,
        video_path: str,
        output_path: str,
        threshold: float = 0.006,
        step: float = 0.5,
        scale: str = "-2:720",
        progress_cb: Optional[Callable[[float], None]] = None
    ) -> str:
        """移除靜音並輸出影片"""
        # 偵測語音片段
        speech_segments = self.detect_speech_segments(
            video_path, threshold, step, progress_cb
        )
        
        if not speech_segments:
            raise ValueError("未偵測到語音片段")
        
        logger.info(f"開始移除靜音，保留 {len(speech_segments)} 個片段")
        
        # 使用 ffmpeg 切割和拼接
        filter_complex = []
        inputs = []
        
        for i, seg in enumerate(speech_segments):
            inputs.extend(["-ss", str(seg.start), "-to", str(seg.end), "-i", video_path])
            filter_complex.append(f"[{i}:v]scale={scale}[v{i}]")
            filter_complex.append(f"[{i}:a]anull[a{i}]")
        
        # 拼接所有片段
        concat_v = "".join(f"[v{i}]" for i in range(len(speech_segments)))
        concat_a = "".join(f"[a{i}]" for i in range(len(speech_segments)))
        filter_complex.append(f"{concat_v}{concat_a}concat=n={len(speech_segments)}:v=1:a=1[outv][outa]")
        
        cmd = [
            self.ffmpeg_path,
            *inputs,
            "-filter_complex", ";".join(filter_complex),
            "-map", "[outv]",
            "-map", "[outa]",
            "-c:v", "libx264",
            "-crf", "26",
            "-c:a", "aac",
            "-y",
            output_path
        ]
        
        logger.info(f"執行 ffmpeg: {' '.join(cmd[:10])}...")
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=600
            )
            
            if result.returncode != 0:
                logger.error(f"ffmpeg 錯誤: {result.stderr[:500]}")
                raise RuntimeError(f"ffmpeg 失敗: {result.returncode}")
            
            logger.info(f"靜音移除完成: {output_path}")
            return output_path
            
        except subprocess.TimeoutExpired:
            logger.error("ffmpeg 超時")
            raise
    
    def get_audio_energy(
        self,
        video_path: str,
        step: float = 0.5
    ) -> List[Tuple[float, float]]:
        """取得音訊能量分佈（用於視覺化）"""
        self._load_librosa()
        
        y, sr = self._librosa.load(video_path, sr=None)
        duration = self._librosa.get_duration(y=y, sr=sr)
        
        energy_data = []
        for t in self._np.arange(0, duration, step):
            chunk = y[int(t*sr):int((t+step)*sr)]
            rms = float(self._np.sqrt(self._np.mean(chunk ** 2)))
            energy_data.append((round(t, 1), round(rms, 4)))
        
        return energy_data


# 全域實例
_editor_instance = None


def get_ai_editor() -> AIVideoEditor:
    """取得 AI 視頻編輯器實例"""
    global _editor_instance
    if _editor_instance is None:
        _editor_instance = AIVideoEditor()
    return _editor_instance
