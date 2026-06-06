"""
綜合視頻編輯工具包
整合多個開源項目的功能
"""

import os
import sys
import json
import tempfile
import subprocess
from pathlib import Path
from typing import List, Tuple, Optional, Callable, Dict, Any
from dataclasses import dataclass, field
import logging

logger = logging.getLogger("smart_aroll")


@dataclass
class VideoInfo:
    """視頻資訊"""
    path: str
    duration: float
    fps: float
    width: int
    height: int
    codec: str
    audio_codec: str
    file_size: int


@dataclass
class EditOperation:
    """編輯操作"""
    type: str  # cut, concat, speed, fade, text, image, etc.
    params: Dict[str, Any]
    start_time: float = 0
    end_time: float = 0


@dataclass
class SubtitleEntry:
    """字幕條目"""
    index: int
    start_time: float
    end_time: float
    text: str
    position: Tuple[int, int] = None


class ComprehensiveVideoEditor:
    """綜合視頻編輯器"""
    
    def __init__(self, ffmpeg_path: str = None):
        self.ffmpeg_path = ffmpeg_path or self._find_ffmpeg()
        self._librosa = None
        self._whisper = None
        self._moviepy = None
    
    def _find_ffmpeg(self) -> str:
        """查找 ffmpeg 路徑"""
        import shutil
        path = shutil.which("ffmpeg")
        if path:
            return path
        
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
                raise ImportError(f"無法載入 librosa: {e}")
    
    def get_video_info(self, video_path: str) -> VideoInfo:
        """取得視頻資訊"""
        cmd = [
            self.ffmpeg_path,
            "-i", video_path,
            "-hide_banner",
            "-v", "error",
            "-show_entries", "stream=codec_name,width,height,r_frame_rate,codec_type",
            "-show_entries", "format=duration,size",
            "-of", "json"
        ]
        
        try:
            result = subprocess.run(
                [self.ffmpeg_path, "-i", video_path, "-hide_banner"],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            # 解析輸出
            output = result.stderr
            
            # 提取時長
            duration = 0.0
            if "Duration:" in output:
                duration_str = output.split("Duration:")[1].split(",")[0].strip()
                h, m, s = duration_str.split(":")
                duration = float(h) * 3600 + float(m) * 60 + float(s)
            
            # 提取 FPS
            fps = 30.0
            if "fps" in output:
                fps_str = output.split("fps")[0].split()[-1]
                try:
                    fps = float(fps_str)
                except:
                    pass
            
            # 提取解析度
            width, height = 1920, 1080
            if "x" in output:
                for part in output.split():
                    if "x" in part and part.replace("x", "").replace(",", "").isdigit():
                        try:
                            w, h = part.split("x")
                            width, height = int(w), int(h.replace(",", ""))
                            break
                        except:
                            pass
            
            return VideoInfo(
                path=video_path,
                duration=duration,
                fps=fps,
                width=width,
                height=height,
                codec="h264",
                audio_codec="aac",
                file_size=os.path.getsize(video_path)
            )
        except Exception as e:
            logger.error(f"無法取得視頻資訊: {e}")
            raise
    
    def cut_video(
        self,
        video_path: str,
        output_path: str,
        start_time: float,
        end_time: float,
        codec: str = "copy"
    ) -> str:
        """切割視頻"""
        cmd = [
            self.ffmpeg_path,
            "-i", video_path,
            "-ss", str(start_time),
            "-to", str(end_time),
            "-c", codec,
            "-y",
            output_path
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            if result.returncode != 0:
                raise RuntimeError(f"切割失敗: {result.stderr[:200]}")
            return output_path
        except Exception as e:
            logger.error(f"切割視頻失敗: {e}")
            raise
    
    def concat_videos(
        self,
        video_paths: List[str],
        output_path: str,
        scale: str = None
    ) -> str:
        """拼接多個視頻"""
        # 建立臨時檔案列表
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            for path in video_paths:
                f.write(f"file '{path}'\n")
            list_file = f.name
        
        try:
            cmd = [
                self.ffmpeg_path,
                "-f", "concat",
                "-safe", "0",
                "-i", list_file,
            ]
            
            if scale:
                cmd.extend(["-vf", f"scale={scale}"])
            
            cmd.extend(["-c", "copy", "-y", output_path])
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
            if result.returncode != 0:
                raise RuntimeError(f"拼接失敗: {result.stderr[:200]}")
            return output_path
        finally:
            os.unlink(list_file)
    
    def change_speed(
        self,
        video_path: str,
        output_path: str,
        speed_factor: float
    ) -> str:
        """改變視頻速度"""
        # 計算 PTS
        pts = 1.0 / speed_factor
        
        cmd = [
            self.ffmpeg_path,
            "-i", video_path,
            "-filter:v", f"setpts={pts}*PTS",
            "-filter:a", f"atempo={speed_factor}",
            "-y",
            output_path
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            if result.returncode != 0:
                raise RuntimeError(f"速度調整失敗: {result.stderr[:200]}")
            return output_path
        except Exception as e:
            logger.error(f"改變速度失敗: {e}")
            raise
    
    def add_fade(
        self,
        video_path: str,
        output_path: str,
        fade_in: float = 0,
        fade_out: float = 0
    ) -> str:
        """添加淡入淡出效果"""
        filters = []
        
        if fade_in > 0:
            filters.append(f"fade=t=in:st=0:d={fade_in}")
        
        if fade_out:
            # 需要先獲取視頻時長
            info = self.get_video_info(video_path)
            fade_start = info.duration - fade_out
            filters.append(f"fade=t=out:st={fade_start}:d={fade_out}")
        
        if not filters:
            # 沒有效果，直接複製
            import shutil
            shutil.copy2(video_path, output_path)
            return output_path
        
        cmd = [
            self.ffmpeg_path,
            "-i", video_path,
            "-vf", ",".join(filters),
            "-y",
            output_path
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            if result.returncode != 0:
                raise RuntimeError(f"淡入淡出失敗: {result.stderr[:200]}")
            return output_path
        except Exception as e:
            logger.error(f"添加淡入淡出失敗: {e}")
            raise
    
    def add_text_overlay(
        self,
        video_path: str,
        output_path: str,
        text: str,
        x: int = 10,
        y: int = 10,
        fontsize: int = 24,
        fontcolor: str = "white",
        fontfile: str = None,
        start_time: float = 0,
        end_time: float = None
    ) -> str:
        """添加文字覆蓋"""
        # 建立 drawtext 濾鏡
        drawtext = f"drawtext=text='{text}':x={x}:y={y}:fontsize={fontsize}:fontcolor={fontcolor}"
        
        if fontfile:
            drawtext += f":fontfile='{fontfile}'"
        
        if start_time > 0 or end_time:
            if end_time:
                drawtext += f":enable='between(t,{start_time},{end_time})'"
            else:
                drawtext += f":enable='gte(t,{start_time})'"
        
        cmd = [
            self.ffmpeg_path,
            "-i", video_path,
            "-vf", drawtext,
            "-y",
            output_path
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            if result.returncode != 0:
                raise RuntimeError(f"添加文字失敗: {result.stderr[:200]}")
            return output_path
        except Exception as e:
            logger.error(f"添加文字覆蓋失敗: {e}")
            raise
    
    def add_image_overlay(
        self,
        video_path: str,
        output_path: str,
        image_path: str,
        x: int = 0,
        y: int = 0,
        start_time: float = 0,
        end_time: float = None
    ) -> str:
        """添加圖片覆蓋"""
        # 建立 overlay 濾鏡
        overlay = f"overlay={x}:{y}"
        
        if start_time > 0 or end_time:
            if end_time:
                overlay += f":enable='between(t,{start_time},{end_time})'"
            else:
                overlay += f":enable='gte(t,{start_time})'"
        
        cmd = [
            self.ffmpeg_path,
            "-i", video_path,
            "-i", image_path,
            "-filter_complex", f"[0:v][1:v]{overlay}",
            "-y",
            output_path
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            if result.returncode != 0:
                raise RuntimeError(f"添加圖片失敗: {result.stderr[:200]}")
            return output_path
        except Exception as e:
            logger.error(f"添加圖片覆蓋失敗: {e}")
            raise
    
    def extract_audio(
        self,
        video_path: str,
        output_path: str,
        format: str = "wav",
        sample_rate: int = 16000
    ) -> str:
        """提取音訊"""
        cmd = [
            self.ffmpeg_path,
            "-i", video_path,
            "-vn",
            "-ac", "1",
            "-ar", str(sample_rate),
            "-f", format,
            "-y",
            output_path
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            if result.returncode != 0:
                raise RuntimeError(f"提取音訊失敗: {result.stderr[:200]}")
            return output_path
        except Exception as e:
            logger.error(f"提取音訊失敗: {e}")
            raise
    
    def detect_silence_librosa(
        self,
        video_path: str,
        threshold: float = 0.006,
        step: float = 0.5
    ) -> List[Tuple[float, float]]:
        """使用 librosa 偵測靜音"""
        self._load_librosa()
        
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
                    silence_segments.append((round(current_silence, 1), round(t, 1)))
                    current_silence = None
        
        if current_silence is not None:
            silence_segments.append((round(current_silence, 1), round(duration, 1)))
        
        return silence_segments
    
    def detect_speech_segments_librosa(
        self,
        video_path: str,
        threshold: float = 0.006,
        step: float = 0.5
    ) -> List[Tuple[float, float]]:
        """使用 librosa 偵測語音片段"""
        self._load_librosa()
        
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
                    speech_segments.append((round(current_speech, 1), round(t, 1)))
                    current_speech = None
        
        if current_speech is not None:
            speech_segments.append((round(current_speech, 1), round(duration, 1)))
        
        return speech_segments
    
    def remove_silence(
        self,
        video_path: str,
        output_path: str,
        threshold: float = 0.006,
        step: float = 0.5,
        padding: float = 0.1
    ) -> str:
        """移除靜音片段"""
        # 偵測語音片段
        speech_segments = self.detect_speech_segments_librosa(
            video_path, threshold, step
        )
        
        if not speech_segments:
            raise ValueError("未偵測到語音片段")
        
        logger.info(f"偵測到 {len(speech_segments)} 個語音片段")
        
        # 切割並拼接語音片段
        temp_files = []
        try:
            for i, (start, end) in enumerate(speech_segments):
                # 添加 padding
                start = max(0, start - padding)
                end = end + padding
                
                temp_file = tempfile.mktemp(suffix='.mp4')
                self.cut_video(video_path, temp_file, start, end)
                temp_files.append(temp_file)
            
            # 拼接所有片段
            self.concat_videos(temp_files, output_path)
            
            logger.info(f"靜音移除完成: {output_path}")
            return output_path
        finally:
            # 清理臨時檔案
            for f in temp_files:
                try:
                    os.unlink(f)
                except:
                    pass
    
    def generate_srt(
        self,
        video_path: str,
        output_path: str,
        language: str = "zh",
        model_size: str = "base"
    ) -> str:
        """生成 SRT 字幕"""
        try:
            import whisper
        except ImportError:
            raise ImportError("無法載入 whisper，請執行: pip install openai-whisper")
        
        logger.info(f"開始生成字幕: {video_path}")
        
        # 載入模型
        model = whisper.load_model(model_size)
        
        # 轉錄
        result = model.transcribe(video_path, language=language, verbose=False)
        
        # 生成 SRT
        srt_lines = []
        for i, segment in enumerate(result["segments"], 1):
            start = segment["start"]
            end = segment["end"]
            text = segment["text"].strip()
            
            # 轉換為 SRT 時間格式
            start_str = self._seconds_to_srt_time(start)
            end_str = self._seconds_to_srt_time(end)
            
            srt_lines.append(f"{i}")
            srt_lines.append(f"{start_str} --> {end_str}")
            srt_lines.append(text)
            srt_lines.append("")
        
        # 寫入檔案
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(srt_lines))
        
        logger.info(f"字幕生成完成: {output_path}")
        return output_path
    
    def _seconds_to_srt_time(self, seconds: float) -> str:
        """將秒數轉換為 SRT 時間格式"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millis = int((seconds % 1) * 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"
    
    def burn_subtitles(
        self,
        video_path: str,
        srt_path: str,
        output_path: str,
        font: str = "Arial",
        fontsize: int = 24,
        fontcolor: str = "white"
    ) -> str:
        """燒錄字幕到視頻"""
        cmd = [
            self.ffmpeg_path,
            "-i", video_path,
            "-vf", f"subtitles={srt_path}:force_style='FontSize={fontsize},FontName={font},PrimaryColour=&H00FFFFFF'",
            "-y",
            output_path
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
            if result.returncode != 0:
                raise RuntimeError(f"燒錄字幕失敗: {result.stderr[:200]}")
            return output_path
        except Exception as e:
            logger.error(f"燒錄字幕失敗: {e}")
            raise
    
    def extract_frames(
        self,
        video_path: str,
        output_dir: str,
        fps: float = 1
    ) -> List[str]:
        """提取視頻幀"""
        os.makedirs(output_dir, exist_ok=True)
        
        cmd = [
            self.ffmpeg_path,
            "-i", video_path,
            "-vf", f"fps={fps}",
            "-q:v", "2",
            os.path.join(output_dir, "frame_%04d.jpg")
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            if result.returncode != 0:
                raise RuntimeError(f"提取幀失敗: {result.stderr[:200]}")
            
            # 取得生成的檔案列表
            frames = sorted([
                os.path.join(output_dir, f)
                for f in os.listdir(output_dir)
                if f.endswith('.jpg')
            ])
            
            return frames
        except Exception as e:
            logger.error(f"提取視頻幀失敗: {e}")
            raise
    
    def create_video_from_frames(
        self,
        frames_dir: str,
        output_path: str,
        fps: float = 24
    ) -> str:
        """從幀創建視頻"""
        cmd = [
            self.ffmpeg_path,
            "-framerate", str(fps),
            "-i", os.path.join(frames_dir, "frame_%04d.jpg"),
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-y",
            output_path
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            if result.returncode != 0:
                raise RuntimeError(f"創建視頻失敗: {result.stderr[:200]}")
            return output_path
        except Exception as e:
            logger.error(f"從幀創建視頻失敗: {e}")
            raise
    
    def compress_video(
        self,
        video_path: str,
        output_path: str,
        crf: int = 23,
        preset: str = "medium",
        scale: str = None
    ) -> str:
        """壓縮視頻"""
        cmd = [
            self.ffmpeg_path,
            "-i", video_path,
            "-c:v", "libx264",
            "-crf", str(crf),
            "-preset", preset,
            "-c:a", "aac",
            "-b:a", "128k",
        ]
        
        if scale:
            cmd.extend(["-vf", f"scale={scale}"])
        
        cmd.extend(["-y", output_path])
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
            if result.returncode != 0:
                raise RuntimeError(f"壓縮失敗: {result.stderr[:200]}")
            return output_path
        except Exception as e:
            logger.error(f"壓縮視頻失敗: {e}")
            raise
    
    def convert_format(
        self,
        video_path: str,
        output_path: str,
        codec: str = None
    ) -> str:
        """轉換格式"""
        cmd = [self.ffmpeg_path, "-i", video_path]
        
        if codec:
            cmd.extend(["-c", codec])
        else:
            cmd.extend(["-c", "copy"])
        
        cmd.extend(["-y", output_path])
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            if result.returncode != 0:
                raise RuntimeError(f"格式轉換失敗: {result.stderr[:200]}")
            return output_path
        except Exception as e:
            logger.error(f"轉換格式失敗: {e}")
            raise
    
    def merge_audio_video(
        self,
        video_path: str,
        audio_path: str,
        output_path: str,
        replace: bool = True
    ) -> str:
        """合併音訊和視頻"""
        cmd = [
            self.ffmpeg_path,
            "-i", video_path,
            "-i", audio_path,
        ]
        
        if replace:
            cmd.extend(["-c:v", "copy", "-map", "0:v:0", "-map", "1:a:0"])
        else:
            cmd.extend(["-filter_complex", "[0:a][1:a]amix=inputs=2:duration=longest"])
        
        cmd.extend(["-shortest", "-y", output_path])
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            if result.returncode != 0:
                raise RuntimeError(f"合併失敗: {result.stderr[:200]}")
            return output_path
        except Exception as e:
            logger.error(f"合併音訊視頻失敗: {e}")
            raise


# 全域實例
_editor_instance = None


def get_comprehensive_editor() -> ComprehensiveVideoEditor:
    """取得綜合視頻編輯器實例"""
    global _editor_instance
    if _editor_instance is None:
        _editor_instance = ComprehensiveVideoEditor()
    return _editor_instance
