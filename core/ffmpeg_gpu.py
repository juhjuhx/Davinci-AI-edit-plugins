"""
統一 FFmpeg 介面
- 自動偵測 GPU 編碼器 (h264_nvenc)
- 自動 fallback 至 CPU (libx264)
- 視頻資訊查詢 (ffprobe)
- 音訊/視訊提取
- 視訊編碼 (NVENC 加速)
- 音訊增強濾鏡鏈
"""

import json
import logging
import os
import shutil
import subprocess
import sys
from typing import Any, Dict, List, Optional, Tuple

from .models import VideoInfo

logger = logging.getLogger("smart_aroll.ffmpeg")

IS_WINDOWS = sys.platform == "win32"


def _quote_arg(arg: str) -> str:
    """Windows 下的引號轉義"""
    if IS_WINDOWS and " " in arg and not (arg.startswith('"') and arg.endswith('"')):
        return f'"{arg}"'
    return arg


def _run(cmd: List[str], timeout: int = None, capture: bool = True) -> Tuple[int, str, str]:
    """執行指令,處理 cp950 編碼問題"""
    try:
        if IS_WINDOWS:
            proc = subprocess.run(
                cmd,
                capture_output=capture,
                text=False,
                timeout=timeout,
                creationflags=subprocess.CREATE_NO_WINDOW if hasattr(subprocess, "CREATE_NO_WINDOW") else 0,
            )
            try:
                stdout = proc.stdout.decode("utf-8", errors="replace") if proc.stdout else ""
            except Exception:
                stdout = proc.stdout.decode("cp950", errors="replace") if proc.stdout else ""
            try:
                stderr = proc.stderr.decode("utf-8", errors="replace") if proc.stderr else ""
            except Exception:
                stderr = proc.stderr.decode("cp950", errors="replace") if proc.stderr else ""
        else:
            proc = subprocess.run(
                cmd,
                capture_output=capture,
                text=True,
                timeout=timeout,
            )
            stdout = proc.stdout or ""
            stderr = proc.stderr or ""

        return proc.returncode, stdout, stderr
    except subprocess.TimeoutExpired:
        return -1, "", "timeout"
    except FileNotFoundError as e:
        return -1, "", str(e)
    except Exception as e:
        return -1, "", str(e)


class FFmpegRunner:
    """FFmpeg 執行器"""

    def __init__(
        self,
        ffmpeg_path: str = None,
        ffprobe_path: str = None,
        prefer_nvenc: bool = True,
        nvenc_preset: str = "p4",
        crf: int = 23,
    ):
        self.ffmpeg_path = ffmpeg_path or self._find_ffmpeg()
        self.ffprobe_path = ffprobe_path or self._find_ffprobe(self.ffmpeg_path)
        self.prefer_nvenc = prefer_nvenc
        self.nvenc_available = False
        self._nvenc_preset = nvenc_preset
        self._crf = crf

        if self.prefer_nvenc:
            self._probe_nvenc()

    def _find_ffmpeg(self) -> str:
        path = shutil.which("ffmpeg")
        if path:
            return path
        candidates = [
            r"C:\ffmpeg\bin\ffmpeg.exe",
            r"C:\Program Files\ffmpeg\bin\ffmpeg.exe",
            "/usr/bin/ffmpeg",
            "/usr/local/bin/ffmpeg",
        ]
        ffmpeg_env = os.environ.get("FFMPEG_PATH", "")
        if ffmpeg_env:
            candidates.insert(0, os.path.join(ffmpeg_env, "ffmpeg.exe"))
        for p in candidates:
            if os.path.isfile(p):
                return p
        return "ffmpeg"

    def _find_ffprobe(self, ffmpeg_path: str) -> str:
        if ffmpeg_path and ffmpeg_path != "ffmpeg":
            candidate = ffmpeg_path.replace("ffmpeg.exe", "ffprobe.exe").replace("ffmpeg", "ffprobe")
            if os.path.isfile(candidate):
                return candidate
        path = shutil.which("ffprobe")
        if path:
            return path
        candidates = [
            r"C:\ffmpeg\bin\ffprobe.exe",
            "/usr/bin/ffprobe",
        ]
        ffmpeg_env = os.environ.get("FFMPEG_PATH", "")
        if ffmpeg_env:
            candidates.insert(0, os.path.join(ffmpeg_env, "ffprobe.exe"))
        for p in candidates:
            if os.path.isfile(p):
                return p
        return "ffprobe"

    def _probe_nvenc(self):
        """偵測 h264_nvenc 是否可用"""
        try:
            rc, out, _ = _run([self.ffmpeg_path, "-hide_banner", "-encoders"])
            self.nvenc_available = "h264_nvenc" in out
            if self.nvenc_available:
                logger.info("✓ NVIDIA NVENC 編碼器可用")
            else:
                logger.info("✗ NVENC 不可用,將使用 libx264")
        except Exception as e:
            logger.warning(f"無法偵測 NVENC: {e}")
            self.nvenc_available = False

    def get_video_encoder(self) -> Tuple[str, Dict[str, str]]:
        """回傳 (encoder_name, extra_args)"""
        if self.nvenc_available and self.prefer_nvenc:
            return "h264_nvenc", {
                "preset": self._nvenc_preset,
                "rc": "vbr",
                "cq": str(self._crf),
                "b:v": "0",
            }
        return "libx264", {
            "preset": "medium",
            "crf": str(self._crf),
        }

    def get_info(self, video_path: str) -> VideoInfo:
        """取得視頻資訊"""
        if not os.path.isfile(video_path):
            raise FileNotFoundError(f"找不到檔案: {video_path}")

        cmd = [
            self.ffprobe_path,
            "-v",
            "error",
            "-print_format",
            "json",
            "-show_format",
            "-show_streams",
            video_path,
        ]
        rc, out, err = _run(cmd)
        if rc != 0:
            raise RuntimeError(f"ffprobe 失敗: {err}")

        try:
            data = json.loads(out)
        except json.JSONDecodeError as e:
            raise RuntimeError(f"ffprobe 輸出解析失敗: {e}")

        fmt = data.get("format", {})
        duration = float(fmt.get("duration", 0.0))
        file_size = int(fmt.get("size", 0))

        video_stream = None
        audio_stream = None
        for s in data.get("streams", []):
            if s.get("codec_type") == "video" and video_stream is None:
                video_stream = s
            elif s.get("codec_type") == "audio" and audio_stream is None:
                audio_stream = s

        if video_stream is None:
            raise RuntimeError("找不到視訊流")

        fps = self._parse_fps(video_stream.get("r_frame_rate", "30/1"))
        width = int(video_stream.get("width", 0))
        height = int(video_stream.get("height", 0))
        codec = video_stream.get("codec_name", "")
        audio_codec = audio_stream.get("codec_name", "") if audio_stream else ""

        return VideoInfo(
            path=video_path,
            duration=duration,
            fps=fps,
            width=width,
            height=height,
            codec=codec,
            audio_codec=audio_codec,
            file_size=file_size,
        )

    def _parse_fps(self, fps_str: str) -> float:
        """解析 '30000/1001' 格式"""
        try:
            if "/" in fps_str:
                n, d = fps_str.split("/")
                if float(d) == 0:
                    return 30.0
                return float(n) / float(d)
            return float(fps_str)
        except Exception:
            return 30.0

    def extract_audio(self, video_path: str, output_path: str, sample_rate: int = 16000, channels: int = 1) -> str:
        """提取音訊為 PCM 16k mono (Whisper 友善)"""
        cmd = [
            self.ffmpeg_path,
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            video_path,
            "-vn",
            "-acodec",
            "pcm_s16le",
            "-ar",
            str(sample_rate),
            "-ac",
            str(channels),
            output_path,
        ]
        rc, _, err = _run(cmd)
        if rc != 0:
            raise RuntimeError(f"音訊提取失敗: {err}")
        return output_path

    def encode_video(
        self,
        input_path: str,
        output_path: str,
        video_filters: str = None,
        audio_filters: str = None,
        crf: int = None,
        preset: str = None,
        copy_audio: bool = True,
    ) -> str:
        """編碼視訊 (自動選 NVENC 或 libx264)"""
        encoder, extra_args = self.get_video_encoder()

        if crf is not None:
            if "crf" in extra_args:
                extra_args["crf"] = str(crf)
            elif "cq" in extra_args:
                extra_args["cq"] = str(crf)
        if preset is not None:
            extra_args["preset"] = preset

        cmd = [
            self.ffmpeg_path,
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            input_path,
        ]
        if video_filters:
            cmd += ["-vf", video_filters]
        cmd += ["-c:v", encoder]
        for k, v in extra_args.items():
            cmd += [f"-{k}", v]
        if copy_audio:
            cmd += ["-c:a", "aac", "-b:a", "192k"]
        if audio_filters:
            cmd += ["-af", audio_filters]
        cmd.append(output_path)

        rc, _, err = _run(cmd)
        if rc != 0:
            raise RuntimeError(f"視訊編碼失敗: {err}")
        return output_path

    def concat_segments(self, segment_files: List[str], output_path: str) -> str:
        """拼接多段視訊 (使用 concat demuxer)"""
        if not segment_files:
            raise ValueError("沒有片段可拼接")

        list_file = output_path + ".list.txt"
        try:
            with open(list_file, "w", encoding="utf-8") as f:
                for seg in segment_files:
                    f.write(f"file '{seg.replace(chr(39), chr(39) + chr(92) + chr(39))}'\n")

            cmd = [
                self.ffmpeg_path,
                "-y",
                "-hide_banner",
                "-loglevel",
                "error",
                "-f",
                "concat",
                "-safe",
                "0",
                "-i",
                list_file,
                "-c",
                "copy",
                output_path,
            ]
            rc, _, err = _run(cmd)
            if rc != 0:
                raise RuntimeError(f"拼接失敗: {err}")
            return output_path
        finally:
            if os.path.isfile(list_file):
                try:
                    os.remove(list_file)
                except OSError:
                    pass

    def apply_audio_filters(self, input_path: str, output_path: str, filters: List[str]) -> str:
        """套用音訊濾鏡鏈"""
        cmd = [
            self.ffmpeg_path,
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            input_path,
            "-af",
            ",".join(filters),
            "-c:a",
            "pcm_s16le",
            output_path,
        ]
        rc, _, err = _run(cmd)
        if rc != 0:
            raise RuntimeError(f"音訊處理失敗: {err}")
        return output_path

    def burn_subtitles(self, video_path: str, srt_path: str, output_path: str, font: str = "Microsoft JhengHei") -> str:
        """燒錄字幕"""
        srt_escaped = srt_path.replace("\\", "/").replace(":", "\\:")

        video_filter = (
            f"subtitles='{srt_escaped}':"
            f"force_style='FontName={font},FontSize=24,"
            f"PrimaryColour=&H00FFFFFF,OutlineColour=&H00000000,Outline=2'"
        )

        encoder, extra_args = self.get_video_encoder()
        cmd = [
            self.ffmpeg_path,
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            video_path,
            "-vf",
            video_filter,
            "-c:v",
            encoder,
        ]
        for k, v in extra_args.items():
            cmd += [f"-{k}", v]
        cmd += ["-c:a", "copy", output_path]

        rc, _, err = _run(cmd)
        if rc != 0:
            raise RuntimeError(f"字幕燒錄失敗: {err}")
        return output_path

    def get_progress(self, video_path: str, log_callback=None) -> Dict[str, Any]:
        """估算處理時間 (用於 UI 顯示)"""
        try:
            info = self.get_info(video_path)
            return {
                "duration": info.duration,
                "fps": info.fps,
                "resolution": f"{info.width}x{info.height}",
                "file_size_mb": round(info.file_size / 1024 / 1024, 2),
            }
        except Exception as e:
            return {"error": str(e)}


# 全域單例
_runner: Optional[FFmpegRunner] = None


def get_runner(config=None) -> FFmpegRunner:
    global _runner
    if _runner is None:
        if config:
            _runner = FFmpegRunner(
                ffmpeg_path=config.ffmpeg.get("path"),
                ffprobe_path=config.ffmpeg.get("ffprobe_path"),
                prefer_nvenc=config.ffmpeg.get("prefer_nvenc", True),
                nvenc_preset=config.ffmpeg.get("nvenc_preset", "p4"),
                crf=config.ffmpeg.get("crf", 23),
            )
        else:
            _runner = FFmpegRunner()
    return _runner
