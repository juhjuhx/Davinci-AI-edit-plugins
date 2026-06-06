"""
語音合成修補 (TTS)
- 使用 edge-tts 將被剪掉的填充詞位置補上靜默 (placeholder)
- 未來可擴展為生成替代音訊

主要功能:
- generate_placeholder: 生成對應長度的靜音音訊,用於填補被剪的位置
- 預留: 之後可整合真正的 TTS 引擎
"""
import logging
import os
import subprocess
from typing import Dict, List, Optional

logger = logging.getLogger("smart_aroll.tts")

_edge_tts = None


def _try_import_edge_tts():
    global _edge_tts
    if _edge_tts is not None:
        return _edge_tts
    try:
        import edge_tts
        _edge_tts = edge_tts
        return _edge_tts
    except ImportError:
        return None


def _run_ffmpeg_silence(duration: float, output_path: str,
                        sample_rate: int = 16000, channels: int = 1,
                        ffmpeg_path: str = "") -> str:
    cmd = [
        ffmpeg_path, "-y", "-hide_banner", "-loglevel", "error",
        "-f", "lavfi",
        "-i", f"anullsrc=channel_layout={'mono' if channels==1 else 'stereo'}:sample_rate={sample_rate}",
        "-t", str(duration),
        "-c:a", "pcm_s16le",
        output_path,
    ]
    proc = subprocess.run(cmd, capture_output=True)
    if proc.returncode != 0:
        raise RuntimeError(f"生成靜音失敗: {proc.stderr.decode(errors='replace')}")
    return output_path


def generate_placeholder(duration: float, output_path: str,
                         sample_rate: int = 16000,
                         ffmpeg_path: str = "") -> str:
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    _run_ffmpeg_silence(duration, output_path, sample_rate=sample_rate, ffmpeg_path=ffmpeg_path)
    logger.info(f"✓ 生成靜音 placeholder: {output_path} ({duration:.2f}s)")
    return output_path


def text_to_speech(text: str, output_path: str, voice: str = "zh-TW-HsiaoChenNeural",
                   rate: str = "+0%", pitch: str = "+0Hz") -> Optional[str]:
    edge_tts = _try_import_edge_tts()
    if edge_tts is None:
        logger.warning("edge-tts 未安裝,跳過 TTS")
        return None

    import asyncio

    async def _synth():
        communicate = edge_tts.Communicate(text, voice=voice, rate=rate, pitch=pitch)
        await communicate.save(output_path)

    try:
        asyncio.run(_synth())
        return output_path
    except Exception as e:
        logger.error(f"TTS 失敗: {e}")
        return None


def patch_silence(audio_path: str, silence_segments: List[Dict[str, float]],
                  output_path: str,
                  ffmpeg_path: str = "") -> str:
    if not silence_segments:
        return audio_path

    filters = []
    for seg in silence_segments:
        s = max(0, seg["start"])
        e = seg["end"]
        filters.append(f"volume=enable='between(t,{s},{e})':volume=0")

    filter_str = ",".join(filters)

    cmd = [
        ffmpeg_path, "-y", "-hide_banner", "-loglevel", "error",
        "-i", audio_path,
        "-af", filter_str,
        "-c:a", "pcm_s16le",
        output_path,
    ]
    proc = subprocess.run(cmd, capture_output=True)
    if proc.returncode != 0:
        raise RuntimeError(f"靜音化失敗: {proc.stderr.decode(errors='replace')}")
    logger.info(f"✓ 音訊靜音化完成: {output_path}")
    return output_path
