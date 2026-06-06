import logging
import os
import shutil
from typing import Any, Dict, Optional

from .ffmpeg_gpu import get_runner

logger = logging.getLogger("smart_aroll.enhance")


def _get_ffmpeg() -> str:
    return get_runner().ffmpeg_path


def get_audio_loudness(audio_path: str) -> Optional[Dict[str, float]]:
    import json
    import subprocess as _sp

    ffmpeg = _get_ffmpeg()
    cmd = [
        ffmpeg,
        "-hide_banner",
        "-i",
        audio_path,
        "-af",
        "loudnorm=I=-16:TP=-1.5:LRA=11:print_format=json",
        "-f",
        "null",
        "-",
    ]
    try:
        proc = _sp.run(cmd, capture_output=True)
        output = proc.stderr.decode(errors="replace") if proc.stderr else ""
        json_start = output.rfind("{")
        if json_start == -1:
            return None
        return json.loads(output[json_start:])
    except Exception:
        return None


def normalize_loudness(input_path: str, output_path: str, target_lufs: float = -16.0) -> str:
    measure = get_audio_loudness(input_path)
    if measure is None:
        logger.warning("Cannot measure loudness, skipping normalization")
        shutil.copy2(input_path, output_path)
        return output_path

    measured_i = measure.get("input_i", "-23")
    measured_tp = measure.get("input_tp", "-2")
    measured_lra = measure.get("input_lra", "0")
    measured_thresh = measure.get("input_thresh", "-34")

    filter_str = (
        f"loudnorm=I={target_lufs}:TP=-1.5:LRA=11:"
        f"measured_I={measured_i}:measured_TP={measured_tp}:"
        f"measured_LRA={measured_lra}:measured_thresh={measured_thresh}:"
        f"linear=true:print_format=summary"
    )

    runner = get_runner()
    return runner.apply_audio_filters(input_path, output_path, [filter_str])


def apply_highpass_lowpass(
    input_path: str, output_path: str, highpass_freq: int = 80, lowpass_freq: int = 12000
) -> str:
    runner = get_runner()
    return runner.apply_audio_filters(input_path, output_path, [f"highpass=f={highpass_freq},lowpass=f={lowpass_freq}"])


def denoise_audio(input_path: str, output_path: str, strength: float = 0.01) -> str:
    runner = get_runner()
    return runner.apply_audio_filters(input_path, output_path, [f"anlmdn=s={strength}"])


def compress_dynamic_range(input_path: str, output_path: str, threshold: float = -20.0, ratio: float = 4.0) -> str:
    runner = get_runner()
    return runner.apply_audio_filters(
        input_path, output_path, [f"acompressor=threshold={threshold}:ratio={ratio}:attack=20:release=200"]
    )


def enhance_audio_pipeline(input_path: str, output_path: str, config: Dict[str, Any]) -> str:
    if not config.get("enabled", False):
        shutil.copy2(input_path, output_path)
        return output_path

    current = input_path
    temp_files = []
    try:
        if config.get("highpass_freq") or config.get("lowpass_freq"):
            tmp = output_path + ".tmp1.wav"
            apply_highpass_lowpass(
                current,
                tmp,
                highpass_freq=config.get("highpass_freq", 80),
                lowpass_freq=config.get("lowpass_freq", 12000),
            )
            if current != input_path:
                temp_files.append(current)
            current = tmp
            temp_files.append(tmp)

        if config.get("denoise", False):
            tmp = output_path + ".tmp2.wav"
            denoise_audio(current, tmp)
            if current != input_path:
                temp_files.append(current)
            current = tmp
            temp_files.append(tmp)

        tmp = output_path + ".tmp3.wav"
        compress_dynamic_range(current, tmp)
        if current != input_path:
            temp_files.append(current)
        current = tmp
        temp_files.append(tmp)

        if config.get("normalize", True):
            normalize_loudness(current, output_path, target_lufs=config.get("target_lufs", -16.0))
            if current != output_path:
                temp_files.append(current)
        else:
            shutil.copy2(current, output_path)

        logger.info(f"Audio enhancement pipeline complete: {output_path}")
        return output_path
    finally:
        for f in temp_files:
            if f != output_path and os.path.isfile(f):
                try:
                    os.remove(f)
                except OSError:
                    pass
