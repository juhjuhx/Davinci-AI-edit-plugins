"""
Flask 主程式
- 整合所有 API
- 服務前端 HTML
- SSE 進度推送
- 統一日誌
"""

import argparse
import base64
import json
import logging
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flask import Flask, Response, jsonify, request

from core.analyzer import Analyzer
from core.config import get_config
from core.edl import CSVExporter, EDLExporter, TranscriptExporter, export_all
from core.ffmpeg_gpu import get_runner
from core.llm import get_llm
from core.utils import clean_path, fix_dll_path
from workers.analyze_worker import get_worker

fix_dll_path()

config = get_config()

log_dir = config.export.get("log_dir", "logs")
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, "smart_aroll.log")
file_handler = RotatingFileHandler(log_file, maxBytes=10 * 1024 * 1024, backupCount=5, encoding="utf-8")
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(logging.Formatter("%(asctime)s [%(name)s] %(levelname)s: %(message)s"))
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(logging.Formatter("%(asctime)s [%(name)s] %(levelname)s: %(message)s"))
logging.basicConfig(level=logging.DEBUG, handlers=[file_handler, console_handler])
logger = logging.getLogger("smart_aroll.api")

result_dir = config.export.get("result_dir", "results")
os.makedirs(result_dir, exist_ok=True)

app = Flask(__name__, static_folder=None)
app.config["JSON_AS_ASCII"] = False
app.config["JSONIFY_PRETTYPRINT_REGULAR"] = False

_runner = None
_worker = None


def get_app_runner():
    global _runner
    if _runner is None:
        _runner = get_runner(config)
    return _runner


def get_app_worker():
    global _worker
    if _worker is None:
        _worker = get_worker()
    return _worker


@app.route("/")
def index():
    ui_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "ui", "index.html")
    if os.path.isfile(ui_path):
        with open(ui_path, "r", encoding="utf-8") as f:
            return f.read()
    return "<h1>UI not found</h1><p>請確認 ui/index.html 存在</p>", 404


@app.route("/health")
def health():
    return jsonify(
        {
            "status": "ok",
            "version": "4.0",
            "time": datetime.now().isoformat(),
            "gpu_nvenc": get_app_runner().nvenc_available,
        }
    )


@app.route("/api/analyze", methods=["POST"])
def api_analyze():
    data = request.json or {}
    video_path = data.get("video_path")
    if not video_path:
        return jsonify({"error": "缺少 video_path"}), 400
    video_path = clean_path(video_path)
    if not os.path.isfile(video_path):
        return jsonify({"error": f"找不到檔案: {video_path}"}), 404

    worker = get_app_worker()
    task_id = worker.submit(video_path)
    return jsonify({"task_id": task_id, "status": "queued"})


@app.route("/api/task/<task_id>")
def api_task_status(task_id):
    worker = get_app_worker()
    task = worker.get_task(task_id)
    if not task:
        return jsonify({"error": "task not found"}), 404
    result = worker.get_result(task_id)
    return jsonify(
        {
            "task": {
                "id": task.id,
                "video_path": task.video_path,
                "status": task.status,
                "progress": task.progress,
                "stage": task.stage,
                "message": task.message,
                "error": task.error,
                "created_at": task.created_at,
                "started_at": task.started_at,
                "finished_at": task.finished_at,
            },
            "result": result.to_dict() if result else None,
        }
    )


@app.route("/api/queue")
def api_queue():
    worker = get_app_worker()
    return jsonify(worker.get_status())


@app.route("/api/task/<task_id>/stream")
def api_task_stream(task_id):
    worker = get_app_worker()

    def generate():
        last_msg = None
        for _ in range(3600):
            task = worker.get_task(task_id)
            if not task:
                yield f"event: error\ndata: {json.dumps({'error': 'task not found'})}\n\n"
                break
            msg = json.dumps(
                {
                    "status": task.status,
                    "progress": task.progress,
                    "stage": task.stage,
                    "message": task.message,
                }
            )
            if msg != last_msg:
                yield f"event: progress\ndata: {msg}\n\n"
                last_msg = msg
            if task.status in ("done", "error"):
                if task.status == "done":
                    result = worker.get_result(task_id)
                    yield f"event: done\ndata: {json.dumps(result.to_dict() if result else {}, default=str)}\n\n"
                else:
                    yield f"event: error\ndata: {json.dumps({'error': task.error})}\n\n"
                break
            time.sleep(0.5)
        else:
            yield f"event: timeout\ndata: {json.dumps({'error': 'stream timeout'})}\n\n"

    return Response(
        generate(), mimetype="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
    )


@app.route("/api/segment/toggle", methods=["POST"])
def api_toggle_segment():
    data = request.json or {}
    task_id = data.get("task_id")
    seg_id = data.get("segment_id")
    override = data.get("override")

    worker = get_app_worker()
    result = worker.get_result(task_id)
    if not result:
        return jsonify({"error": "找不到分析結果"}), 404

    target = None
    for s in result.segments:
        if s.id == seg_id:
            target = s
            break
    if not target:
        return jsonify({"error": "找不到片段"}), 404

    if override == "usable":
        target.manual_override = True
    elif override == "cut":
        target.manual_override = False
    else:
        target.manual_override = None
    result.used_duration = sum(s.duration for s in result.segments if s.is_kept)
    result.cut_duration = sum(s.duration for s in result.segments if not s.is_kept)

    return jsonify(
        {
            "success": True,
            "segment": target.to_dict(),
            "used_duration": result.used_duration,
            "cut_duration": result.cut_duration,
        }
    )


@app.route("/api/segment/apply", methods=["POST"])
def api_apply_segments():
    data = request.json or {}
    task_id = data.get("task_id")
    overrides = data.get("overrides", [])
    worker = get_app_worker()
    result = worker.get_result(task_id)
    if not result:
        return jsonify({"error": "找不到分析結果"}), 404
    for o in overrides:
        seg_id = o.get("segment_id")
        override = o.get("override")
        for s in result.segments:
            if s.id == seg_id:
                if override == "clear":
                    s.manual_override = None
                else:
                    s.manual_override = override == "usable"
                break
    result.used_duration = sum(s.duration for s in result.segments if s.is_kept)
    result.cut_duration = sum(s.duration for s in result.segments if not s.is_kept)
    return jsonify({"success": True, "segments": [s.to_dict() for s in result.segments]})


@app.route("/api/segment/split", methods=["POST"])
def api_split_segment():
    data = request.json or {}
    task_id = data.get("task_id")
    seg_id = data.get("segment_id")
    split_time = float(data.get("split_time", 0))
    worker = get_app_worker()
    result = worker.get_result(task_id)
    if not result:
        return jsonify({"error": "找不到分析結果"}), 404
    fps = result.fps or 30.0
    frame = round(split_time * fps)
    split_time = frame / fps
    target = None
    target_idx = -1
    for i, s in enumerate(result.segments):
        if s.id == seg_id:
            target = s
            target_idx = i
            break
    if not target or not (target.start < split_time < target.end):
        return jsonify({"error": "無效的分割點"}), 400
    ratio = (split_time - target.start) / (target.end - target.start)
    text = target.text or ""
    split_char = max(1, int(len(text) * ratio))
    text_a = text[:split_char]
    text_b = text[split_char:]
    new_id = max(s.id for s in result.segments) + 1
    from core.models import AnalysisSegment as _Seg

    seg_a = _Seg(
        id=target.id,
        start=target.start,
        end=split_time,
        text=text_a,
        confidence=target.confidence,
        type=target.type,
        priority=target.priority,
        reason=target.reason,
        rules_hit=list(target.rules_hit),
        manual_override=target.manual_override,
    )
    seg_b = _Seg(
        id=new_id,
        start=split_time,
        end=target.end,
        text=text_b,
        confidence=target.confidence,
        type=target.type,
        priority=target.priority,
        reason=target.reason,
        rules_hit=list(target.rules_hit),
        manual_override=target.manual_override,
    )
    result.segments[target_idx : target_idx + 1] = [seg_a, seg_b]
    result.used_duration = sum(s.duration for s in result.segments if s.is_kept)
    result.cut_duration = sum(s.duration for s in result.segments if not s.is_kept)
    return jsonify(
        {
            "success": True,
            "segments": [s.to_dict() for s in result.segments],
            "total_duration": result.total_duration,
        }
    )


@app.route("/api/export", methods=["POST"])
def api_export():
    data = request.json or {}
    task_id = data.get("task_id")
    formats = data.get("formats", ["edl", "csv", "srt", "transcript"])

    worker = get_app_worker()
    result = worker.get_result(task_id)
    if not result:
        return jsonify({"error": "找不到分析結果"}), 404

    output_dir = config.export.get("result_dir", "results")
    base_name = Path(result.video_path).stem

    paths = {}
    try:
        if "edl" in formats:
            edl = EDLExporter(fps=result.fps or 30.0, track_index=config.export.get("dv_track_index", 1))
            paths["edl"] = edl.export(result, os.path.join(output_dir, f"{base_name}.edl"))
        if "csv" in formats:
            paths["csv"] = CSVExporter().export(result, os.path.join(output_dir, f"{base_name}.csv"))
        if "srt" in formats:
            paths["srt"] = TranscriptExporter().export_srt(result, os.path.join(output_dir, f"{base_name}.srt"))
        if "transcript" in formats:
            paths["transcript"] = TranscriptExporter().export_text(
                result, os.path.join(output_dir, f"{base_name}_transcript.txt")
            )
        if "transcript_kept" in formats:
            paths["transcript_kept"] = TranscriptExporter().export_text(
                result, os.path.join(output_dir, f"{base_name}_kept.txt"), kept_only=True
            )
        if "markers" in formats:
            edl = EDLExporter(fps=result.fps or 30.0)
            paths["markers"] = edl.export_markers(result, os.path.join(output_dir, f"{base_name}_markers.csv"))
        if "all" in formats:
            paths = export_all(result, output_dir, fps=result.fps, track_index=config.export.get("dv_track_index", 1))
    except Exception as e:
        logger.error(f"匯出失敗: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500

    return jsonify({"success": True, "paths": paths})


@app.route("/api/video/info", methods=["POST"])
def api_video_info():
    data = request.json or {}
    video_path = data.get("video_path")
    if not video_path or not os.path.isfile(video_path):
        return jsonify({"error": "找不到影片"}), 404
    try:
        info = get_app_runner().get_info(video_path)
        return jsonify(
            {
                "path": info.path,
                "duration": info.duration,
                "fps": info.fps,
                "width": info.width,
                "height": info.height,
                "codec": info.codec,
                "audio_codec": info.audio_codec,
                "file_size_mb": round(info.file_size / 1024 / 1024, 2),
            }
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/audio/silence", methods=["POST"])
def api_detect_silence():
    from core.analyzer import AudioAnalyzer

    data = request.json or {}
    video_path = data.get("video_path")
    if not video_path or not os.path.isfile(video_path):
        return jsonify({"error": "找不到影片"}), 404

    threshold = float(data.get("threshold", 0.006))
    min_duration = float(data.get("min_duration", 0.5))

    runner = get_app_runner()
    audio_temp = os.path.join(os.path.dirname(video_path), f".{os.path.basename(video_path)}.audio.wav")
    try:
        runner.extract_audio(video_path, audio_temp)
        analyzer = AudioAnalyzer(config)
        silences = analyzer.detect_silence(audio_temp, threshold=threshold, min_duration=min_duration)
        return jsonify(
            {
                "success": True,
                "silences": silences,
                "total_silence": sum(s["duration"] for s in silences),
            }
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        if os.path.isfile(audio_temp):
            try:
                os.remove(audio_temp)
            except Exception:
                pass


@app.route("/api/audio/energy", methods=["POST"])
def api_audio_energy():
    from core.analyzer import AudioAnalyzer

    data = request.json or {}
    video_path = data.get("video_path")
    if not video_path or not os.path.isfile(video_path):
        return jsonify({"error": "找不到影片"}), 404

    runner = get_app_runner()
    audio_temp = os.path.join(os.path.dirname(video_path), f".{os.path.basename(video_path)}.audio.wav")
    try:
        runner.extract_audio(video_path, audio_temp)
        analyzer = AudioAnalyzer(config)
        energy = analyzer.get_energy_profile(audio_temp, step=0.5)
        return jsonify({"success": True, "energy": energy})
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        if os.path.isfile(audio_temp):
            try:
                os.remove(audio_temp)
            except Exception:
                pass


@app.route("/api/video/frame", methods=["POST"])
def api_video_frame():
    data = request.json or {}
    video_path = data.get("video_path")
    time_sec = float(data.get("time", 0))
    if not video_path or not os.path.isfile(video_path):
        return jsonify({"error": "找不到影片"}), 404
    runner = get_app_runner()
    cmd = [
        runner.ffmpeg_path,
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-ss",
        str(time_sec),
        "-i",
        video_path,
        "-vframes",
        "1",
        "-vf",
        "scale=iw/4:ih/4",
        "-f",
        "image2pipe",
        "-vcodec",
        "mjpeg",
        "pipe:1",
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, timeout=10)
        if proc.returncode != 0 or not proc.stdout:
            return jsonify({"error": "frame extract failed"}), 500
        b64 = base64.b64encode(proc.stdout).decode("ascii")
        return jsonify({"success": True, "frame": "data:image/jpeg;base64," + b64})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/video/cut", methods=["POST"])
def api_video_cut():
    data = request.json or {}
    video_path = data.get("video_path")
    start_time = float(data.get("start_time", 0))
    end_time = float(data.get("end_time", 0))
    if not video_path or not os.path.isfile(video_path):
        return jsonify({"error": "找不到影片"}), 404
    if end_time <= start_time:
        return jsonify({"error": "結束時間必須大於開始時間"}), 400

    base_name = Path(video_path).stem
    output_path = os.path.join(result_dir, f"{base_name}_cut.mp4")
    try:
        runner = get_app_runner()
        encoder, _ = runner.get_video_encoder()
        cmd = [
            runner.ffmpeg_path,
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-ss",
            str(start_time),
            "-i",
            video_path,
            "-to",
            str(end_time - start_time),
            "-c:v",
            encoder,
            "-c:a",
            "aac",
            "-b:a",
            "192k",
        ]
        if encoder == "h264_nvenc":
            cmd += ["-preset", runner._nvenc_preset, "-cq", str(runner._crf), "-b:v", "0"]
        else:
            cmd += ["-preset", "medium", "-crf", str(runner._crf)]
        cmd.append(output_path)
        proc = subprocess.run(cmd, capture_output=True)
        if proc.returncode != 0:
            raise RuntimeError(proc.stderr.decode(errors="replace"))
        return jsonify({"success": True, "output_path": output_path})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/video/compress", methods=["POST"])
def api_video_compress():
    data = request.json or {}
    video_path = data.get("video_path")
    crf = int(data.get("crf", 23))
    if not video_path or not os.path.isfile(video_path):
        return jsonify({"error": "找不到影片"}), 404

    base_name = Path(video_path).stem
    output_path = os.path.join(result_dir, f"{base_name}_compressed.mp4")
    try:
        runner = get_app_runner()
        runner.encode_video(video_path, output_path, crf=crf)
        return jsonify({"success": True, "output_path": output_path})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/video/speed", methods=["POST"])
def api_video_speed():
    data = request.json or {}
    video_path = data.get("video_path")
    speed = float(data.get("speed_factor", 1.0))
    if not video_path or not os.path.isfile(video_path):
        return jsonify({"error": "找不到影片"}), 404
    if speed <= 0:
        return jsonify({"error": "速度必須大於 0"}), 400

    base_name = Path(video_path).stem
    output_path = os.path.join(result_dir, f"{base_name}_speed_{speed}x.mp4")
    try:
        runner = get_app_runner()
        encoder, _ = runner.get_video_encoder()
        video_filter = f"setpts=PTS/{speed}"
        audio_filter = f"atempo={speed}"
        if speed > 2.0:
            audio_filter = f"atempo=2.0,atempo={speed / 2.0}"
        elif speed < 0.5:
            audio_filter = f"atempo=0.5,atempo={speed / 0.5}"

        cmd = [
            runner.ffmpeg_path,
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            video_path,
            "-vf",
            video_filter,
            "-af",
            audio_filter,
            "-c:v",
            encoder,
            "-c:a",
            "aac",
        ]
        if encoder == "h264_nvenc":
            cmd += ["-preset", runner._nvenc_preset, "-cq", str(runner._crf), "-b:v", "0"]
        else:
            cmd += ["-preset", "medium", "-crf", str(runner._crf)]
        cmd.append(output_path)
        proc = subprocess.run(cmd, capture_output=True)
        if proc.returncode != 0:
            raise RuntimeError(proc.stderr.decode(errors="replace"))
        return jsonify({"success": True, "output_path": output_path})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/video/auto-edit", methods=["POST"])
def api_auto_edit():
    data = request.json or {}
    task_id = data.get("task_id")
    quality = data.get("quality", "copy")
    apply_enhance = data.get("enhance", False)
    worker = get_app_worker()
    result = worker.get_result(task_id)
    if not result:
        return jsonify({"error": "找不到分析結果"}), 404
    video_path = result.video_path
    base_name = Path(video_path).stem
    ext = Path(video_path).suffix or ".mp4"
    output_path = os.path.join(result_dir, f"{base_name}_cut{ext}")
    kept_segments = [s for s in result.segments if s.is_kept]
    if not kept_segments:
        return jsonify({"error": "沒有可保留的片段"}), 400
    runner = get_app_runner()
    ffmpeg = runner.ffmpeg_path
    segment_files = []
    temp_dir = os.path.join(result_dir, ".temp_segments")
    os.makedirs(temp_dir, exist_ok=True)
    try:
        for i, seg in enumerate(kept_segments):
            seg_path = os.path.join(temp_dir, f"seg_{i:04d}{ext}")
            if quality == "copy":
                cmd = [
                    ffmpeg,
                    "-y",
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-ss",
                    str(seg.start),
                    "-i",
                    video_path,
                    "-t",
                    str(seg.duration),
                    "-avoid_negative_ts",
                    "make_zero",
                    "-c",
                    "copy",
                    seg_path,
                ]
            elif quality == "h265_10bit_422":
                cmd = [
                    ffmpeg,
                    "-y",
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-ss",
                    str(seg.start),
                    "-i",
                    video_path,
                    "-t",
                    str(seg.duration),
                    "-avoid_negative_ts",
                    "make_zero",
                    "-c:v",
                    "libx265",
                    "-pix_fmt",
                    "yuv422p10le",
                    "-preset",
                    "medium",
                    "-crf",
                    "18",
                    "-c:a",
                    "pcm_s16le",
                    seg_path,
                ]
            else:
                cmd = [
                    ffmpeg,
                    "-y",
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-ss",
                    str(seg.start),
                    "-i",
                    video_path,
                    "-t",
                    str(seg.duration),
                    "-avoid_negative_ts",
                    "make_zero",
                    "-c:v",
                    "libx264",
                    "-preset",
                    "medium",
                    "-crf",
                    "18",
                    "-c:a",
                    "aac",
                    "-b:a",
                    "192k",
                    seg_path,
                ]
            proc = subprocess.run(cmd, capture_output=True)
            if proc.returncode != 0:
                raise RuntimeError(f"切片段失敗: {proc.stderr.decode(errors='replace')}")
            segment_files.append(seg_path)
        runner.concat_segments(segment_files, output_path)
        if quality != "copy":
            tmp = output_path + ".remux.mp4"
            cmd = [
                ffmpeg,
                "-y",
                "-hide_banner",
                "-loglevel",
                "error",
                "-i",
                output_path,
                "-c",
                "copy",
                "-movflags",
                "+faststart",
                tmp,
            ]
            proc = subprocess.run(cmd, capture_output=True)
            if proc.returncode == 0:
                os.replace(tmp, output_path)
        if apply_enhance:
            enhanced = output_path + ".enh" + ext
            cmd = [
                ffmpeg,
                "-y",
                "-hide_banner",
                "-loglevel",
                "error",
                "-i",
                output_path,
                "-c:v",
                "copy",
                "-af",
                "loudnorm=I=-16:TP=-1.5:LRA=11",
                "-c:a",
                "aac",
                "-b:a",
                "192k",
                enhanced,
            ]
            proc = subprocess.run(cmd, capture_output=True)
            if proc.returncode == 0:
                os.replace(enhanced, output_path)
    finally:
        for f in segment_files:
            try:
                os.remove(f)
            except Exception:
                pass
        try:
            shutil.rmtree(temp_dir, ignore_errors=True)
        except Exception:
            pass
    return jsonify(
        {
            "success": True,
            "output_path": output_path,
            "kept_count": len(kept_segments),
            "original_duration": result.total_duration,
            "output_duration": result.used_duration,
        }
    )


@app.route("/api/config")
def api_get_config():
    safe = {
        "whisper": {k: v for k, v in config.whisper.items() if k != "api_key"},
        "llm": {**config.llm, "model_path": "***" if config.llm.get("model_path") else ""},
        "ffmpeg": {**config.ffmpeg, "path": "***", "ffprobe_path": "***"},
    }
    return jsonify(safe)


@app.route("/api/browse", methods=["GET"])
def api_browse():
    if sys.platform != "win32":
        return jsonify({"error": "此功能僅支援 Windows"}), 400
    try:
        import tkinter as tk
        from tkinter import filedialog

        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        path = filedialog.askopenfilename(
            title="選擇影片/音訊",
            filetypes=[
                ("Media", "*.mp4 *.mov *.mkv *.avi *.mxf *.webm *.mp3 *.wav *.flac *.aac *.m4a"),
                ("Video", "*.mp4 *.mov *.mkv *.avi *.mxf *.webm"),
                ("Audio", "*.mp3 *.wav *.flac *.aac *.m4a"),
                ("All", "*.*"),
            ],
        )
        root.destroy()
        return jsonify({"path": path or ""})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


def main():
    host = config.server.get("host", "127.0.0.1")
    port = config.server.get("port", 8888)

    print("=" * 60)
    print("  Smart A-Roll v4.0")
    print(f"  http://{host}:{port}")
    print(f"  GPU NVENC: {'✓' if get_app_runner().nvenc_available else '✗ (using libx264)'}")
    print(f"  LLM: {'✓' if config.llm.get('enabled') else '✗ (disabled)'}")
    print("=" * 60)

    if host not in ("127.0.0.1", "localhost", "::1"):
        logger.warning(
            f"伺服器綁定到 {host},將對外網路開放且無身份驗證。"
            "請確認你了解風險,或改用 127.0.0.1 + reverse proxy。"
        )

    get_app_worker()
    app.run(host=host, port=port, debug=False, threaded=True, use_reloader=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Smart A-Roll v4.0")
    parser.add_argument("--video", help="Analyze a video file directly")
    parser.add_argument("--output", help="Output directory (default: results/)")
    parser.add_argument("--no-browser", action="store_true", help="Don't open browser")
    args = parser.parse_args()

    if args.video:
        video_path = clean_path(args.video)
        if not os.path.isfile(video_path):
            print(f"Error: File not found: {video_path}")
            sys.exit(1)
        output_dir = args.output or config.export.get("result_dir", "results")
        os.makedirs(output_dir, exist_ok=True)

        analyzer = Analyzer(config)
        if config.llm.get("enabled", False):
            llm = get_llm(config)
            if llm.is_ready:
                analyzer.set_llm(llm)

        def progress_cb(stage, message, percent):
            print(f"\r[{stage}] {percent}% - {message}", end="", flush=True)

        print(f"Analyzing: {video_path}")
        result = analyzer.analyze(video_path, progress_callback=progress_cb)
        print()

        from core.edl import export_all

        paths = export_all(result, output_dir, fps=result.fps)

        usable = sum(1 for s in result.segments if s.is_kept)
        cut = len(result.segments) - usable
        print(f"Done: {len(result.segments)} segments ({usable} usable, {cut} cut)")
        print(f"Output: {output_dir}")
        for fmt, path in paths.items():
            print(f"  {fmt}: {path}")
    else:
        main()
