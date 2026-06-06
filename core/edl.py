import os
import json
import logging
from pathlib import Path
from typing import List, Optional, Dict, Any
from datetime import datetime

from .models import AnalysisSegment, AnalysisResult, SegmentType

logger = logging.getLogger("smart_aroll.edl")


def seconds_to_tc(seconds: float, fps: float = 30.0, drop_frame: bool = False) -> str:
    if seconds < 0:
        seconds = 0
    fps = max(1.0, float(fps))

    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    frames = int(round((seconds - int(seconds)) * fps))

    if frames >= fps:
        frames = 0
        secs += 1
    if secs >= 60:
        secs = 0
        minutes += 1
    if minutes >= 60:
        minutes = 0
        hours += 1

    sep = ";" if drop_frame else ":"
    return f"{hours:02d}:{minutes:02d}:{secs:02d}{sep}{frames:02d}"


class EDLExporter:

    def __init__(self, fps: float = 30.0, track_index: int = 1,
                 title: str = "Smart A-Roll Edit"):
        self.fps = fps
        self.track_index = track_index
        self.title = title

    def export(self, result: AnalysisResult, output_path: str,
               export_all: bool = True) -> str:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

        lines = []
        lines.append(f"TITLE: {self.title}")
        lines.append("FCM: NON-DROP FRAME")
        lines.append("")

        record_tc = 0.0
        clip_num = 1

        for seg in result.segments:
            duration = seg.duration
            src_in = seg.start
            src_out = seg.end
            rec_in = record_tc
            rec_out = record_tc + duration

            if seg.is_kept:
                comment_type = seg.type.value.upper()
                comment_text = seg.text[:50]
                comment_line = f"* COMMENT: {comment_type} - {comment_text}"
            else:
                reason_tag = seg.type.value
                comment_text = seg.text[:50]
                comment_line = f"* COMMENT: CUT [{reason_tag}] - {comment_text}"

            lines.append(f"{clip_num:03d}  AX       AA/V  C")
            lines.append(
                f"  {seconds_to_tc(src_in, self.fps)} "
                f"{seconds_to_tc(src_out, self.fps)} "
                f"{seconds_to_tc(rec_in, self.fps)} "
                f"{seconds_to_tc(rec_out, self.fps)}"
            )
            lines.append(comment_line)
            lines.append("")

            record_tc += duration
            clip_num += 1

        with open(output_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

        logger.info(f"EDL exported: {output_path} ({clip_num - 1} entries)")
        return output_path

    def export_markers(self, result: AnalysisResult, output_path: str) -> str:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        lines = ["# DaVinci Resolve Markers (import via Console script)"]
        lines.append("frame,fps,color,name,note,start,end")

        for seg in result.segments:
            frame = int(seg.start * self.fps)
            color = "Red" if seg.type == SegmentType.CUT else (
                "Yellow" if seg.type in (SegmentType.FILLER, SegmentType.HESITATION) else (
                "Orange" if seg.type == SegmentType.REPEAT else "Green"))
            name = seg.type.value
            note = seg.text[:100].replace('"', "'")
            lines.append(f"{frame},{self.fps},{color},{name},\"{note}\",{seg.start:.3f},{seg.end:.3f}")

        with open(output_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        logger.info(f"Markers exported: {output_path}")
        return output_path


class CSVExporter:

    def export(self, result: AnalysisResult, output_path: str) -> str:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

        with open(output_path, "w", encoding="utf-8-sig", newline="") as f:
            import csv
            writer = csv.writer(f)
            writer.writerow([
                "id", "start", "end", "duration", "type", "priority",
                "confidence", "is_kept", "reason", "text", "manual_override", "rules_hit"
            ])
            for s in result.segments:
                writer.writerow([
                    s.id, f"{s.start:.3f}", f"{s.end:.3f}", f"{s.duration:.3f}",
                    s.type.value, s.priority.value,
                    f"{s.confidence:.3f}", s.is_kept,
                    s.reason, s.text, s.manual_override, "|".join(s.rules_hit),
                ])
        logger.info(f"CSV exported: {output_path}")
        return output_path


class TranscriptExporter:

    def export_srt(self, result: AnalysisResult, output_path: str) -> str:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            for i, s in enumerate(result.segments, 1):
                f.write(f"{i}\n")
                f.write(f"{self._srt_time(s.start)} --> {self._srt_time(s.end)}\n")
                if s.type != s.effective_type or s.type.value != "usable":
                    f.write(f"[{s.type.value}] {s.text}\n\n")
                else:
                    f.write(f"{s.text}\n\n")
        logger.info(f"SRT exported: {output_path}")
        return output_path

    def export_text(self, result: AnalysisResult, output_path: str,
                    kept_only: bool = False) -> str:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(f"# Transcript - {os.path.basename(result.video_path)}\n")
            f.write(f"# Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"# Duration: {result.total_duration:.1f}s | Kept: {result.used_duration:.1f}s | Cut: {result.cut_duration:.1f}s\n")
            f.write(f"# FPS: {result.fps:.2f}\n")
            f.write(f"# {'Kept only' if kept_only else 'All segments'}\n")
            f.write("=" * 60 + "\n\n")

            for s in result.segments:
                if kept_only and not s.is_kept:
                    continue
                marker = "+" if s.is_kept else "x"
                f.write(f"[{seconds_to_tc(s.start, result.fps)}] {marker} [{s.type.value}] {s.text}\n")
                if s.reason and not s.is_kept:
                    f.write(f"  -> {s.reason}\n")
        logger.info(f"Transcript exported: {output_path}")
        return output_path

    def _srt_time(self, seconds: float) -> str:
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        s = seconds % 60
        return f"{h:02d}:{m:02d}:{s:06.3f}".replace(".", ",")


def export_all(result: AnalysisResult, output_dir: str,
               fps: float = None, track_index: int = 1) -> Dict[str, str]:
    os.makedirs(output_dir, exist_ok=True)
    base_name = Path(result.video_path).stem

    fps = fps or result.fps or 30.0

    edl = EDLExporter(fps=fps, track_index=track_index)
    csv_exp = CSVExporter()
    trans = TranscriptExporter()

    paths = {
        "edl": edl.export(result, os.path.join(output_dir, f"{base_name}.edl")),
        "edl_markers": edl.export_markers(result, os.path.join(output_dir, f"{base_name}_markers.csv")),
        "csv": csv_exp.export(result, os.path.join(output_dir, f"{base_name}.csv")),
        "srt": trans.export_srt(result, os.path.join(output_dir, f"{base_name}.srt")),
        "transcript": trans.export_text(result, os.path.join(output_dir, f"{base_name}_transcript.txt")),
        "transcript_kept": trans.export_text(result, os.path.join(output_dir, f"{base_name}_kept.txt"), kept_only=True),
    }

    json_path = os.path.join(output_dir, f"{base_name}_result.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result.to_dict(), f, ensure_ascii=False, indent=2, default=str)
    paths["json"] = json_path

    return paths
