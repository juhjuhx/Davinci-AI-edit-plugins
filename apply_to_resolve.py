import json
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from core.models import AnalysisResult, AnalysisSegment, Priority, SegmentType  # noqa: E402
from core.resolve import ResolveIntegration  # noqa: E402

RESULT_FILE = os.path.join(SCRIPT_DIR, "results", "last_result.json")

print("[SmartAroll] Loading analysis result...")

try:
    with open(RESULT_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"[SmartAroll] Loaded {len(data.get('segments', []))} segments")
except Exception as e:
    print(f"[SmartAroll] Error: Cannot load result - {e}")
    sys.exit(1)

try:
    import DaVinciResolveScript as dvr  # noqa: N813
    resolve = dvr.scriptapp("Resolve")
    if not resolve:
        print("[SmartAroll] Error: Not connected to Resolve")
        sys.exit(1)
except Exception as e:
    print(f"[SmartAroll] Error: Cannot connect to Resolve - {e}")
    sys.exit(1)

project = resolve.GetProjectManager().GetCurrentProject()
if not project:
    print("[SmartAroll] Error: Please open a project first")
    sys.exit(1)

timeline = project.GetCurrentTimeline()
if not timeline:
    print("[SmartAroll] Error: Please open a timeline first")
    sys.exit(1)

segments = []
for s in data.get("segments", []):
    seg = AnalysisSegment(
        id=s["id"],
        start=s["start"],
        end=s["end"],
        text=s.get("text", ""),
        confidence=s.get("confidence", 1.0),
        type=SegmentType(s.get("type", "usable")),
        priority=Priority(s.get("priority", "low")),
        reason=s.get("reason", ""),
        rules_hit=s.get("rules_hit", []),
    )
    if s.get("manual_override") is not None:
        seg.manual_override = s["manual_override"]
    segments.append(seg)

result = AnalysisResult(
    video_path=data.get("video_path", ""),
    segments=segments,
    fps=data.get("fps", 30.0),
    total_duration=data.get("total_duration", 0.0),
    used_duration=data.get("used_duration", 0.0),
    cut_duration=data.get("cut_duration", 0.0),
)

cnt = ResolveIntegration.apply(result, project)
print(f"[SmartAroll] Applied {cnt} markers to timeline")
