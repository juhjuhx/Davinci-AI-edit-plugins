import logging

from .models import AnalysisResult, SegmentType

logger = logging.getLogger("smart_aroll.resolve")


class ResolveIntegration:
    _CLIP_COLOR = {"unusable": "Red", "usable": "Green"}
    _MARKER_COLORS = {
        SegmentType.CUT: "Red",
        SegmentType.FILLER: "Orange",
        SegmentType.REPEAT: "Yellow",
        SegmentType.HESITATION: "Yellow",
        SegmentType.NERVOUS: "Yellow",
        SegmentType.CORRECTION: "Orange",
        SegmentType.SILENCE: "Blue",
        SegmentType.UNCERTAIN: "Purple",
        SegmentType.USABLE: "Green",
    }

    @classmethod
    def apply(cls, result: AnalysisResult, project, track_index: int = 1) -> int:
        timeline = project.GetCurrentTimeline()
        if not timeline:
            logger.error("No timeline found")
            return 0

        fps_str = timeline.GetSetting("timelineFrameRate")
        fps = float(fps_str) if fps_str else 30.0

        clips = timeline.GetItemListInTrack("video", track_index)
        if not clips:
            logger.error("No clips found on track")
            return 0

        applied = 0
        for seg in result.segments:
            seg_sf = int(seg.start * fps)
            seg_ef = int(seg.end * fps)

            for clip in clips:
                if clip.GetStart() > seg_ef or clip.GetEnd() < seg_sf:
                    continue

                color_key = "unusable" if not seg.is_kept else "usable"
                clip.SetClipColor(cls._CLIP_COLOR[color_key])

                if not seg.is_kept:
                    marker_color = cls._MARKER_COLORS.get(seg.type, "Yellow")
                    note = " | ".join(filter(None, [
                        seg.type.value,
                        seg.reason,
                        seg.text[:60]
                    ]))
                    clip.AddMarker(
                        clip.GetStart(),
                        marker_color,
                        "AI",
                        note,
                        max(1, seg_ef - seg_sf)
                    )

                applied += 1

        logger.info(f"Applied {applied} markers to timeline")
        return applied
