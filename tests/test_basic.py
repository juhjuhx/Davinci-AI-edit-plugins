"""
Basic smoke tests for Smart A-Roll core module.
"""

import core
import core.utils as utils


def test_version():
    """Ensure version is set to expected 4.0.0."""
    assert core.__version__ == "4.0.0"


def test_clean_path_basic():
    """clean_path should strip leading/trailing whitespace."""
    assert utils.clean_path("test") == "test"
    assert utils.clean_path("  test  ") == "test"
    assert utils.clean_path("\t test\n") == "test"


def test_clean_path_strips_bidi():
    """clean_path should strip Unicode bidi/invisible characters."""
    p = "\u200bhello\u200b"
    assert utils.clean_path(p) == "hello"


def test_sigmoid_confidence():
    """sigmoid_confidence should map logprob to [0, 1] range."""
    from core.analyzer import sigmoid_confidence

    # High confidence (good logprob) -> close to 1
    assert sigmoid_confidence(1.0) > 0.95
    # Low confidence (bad logprob) -> close to 0
    assert sigmoid_confidence(-2.0) < 0.05
    # None should return 0
    assert sigmoid_confidence(None) == 0.0


def test_config_defaults():
    """Config should provide defaults even when file is missing."""
    from core.config import Config

    config = Config(raw={})
    assert isinstance(config.whisper, dict)
    assert isinstance(config.llm, dict)


def test_models_segment_type():
    """SegmentType should expose expected values."""
    from core.models import SegmentType

    assert SegmentType.USABLE.value == "usable"
    assert SegmentType.CUT.value == "cut"
    assert SegmentType.FILLER.value == "filler"


def test_analysis_segment_default():
    """AnalysisSegment default type is USABLE."""
    from core.models import AnalysisSegment

    seg = AnalysisSegment(id=1, start=0.0, end=1.0, text="hello")
    assert seg.type == core.models.SegmentType.USABLE
    assert seg.is_kept is True
    assert seg.duration == 1.0


def test_analysis_segment_roundtrip():
    """AnalysisSegment.to_dict() -> from_dict() must round-trip losslessly.

    Regression test: previously to_dict() added computed fields (effective_type,
    is_kept, duration) that the dataclass did not accept, crashing the cache reload.
    """
    from core.models import AnalysisSegment, Priority, SegmentType

    original = AnalysisSegment(
        id=42,
        start=1.5,
        end=4.25,
        text="hello world",
        confidence=0.87,
        type=SegmentType.FILLER,
        priority=Priority.HIGH,
        reason="filler word",
        rules_hit=["filler_detector", "short_duration"],
        llm_verified=True,
        llm_reason="looks like filler",
        manual_override=False,
    )
    d = original.to_dict()
    assert "effective_type" not in d, "to_dict must not leak computed fields"
    assert "is_kept" not in d, "to_dict must not leak computed fields"
    assert "duration" not in d, "to_dict must not leak computed fields"
    assert d["type"] == "filler", "type must be serialized as plain string"
    assert d["priority"] == "high", "priority must be serialized as plain string"

    restored = AnalysisSegment.from_dict(d)
    assert restored.id == original.id
    assert restored.start == original.start
    assert restored.end == original.end
    assert restored.text == original.text
    assert restored.confidence == original.confidence
    assert restored.type == original.type, "type must be restored as enum"
    assert restored.priority == original.priority, "priority must be restored as enum"
    assert restored.reason == original.reason
    assert restored.rules_hit == original.rules_hit
    assert restored.llm_verified == original.llm_verified
    assert restored.llm_reason == original.llm_reason
    assert restored.manual_override == original.manual_override
    assert restored.is_kept == original.is_kept
    assert restored.duration == original.duration


def test_from_dict_handles_missing_optional_fields():
    """from_dict should tolerate older cache files that lack newer fields."""
    from core.models import AnalysisSegment, SegmentType

    minimal = {"id": 1, "start": 0.0, "end": 1.0, "text": "x"}
    seg = AnalysisSegment.from_dict(minimal)
    assert seg.type == SegmentType.USABLE
    assert seg.confidence == 1.0
    assert seg.manual_override is None
    assert seg.rules_hit == []


def test_to_dict_strips_compat_with_legacy_cache_files():
    """Cache files written before the from_dict fix may still contain extra fields.

    from_dict() should ignore them rather than crash, so old caches can be migrated.
    """
    from core.models import AnalysisSegment

    legacy = {
        "id": 1,
        "start": 0.0,
        "end": 1.0,
        "text": "hi",
        "type": "usable",
        "priority": "low",
        "effective_type": "usable",
        "is_kept": True,
        "duration": 1.0,
    }
    seg = AnalysisSegment.from_dict(legacy)
    assert seg.id == 1
    assert seg.is_kept is True
