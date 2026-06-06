"""
Basic smoke tests for Smart A-Roll core module.
"""
import pytest

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
