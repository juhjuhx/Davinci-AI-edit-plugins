import os
import re
import math
import time
import logging
import hashlib
import json
from pathlib import Path
from typing import Optional, List, Dict, Any, Callable

from .config import Config
from .models import (
    AnalysisSegment, TranscriptionSegment, AnalysisResult,
    SegmentType, Priority, VideoInfo
)
from .ffmpeg_gpu import get_runner
from .utils import clean_path, lazy_import

logger = logging.getLogger("smart_aroll.analyzer")

_whisper = lazy_import("faster_whisper", "WhisperModel")
_librosa = lazy_import("librosa")
_np = lazy_import("numpy")


def sigmoid_confidence(avg_logprob: float, k: float = 4.0, threshold: float = -0.5) -> float:
    if avg_logprob is None:
        return 0.0
    try:
        z = k * (avg_logprob - threshold)
        z = max(-20.0, min(20.0, z))
        return 1.0 / (1.0 + math.exp(-z))
    except (OverflowError, ValueError):
        return 0.0


class RulesEngine:

    def __init__(self, rules_config: Dict[str, Any]):
        self.filler_patterns = [re.compile(p) for p in rules_config.get("filler_patterns", [])]
        self.repeat_patterns = [re.compile(p) for p in rules_config.get("repeat_patterns", [])]
        self.cue_patterns = {k: re.compile(v) for k, v in rules_config.get("cue_patterns", {}).items()}
        self.throat_patterns = [re.compile(p) for p in rules_config.get("throat_patterns", [])]
        self.correction_patterns = [re.compile(p) for p in rules_config.get("correction_patterns", [])]

    def analyze(self, text: str) -> Dict[str, Any]:
        if not text or not text.strip():
            return {
                "type": SegmentType.USABLE,
                "priority": Priority.LOW,
                "reason": "",
                "rules_hit": [],
            }

        text_clean = text.strip()

        for name, pattern in self.cue_patterns.items():
            if pattern.search(text_clean):
                return {
                    "type": SegmentType.CUT,
                    "priority": Priority.HIGH,
                    "reason": f"{'拍手指令' if name == 'clap_cue' else '剪輯指令'} ({name})",
                    "rules_hit": [name],
                }

        for pattern in self.repeat_patterns:
            if pattern.search(text_clean):
                return {
                    "type": SegmentType.REPEAT,
                    "priority": Priority.HIGH,
                    "reason": "重複詞",
                    "rules_hit": ["repetition"],
                }

        for pattern in self.correction_patterns:
            if pattern.search(text_clean):
                return {
                    "type": SegmentType.CORRECTION,
                    "priority": Priority.MID,
                    "reason": "自我修正",
                    "rules_hit": ["self_correction"],
                }

        for pattern in self.throat_patterns:
            if pattern.search(text_clean):
                return {
                    "type": SegmentType.CUT,
                    "priority": Priority.MID,
                    "reason": "喉音/清嗓",
                    "rules_hit": ["throat_clearing"],
                }

        for pattern in self.filler_patterns:
            if pattern.search(text_clean):
                return {
                    "type": SegmentType.FILLER,
                    "priority": Priority.MID,
                    "reason": "填充詞/口頭禪",
                    "rules_hit": ["filler"],
                }

        return {
            "type": SegmentType.USABLE,
            "priority": Priority.LOW,
            "reason": "正常語句",
            "rules_hit": [],
        }


class AudioAnalyzer:

    def __init__(self, config: Config):
        self.config = config

    def _import_deps(self):
        if not _librosa.is_available:
            raise ImportError("librosa 未安裝")
        if not _np.is_available:
            raise ImportError("numpy 未安裝")

    def detect_silence(self, audio_path: str, threshold: float = 0.006,
                       step: float = 0.5, min_duration: float = 0.5) -> List[Dict[str, float]]:
        self._import_deps()
        librosa = _librosa.load()
        y, sr = librosa.load(audio_path, sr=16000, mono=True)
        hop_length = int(step * sr)
        energy = librosa.feature.rms(y=y, hop_length=hop_length)[0]

        silences = []
        in_silence = False
        start_idx = 0

        for i, e in enumerate(energy):
            if e < threshold:
                if not in_silence:
                    in_silence = True
                    start_idx = i
            else:
                if in_silence:
                    end_idx = i
                    duration = (end_idx - start_idx) * step
                    if duration >= min_duration:
                        silences.append({
                            "start": start_idx * step,
                            "end": end_idx * step,
                            "duration": duration,
                        })
                    in_silence = False

        if in_silence:
            end_idx = len(energy)
            duration = (end_idx - start_idx) * step
            if duration >= min_duration:
                silences.append({
                    "start": start_idx * step,
                    "end": end_idx * step,
                    "duration": duration,
                })

        return silences

    def get_energy_profile(self, audio_path: str, step: float = 0.5) -> List[Dict[str, float]]:
        self._import_deps()
        librosa = _librosa.load()
        y, sr = librosa.load(audio_path, sr=16000, mono=True)
        hop_length = int(step * sr)
        energy = librosa.feature.rms(y=y, hop_length=hop_length)[0]

        return [
            {"t": i * step, "energy": float(e)}
            for i, e in enumerate(energy)
        ]


class WhisperTranscriber:

    def __init__(self, config: Config):
        self.config = config
        self._model = None
        self._model_name = None
        self._device = None

    def _load_model(self):
        if self._model is not None:
            return

        WhisperModel = _whisper.get_attr("WhisperModel")
        if WhisperModel is None:
            raise ImportError("faster-whisper 未安裝")

        wcfg = self.config.whisper
        model_name = wcfg.get("model_size", "large-v3")
        device = wcfg.get("device", "cuda")
        compute_type = wcfg.get("compute_type", "float16")

        if device == "cuda":
            try:
                import torch
                if not torch.cuda.is_available():
                    logger.warning("CUDA 不可用,改用 CPU")
                    device = "cpu"
                    compute_type = "int8"
            except ImportError:
                device = "cpu"
                compute_type = "int8"

        logger.info(f"載入 Whisper 模型: {model_name} (device={device}, compute={compute_type})")

        try:
            self._model = WhisperModel(model_name, device=device, compute_type=compute_type)
            self._model_name = model_name
            self._device = device
            logger.info("✓ Whisper 模型載入完成")
        except Exception as e:
            logger.error(f"Whisper 載入失敗 ({device}): {e},改用 CPU int8")
            self._model = WhisperModel(model_name, device="cpu", compute_type="int8")
            self._model_name = model_name
            self._device = "cpu"

    def unload(self):
        if self._model is not None:
            del self._model
            self._model = None
            try:
                import torch
                torch.cuda.empty_cache()
            except ImportError:
                pass

    def transcribe(self, audio_path: str, vad_filter: bool = None) -> List[TranscriptionSegment]:
        self._load_model()
        wcfg = self.config.whisper

        if vad_filter is None:
            vad_filter = wcfg.get("vad_filter", True)

        kwargs = {
            "language": wcfg.get("language", "zh"),
            "beam_size": wcfg.get("beam_size", 1),
            "condition_on_previous_text": wcfg.get("condition_on_previous_text", False),
            "vad_filter": vad_filter,
            "vad_parameters": {
                "threshold": wcfg.get("vad_threshold", 0.5),
                "min_silence_duration_ms": wcfg.get("vad_min_silence_ms", 500),
                "speech_pad_ms": wcfg.get("vad_speech_pad_ms", 200),
            } if vad_filter else None,
        }

        logger.info(f"開始轉錄: {audio_path}")
        t0 = time.time()
        segments_iter, info = self._model.transcribe(audio_path, **kwargs)
        logger.info(f"Whisper 偵測語言: {info.language} (prob={info.language_probability:.2f})")

        sig_k = self.config.analysis.get("sigmoid_k", 4.0)
        sig_th = self.config.analysis.get("sigmoid_threshold", -0.5)

        segments = []
        for i, seg in enumerate(segments_iter):
            confidence = sigmoid_confidence(seg.avg_logprob, k=sig_k, threshold=sig_th)
            segments.append(TranscriptionSegment(
                id=i,
                start=seg.start,
                end=seg.end,
                text=seg.text.strip(),
                confidence=confidence,
                avg_logprob=getattr(seg, "avg_logprob", 0.0) or 0.0,
                no_speech_prob=getattr(seg, "no_speech_prob", 0.0) or 0.0,
            ))

        logger.info(f"轉錄完成: {len(segments)} 個片段,耗時 {time.time()-t0:.1f}s")
        return segments


class Analyzer:

    def __init__(self, config: Config):
        self.config = config
        self.whisper = WhisperTranscriber(config)
        self.audio = AudioAnalyzer(config)
        self.rules = RulesEngine(config.rules)
        self.llm = None
        self._runner = get_runner(config)

    def set_llm(self, llm_analyzer):
        self.llm = llm_analyzer

    def _get_cache_path(self, video_path: str) -> Optional[Path]:
        cache_dir = self.config.export.get("cache_dir")
        if not cache_dir or not self.config.export.get("enable_cache", False):
            return None
        os.makedirs(cache_dir, exist_ok=True)
        m = hashlib.md5()
        m.update(video_path.encode("utf-8"))
        m.update(str(os.path.getmtime(video_path)).encode("utf-8"))
        return Path(cache_dir) / f"{m.hexdigest()}.json"

    def _load_cache(self, cache_path: Path) -> Optional[Dict]:
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None

    def _save_cache(self, cache_path: Path, data: Dict):
        try:
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2, default=str)
        except Exception as e:
            logger.warning(f"無法寫入快取: {e}")

    def _extract_audio(self, video_path: str) -> str:
        audio_temp = os.path.join(
            os.path.dirname(video_path),
            f".{os.path.basename(video_path)}.audio.wav"
        )
        try:
            self._runner.extract_audio(video_path, audio_temp)
        except Exception as e:
            raise RuntimeError(f"音訊提取失敗: {e}")
        return audio_temp

    def _transcribe(self, audio_path: str, result: AnalysisResult,
                    progress_callback: Optional[Callable] = None):
        if progress_callback:
            progress_callback("whisper", "Whisper 轉錄中", 20)
        transcriptions = self.whisper.transcribe(audio_path)
        result.transcriptions = transcriptions
        return transcriptions

    def _apply_rules(self, transcriptions: List[TranscriptionSegment],
                     result: AnalysisResult,
                     progress_callback: Optional[Callable] = None) -> List[AnalysisSegment]:
        if progress_callback:
            progress_callback("filter", "信心度過濾", 60)
        threshold = self.config.analysis.get("confidence_threshold", 0.2)
        analysis_segments = []

        for t in transcriptions:
            seg = AnalysisSegment(
                id=t.id,
                start=t.start,
                end=t.end,
                text=t.text,
                confidence=t.confidence,
            )
            if t.confidence < threshold:
                seg.type = SegmentType.UNCERTAIN
                seg.priority = Priority.MID
                seg.reason = f"信心度低 ({t.confidence:.2f} < {threshold})"
                seg.rules_hit = ["low_confidence"]

            rule_result = self.rules.analyze(t.text)
            if rule_result["rules_hit"] and seg.type == SegmentType.USABLE:
                seg.type = rule_result["type"]
                seg.priority = rule_result["priority"]
                seg.reason = rule_result["reason"]
                seg.rules_hit = rule_result["rules_hit"]

            if seg.duration < self.config.analysis.get("min_segment_duration", 0.3):
                if seg.type == SegmentType.USABLE:
                    seg.type = SegmentType.UNCERTAIN
                    seg.reason = f"片段過短 ({seg.duration:.2f}s)"

            analysis_segments.append(seg)

        return analysis_segments

    def _compute_energy(self, audio_path: str) -> List[Dict[str, float]]:
        try:
            return self.audio.get_energy_profile(audio_path, step=0.05)
        except Exception as e:
            logger.warning(f"Energy profile failed: {e}")
            return []

    def _llm_verify(self, segments: List[AnalysisSegment],
                    progress_callback: Optional[Callable] = None):
        if not self.llm or not self.llm.is_ready:
            return

        if progress_callback:
            progress_callback("llm", "LLM 二次確認中", 75)

        uncertain_texts = [
            (i, s.text) for i, s in enumerate(segments)
            if s.type in (SegmentType.UNCERTAIN, SegmentType.FILLER, SegmentType.HESITATION)
        ]
        for idx, (i, text) in enumerate(uncertain_texts):
            r = self.llm.analyze_segment(text)
            if r["verified"]:
                seg = segments[i]
                seg.llm_verified = True
                seg.llm_reason = r["reason"]
                if r["should_cut"] and seg.type in (SegmentType.UNCERTAIN, SegmentType.FILLER):
                    seg.type = SegmentType.CUT
                    seg.reason = f"LLM 確認剪輯: {r['reason']}"
            if progress_callback:
                progress_callback("llm", f"LLM 確認中 {idx+1}/{len(uncertain_texts)}",
                                  75 + int(20 * idx / max(1, len(uncertain_texts))))

    def _cleanup_temp(self, audio_temp: str):
        if os.path.isfile(audio_temp):
            try:
                os.remove(audio_temp)
            except OSError:
                pass

    def analyze(self, video_path: str,
                progress_callback: Optional[Callable] = None) -> AnalysisResult:
        video_path = clean_path(video_path)
        t0 = time.time()
        result = AnalysisResult(video_path=video_path)

        # Stage 1: Video info
        if progress_callback:
            progress_callback("video_info", "讀取視訊資訊", 5)
        try:
            video_info = self._runner.get_info(video_path)
            result.video_info = video_info
            result.fps = video_info.fps
        except Exception as e:
            logger.warning(f"ffprobe 失敗,使用預設 FPS=30: {e}")
            result.fps = 30.0
        result.total_duration = result.video_info.duration if result.video_info else 0.0

        # Stage 2: Cache check
        cache_path = self._get_cache_path(video_path)
        if cache_path and cache_path.is_file():
            cached = self._load_cache(cache_path)
            if cached:
                logger.info("✓ 使用快取結果")
                result.cache_hit = True
                result.fps = cached.get("fps", result.fps)
                result.segments = [AnalysisSegment(**s) for s in cached.get("segments", [])]
                result.used_duration = sum(s.duration for s in result.segments if s.is_kept)
                result.cut_duration = sum(s.duration for s in result.segments if not s.is_kept)
                if progress_callback:
                    progress_callback("done", "從快取載入", 100)
                return result

        # Stage 3: Audio extraction
        if progress_callback:
            progress_callback("audio", "提取音訊", 10)
        audio_temp = self._extract_audio(video_path)

        try:
            # Stage 4: Transcription
            transcriptions = self._transcribe(audio_temp, result, progress_callback)

            # Stage 5: Rules engine
            analysis_segments = self._apply_rules(transcriptions, result, progress_callback)

            # Stage 6: Energy profile
            if progress_callback:
                progress_callback("energy", "計算音訊波形", 55)
            result.energy_profile = self._compute_energy(audio_temp)

            # Stage 7: LLM verification
            self._llm_verify(analysis_segments, progress_callback)

            # Stage 8: Finalize
            result.segments = analysis_segments
            result.used_duration = sum(s.duration for s in analysis_segments if s.is_kept)
            result.cut_duration = sum(s.duration for s in analysis_segments if not s.is_kept)
            result.llm_enabled = self.llm is not None and self.llm.is_ready

            if cache_path:
                self._save_cache(cache_path, {
                    "fps": result.fps,
                    "segments": [s.to_dict() for s in analysis_segments],
                })

        finally:
            self.whisper.unload()
            self._cleanup_temp(audio_temp)

        result.analysis_time = time.time() - t0
        if progress_callback:
            progress_callback("done", "分析完成", 100)

        return result
