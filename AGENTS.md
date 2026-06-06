# AGENTS.md — Smart A-Roll v4.0

AI video editing tool. Analyzes video audio with Whisper + rules engine + optional LLM, lets users select/deselect segments in a WebUI, then exports cut video via ffmpeg (stream copy or re-encode).

## Environment

- **Python 3.10+** with conda env `smart_aroll` or system Python
- **OS:** Windows 10/11 (also works on Linux/macOS with minor changes)
- **FFmpeg:** Auto-detected from PATH, or set `FFMPEG_PATH` env var
- **LLM (optional):** Set `LLAMA_CPP_PATH` env var, model in project root

## Commands

```bat
:: One-click launch (kills old process, checks deps, opens browser)
launch.bat

:: Install dependencies
install.bat

:: Build .exe + .zip + NSIS installer
build.bat

:: CLI mode
python api\app.py --video path\to\video.mp4

:: CLI mode without opening browser
python api\app.py --video path\to\video.mp4 --no-browser

:: Lint check
ruff check .

:: Type check
mypy core/ --ignore-missing-imports

:: Run tests
pytest
```

## Architecture

```
core/
  utils.py          — Shared utilities: clean_path(), fix_dll_path(), lazy_import()
  config.py         — Nested JSON config, deep_merge, lazy singleton via get_config()
  models.py         — SegmentType enum, AnalysisSegment, AnalysisResult (has energy_profile)
  analyzer.py       — WhisperTranscriber + RulesEngine + AudioAnalyzer + Analyzer
  ffmpeg_gpu.py     — FFmpegRunner with NVENC auto-detect, _run() for cp950 encoding
  edl.py            — EDL/CSV/SRT/Marker/Transcript exporters
  llm.py            — llama.cpp via LazyImport, graceful fallback
  tts.py            — edge-tts + silence placeholder
  enhance.py        — Audio pipeline via FFmpegRunner: highpass → denoise → compress → loudnorm
  resolve.py        — DaVinci Resolve Console integration (markers + clip colors)

workers/
  analyze_worker.py — Single-thread background queue, SSE progress, auto-export

api/
  app.py            — Flask server (routes + CLI mode + DLL fix + RotatingFileHandler)

ui/
  index.html        — Single-page WebUI: waveform timeline, draggable panels, preview, export

config.json         — Nested format: whisper/llm/analysis/rules/export/ffmpeg/tts/enhance/server
pyproject.toml      — Project metadata + pytest/mypy/ruff config
```

## Critical constraints

1. **`manual_override` must be `bool` or `None`**, never string. `True` = usable, `False` = cut, `None` = clear.
2. **faster-whisper `avg_logprob` needs sigmoid normalization.** `1.0 / (1.0 + exp(-4.0 * (avg_logprob - -0.5)))`. Threshold 0.2.
3. **LLM may crash on older GPUs** — `0xc000001d` illegal instruction. Must be rebuilt from source or disabled.
4. **EDL timecodes are cumulative** — `record_tc` increases with each kept segment. Format: `AA/V` track, `* COMMENT:` prefix.
5. **config.json uses nested format** — `{"whisper": {"model_size": "large-v3"}}`, not flat keys.
6. **`Analyzer.analyze()` returns `AnalysisResult`** object, not a tuple.
7. **`clean_path()` strips Unicode bidi/invisible chars** from paths. Import from `core.utils`.
8. **`fix_dll_path()` runs before importing llama_cpp** — searches CUDA DLL directories. Import from `core.utils`.
9. **`enable_cache: false`** forces re-analysis every time.

## Key API routes

| Route | Method | Purpose |
|-------|--------|---------|
| `/api/analyze` | POST | Submit video for analysis |
| `/api/task/<id>/stream` | GET | SSE progress |
| `/api/segment/toggle` | POST | Toggle keep/cut (sets `manual_override` as bool) |
| `/api/segment/apply` | POST | Batch toggle |
| `/api/segment/split` | POST | Razor split at time (frame-aligned) |
| `/api/video/frame` | POST | Extract single frame as base64 JPEG (1/4 resolution) |
| `/api/video/auto-edit` | POST | ffmpeg export: copy/h265_10bit_422/h264 |
| `/api/export` | POST | Export EDL/CSV/SRT/JSON |
| `/api/browse` | GET | Windows file picker |

## Key patterns

- `is_kept` checks `effective_type` → `manual_override` (bool) → `type`
- Energy profile: `AudioAnalyzer.get_energy_profile(path, step=0.05)`
- Razor split: `round(time * fps) / fps` for frame alignment
- ffmpeg cut: `-ss` before `-i`, `-avoid_negative_ts make_zero`
- ffmpeg concat: concat demuxer + `-c copy` + remux with `-movflags +faststart`
- Audio enhance: `-c:v copy` + `-af "loudnorm"` (video untouched)
- Preview: `/api/video/frame` returns base64 JPEG at 1/4 resolution, called on timeline click
- Lazy imports: use `LazyImport` from `core.utils` for optional dependencies

## Dependencies

```
faster-whisper flask edge-tts librosa numpy pydub
```

Optional: `llama-cpp-python` for LLM verification.

## Logs

`logs/smart_aroll.log` — RotatingFileHandler, 10MB, 5 backups. Excluded from git.
