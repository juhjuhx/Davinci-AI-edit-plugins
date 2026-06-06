# AGENTS.md — Smart A-Roll v4.0

AI video editing tool. Whisper + rules engine + optional LLM analyze a video's audio, users keep/cut segments in a WebUI, then ffmpeg exports the cut (stream copy or re-encode with NVENC).

## Environment

- **Python 3.10+**, tested 3.10.11 (matches CI). Optional `conda env smart_aroll` is what `launch.bat` looks for; otherwise any system Python works.
- **Windows 10/11** primary, Linux/macOS run but `api/browse` (tkinter file picker) is Windows-only.
- **FFmpeg** auto-detected from `PATH`; override with env `FFMPEG_PATH` (read in `core/ffmpeg_gpu.py:95,115`).
- **LLM (optional):** set `config.llm.model_path` in `config.json` (NOT an env var). Needs `llama-cpp-python` installed via `pip install -e .[llm]`.

## Commands

```bat
:: One-click launch (kills :8888, checks flask, opens browser)
launch.bat

:: Install Python deps (no LLM)
install.bat

:: Build .exe + .zip + NSIS installer
build.bat

:: CLI mode (no WebUI)
python api\app.py --video path\to\video.mp4
python api\app.py --video path\to\video.mp4 --no-browser

:: Verification (use CI-scoped paths to avoid the nested pollution dir)
ruff check core/ api/ workers/ tests/ apply_to_resolve.py
ruff format core/ api/ workers/ tests/ apply_to_resolve.py
mypy core/ --ignore-missing-imports
pytest                                                :: all tests
pytest tests/test_basic.py::test_X                    :: single test
pytest -k sigmoid                                     :: pattern match
```

`ruff check .` from repo root will descend into the **untracked nested `Davinci剪口播腳本/`** copy and produce noise. Always scope the path explicitly.

## Repository layout

```
core/
  utils.py          clean_path(), fix_dll_path(), lazy_import() factory
  config.py         Nested JSON config, deep_merge, get_config() singleton
  models.py         SegmentType, Priority, AnalysisSegment (to_dict/from_dict), AnalysisResult
  analyzer.py       WhisperTranscriber, AudioAnalyzer, RulesEngine, Analyzer
  ffmpeg_gpu.py     FFmpegRunner with NVENC auto-detect, _run() handles cp950
  edl.py            EDL / CSV / SRT / marker / transcript exporters, export_all()
  llm.py            llama.cpp via lazy_import, graceful fallback
  tts.py            edge-tts + silence placeholder
  enhance.py        highpass → denoise → compress → loudnorm pipeline
  resolve.py        DaVinci Resolve console integration (markers + clip colors)
workers/
  analyze_worker.py Single-thread queue, SSE progress, auto-export on done
api/
  app.py            Flask server, routes, CLI mode, top-level fix_dll_path(), RotatingFileHandler
apply_to_resolve.py Standalone: reads results/last_result.json, applies markers to current Resolve timeline
ui/index.html       Single-page WebUI (no build step)
config.json         Nested format: whisper/llm/analysis/rules/export/ffmpeg/tts/enhance/server
pyproject.toml      Package metadata, [project.optional-dependencies] llm/dev, pytest/mypy/ruff config
requirements.txt    Pinned floor versions for CI install (6 deps, no LLM)
.github/workflows/ci.yml  3 jobs: lint (ruff check+format with auto-commit), type-check (mypy), test (pytest)
```

⚠ **Trap:** `Davinci剪口播腳本/` (inside repo root) is a stale untracked copy. It is **not** in git. CI scopes to root paths so it is ignored, but `git status` / `find` / IDEs may show it. Don't edit files there — changes are lost.

## Critical constraints (will break if violated)

1. **`manual_override` must be `bool` or `None`**, never string. `True`=usable, `False`=cut, `None`=cleared. UI never sends the string `"None"`.
2. **`AnalysisSegment.to_dict()` ↔ `from_dict()` is the only supported round-trip.** `to_dict()` returns a clean dict of source-of-truth fields only (no `effective_type`/`is_kept`/`duration` — those are `@property`). Cache files are written via `to_dict()` and reloaded via `from_dict()` in `core/analyzer.py:441`. **Adding a new dataclass field requires updating both methods.**
3. **faster-whisper `avg_logprob` needs sigmoid normalization** before thresholding: `1 / (1 + exp(-k * (x - threshold)))`, default `k=4.0`, `threshold=-0.5`. Default `confidence_threshold=0.2` for cut.
4. **`enable_cache: false` by default** — every run re-analyzes. To use cache, set `export.enable_cache=true` and `export.cache_dir` in `config.json`. Cached segments must be JSON-clean (use `from_dict`).
5. **`concat_segments(segment_files, output_path)` requires all segment paths in a single directory** (post-S1 hardening). List file is placed in `commonpath(segment_files)`. Cross-directory concat will raise.
6. **LLM may crash older CPUs** with `0xc000001d` (illegal instruction SSE4.2) — must be rebuilt from source or `llm.enabled=false`.
7. **`fix_dll_path()` is called at module import time** in `api/app.py:34` and `core/llm.py:9` (BEFORE `llama_cpp` is loaded). It mutates `PATH` to find CUDA/cuDNN DLLs. Do not call lazily.
8. **EDL timecodes are cumulative** — `record_tc` increases with each kept segment. Format: `AA/V` track, `* COMMENT:` prefix.
9. **config.json is nested** — `{"whisper": {"model_size": "large-v3"}}`, not flat. `Config` dataclass in `core/config.py` does deep merge with `DEFAULT_CONFIG`.
10. **`Analyzer.analyze()` returns `AnalysisResult`** object (with `segments`, `video_info`, `energy_profile`, `transcriptions`). Not a tuple.
11. **`clean_path()` strips Unicode bidi/invisible chars** (`\u200b` etc.). Always apply to user-supplied paths from the WebUI.

## Key API routes (Flask)

| Route | Method | Purpose |
|---|---|---|
| `/api/analyze` | POST | Submit video → task_id |
| `/api/task/<id>/stream` | GET | SSE progress (`started`/`progress`/`exported`/`done`/`error`) |
| `/api/task/<id>` | GET | Status snapshot |
| `/api/segment/toggle` | POST | Toggle keep/cut (sets `manual_override` as bool) |
| `/api/segment/apply` | POST | Batch toggle |
| `/api/segment/split` | POST | Razor split at time, frame-aligned |
| `/api/video/frame` | POST | Single frame as base64 JPEG @ 1/4 resolution |
| `/api/video/auto-edit` | POST | ffmpeg export: `copy` / `h265_10bit_422` / `h264` |
| `/api/export` | POST | EDL/CSV/SRT/JSON/Transcript |
| `/api/browse` | GET | Windows tkinter file picker (returns 400 elsewhere) |

## Key patterns

- **is_kept chain:** `effective_type = manual_override if not None else type` → `is_kept = effective_type == USABLE`.
- **Razor split:** `round(time * fps) / fps` for frame alignment.
- **ffmpeg cut:** `-ss` BEFORE `-i` (fast seek), `-avoid_negative_ts make_zero`.
- **ffmpeg concat:** demuxer + `-c copy`, then remux with `-movflags +faststart`.
- **Audio enhance:** `-c:v copy` + `-af "loudnorm=I=-16:TP=-1.5:LRA=11"` (video untouched).
- **Lazy imports:** `from core.utils import lazy_import` and use the lowercase **function** (not the `LazyImport` class directly). Examples: `_whisper = lazy_import("faster_whisper", "WhisperModel")` in `core/analyzer.py:24`.
- **Preview frame:** `/api/video/frame` is polled on timeline click, returns base64 JPEG at 1/4 scale for fast UI.
- **`api/browse` is Windows-only** — Linux/macOS get 400 from `sys.platform != "win32"` check.

## CI quirks (`.github/workflows/ci.yml`)

- **3 independent jobs** on `windows-latest`, Python 3.10. No cross-job dependencies; all 3 must pass.
- **Lint job auto-commits format fixes** via `github-actions[bot]`. If your push fails lint, a follow-up commit `style: apply ruff format` may appear on `main` from CI.
- **mypy is gated by `[[tool.mypy.overrides]]` in `pyproject.toml`** that sets `ignore_errors = true` for `core.*`, `api.*`, `workers.*`, `apply_to_resolve`. CI `mypy` mostly confirms the project type-loads; it does NOT enforce strict typing on runtime modules. Top-level `api/app.py` is NOT in overrides but is also not checked (CI runs `mypy core/`).
- **Test job installs from `requirements.txt`** (6 deps, no LLM) + `pytest` + `pytest-cov`. No ffmpeg, no CUDA, no model files in CI.
- **Path scoping** in `ruff check` / `ruff format` explicitly enumerates `core/ api/ workers/ tests/ apply_to_resolve.py` to skip the nested pollution dir.
- **`permissions: contents: write`** is required for the auto-format-commit step. Do not remove.

## Testing notes

- **Tests are smoke tests, not coverage.** Only `tests/test_basic.py` (10 tests). Bugs in non-imported code paths (like the old `to_dict` round-trip) will slip through.
- **Always add a regression test when fixing a bug** — see `test_analysis_segment_roundtrip` for the pattern.
- **No fixtures, no mocking framework, no snapshot tests.** Tests are deterministic, no ffmpeg, no audio files.
- Tests run in <1s. If a test takes >5s, it's probably trying to import a heavy dep it shouldn't.

## Dependencies

- Runtime (`requirements.txt`): `faster-whisper`, `flask`, `edge-tts`, `librosa`, `numpy`, `pydub`.
- Optional `llm` extra (`pyproject.toml`): `llama-cpp-python`.
- Dev (`pyproject.toml [project.optional-dependencies]`): `pytest`, `pytest-cov`, `mypy`, `ruff`. CI installs `ruff`/`mypy` separately with bare `pip install` (not via `pip install -e .[dev]`).

## Logs

`logs/smart_aroll.log` — `RotatingFileHandler` 10MB × 5 backups, UTF-8. Excluded from git (`.gitignore`). Logger namespace: `smart_aroll.*` (ffmpeg/worker/etc).

## Known layout / release notes

- v4.0.0 is the first public release. No prior v3.x artifacts should be referenced.
- GitHub Release v4.0.0 + Issues / Discussions / Projects tabs are all enabled on `juhjuhx/Davinci-AI-edit-plugins`.
- Topics set on repo: `video-editing`, `whisper`, `davinci-resolve`, `ai`, `python`, `asr`, `offline`.
