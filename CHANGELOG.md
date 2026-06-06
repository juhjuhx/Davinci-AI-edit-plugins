# Changelog

## v4.0.0 (2025-07)

### Initial Release

- **Whisper Transcription** — supports tiny to large-v3 models with CUDA/Metal/Vulkan GPU acceleration
- **Rules Engine** — auto-detects filler words, repeats, self-corrections, throat clearing, and editing cues (clap/cut)
- **Optional LLM Verification** — Qwen2.5-0.5B GGUF integration via llama-cpp-python for secondary confirmation
- **Silence Detection** — librosa-based RMS energy thresholding for pause/stop detection
- **Web UI** — single-page application with waveform timeline, draggable panels, razor split, keyboard shortcuts
- **DaVinci Resolve Integration** — clip coloring (green/red) + timeline markers via `apply_to_resolve.py`
- **FFmpeg Export** — stream copy (lossless), H265 10bit 422, H264, with NVENC GPU acceleration
- **Audio Enhancement** — highpass/lowpass filtering, loudness normalization (loudnorm), denoise, dynamic range compression
- **EDL / CSV / SRT / JSON Export** — multiple interchange formats for external editing workflows
- **Background Worker** — single-thread queue with SSE progress streaming to frontend
- **Edge-TTS Support** — placeholder silence generation for cut segments
- **Windows Installer** — PyInstaller packaging with NSIS setup builder
