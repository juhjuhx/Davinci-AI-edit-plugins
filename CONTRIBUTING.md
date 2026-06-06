# Contributing

## Development Setup

```bash
# Clone
git clone https://github.com/juhjuhx/Davinci-AI-edit-plugins.git
cd Davinci-AI-edit-plugins

# Create environment (conda recommended)
conda create -n smart_aroll python=3.10
conda activate smart_aroll

# Install dependencies
pip install -r requirements.txt

# Optional: LLM support (CUDA)
pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu

# Optional: dev tools
pip install pytest mypy ruff
```

## Code Style

- **Python 3.10+** — uses `from __future__ import annotations` and `dataclass` patterns
- **Line length**: 120 characters
- **Formatting**: double-quoted strings, 4-space indentation
- **Linting**: `ruff` (config in `pyproject.toml`)
- **Type hints**: recommended for public APIs, optional for internals

Run linting before submitting:

```bash
ruff check .
mypy core/ --ignore-missing-imports
```

## Commit Convention

Use [Angular Commit Convention](https://www.conventionalcommits.org/):

```
feat: add razor split frame alignment
fix: handle empty energy profile gracefully
refactor: extract _fix_dll_path to core.utils
docs: update README with FFmpeg setup guide
```

## Pull Request Process

1. Fork the repo and create a feature branch from `main`
2. Make your changes with clear commit messages
3. Run `ruff check .` to ensure code quality
4. If adding features, include a brief usage example in the PR description
5. PRs require at least one review before merging

## Project Structure

```
SmartARoll/
├── core/           # Analysis engine, config, models, FFmpeg, exporters
├── workers/        # Background task queue
├── api/            # Flask HTTP server
├── ui/             # Web frontend (single HTML file)
├── config.json     # User settings (Whisper, LLM, FFmpeg, etc.)
├── launch.bat      # One-click launcher
├── build.bat       # PyInstaller packaging
└── apply_to_resolve.py  # DaVinci Console integration script
```

## Testing

Tests are located in `tests/`. Run with:

```bash
pytest
```

## Reporting Issues

Use the issue templates — include your `config.json`, `logs/smart_aroll.log`, and the Whisper model size you're using.
