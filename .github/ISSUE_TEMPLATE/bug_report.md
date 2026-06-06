---
name: Bug Report
about: Report a bug to help us improve Smart A-Roll
title: "[BUG] "
labels: bug
assignees: ""
---

## Describe the Bug

A clear and concise description of what the bug is.

## To Reproduce

Steps to reproduce the behavior:
1. Run `launch.bat`
2. Select video `...`
3. Click on `...`
4. See error

## Expected Behavior

What did you expect to happen?

## Screenshots / Logs

Attach relevant portions of `logs/smart_aroll.log` or screenshots.

## Environment

- **OS**: Windows 10 / Windows 11 / macOS / Linux
- **Python Version**: 3.10 / 3.11 / 3.12
- **Whisper Model**: tiny / base / small / medium / large-v3
- **LLM Enabled**: Yes / No
- **GPU**: NVIDIA (model) / AMD / Intel / None
- **FFmpeg**: System PATH / bundled / not found

## Config

Paste your `config.json` (remove any sensitive paths):

```json
{
  "whisper": { "model_size": "large-v3" },
  ...
}
```

## Additional Context

Add any other context about the problem here.
