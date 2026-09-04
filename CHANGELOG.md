# Changelog
All notable changes to **Voicesolate** will be documented in this file.

## [0.3.1] - 2026-09-03

### Added
- **Modern Voice Studio Web UI (Gradio)**:
  - Replaced terminal TUI audition suite with a clean, browser-based Voice Studio GUI (`http://localhost:7860`).
  - Real-time in-process neural voice synthesis with **F5-TTS** (flow-matching DiT) running on Apple Silicon / CUDA.
  - Side-by-side audio comparison: Actor Original Reference Voice prompt vs. AI Cloned Speech.
  - Live engine status detection badges (`🟢 Ready`, `🟡 Dataset Ready`, `🔴 Not Installed`).
  - Built-in one-click engine installer directly from the browser.
- **Multi-Episode Dataset Aggregation**:
  - Automatically discovers and merges isolated dialogue clips for a character across multiple episodes in `output/` into a single, high-substance training corpus.
- **Explicit System Dependency Validation**:
  - Loud, clear error panels if system binaries (`ffmpeg`, `ffprobe`) are missing, complete with copy-paste install commands for macOS (Homebrew), Debian/Ubuntu, Arch Linux, and Windows.
- **Package Dependencies**:
  - Added `f5-tts>=0.1.0` and `piper-tts>=1.2.0` directly to `pyproject.toml` dependencies.

### Removed
- Removed terminal TUI (`interactive_tester.py`) and all silent fallbacks in favor of the Web UI and loud, informative error reporting.
- Removed `--tui` CLI argument; added `--no-web-ui` flag for headless pipelines.

## [0.3.0] - 2026-09-03

### Added
- **End-to-End TTS Dataset Generation & Model Packaging (`DatasetBuilder` & `ModelTrainer`)**:
  - Automatically formats training datasets for **Piper (VITS / ONNX)**, **Coqui XTTS-v2 / Chatterbox**, and **F5-TTS (Flow-Matching DiT)**.
  - Automatically generates 22,050 Hz LJSpeech-formatted audio with pipe-separated `metadata.csv`.
  - Automatically generates 24,000 Hz XTTS / F5-TTS datasets with optimal reference prompt audio selection (`ref_audio/ref.wav` and `ref.txt`).
- **Idempotent & Re-entrant Execution**:
  - Full run safely re-uses existing audio slices and enhanced files on disk, picking up immediately if interrupted.
- **Granular Multi-Tier Cache Bypass Flags**:
  - Added `--no-cache-stt`, `--no-cache-align`, `--no-cache-audio`, `--no-cache-enhance`, and `--no-cache-script` to allow surgical re-processing without losing valid cache layers.
- **CLI Options**:
  - Added `--targets` (`all`, `piper`, `xtts`, `f5`).
  - Added `--no-train` (dataset preparation only).
  - Added `--no-web-ui` (headless mode).
  - Added `--min-duration` (default: 5.0s).

### Changed
- Refactored output directory structure to cleanly isolate `<CHARACTER>/raw/`, `<CHARACTER>/enhanced/`, `<CHARACTER>/datasets/`, and `<CHARACTER>/models/`.
- Made `ScriptParser` completely franchise-agnostic with pluggable providers (`--provider startrek`).
- Implemented Acoustic Energy Valley Snapping and strict final-word matching to eliminate trailing speaker bleed.

## [0.2.0] - 2026-09-03
- Initial public release of `Voicesolate`.
- Remote zero-download chunk seeking over SSH/SFTP.
- Whisper sub-second word-level timestamp snapping.
- Discrete Front Center (`FC`) surround channel extraction.
- Neural Demucs vocal stem isolation with shift-averaging.
