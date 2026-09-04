# Changelog
All notable changes to **Voicesolate** will be documented in this file.

## [0.3.0] - 2026-09-03

### Added
- **End-to-End TTS Dataset Generation & Model Packaging (`DatasetBuilder` & `ModelTrainer`)**:
  - Automatically formats training datasets for **Piper (VITS / ONNX)**, **Coqui XTTS-v2 / Chatterbox**, and **F5-TTS (Flow-Matching DiT)**.
  - Automatically generates 22,050 Hz LJSpeech-formatted audio with pipe-separated `metadata.csv`.
  - Automatically generates 24,000 Hz XTTS / F5-TTS datasets with optimal reference prompt audio selection (`ref_audio/ref.wav` and `ref.txt`).
- **Interactive Voice Model Audition TUI (`InteractiveTester`)**:
  - Terminal-based audition suite enabling live playback, model switching (Piper, XTTS, F5-TTS), and real-time speech parameter tuning (speed, temperature, NFE steps).
- **Idempotent & Re-entrant Execution**:
  - Full run now safely re-uses existing audio slices and enhanced files on disk, picking up immediately if interrupted.
- **Granular Multi-Tier Cache Bypass Flags**:
  - Added `--no-cache-stt`, `--no-cache-align`, `--no-cache-audio`, `--no-cache-enhance`, and `--no-cache-script` to allow surgical re-processing without losing valid cache layers.
- **CLI Options**:
  - Added `--targets` (`all`, `piper`, `xtts`, `f5`).
  - Added `--no-train` (dataset preparation only).
  - Added `--no-interactive` (headless mode).
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
