# 🎙️ Voicesolate

**Voicesolate** is an automated, end-to-end character voice extraction, neural isolation, and TTS model training pipeline. It maps script lines directly to video or audio media (local files or over zero-download remote SSH/SFTP), segments exact character lines using whole-span Levenshtein matching and sub-second Whisper word timestamps, isolates studio-grade vocal stems using GPU neural separation, aggregates character corpora across episodes, and **automatically packages ready-to-train datasets and fine-tunes voice models for Piper (VITS/ONNX), Coqui XTTS-v2 / Chatterbox, and F5-TTS**.

---

## ✨ Key Features

1. **End-to-End "Media-to-Model" Pipeline**:
   - Single command extracts character speech, masters vocal stems, builds LJSpeech/TTS datasets, configures voice profiles, and prepares models for inference.
2. **Multi-Target TTS Support (Piper • XTTS • F5-TTS)**:
   - **F5-TTS**: Formats 24kHz diffusion dataset + optimal prompt audio/text for flow-matching DiT cloning with live in-browser synthesis.
   - **Piper (VITS / ONNX)**: Generates 22,050 Hz LJSpeech dataset + `metadata.csv` for ultra-low latency CPU deployment (e.g. Home Assistant / Raspberry Pi).
   - **Coqui XTTS-v2 / Chatterbox**: Formats 24kHz dataset + auto-selects pristine reference prompt packs for immediate zero-shot or fine-tuned cloning.
3. **Interactive Voice Studio Web UI**:
   - Modern browser-based audition studio (`http://localhost:7860`) to test models, run real-time neural synthesis, tweak speech parameters (speed, seed), and compare original actor reference audio against AI synthesis side-by-side.
4. **Multi-Episode Dataset Aggregation**:
   - Automatically scans and aggregates all isolated dialogue clips for a character across multiple episodes into a unified, high-substance training corpus.
5. **Remote Zero-Download Chunk Seeking (SSH/SFTP)**:
   - Slices and processes audio chunks on-the-fly over SSH without downloading multi-gigabyte video files.
6. **Multi-Channel Discrete Dialogue Isolation**:
   - For 5.1/7.1 surround mixes, pulls discrete Front Center (`FC`) dialogue with **zero comb filtering** and zero destructive phase arithmetic.
7. **Sub-Second Whisper Word Snapping & Energy Valley Detection**:
   - Snaps start/end boundaries directly to spoken words and silence gaps, eliminating adjacent speaker bleed.
8. **GPU Neural Stem Separation (Meta HTDemucs)**:
   - Eliminates background music and ambient sound effects using Apple Silicon GPU / CUDA with test-time shift averaging (`shifts=2, overlap=0.25`).
9. **Re-entrant, Idempotent & Granular Caching (`CacheManager`)**:
   - Caches Whisper STT, alignments, audio slices, and neural stems. Failed or resumed runs re-use existing files instantly. Includes granular bypass flags (`--no-cache-stt`, `--no-cache-align`, `--no-cache-audio`, `--no-cache-enhance`, `--no-cache-script`).

---

## ⚙️ System Requirements

- **Python**: `>=3.10`
- **FFmpeg & FFprobe**: Required for audio extraction, chunk slicing, and stem alignment.
  - **macOS (Homebrew)**: `brew install ffmpeg`
  - **Debian / Ubuntu**: `sudo apt-get update && sudo apt-get install -y ffmpeg`
  - **Arch Linux**: `sudo pacman -S ffmpeg`
  - **Windows (Winget)**: `winget install Gyan.FFmpeg`
  - [Official Download Guide](https://ffmpeg.org/download.html)

---

## 🚀 Installation

```bash
# Clone the repository
git clone https://github.com/axiomantic/voicesolate.git
cd voicesolate

# Create virtual environment and install all dependencies
uv venv
source .venv/bin/activate
uv pip install -e .
```

---

## 💻 CLI Usage

### Basic Extraction (Local File with Screenplay or Subtitles)
```bash
voicesolate -i "path/to/movie.mkv" -s "path/to/screenplay.txt" -c "NEO"
```

### Remote Zero-Download Extraction (Over SSH / SFTP)
```bash
voicesolate \
  -i "sftp://nas.lan/mnt/media/film.mkv" \
  -c "CHARACTER"
```

### Options & Flags
| Flag | Description | Default |
| :--- | :--- | :--- |
| `-i, --input` | Path or SFTP/SSH URL to video or audio file | *(Required)* |
| `-c, --character` | Target character name(s) to export | Interactive prompt |
| `-s, --script` | Script path (`.txt`, `.json`, `.srt`), or web URL | Auto-detected from subtitles |
| `--provider` | Optional script provider (e.g. `startrek`) | `None` |
| `-o, --output-dir` | Directory where audio clips and manifest are saved | `./output` |
| `--targets` | TTS targets to prepare & train: `all`, `piper`, `xtts`, `f5` | `all` |
| `--min-duration` | Discard audio clips shorter than N seconds (pass `0` to keep all) | `5.0` (discards <= 5s) |
| `--no-train` | Prepare training-ready datasets only (skip tuning) | `False` |
| `--no-web-ui` | Skip launching the Voice Studio Web UI at the end | `False` |
| `--no-aggregate` | Do not aggregate clips from other episodes in output directory | `False` (aggregates all) |
| `--no-enhance` | Skip neural Demucs isolation (export discrete raw slices) | `False` |
| `--no-cache-stt` | Bypass Whisper STT cache (re-transcribes context windows) | `False` |
| `--no-cache-align` | Bypass alignment cache (re-runs character sequence matching) | `False` |
| `--no-cache-audio` | Force re-slicing raw discrete audio from media source | `False` |
| `--no-cache-enhance`| Force re-running GPU Demucs isolation on slices | `False` |
| `--no-cache-script` | Bypass cached script JSON and re-fetch / re-parse | `False` |
| `--wyoming-host` | Optional Wyoming Whisper STT IP | `10.0.2.141` |
| `--wyoming-port` | Optional Wyoming Whisper STT Port | `10300` |

---

## 📁 Output Structure

Voicesolate automatically produces an end-to-end training and inference layout:
```text
output/<media_name>/
├── manifest.json
└── <CHARACTER>/
    ├── raw/                                        # Discrete dialogue slices
    │   └── 00_00_35_915-00_00_36_855.wav
    ├── enhanced/                                   # Clean neural vocal masters
    │   └── 00_00_35_915-00_00_36_855_enhanced.wav
    ├── datasets/
    │   ├── piper/                                  # 22.05kHz mono LJSpeech dataset
    │   │   ├── metadata.csv                        # id|text|normalized_text
    │   │   ├── dataset.json
    │   │   └── wavs/
    │   ├── xtts/                                   # 24kHz Coqui XTTS-v2 dataset
    │   │   ├── metadata.csv                        # audio_path|text|speaker_name
    │   │   ├── reference_audio/                    # Curated 6-12s speaker audio prompts
    │   │   └── wavs/
    │   └── f5tts/                                  # 24kHz F5-TTS DiT dataset
    │       ├── metadata.csv                        # audio_path|transcript
    │       ├── ref_audio/                          # Optimal reference voice prompt (WAV + TXT)
    │       └── wavs/
    └── models/
        ├── piper/                                  # voice.json & .onnx model package
        ├── xtts/                                   # speaker_profile.json
        └── f5tts/                                  # f5_profile.json
```
