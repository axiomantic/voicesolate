# 🎙️ Voicesolate

**Voicesolate** is an automated, high-fidelity character dialogue extraction and neural audio isolation pipeline. It maps script lines directly to video/audio media (local files or over remote SSH/SFTP), segments exact character lines using whole-span Levenshtein matching and sub-second Whisper word timestamps, and isolates clean, studio-grade vocal stems using GPU neural separation.

---

## ✨ Key Features

1. **Remote Zero-Download Chunk Seeking (SSH/SFTP)**:
   - Slices and processes audio chunks on-the-fly over SSH without downloading multi-gigabyte video files.
2. **Multi-Channel Discrete Dialogue Isolation**:
   - For 5.1/7.1 surround mixes, pulls discrete Front Center (`FC`) dialogue with **zero comb filtering** and zero destructive phase arithmetic.
3. **Whole-Span Levenshtein & Context Window Alignment**:
   - Uses normalized Levenshtein ratios on candidate subtitle spans to eliminate wrong-character bleed (e.g. short words like *"Starship?"* inside another character's sentence).
   - Expands candidate zones into a context window ($[-2.0\text{s}, +1.5\text{s}]$) to capture full conversational turns.
4. **Sub-Second Whisper Word-Level Snapping**:
   - Integrates local `faster-whisper` to snap start and end timestamps directly to the exact words spoken by the character.
5. **GPU Neural Stem Separation (Meta HTDemucs)**:
   - Eliminates background music and ambient effects using Apple Silicon GPU / CUDA with test-time shift averaging (`shifts=2, overlap=0.25`).
6. **Zero-Gating Natural Dynamic Mastering**:
   - No aggressive noise gates (`agate`) or artificial energy duckers. Dialogue tails, natural room decay, and breath dynamics are preserved cleanly.
7. **Multi-Tier Persistent Caching (`CacheManager`)**:
   - Caches Whisper context window transcripts, word timestamps, and full character alignments in `cache/stt/`.
   - Repeated runs resolve in **under 0.5 seconds**!

---

## 🚀 Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/voicesolate.git
cd voicesolate

# Create environment and install
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

### Interactive TUI Mode
If you omit the `-c` argument, Voicesolate displays an interactive character selection table ranked by total lines and words spoken:
```bash
voicesolate -i "path/to/movie.mkv"
```

### Options & Flags
| Flag | Description | Default |
| :--- | :--- | :--- |
| `-i, --input` | Path or SFTP/SSH URL to video or audio file | *(Required)* |
| `-c, --character` | Target character name(s) to export | Interactive TUI |
| `-s, --script` | Script path (`.txt`, `.json`, `.srt`), or web URL | Auto-detected from video subtitles |
| `--provider` | Optional script provider (e.g. `startrek`) | `None` |
| `-o, --output-dir` | Directory where audio clips and manifest are saved | `./output` |
| `--targets` | TTS targets to prepare & train: `all`, `piper`, `xtts`, `f5` | `all` |
| `--min-duration` | Discard audio clips shorter than N seconds (pass `0` to keep all) | `5.0` (discards <= 5s) |
| `--no-train` | Prepare training-ready datasets only (skip tuning) | `False` |
| `--no-enhance` | Skip neural Demucs isolation (export discrete raw slices) | `False` |
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
    │   │   ├── metadata.csv
    │   │   └── wavs/
    │   ├── xtts/                                   # 24kHz Coqui XTTS & Chatterbox format
    │   │   ├── metadata.csv
    │   │   ├── reference_audio/
    │   │   └── wavs/
    │   └── f5tts/                                  # 24kHz F5-TTS DiT diffusion format
    │       ├── metadata.csv
    │       ├── ref_audio/ (ref.wav & ref.txt)
    │       └── wavs/
    └── models/                                     # Output voice models & configs
        ├── piper/                                  # Piper VITS model & voice.json
        ├── xtts/                                   # XTTS speaker profile & latents
        └── f5tts/                                  # F5-TTS reference prompt pack & DiT weights
```

`manifest.json` provides comprehensive episode-level metadata with script dialogue, exact timecodes, and transcription confidence scores.
