# 🧪 Voicesolate Test Suite & Testing Strategy

Voicesolate employs a tiered testing architecture designed for sub-second developer feedback, deterministic isolated audio processing, and reliable continuous integration.

---

## 🏛️ Test Architecture & Tiers

| Tier | Duration | Scope | Dependencies | Markers | Command |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Tier 1: Unit** | `< 1s` per test | Pure algorithmic logic, text parsing, key generation, boundary math | In-memory synthetic fixtures, no I/O or network | `@pytest.mark.unit` | `pytest -m unit` |
| **Tier 2: Integration** | `1 - 5s` | FastAPI endpoint contracts, dataset compilation, filesystem manifests | Isolated `tmp_path` directories, `TestClient` | `@pytest.mark.integration` | `pytest -m integration` |
| **Tier 3: E2E / Hardware** | `> 5s` | Real Demucs stem separation, Whisper STT inference, DiT flow-matching | CUDA / Apple Silicon MPS, GPU weights | `@pytest.mark.gpu`, `@pytest.mark.slow` | `pytest -m gpu` |

---

## 📂 Test Directory Layout

```text
tests/
├── conftest.py                     # Shared synthetic audio generators & script fixtures
├── pytest.ini                      # Marker configurations and runner options
├── unit/                           # Isolated unit test suites (<1s)
│   ├── test_script_parser.py       # Screenplay, SRT, and JSON script parsing
│   ├── test_cache_manager.py       # Deterministic hashing & granular bypass flags
│   ├── test_audio_extractor.py     # Local path and remote SFTP/SSH URL resolution
│   ├── test_search_aligner.py      # Subtitle anchors, timecodes, and fuzzy matching
│   └── test_dataset_builder.py     # Piper, XTTS, and F5-TTS dataset packaging
└── integration/                    # Contract & pipeline integration tests (1-5s)
    └── test_api_endpoints.py       # FastAPI REST endpoints & engine telemetry
```

---

## 🚀 Running Tests

### Fast Developer Loop (All Unit & Integration Tests)
```bash
uv run pytest
```

### Run Only Fast Unit Tests
```bash
uv run pytest -m unit
```

### Run Integration Tests
```bash
uv run pytest -m integration
```

### Run Tests with Coverage Report
```bash
uv run pytest --cov=voicesolate --cov-report=term-missing
```

---

## 🎙️ Synthetic Audio Fixtures (`make_wav_file`)

To avoid committing multi-megabyte binary audio files into git history, tests utilize the `make_wav_file` fixture in `tests/conftest.py`. It synthesizes mathematical sine waves directly into standard 16-bit PCM WAV headers in milliseconds:

```python
def test_resample(temp_dir, make_wav_file):
    # Generates a clean 1.5s 16kHz WAV in tmp_path
    wav_path = make_wav_file(temp_dir / "sample.wav", duration_sec=1.5, sample_rate=16000)
    ...
```

---

## 🤖 Continuous Integration (`.github/workflows/tests.yml`)

The unit and integration suites run automatically on every pull request and push to `main` across Linux (`ubuntu-latest`) and macOS (`macos-latest`):
1. Sets up Python 3.11 with `uv`.
2. Installs FFmpeg via OS package manager.
3. Executes `pytest -m "unit or integration"`.
