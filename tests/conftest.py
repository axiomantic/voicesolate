import os
import math
import struct
import wave
import pytest
from pathlib import Path
from typing import List, Dict, Any

@pytest.fixture
def temp_dir(tmp_path: Path) -> Path:
    """Provides an isolated temporary directory for test outputs."""
    return tmp_path

@pytest.fixture
def make_wav_file():
    """
    Factory fixture to generate clean synthetic PCM WAV files in-memory
    without needing external media files.
    """
    def _generator(path: Path, duration_sec: float = 1.0, sample_rate: int = 16000, num_channels: int = 1) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        total_samples = int(duration_sec * sample_rate)
        # Generate 440 Hz sine wave tone
        samples = []
        for i in range(total_samples):
            val = int(32767.0 * 0.5 * math.sin(2.0 * math.pi * 440.0 * (i / sample_rate)))
            samples.append(val)

        with wave.open(str(path), "wb") as wf:
            wf.setnchannels(num_channels)
            wf.setsampwidth(2) # 16-bit
            wf.setframerate(sample_rate)
            # Pack frames
            if num_channels == 1:
                data = struct.pack(f"<{len(samples)}h", *samples)
            else:
                interleaved = []
                for s in samples:
                    for _ in range(num_channels):
                        interleaved.append(s)
                data = struct.pack(f"<{len(interleaved)}h", *interleaved)
            wf.writeframes(data)
        return path

    return _generator

@pytest.fixture
def sample_script_text() -> str:
    return """
DATA
Captain, sensors are detecting unusual spatial distortions.

PICARD
Can you identify the source, Mister Data?

DATA
It appears to be a temporal rift originating from late nineteenth century Earth.

PICARD
(thoughtfully)
Nineteenth century. We must proceed with extreme caution.
"""

@pytest.fixture
def sample_srt_content() -> str:
    return """1
00:00:01,500 --> 00:00:04,200
Captain, sensors are detecting unusual spatial distortions.

2
00:00:05,100 --> 00:00:07,800
Can you identify the source, Mister Data?

3
00:00:08,500 --> 00:00:12,000
It appears to be a temporal rift originating from late nineteenth century Earth.
"""

@pytest.fixture
def sample_whisper_segments() -> List[Dict[str, Any]]:
    return [
        {
            "start": 1.5,
            "end": 4.2,
            "text": " Captain, sensors are detecting unusual spatial distortions.",
            "words": [
                {"word": "Captain,", "start": 1.52, "end": 1.95, "probability": 0.98},
                {"word": "sensors", "start": 2.10, "end": 2.45, "probability": 0.99},
                {"word": "are", "start": 2.48, "end": 2.62, "probability": 0.97},
                {"word": "detecting", "start": 2.65, "end": 3.10, "probability": 0.99},
                {"word": "unusual", "start": 3.15, "end": 3.60, "probability": 0.96},
                {"word": "spatial", "start": 3.65, "end": 4.00, "probability": 0.98},
                {"word": "distortions.", "start": 4.02, "end": 4.20, "probability": 0.99},
            ]
        },
        {
            "start": 5.1,
            "end": 7.8,
            "text": " Can you identify the source, Mister Data?",
            "words": [
                {"word": "Can", "start": 5.10, "end": 5.30, "probability": 0.95},
                {"word": "you", "start": 5.32, "end": 5.45, "probability": 0.98},
                {"word": "identify", "start": 5.50, "end": 6.10, "probability": 0.99},
                {"word": "the", "start": 6.12, "end": 6.22, "probability": 0.96},
                {"word": "source,", "start": 6.25, "end": 6.75, "probability": 0.99},
                {"word": "Mister", "start": 6.85, "end": 7.20, "probability": 0.98},
                {"word": "Data?", "start": 7.25, "end": 7.80, "probability": 0.99},
            ]
        }
    ]
