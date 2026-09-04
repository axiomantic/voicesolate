import csv
import json
import pytest
import soundfile as sf
import numpy as np
from pathlib import Path
from voicesolate.dataset_builder import DatasetBuilder

@pytest.mark.unit
class TestDatasetBuilder:
    def test_build_piper_dataset_consumes_and_validates_audio(self, temp_dir: Path, make_wav_file):
        char_dir = temp_dir / "output" / "ep1" / "CLEMENS"
        enh_dir = char_dir / "enhanced"
        enh_dir.mkdir(parents=True, exist_ok=True)

        wav1 = make_wav_file(enh_dir / "clip1_enhanced.wav", duration_sec=1.5, sample_rate=16000)
        wav2 = make_wav_file(enh_dir / "clip2_enhanced.wav", duration_sec=2.0, sample_rate=16000)

        clips = [
            {"file": str(wav1), "enhanced_file": str(wav1), "text": "First test dialogue."},
            {"file": str(wav2), "enhanced_file": str(wav2), "text": "Second test dialogue."}
        ]

        builder = DatasetBuilder(char_dir)
        piper_path = builder.build_piper_ljspeech(clips)

        assert piper_path.exists()
        metadata_file = piper_path / "metadata.csv"
        assert metadata_file.exists()

        # 1. Full CSV Consumption: Verify every single cell and column
        with open(metadata_file, "r", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="|")
            rows = list(reader)
        assert len(rows) == 2
        for row in rows:
            assert len(row) == 3, "LJSpeech must have exactly id|text|normalized_text"
            assert row[1] == row[2], "Default normalization should match text"
        assert rows[0][1] == "First test dialogue."
        assert rows[1][1] == "Second test dialogue."

        # 2. Audio Binary Consumption: Read resampled audio files with soundfile
        wavs_dir = piper_path / "wavs"
        wav_files = sorted(list(wavs_dir.glob("*.wav")))
        assert len(wav_files) == 2

        for wf in wav_files:
            info = sf.info(str(wf))
            # Must strictly be 22,050 Hz mono 16-bit PCM for Piper VITS
            assert info.samplerate == 22050, f"Expected 22050Hz, got {info.samplerate}Hz"
            assert info.channels == 1, f"Expected mono, got {info.channels} channels"
            assert info.subtype == "PCM_16"
            assert info.duration > 1.0

            # Verify audio is non-silent and normalized (not clipped)
            data, sr = sf.read(str(wf))
            assert np.max(np.abs(data)) <= 0.95, "Audio must not exceed normalized ceiling"
            assert np.sqrt(np.mean(data**2)) > 0.01, "Audio must not be silent zeros"

    def test_build_f5tts_dataset_consumes_and_validates_audio(self, temp_dir: Path, make_wav_file):
        char_dir = temp_dir / "output" / "ep1" / "CLEMENS"
        enh_dir = char_dir / "enhanced"
        enh_dir.mkdir(parents=True, exist_ok=True)

        wav1 = make_wav_file(enh_dir / "clip1_enhanced.wav", duration_sec=8.5, sample_rate=16000)
        clips = [
            {"file": str(wav1), "enhanced_file": str(wav1), "text": "The secret of getting ahead."}
        ]

        builder = DatasetBuilder(char_dir)
        f5_path = builder.build_f5tts_dataset(clips)

        assert f5_path.exists()
        metadata_file = f5_path / "metadata.csv"
        assert metadata_file.exists()

        # 1. Full CSV Consumption
        with open(metadata_file, "r", encoding="utf-8") as f:
            lines = [line.strip().split("|") for line in f if line.strip()]
        assert len(lines) == 1
        target_path, transcript = lines[0]
        assert Path(target_path).exists()
        assert transcript == "The secret of getting ahead."

        # 2. Audio Binary Consumption
        info = sf.info(target_path)
        assert info.samplerate == 24000, "F5-TTS requires 24kHz audio"
        assert info.channels == 1

        # 3. Reference prompt pack validation
        ref_dir = f5_path / "ref_audio"
        ref_wav = ref_dir / "ref.wav"
        ref_txt = ref_dir / "ref.txt"
        assert ref_wav.exists()
        assert ref_txt.exists()
        assert ref_txt.read_text(encoding="utf-8").strip() == "The secret of getting ahead."

        ref_info = sf.info(str(ref_wav))
        assert ref_info.samplerate == 24000
        assert ref_info.channels == 1

    def test_build_xtts_dataset_consumes_and_validates_audio(self, temp_dir: Path, make_wav_file):
        char_dir = temp_dir / "output" / "ep1" / "CLEMENS"
        enh_dir = char_dir / "enhanced"
        enh_dir.mkdir(parents=True, exist_ok=True)

        wav1 = make_wav_file(enh_dir / "clip1_enhanced.wav", duration_sec=2.5, sample_rate=16000)
        clips = [
            {"file": str(wav1), "enhanced_file": str(wav1), "text": "Always do right."}
        ]

        builder = DatasetBuilder(char_dir)
        xtts_path = builder.build_xtts_dataset(clips)

        assert xtts_path.exists()
        metadata_file = xtts_path / "metadata.csv"

        with open(metadata_file, "r", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="|")
            rows = list(reader)

        assert len(rows) == 2
        assert rows[0] == ["audio_file", "text", "speaker_name"]
        rel_path, text, speaker = rows[1]
        assert text == "Always do right."
        assert speaker == "CLEMENS"

        target_wav = xtts_path / rel_path
        assert target_wav.exists()
        info = sf.info(str(target_wav))
        assert info.samplerate == 24000
        assert info.channels == 1

    def test_negative_control_missing_audio_files_handled_gracefully(self, temp_dir: Path):
        """Negative control: clips pointing to nonexistent files should not crash builder."""
        char_dir = temp_dir / "output" / "ep1" / "CLEMENS"
        char_dir.mkdir(parents=True, exist_ok=True)

        clips = [
            {"file": "/nonexistent/path/clip999.wav", "enhanced_file": None, "text": "Ghost speech."}
        ]

        builder = DatasetBuilder(char_dir)
        piper_path = builder.build_piper_ljspeech(clips)
        assert piper_path.exists()
        # Should generate valid metadata with 0 audio rows rather than crashing or writing corrupted files
        with open(piper_path / "metadata.csv", "r", encoding="utf-8") as f:
            rows = [r for r in csv.reader(f, delimiter="|") if r]
        assert len(rows) == 0
