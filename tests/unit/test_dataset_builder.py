import csv
import json
import pytest
from pathlib import Path
from voicesolate.dataset_builder import DatasetBuilder

@pytest.mark.unit
class TestDatasetBuilder:
    def test_build_piper_dataset(self, temp_dir: Path, make_wav_file):
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

        # Check LJSpeech pipe-separated format: id|text|normalized_text
        with open(metadata_file, "r", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="|")
            rows = list(reader)
        assert len(rows) == 2
        assert "First test dialogue." in rows[0][1]

        # Check wav files resampled to 22.05kHz in wavs/
        wavs_dir = piper_path / "wavs"
        assert wavs_dir.exists()
        assert len(list(wavs_dir.glob("*.wav"))) == 2

    def test_build_f5tts_dataset(self, temp_dir: Path, make_wav_file):
        char_dir = temp_dir / "output" / "ep1" / "CLEMENS"
        enh_dir = char_dir / "enhanced"
        enh_dir.mkdir(parents=True, exist_ok=True)

        wav1 = make_wav_file(enh_dir / "clip1_enhanced.wav", duration_sec=3.5, sample_rate=16000)
        clips = [
            {"file": str(wav1), "enhanced_file": str(wav1), "text": "The secret of getting ahead."}
        ]

        builder = DatasetBuilder(char_dir)
        f5_path = builder.build_f5tts_dataset(clips)

        assert f5_path.exists()
        ref_audio_dir = f5_path / "ref_audio"
        assert ref_audio_dir.exists()
        assert (ref_audio_dir / "ref.wav").exists()
        assert (ref_audio_dir / "ref.txt").exists()
        assert "The secret of getting ahead." in (ref_audio_dir / "ref.txt").read_text(encoding="utf-8")

    def test_build_xtts_dataset(self, temp_dir: Path, make_wav_file):
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
        assert metadata_file.exists()
        # Check pipe-separated audio|text|speaker
        with open(metadata_file, "r", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="|")
            rows = list(reader)
        assert len(rows) == 2
        assert rows[0] == ["audio_file", "text", "speaker_name"]
        assert rows[1][2] == "CLEMENS"
