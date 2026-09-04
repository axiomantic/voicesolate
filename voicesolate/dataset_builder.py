import os
import csv
import json
import shutil
from pathlib import Path
from typing import List, Dict, Any, Optional
import soundfile as sf
import numpy as np

class DatasetBuilder:
    """
    Builds training-ready datasets from Voicesolate enhanced audio files.
    Supports:
    1. Piper (VITS): LJSpeech format, 22,050 Hz 16-bit mono, metadata.csv (pipe-separated)
    2. Coqui XTTS / Chatterbox: 24,000 Hz mono, metadata.csv, speaker latents / reference packs
    3. F5-TTS: 24,000 Hz mono, metadata.csv, ref_audio pack for zero-shot and fine-tuning
    """

    def __init__(self, target_char_dir: Path):
        self.char_dir = Path(target_char_dir)
        self.char_name = self.char_dir.name
        self.enhanced_dir = self.char_dir / "enhanced"
        self.raw_dir = self.char_dir / "raw"
        self.datasets_dir = self.char_dir / "datasets"
        self.models_dir = self.char_dir / "models"

    def _resample_audio(self, input_path: Path, output_path: Path, target_sr: int):
        """Reads WAV and writes to target sample rate, mono 16-bit PCM."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        data, sr = sf.read(str(input_path))
        if data.ndim > 1:
            data = np.mean(data, axis=1)

        if sr != target_sr:
            import torchaudio.transforms as T
            import torch
            tensor = torch.from_numpy(data).float().unsqueeze(0)
            resampler = T.Resample(orig_freq=sr, new_freq=target_sr)
            data = resampler(tensor).squeeze(0).numpy()

        # Peak normalize to -1.0 dB to prevent clipping
        max_val = np.max(np.abs(data))
        if max_val > 0:
            data = data / max_val * 0.90

        sf.write(str(output_path), data, target_sr, subtype="PCM_16")

    def _resolve_clip_file(self, clip: Dict[str, Any]) -> Optional[Path]:
        """Resolves the audio file path across current and other episode directories."""
        candidate = Path(clip.get("enhanced_file") or clip.get("file") or "")
        if candidate.exists():
            return candidate

        # Check in self.enhanced_dir
        if (self.enhanced_dir / candidate.name).exists():
            return self.enhanced_dir / candidate.name

        # Check in self.raw_dir
        if (self.raw_dir / candidate.name).exists():
            return self.raw_dir / candidate.name

        # Search across other episode outputs under output/
        output_root = self.char_dir.parent.parent
        if output_root.exists():
            for ep_dir in output_root.iterdir():
                if ep_dir.is_dir():
                    other_enh = ep_dir / self.char_name / "enhanced" / candidate.name
                    if other_enh.exists():
                        return other_enh
                    other_raw = ep_dir / self.char_name / "raw" / candidate.name
                    if other_raw.exists():
                        return other_raw

        return None

    @classmethod
    def aggregate_all_clips_for_character(cls, char_name: str, output_root: Path, current_clips: Optional[List[Dict[str, Any]]] = None) -> List[Dict[str, Any]]:
        """
        Gathers all valid clips for this character across every episode directory in output_root.
        Prevents duplicates by checking audio content filenames.
        """
        all_clips = []
        seen_stems = set()

        # 1. Include current clips first
        if current_clips:
            for c in current_clips:
                if c.get("character") == char_name:
                    p = Path(c.get("enhanced_file") or c.get("file") or "")
                    stem = p.stem.replace("_enhanced", "")
                    if stem not in seen_stems:
                        seen_stems.add(stem)
                        all_clips.append(dict(c))

        # 2. Discover clips from other episode manifests
        if output_root.exists():
            for manifest_file in sorted(output_root.glob("*/manifest.json")):
                try:
                    with open(manifest_file, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    for c in data.get("clips", []):
                        if c.get("character") == char_name:
                            p = Path(c.get("enhanced_file") or c.get("file") or "")
                            stem = p.stem.replace("_enhanced", "")
                            if stem not in seen_stems:
                                # Verify file actually exists
                                enh_path = manifest_file.parent / char_name / "enhanced" / p.name
                                raw_path = manifest_file.parent / char_name / "raw" / p.name
                                resolved = None
                                if p.exists():
                                    resolved = p
                                elif enh_path.exists():
                                    resolved = enh_path
                                elif raw_path.exists():
                                    resolved = raw_path

                                if resolved and resolved.exists():
                                    c_copy = dict(c)
                                    c_copy["enhanced_file"] = str(resolved)
                                    seen_stems.add(stem)
                                    all_clips.append(c_copy)
                except Exception:
                    continue

        return all_clips

    # Aliases for backwards compatibility and ergonomic pipeline invocation
    aggregate_character_clips = aggregate_all_clips_for_character

    def build_piper_ljspeech(self, clips: List[Dict[str, Any]]) -> Path:
        """
        Builds LJSpeech-compatible dataset for Piper VITS training.
        Format:
        datasets/piper/
        ├── metadata.csv (id|text|normalized_text)
        └── wavs/
            └── <id>.wav (22,050 Hz mono 16-bit PCM)
        """
        piper_dir = self.datasets_dir / "piper"
        wavs_dir = piper_dir / "wavs"
        if wavs_dir.exists():
            shutil.rmtree(wavs_dir)
        wavs_dir.mkdir(parents=True, exist_ok=True)

        metadata_rows = []
        for i, clip in enumerate(clips):
            source_file = self._resolve_clip_file(clip)
            if not source_file:
                continue

            clip_id = f"clip_{i:04d}_{source_file.stem.replace('_enhanced', '')}"
            target_wav = wavs_dir / f"{clip_id}.wav"
            self._resample_audio(source_file, target_wav, target_sr=22050)

            clean_text = clip["text"].strip().replace("|", " ")
            metadata_rows.append(f"{clip_id}|{clean_text}|{clean_text}")

        metadata_csv = piper_dir / "metadata.csv"
        with open(metadata_csv, "w", encoding="utf-8") as f:
            f.write("\n".join(metadata_rows) + "\n")

        # Save dataset configuration
        dataset_info = {
            "dataset_format": "ljspeech",
            "sample_rate": 22050,
            "channels": 1,
            "num_clips": len(metadata_rows),
            "target": "piper"
        }
        with open(piper_dir / "dataset.json", "w", encoding="utf-8") as f:
            json.dump(dataset_info, f, indent=2)

        return piper_dir

    def build_xtts_dataset(self, clips: List[Dict[str, Any]]) -> Path:
        """
        Builds dataset for Coqui XTTS-v2 & Chatterbox.
        Format:
        datasets/xtts/
        ├── metadata.csv (audio_file|text|speaker_name)
        └── wavs/ (24,000 Hz mono 16-bit PCM)
        """
        xtts_dir = self.datasets_dir / "xtts"
        wavs_dir = xtts_dir / "wavs"
        if wavs_dir.exists():
            shutil.rmtree(wavs_dir)
        wavs_dir.mkdir(parents=True, exist_ok=True)

        metadata_rows = [["audio_file", "text", "speaker_name"]]
        character_name = self.char_name

        for i, clip in enumerate(clips):
            source_file = self._resolve_clip_file(clip)
            if not source_file:
                continue

            clip_id = f"xtts_{i:04d}_{source_file.stem.replace('_enhanced', '')}"
            target_wav = wavs_dir / f"{clip_id}.wav"
            self._resample_audio(source_file, target_wav, target_sr=24000)

            clean_text = clip["text"].strip()
            metadata_rows.append([f"wavs/{clip_id}.wav", clean_text, character_name])

        metadata_csv = xtts_dir / "metadata.csv"
        with open(metadata_csv, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f, delimiter="|")
            writer.writerows(metadata_rows)

        # Select the top 3 cleanest 6-12s clips as speaker reference audio
        reference_dir = xtts_dir / "reference_audio"
        reference_dir.mkdir(exist_ok=True)
        ref_candidates = []
        for f in wavs_dir.glob("*.wav"):
            dur = sf.info(str(f)).duration
            if 6.0 <= dur <= 14.0:
                ref_candidates.append((f, dur))
        ref_candidates.sort(key=lambda x: abs(x[1] - 9.0))
        for ref_wav, _ in ref_candidates[:3]:
            shutil.copy(str(ref_wav), str(reference_dir / ref_wav.name))

        return xtts_dir

    def build_f5tts_dataset(self, clips: List[Dict[str, Any]]) -> Path:
        """
        Builds dataset for F5-TTS (Flow-Matching Non-Autoregressive DiT).
        Format:
        datasets/f5tts/
        ├── metadata.csv (audio_path|transcript)
        ├── wavs/ (24,000 Hz mono 16-bit PCM)
        └── ref_audio/ (Optimal reference voice prompt)
        """
        f5_dir = self.datasets_dir / "f5tts"
        wavs_dir = f5_dir / "wavs"
        if wavs_dir.exists():
            shutil.rmtree(wavs_dir)
        wavs_dir.mkdir(parents=True, exist_ok=True)

        metadata_rows = []
        for i, clip in enumerate(clips):
            source_file = self._resolve_clip_file(clip)
            if not source_file:
                continue

            clip_id = f"f5_{i:04d}_{source_file.stem.replace('_enhanced', '')}"
            target_wav = wavs_dir / f"{clip_id}.wav"
            self._resample_audio(source_file, target_wav, target_sr=24000)

            clean_text = clip["text"].strip().replace("|", " ")
            metadata_rows.append(f"{target_wav.resolve()}|{clean_text}")

        metadata_csv = f5_dir / "metadata.csv"
        with open(metadata_csv, "w", encoding="utf-8") as f:
            f.write("\n".join(metadata_rows) + "\n")

        # Pick best single reference prompt (~8-12 seconds)
        ref_dir = f5_dir / "ref_audio"
        ref_dir.mkdir(exist_ok=True)
        best_ref = None
        best_diff = float("inf")
        best_clip_text = ""
        for i, clip in enumerate(clips):
            source_file = self._resolve_clip_file(clip)
            if source_file and source_file.exists():
                dur = sf.info(str(source_file)).duration
                if 7.0 <= dur <= 14.0 and abs(dur - 9.5) < best_diff:
                    best_diff = abs(dur - 9.5)
                    best_ref = wavs_dir / f"f5_{i:04d}_{source_file.stem.replace('_enhanced', '')}.wav"
                    best_clip_text = clip["text"].strip()

        # Fall back to longest available clip if no clip in 7-14s window
        if not best_ref and clips:
            longest_dur = -1.0
            for i, clip in enumerate(clips):
                source_file = self._resolve_clip_file(clip)
                if source_file and source_file.exists():
                    dur = sf.info(str(source_file)).duration
                    if dur > longest_dur:
                        longest_dur = dur
                        best_ref = wavs_dir / f"f5_{i:04d}_{source_file.stem.replace('_enhanced', '')}.wav"
                        best_clip_text = clip["text"].strip()

        if best_ref and best_ref.exists():
            shutil.copy(str(best_ref), str(ref_dir / "ref.wav"))
            with open(ref_dir / "ref.txt", "w", encoding="utf-8") as f:
                f.write(best_clip_text + "\n")

        return f5_dir

    def build_all(self, *args, **kwargs) -> Dict[str, Path]:
        """
        Builds all requested target datasets.
        Accepts:
          build_all(clips, targets=None, aggregate_all=True)
          OR
          build_all(char_name, clips, targets=None, aggregate_all=True)
        """
        # Polymorphic argument resolution
        clips = []
        targets = kwargs.get("targets", None)
        aggregate_all = kwargs.get("aggregate_all", True)

        if len(args) >= 2 and isinstance(args[0], str):
            # Form: build_all(char_name, clips, ...)
            clips = args[1]
            if len(args) >= 3 and targets is None:
                targets = args[2]
            if len(args) >= 4 and "aggregate_all" not in kwargs:
                aggregate_all = args[3]
        elif len(args) >= 1:
            # Form: build_all(clips, ...)
            clips = args[0]
            if len(args) >= 2 and targets is None:
                targets = args[1]
            if len(args) >= 3 and "aggregate_all" not in kwargs:
                aggregate_all = args[2]
        else:
            clips = kwargs.get("clips", [])

        final_clips = clips
        if aggregate_all:
            output_root = self.char_dir.parent.parent
            final_clips = self.aggregate_all_clips_for_character(self.char_name, output_root, current_clips=clips)

        selected_targets = [t.lower().strip() for t in (targets or ["all"])]
        do_all = "all" in selected_targets

        results = {}
        if do_all or "piper" in selected_targets or "onnx" in selected_targets:
            results["piper"] = self.build_piper_ljspeech(final_clips)

        if do_all or "xtts" in selected_targets or "coqui" in selected_targets or "chatterbox" in selected_targets:
            results["xtts"] = self.build_xtts_dataset(final_clips)

        if do_all or "f5" in selected_targets or "f5-tts" in selected_targets or "f5tts" in selected_targets:
            results["f5tts"] = self.build_f5tts_dataset(final_clips)

        return results
