import os
import sys
import re
import json
import time
import shutil
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
import soundfile as sf

from ..script_parser import ScriptParser
from ..audio_extractor import AudioExtractor
from ..search_aligner import SearchAligner
from ..audio_enhancer import AudioEnhancer
from ..cache_manager import CacheManager
from ..dataset_builder import DatasetBuilder
from ..model_trainer import ModelTrainer
from ..waveform import generate_macro_waveform_from_manifest, extract_peaks_from_wav
from .job_manager import job_manager

logger = logging.getLogger("voicesolate.pipeline_runner")

def run_scan_job(job_id: str, input_path: str, script_path: Optional[str] = None, provider: Optional[str] = None):
    """
    Asynchronously scans media file, extracts embedded subtitles,
    parses screenplay/script, and ranks speaking characters.
    """
    try:
        job_manager.update_job(job_id, progress=5.0, stage="init", message="Initializing media inspector...")
        extractor = AudioExtractor(input_path)
        
        # Episode naming
        if extractor.is_remote:
            filename = Path(extractor.remote_file_path).stem
        else:
            filename = extractor.local_path.stem
        episode_name = re.sub(r"[^a-zA-Z0-9_\-\.]", "_", filename)[:60]
        output_base_dir = Path("./output").resolve() / episode_name
        output_base_dir.mkdir(parents=True, exist_ok=True)

        job_manager.update_job(job_id, progress=20.0, stage="subtitles", message="Extracting embedded subtitles...")
        embedded_srt = output_base_dir / "embedded_subs.srt"
        found_subs = False
        if not embedded_srt.exists() or embedded_srt.stat().st_size == 0:
            found_subs = extractor.extract_subtitles_to_file(str(embedded_srt))
        else:
            found_subs = True

        active_subs_path = str(embedded_srt) if found_subs else None

        job_manager.update_job(job_id, progress=50.0, stage="script", message="Parsing dialogue and script...")
        parser = ScriptParser(
            script_source=script_path,
            provider=provider,
            subtitles_path=active_subs_path,
            episode_hint=filename
        )
        script_lines = parser.parse()
        char_counts = parser.get_character_counts()

        # Build character table
        characters = []
        for char_name, line_count in char_counts.items():
            if line_count >= 1:
                # Estimate duration (~3.2 seconds per dialogue line average)
                est_sec = line_count * 3.2
                characters.append({
                    "name": char_name,
                    "lines": line_count,
                    "estimated_duration_sec": round(est_sec, 1),
                    "estimated_duration_min": round(est_sec / 60.0, 1)
                })

        duration = extractor.get_duration()

        result = {
            "episode_name": episode_name,
            "filename": filename,
            "duration": round(duration, 2),
            "is_remote": extractor.is_remote,
            "has_subtitles": found_subs,
            "subtitles_path": active_subs_path,
            "script_id": parser.script_id,
            "script_lines_count": len(script_lines),
            "characters": characters[:50]
        }

        job_manager.update_job(
            job_id,
            progress=100.0,
            stage="complete",
            status="completed",
            message=f"Discovered {len(characters)} speaking characters across {len(script_lines)} dialogue lines.",
            result=result
        )

    except Exception as e:
        logger.exception("Scan job failed")
        job_manager.update_job(
            job_id,
            progress=100.0,
            stage="error",
            status="failed",
            message=f"Scan failed: {str(e)}",
            error=str(e)
        )

def run_pipeline_job(job_id: str, params: Dict[str, Any]):
    """
    Executes full pipeline:
    Alignment -> Audio Slicing -> Demucs Isolation -> Dataset Curation -> Model Preparation.
    Emits real-time divide-and-conquer worker telemetry.
    """
    try:
        input_path = params.get("input_path", "").strip()
        target_chars = params.get("characters", [])
        if isinstance(target_chars, str):
            target_chars = [target_chars]
        target_chars = [c.strip().upper() for c in target_chars if c.strip()]
        
        if not target_chars:
            raise ValueError("No character specified for extraction.")

        min_duration = float(params.get("min_duration", 3.0))
        do_enhance = bool(params.get("enhance", True))
        targets = params.get("targets", ["all"])
        no_train = bool(params.get("no_train", False))
        no_aggregate = bool(params.get("no_aggregate", False))

        job_manager.update_job(job_id, progress=5.0, stage="init", message="Initializing media pipeline...")
        extractor = AudioExtractor(input_path)

        if extractor.is_remote:
            filename = Path(extractor.remote_file_path).stem
        else:
            filename = extractor.local_path.stem
        episode_name = re.sub(r"[^a-zA-Z0-9_\-\.]", "_", filename)[:60]
        output_base_dir = Path("./output").resolve() / episode_name
        output_base_dir.mkdir(parents=True, exist_ok=True)

        embedded_srt = output_base_dir / "embedded_subs.srt"
        if not embedded_srt.exists() or embedded_srt.stat().st_size == 0:
            extractor.extract_subtitles_to_file(str(embedded_srt))
        active_subs = str(embedded_srt) if embedded_srt.exists() else None

        job_manager.update_job(job_id, progress=15.0, stage="script", message="Parsing script lines...")
        parser = ScriptParser(
            script_source=params.get("script_path"),
            provider=params.get("provider"),
            subtitles_path=active_subs,
            episode_hint=filename
        )
        script_lines = parser.parse()

        # Alignment
        job_manager.update_job(job_id, progress=25.0, stage="align", message="Executing search & divide-and-conquer alignment...")
        cache_manager = CacheManager()
        aligner = SearchAligner(extractor, cache_manager=cache_manager)

        # Worker state reporter
        def on_align_step(step_info: Dict[str, Any]):
            worker_id = "worker-stt-1"
            stype = step_info.get("type", "worker_scan")
            if stype == "worker_scan":
                job_manager.update_worker_state(
                    job_id=job_id,
                    worker_id=worker_id,
                    state="scanning",
                    chunk_start=step_info.get("start_sec", 0.0),
                    chunk_end=step_info.get("end_sec", 0.0),
                    snippet=step_info.get("target_text", "")
                )
            elif stype == "clip_matched":
                job_manager.update_worker_state(
                    job_id=job_id,
                    worker_id=worker_id,
                    state="matched",
                    chunk_start=step_info.get("start_sec", 0.0),
                    chunk_end=step_info.get("end_sec", 0.0),
                    snippet=step_info.get("text", ""),
                    confidence=step_info.get("confidence")
                )

        aligned_clips = aligner.align_character_dialogue(
            all_script_lines=script_lines,
            target_characters=target_chars,
            subtitles_path=active_subs,
            script_id=parser.script_id,
            callback=on_align_step
        )

        if not aligned_clips:
            job_manager.update_job(
                job_id,
                progress=100.0,
                stage="complete",
                status="completed",
                message=f"No matching dialogue lines found for {', '.join(target_chars)}."
            )
            return

        job_manager.update_job(
            job_id,
            progress=45.0,
            stage="slicing",
            message=f"Found {len(aligned_clips)} dialogue instances. Slicing raw discrete audio stems..."
        )

        # Process per character
        enhancer = AudioEnhancer(cache_manager=cache_manager) if do_enhance else None
        dataset_builder = DatasetBuilder(output_base_dir)
        processed_manifest_clips = []

        total_steps = len(target_chars) * len(aligned_clips)
        step_idx = 0

        for char_name in target_chars:
            if job_manager.is_cancelled(job_id):
                job_manager.update_job(job_id, status="cancelled", message="Cancelled by user.")
                return

            char_dir = output_base_dir / char_name
            raw_dir = char_dir / "raw"
            raw_dir.mkdir(parents=True, exist_ok=True)
            enh_dir = char_dir / "enhanced" if do_enhance else None
            if enh_dir:
                enh_dir.mkdir(parents=True, exist_ok=True)

            char_clips = [c for c in aligned_clips if c.character.upper() == char_name.upper()]
            
            for idx, clip in enumerate(char_clips):
                step_idx += 1
                prog = 45.0 + (step_idx / max(1, total_steps)) * 35.0

                raw_file = raw_dir / f"{clip.timecode_str}.wav"
                dur = clip.end_sec - clip.start_sec

                if dur < min_duration:
                    continue

                if not raw_file.exists():
                    extractor.extract_slice(clip.start_sec, dur, str(raw_file))

                enh_file = None
                if do_enhance and enh_dir:
                    enh_file = enh_dir / f"{clip.timecode_str}_enhanced.wav"
                    if not enh_file.exists():
                        job_manager.update_job(
                            job_id,
                            progress=prog,
                            stage="demucs",
                            message=f"[{idx+1}/{len(char_clips)}] Demucs vocal isolation: {clip.text[:40]}..."
                        )
                        job_manager.update_worker_state(
                            job_id=job_id,
                            worker_id="worker-demucs-1",
                            state="enhancing",
                            chunk_start=clip.start_sec,
                            chunk_end=clip.end_sec,
                            snippet=f"Isolating ({idx+1}/{len(char_clips)}): {clip.text[:35]}..."
                        )
                        enhancer.clean_and_enhance_file(
                            str(raw_file),
                            str(enh_file),
                            media_key=aligner.media_key,
                            timecode_str=clip.timecode_str
                        )
                        job_manager.update_worker_state(
                            job_id=job_id,
                            worker_id="worker-demucs-1",
                            state="matched",
                            chunk_start=clip.start_sec,
                            chunk_end=clip.end_sec,
                            snippet=f"Isolated: {clip.text[:35]}..."
                        )

                clip_dict = {
                    "character": char_name,
                    "text": clip.text,
                    "start_sec": clip.start_sec,
                    "end_sec": clip.end_sec,
                    "confidence": clip.confidence,
                    "file": str(raw_file.resolve()),
                    "enhanced_file": str(enh_file.resolve()) if enh_file and enh_file.exists() else None
                }
                processed_manifest_clips.append(clip_dict)
                job_manager.notify_clip_discovered(job_id, clip_dict)

            # Build datasets
            job_manager.update_job(job_id, progress=85.0, stage="dataset", message=f"Building dataset packs for {char_name}...")
            all_char_clips = char_clips
            if not no_aggregate:
                all_char_clips = dataset_builder.aggregate_character_clips(char_name, Path("./output").resolve())

            datasets = dataset_builder.build_all(char_name, all_char_clips, targets=targets)

            # Train / Configure Models
            if not no_train:
                job_manager.update_job(job_id, progress=92.0, stage="train", message=f"Configuring TTS models for {char_name}...")
                trainer = ModelTrainer(char_dir)
                trainer.train_all(datasets, targets=targets)

        # Write manifest
        manifest_data = {
            "episode": episode_name,
            "selected_characters": target_chars,
            "clips": processed_manifest_clips
        }
        with open(output_base_dir / "manifest.json", "w", encoding="utf-8") as f:
            json.dump(manifest_data, f, indent=2)

        # Generate macro-waveform data
        waveform_data = generate_macro_waveform_from_manifest(output_base_dir / "manifest.json")

        job_manager.update_job(
            job_id,
            progress=100.0,
            stage="complete",
            status="completed",
            message=f"Pipeline finished! Extracted & isolated {len(processed_manifest_clips)} clips for {', '.join(target_chars)}.",
            result={
                "episode": episode_name,
                "characters": target_chars,
                "clips_count": len(processed_manifest_clips),
                "waveform": waveform_data
            }
        )

    except Exception as e:
        logger.exception("Pipeline job failed")
        job_manager.update_job(
            job_id,
            progress=100.0,
            stage="error",
            status="failed",
            message=f"Pipeline error: {str(e)}",
            error=str(e)
        )
