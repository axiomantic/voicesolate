import os
import sys
import re
import json
import time
import shutil
import logging
import threading
import queue
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
            found_subs = extractor.extract_embedded_subtitles(str(embedded_srt))
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
            extractor.extract_embedded_subtitles(str(embedded_srt))
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
        cache_manager = CacheManager(
            use_cache_stt=not bool(params.get("no_cache_stt", False)),
            use_cache_align=not bool(params.get("no_cache_align", False)),
            use_cache_audio=not bool(params.get("no_cache_audio", False)),
            use_cache_enhance=not bool(params.get("no_cache_enhance", False)),
            use_cache_script=not bool(params.get("no_cache_script", False)),
        )
        aligner = SearchAligner(extractor, cache_manager=cache_manager)

        # Build Stage A itemized queue
        target_chars_set = set(c.upper() for c in target_chars)
        target_dialogue_lines = [l for l in script_lines if l.character.upper() in target_chars_set]
        stage_a_items = [
            {
                "id": f"stt-{i}",
                "index": i,
                "character": l.character,
                "text": l.text,
                "state": "pending",
                "status_text": "Queued for STT"
            }
            for i, l in enumerate(target_dialogue_lines)
        ]

        # Prepare Stage B (Demucs / Stem Slicing) queue & worker pool
        stage_b_items: List[Dict[str, Any]] = []
        stage_b_queue: queue.Queue = queue.Queue()
        processed_manifest_clips: List[Dict[str, Any]] = []
        manifest_lock = threading.Lock()
        stage_b_items_lock = threading.Lock()
        enhancer = AudioEnhancer(cache_manager=cache_manager) if do_enhance else None

        num_demucs_workers = max(1, min(8, int(params.get("demucs_workers", 2))))
        num_stt_workers = max(1, min(8, int(params.get("stt_workers", 2))))

        # Initialize Demucs worker telemetry cards
        for w_idx in range(1, num_demucs_workers + 1):
            job_manager.update_worker_state(
                job_id=job_id,
                worker_id=f"worker-demucs-{w_idx}",
                state="idle",
                chunk_start=0.0,
                chunk_end=0.0,
                snippet="Awaiting matched dialogue...",
                queue_count=0,
                queue_items=stage_b_items
            )

        # Initialize STT worker telemetry cards
        for s_idx in range(1, num_stt_workers + 1):
            job_manager.update_worker_state(
                job_id=job_id,
                worker_id=f"worker-stt-{s_idx}",
                state="idle",
                chunk_start=0.0,
                chunk_end=0.0,
                snippet="Awaiting search start...",
                queue_count=len(stage_a_items),
                queue_items=stage_a_items
            )

        def demucs_worker_loop(w_id: str):
            while True:
                work = stage_b_queue.get()
                if work is None:
                    stage_b_queue.task_done()
                    job_manager.update_worker_state(
                        job_id=job_id,
                        worker_id=w_id,
                        state="idle",
                        chunk_start=0.0,
                        chunk_end=0.0,
                        snippet="Isolation complete",
                        queue_count=0,
                        queue_items=stage_b_items
                    )
                    break

                if job_manager.is_cancelled(job_id):
                    stage_b_queue.task_done()
                    break

                char_name, clip, item_b = work
                char_dir = output_base_dir / char_name
                raw_dir = char_dir / "raw"
                raw_dir.mkdir(parents=True, exist_ok=True)
                enh_dir = char_dir / "enhanced" if do_enhance else None
                if enh_dir:
                    enh_dir.mkdir(parents=True, exist_ok=True)

                raw_file = raw_dir / f"{clip.timecode_str}.wav"
                dur = clip.end_sec - clip.start_sec

                with stage_b_items_lock:
                    item_b["state"] = "enhancing" if do_enhance else "slicing"
                    item_b["status_text"] = "Isolating with Demucs v4..." if do_enhance else "Slicing audio stem..."
                    remaining_queue = sum(1 for it in stage_b_items if it["state"] == "pending")

                job_manager.update_worker_state(
                    job_id=job_id,
                    worker_id=w_id,
                    state="enhancing" if do_enhance else "slicing",
                    chunk_start=clip.start_sec,
                    chunk_end=clip.end_sec,
                    snippet=f"Isolating: \"{clip.text[:30]}\"...",
                    queue_count=remaining_queue,
                    queue_items=stage_b_items
                )

                try:
                    if not raw_file.exists():
                        extractor.extract_slice(clip.start_sec, dur, str(raw_file))

                    enh_file = None
                    if do_enhance and enh_dir:
                        enh_file = enh_dir / f"{clip.timecode_str}_enhanced.wav"
                        if not enh_file.exists():
                            enhancer.clean_and_enhance_file(
                                str(raw_file),
                                str(enh_file),
                                media_key=aligner.media_key,
                                timecode_str=clip.timecode_str
                            )
                            with stage_b_items_lock:
                                item_b["state"] = "matched"
                                item_b["status_text"] = "✓ Isolated"
                                item_b["enhanced_file"] = str(enh_file.resolve())
                                item_b["file"] = str(raw_file.resolve())
                            snippet_text = f"Isolated: \"{clip.text[:30]}\""
                        else:
                            with stage_b_items_lock:
                                item_b["state"] = "matched"
                                item_b["status_text"] = "✓ Cached"
                                item_b["enhanced_file"] = str(enh_file.resolve())
                                item_b["file"] = str(raw_file.resolve())
                            snippet_text = f"Cached: \"{clip.text[:30]}\""
                    else:
                        with stage_b_items_lock:
                            item_b["state"] = "matched"
                            item_b["status_text"] = "✓ Sliced"
                            item_b["file"] = str(raw_file.resolve())
                        snippet_text = f"Sliced: \"{clip.text[:30]}\""

                    with stage_b_items_lock:
                        remaining_queue = sum(1 for it in stage_b_items if it["state"] == "pending")

                    job_manager.update_worker_state(
                        job_id=job_id,
                        worker_id=w_id,
                        state="matched",
                        chunk_start=clip.start_sec,
                        chunk_end=clip.end_sec,
                        snippet=snippet_text,
                        queue_count=remaining_queue,
                        queue_items=stage_b_items
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
                    with manifest_lock:
                        processed_manifest_clips.append(clip_dict)
                    job_manager.notify_clip_discovered(job_id, clip_dict)

                except Exception as ex:
                    logger.exception(f"[{w_id}] Error processing clip {clip.timecode_str}")
                    with stage_b_items_lock:
                        item_b["state"] = "failed"
                        item_b["status_text"] = f"Failed: {str(ex)[:30]}"
                        remaining_queue = sum(1 for it in stage_b_items if it["state"] == "pending")
                    job_manager.update_worker_state(
                        job_id=job_id,
                        worker_id=w_id,
                        state="idle",
                        chunk_start=clip.start_sec,
                        chunk_end=clip.end_sec,
                        snippet=f"Failed: {str(ex)[:25]}",
                        queue_count=remaining_queue,
                        queue_items=stage_b_items
                    )
                finally:
                    stage_b_queue.task_done()

        # Start Demucs worker threads
        demucs_threads = []
        for w_idx in range(1, num_demucs_workers + 1):
            t = threading.Thread(target=demucs_worker_loop, args=(f"worker-demucs-{w_idx}",), daemon=True)
            t.start()
            demucs_threads.append(t)

        def enqueue_clip_to_stage_b(clip):
            dur = clip.end_sec - clip.start_sec
            if dur < min_duration:
                return
            with stage_b_items_lock:
                idx = len(stage_b_items)
                dur_s = round(dur, 1)
                item_b = {
                    "id": f"demucs-{idx}",
                    "index": idx,
                    "character": clip.character,
                    "text": clip.text,
                    "timecode": clip.timecode_str,
                    "duration": f"{dur_s}s",
                    "start_sec": clip.start_sec,
                    "end_sec": clip.end_sec,
                    "confidence": clip.confidence,
                    "state": "pending",
                    "status_text": "Queued for Demucs"
                }
                stage_b_items.append(item_b)
                remaining = sum(1 for it in stage_b_items if it["state"] == "pending")

            stage_b_queue.put((clip.character, clip, item_b))

            # Update queue info on idle workers
            job = job_manager.get_job(job_id)
            if job:
                for w_i in range(1, num_demucs_workers + 1):
                    w_id = f"worker-demucs-{w_i}"
                    curr_w = job.workers.get(w_id, {})
                    if curr_w.get("state") == "idle":
                        job_manager.update_worker_state(
                            job_id=job_id,
                            worker_id=w_id,
                            state="idle",
                            chunk_start=0.0,
                            chunk_end=0.0,
                            snippet=f"{remaining} clips queued for isolation",
                            queue_count=remaining,
                            queue_items=stage_b_items
                        )
                        break

        # Worker state reporter
        def on_align_step(step_info: Dict[str, Any]):
            worker_id = step_info.get("worker_id", "worker-stt-1")
            stype = step_info.get("type", "worker_scan")
            if stype == "worker_scan":
                line_idx = step_info.get("index")
                if line_idx is not None and 0 <= line_idx < len(stage_a_items):
                    stage_a_items[line_idx]["state"] = "scanning"
                    stage_a_items[line_idx]["status_text"] = f"Scanning ({worker_id})..."

                job_manager.update_worker_state(
                    job_id=job_id,
                    worker_id=worker_id,
                    state="scanning",
                    chunk_start=step_info.get("start_sec", 0.0),
                    chunk_end=step_info.get("end_sec", 0.0),
                    snippet=step_info.get("target_text", ""),
                    queue_count=sum(1 for it in stage_a_items if it["state"] == "pending"),
                    queue_items=stage_a_items
                )
            elif stype == "clip_matched":
                text_matched = step_info.get("text", "").strip()
                for it in stage_a_items:
                    if it["text"].strip() == text_matched or (it["state"] == "scanning" and it["status_text"].startswith(f"Scanning ({worker_id})")):
                        it["state"] = "matched"
                        it["status_text"] = "Matched"
                        it["start_sec"] = step_info.get("start_sec")
                        it["end_sec"] = step_info.get("end_sec")
                        it["confidence"] = step_info.get("confidence")
                        dur = round((step_info.get("end_sec", 0) - step_info.get("start_sec", 0)), 1)
                        it["duration"] = f"{dur}s"
                        break

                job_manager.update_worker_state(
                    job_id=job_id,
                    worker_id=worker_id,
                    state="matched",
                    chunk_start=step_info.get("start_sec", 0.0),
                    chunk_end=step_info.get("end_sec", 0.0),
                    snippet=step_info.get("text", ""),
                    confidence=step_info.get("confidence"),
                    queue_count=sum(1 for it in stage_a_items if it["state"] == "pending"),
                    queue_items=stage_a_items
                )
                job_manager.notify_clip_discovered(
                    job_id=job_id,
                    clip_data={
                        "character": step_info.get("character"),
                        "text": step_info.get("text"),
                        "start_sec": step_info.get("start_sec"),
                        "end_sec": step_info.get("end_sec"),
                        "confidence": step_info.get("confidence")
                    }
                )
                matched_clip = step_info.get("clip")
                if matched_clip:
                    enqueue_clip_to_stage_b(matched_clip)

            elif stype == "cache_hit":
                for it in stage_a_items:
                    it["state"] = "matched"
                    it["status_text"] = "Cached alignment"
                for s_i in range(1, num_stt_workers + 1):
                    job_manager.update_worker_state(
                        job_id=job_id,
                        worker_id=f"worker-stt-{s_i}",
                        state="matched",
                        chunk_start=0.0,
                        chunk_end=0.0,
                        snippet=f"Loaded {step_info.get('count', 0)} cached speech alignments",
                        queue_count=0,
                        queue_items=stage_a_items
                    )
                for c in step_info.get("clips", []):
                    enqueue_clip_to_stage_b(c)

        # Signal STT Workers starting search
        for s_i in range(1, num_stt_workers + 1):
            job_manager.update_worker_state(
                job_id=job_id,
                worker_id=f"worker-stt-{s_i}",
                state="scanning" if stage_a_items else "idle",
                chunk_start=0.0,
                chunk_end=15.0,
                snippet=f"Starting search across {num_stt_workers} workers...",
                queue_count=len(stage_a_items),
                queue_items=stage_a_items
            )

        similarity_thresh = float(params.get("similarity_threshold", 55.0))
        aligned_clips = aligner.align_character_dialogue(
            all_script_lines=script_lines,
            target_characters=target_chars,
            subtitles_path=active_subs,
            script_id=parser.script_id,
            similarity_threshold=similarity_thresh,
            num_workers=num_stt_workers,
            callback=on_align_step
        )

        if not aligned_clips:
            # Signal Demucs workers to terminate
            for _ in range(num_demucs_workers):
                stage_b_queue.put(None)
            for t in demucs_threads:
                t.join()

            for s_i in range(1, num_stt_workers + 1):
                job_manager.update_worker_state(
                    job_id=job_id,
                    worker_id=f"worker-stt-{s_i}",
                    state="idle",
                    chunk_start=0.0,
                    chunk_end=0.0,
                    snippet="No matching dialogue found",
                    queue_count=0,
                    queue_items=stage_a_items
                )
            job_manager.update_job(
                job_id,
                progress=100.0,
                stage="complete",
                status="completed",
                message=f"No matching dialogue lines found for {', '.join(target_chars)}."
            )
            return

        # Stage A Completed
        matched_texts = {c.text.strip().lower() for c in aligned_clips}
        for it in stage_a_items:
            if it["state"] == "matched" or it.get("text", "").strip().lower() in matched_texts:
                it["state"] = "matched"
                it["status_text"] = "Matched"
            else:
                it["state"] = "unmatched"
                it["status_text"] = "Unmatched"

        for s_i in range(1, num_stt_workers + 1):
            job_manager.update_worker_state(
                job_id=job_id,
                worker_id=f"worker-stt-{s_i}",
                state="matched",
                chunk_start=0.0,
                chunk_end=0.0,
                snippet=f"Matched {len(aligned_clips)} speech instances",
                queue_count=0,
                queue_items=stage_a_items
            )

        job_manager.update_job(
            job_id,
            progress=50.0,
            stage="demucs",
            message=f"Found {len(aligned_clips)} dialogue instances. Processing neural vocal isolation with {num_demucs_workers} workers..."
        )

        # Signal Demucs workers that alignment is done
        for _ in range(num_demucs_workers):
            stage_b_queue.put(None)

        # Wait for all Demucs workers to finish processing queued clips
        for t in demucs_threads:
            t.join()

        if job_manager.is_cancelled(job_id):
            job_manager.update_job(job_id, status="cancelled", message="Cancelled by user.")
            return

        # Process per character datasets and models
        for char_name in target_chars:
            if job_manager.is_cancelled(job_id):
                job_manager.update_job(job_id, status="cancelled", message="Cancelled by user.")
                return

            char_dir = output_base_dir / char_name
            char_dir.mkdir(parents=True, exist_ok=True)
            builder = DatasetBuilder(char_dir)
            this_char_clips = [c for c in processed_manifest_clips if c.get("character", "").upper() == char_name.upper()]
            if not no_aggregate:
                all_char_clips = builder.aggregate_all_clips_for_character(
                    char_name, output_base_dir.parent, current_clips=this_char_clips
                )
            else:
                all_char_clips = this_char_clips

            job_manager.update_job(
                job_id,
                progress=85.0,
                stage="dataset",
                message=f"Building dataset packs for {char_name} ({len(all_char_clips)} clips)..."
            )
            datasets = builder.build_all(all_char_clips, targets=targets, aggregate_all=False)

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

        for w_i in range(1, num_demucs_workers + 1):
            job_manager.update_worker_state(
                job_id=job_id,
                worker_id=f"worker-demucs-{w_i}",
                state="matched",
                chunk_start=0.0,
                chunk_end=0.0,
                snippet="Demucs isolation complete",
                queue_count=0,
                queue_items=stage_b_items
            )

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
        job_manager.update_worker_state(
            job_id=job_id,
            worker_id="worker-stt-1",
            state="error",
            chunk_start=0.0,
            chunk_end=0.0,
            snippet=f"Failed: {str(e)[:40]}"
        )
        for w_i in range(1, num_demucs_workers + 1):
            job_manager.update_worker_state(
                job_id=job_id,
                worker_id=f"worker-demucs-{w_i}",
                state="error",
                chunk_start=0.0,
                chunk_end=0.0,
                snippet="Aborted due to error",
                queue_count=0
            )
        job_manager.update_job(
            job_id,
            progress=100.0,
            stage="error",
            status="failed",
            message=f"Pipeline error: {str(e)}",
            error=str(e)
        )
