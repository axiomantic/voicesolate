import os
import re
import time
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, asdict
from rapidfuzz import fuzz
from rich.progress import Progress
import numpy as np

from .wyoming_client import WyomingSTTClient
from .audio_extractor import AudioExtractor
from .script_parser import DialogueLine
from .cache_manager import CacheManager

@dataclass
class AlignedClip:
    character: str
    text: str
    start_sec: float
    end_sec: float
    confidence: float
    timecode_str: str

@dataclass
class SubtitleAnchor:
    start_sec: float
    end_sec: float
    text: str

class SearchAligner:
    """
    Automated Hierarchical STT Alignment Engine.
    Uses whole-span Levenshtein subtitle candidate matching, expanded context windows,
    sub-second Whisper word-level timestamps, and persistent multi-tier caching.
    """

    def __init__(self, audio_extractor: AudioExtractor, stt_client: Optional[WyomingSTTClient] = None, cache_manager: Optional[CacheManager] = None):
        self.extractor = audio_extractor
        self.stt_client = stt_client
        self.duration = audio_extractor.get_duration()
        self.cache = cache_manager or CacheManager()
        self.media_key = self.cache.get_media_key(audio_extractor.raw_path)
        self._local_whisper = None

    def _get_local_whisper(self):
        """Lazy loader for local faster-whisper model."""
        if self._local_whisper is None:
            try:
                from faster_whisper import WhisperModel
                self._local_whisper = WhisperModel("base", device="cpu", compute_type="int8")
            except Exception as e:
                print(f"Notice: local faster-whisper not available ({e}), using Wyoming STT fallback.")
        return self._local_whisper

    def format_timecode(self, seconds: float) -> str:
        """Formats seconds into HH_MM_SS_mmm."""
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        s = int(seconds % 60)
        ms = int(round((seconds - int(seconds)) * 1000))
        return f"{h:02d}_{m:02d}_{s:02d}_{ms:03d}"

    def parse_srt_anchors(self, srt_path: str) -> List[SubtitleAnchor]:
        """Parses SRT file into timestamped text anchors."""
        if not srt_path or not os.path.exists(srt_path):
            return []

        anchors = []
        with open(srt_path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()

        time_pattern = re.compile(
            r"(\d{2}):(\d{2}):(\d{2})[,.](\d{3})\s*-->\s*(\d{2}):(\d{2}):(\d{2})[,.](\d{3})"
        )

        blocks = content.strip().split("\n\n")
        for block in blocks:
            lines = [l.strip() for l in block.split("\n") if l.strip()]
            for i, line in enumerate(lines):
                m = time_pattern.search(line)
                if m:
                    start_s = int(m.group(1))*3600 + int(m.group(2))*60 + int(m.group(3)) + int(m.group(4))/1000.0
                    end_s = int(m.group(5))*3600 + int(m.group(6))*60 + int(m.group(7)) + int(m.group(8))/1000.0
                    text = " ".join(lines[i+1:])
                    clean_text = re.sub(r"<[^>]+>", "", text).strip()
                    if clean_text:
                        anchors.append(SubtitleAnchor(start_sec=start_s, end_sec=end_s, text=clean_text))
                    break
        return anchors

    def align_character_lines(
        self,
        all_script_lines: List[DialogueLine],
        target_characters: List[str],
        subtitles_path: Optional[str] = None,
        script_id: str = "default",
        similarity_threshold: float = 55.0,
        progress: Optional[Progress] = None
    ) -> List[AlignedClip]:
        """
        Whole-Span Monotonic Levenshtein Alignment across Subtitles & Audio.
        Checks persistent alignment cache first for zero-latency instant re-runs.
        """
        # Check alignment cache
        cached_clips = []
        all_cached = True
        for char in target_characters:
            char_data = self.cache.get_alignment_cache(self.media_key, char, script_id)
            if char_data:
                cached_clips.extend([AlignedClip(**c) for c in char_data])
            else:
                all_cached = False
                break

        if all_cached and cached_clips:
            print(f"✓ Loaded {len(cached_clips)} aligned clips from persistent alignment cache.")
            return cached_clips

        target_chars_set = set(c.upper() for c in target_characters)
        target_lines = [l for l in all_script_lines if l.character.upper() in target_chars_set]

        aligned_clips: List[AlignedClip] = []
        anchors = self.parse_srt_anchors(subtitles_path) if subtitles_path else []

        task_id = None
        if progress:
            task_id = progress.add_task(
                f"[cyan]Aligning {len(target_characters)} character(s)...",
                total=len(target_lines)
            )

        last_anchor_idx = 0

        for line_idx, target in enumerate(target_lines):
            target_text = target.text.strip()
            target_clean = target_text.lower()
            words = target_clean.split()
            num_words = len(words)

            if progress and task_id is not None:
                trunc = (target_text[:30] + "...") if len(target_text) > 30 else target_text
                progress.update(
                    task_id,
                    description=f"[cyan]Aligning [{line_idx+1}/{len(target_lines)}] {target.character}: \"{trunc}\""
                )

            best_start = None
            best_end = None
            best_score = 0.0

            # 1. Search forward monotonically from last_anchor_idx first
            search_ranges = [
                range(last_anchor_idx, min(len(anchors), last_anchor_idx + 80)),
                range(0, len(anchors)) # Fallback if missed in local forward window
            ]

            for s_range in search_ranges:
                for start_i in s_range:
                    max_lookahead = min(len(anchors), start_i + max(3, num_words // 3 + 3))
                    accum = ""
                    for end_i in range(start_i, max_lookahead):
                        accum += (" " if accum else "") + anchors[end_i].text.lower()
                        if num_words <= 4:
                            score = fuzz.ratio(target_clean, accum)
                        else:
                            score = max(fuzz.token_sort_ratio(target_clean, accum), fuzz.ratio(target_clean, accum))

                        if score > best_score:
                            best_score = score
                            best_start = start_i
                            best_end = end_i

                if best_score >= similarity_threshold:
                    break

            if best_start is not None and best_score >= similarity_threshold:
                raw_start = anchors[best_start].start_sec
                raw_end = anchors[best_end].end_sec

                # Update cursor
                last_anchor_idx = best_end + 1

                # 2. Refine boundaries with context window + word-level timestamps
                refined_start, refined_end, conf = self._refine_boundaries_with_stt(
                    raw_start, raw_end, target_text
                )

                timecode = f"{self.format_timecode(refined_start)}-{self.format_timecode(refined_end)}"
                aligned_clips.append(AlignedClip(
                    character=target.character,
                    text=target.text,
                    start_sec=refined_start,
                    end_sec=refined_end,
                    confidence=conf,
                    timecode_str=timecode
                ))

            if progress and task_id is not None:
                progress.advance(task_id, 1)

        # Save to persistent alignment cache
        for char in target_characters:
            char_clips = [asdict(c) for c in aligned_clips if c.character.upper() == char.upper()]
            if char_clips:
                self.cache.save_alignment_cache(self.media_key, char, script_id, char_clips)

        return aligned_clips

    def _refine_boundaries_with_stt(
        self,
        start_sec: float,
        end_sec: float,
        target_text: str
    ) -> Tuple[float, float, float]:
        """
        Refines clip boundaries using an expanded context window, persistent STT cache,
        and sub-second word timestamps to snap to exact words spoken.
        """
        whisper_model = self._get_local_whisper()
        
        # Expand context window: 2.0s before, 1.5s after
        pad_pre = 2.0
        pad_post = 1.5
        probe_start = max(0.0, start_sec - pad_pre)
        probe_end = min(self.duration, end_sec + pad_post)
        probe_dur = probe_end - probe_start

        # Window cache key: rounded to 1 decimal place to hit cache reliably
        win_key = f"{probe_start:.1f}_{probe_dur:.1f}"
        cached_entry = self.cache.get_stt_entry(self.media_key, win_key)

        whisper_words = []
        if cached_entry and "words" in cached_entry:
            whisper_words = cached_entry["words"]
        else:
            try:
                pcm = self.extractor.extract_slice_pcm(probe_start, probe_dur)
                if whisper_model is not None:
                    audio_data = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
                    segments, _ = whisper_model.transcribe(audio_data, word_timestamps=True, language="en")
                    
                    for seg in segments:
                        for w in seg.words:
                            clean_w = "".join(c for c in w.word.lower() if c.isalnum())
                            if clean_w:
                                whisper_words.append({
                                    "word": w.word.strip(),
                                    "clean": clean_w,
                                    "start": probe_start + w.start,
                                    "end": probe_start + w.end,
                                    "prob": w.probability
                                })
                    # Save to persistent STT cache
                    self.cache.save_stt_entry(self.media_key, win_key, {"words": whisper_words})
            except Exception:
                pass

        target_words = ["".join(c for c in w.lower() if c.isalnum()) for w in target_text.split()]
        target_words = [w for w in target_words if w]
        L = len(target_words)

        if whisper_words and target_words:
            best_span = None
            best_score = 0.0
            target_phrase = " ".join(target_words)

            # Sliding window search over word timestamp stream
            for win_size in range(max(1, L - 2), min(len(whisper_words) + 1, L + 4)):
                for i in range(len(whisper_words) - win_size + 1):
                    cand_words = [cw["clean"] for cw in whisper_words[i:i+win_size]]
                    score = fuzz.ratio(target_phrase, " ".join(cand_words))

                    # Exact boundary alignment bonuses:
                    first_match = (cand_words[0] == target_words[0])
                    last_match = (cand_words[-1] == target_words[-1])

                    bonus = 0.0
                    if first_match:
                        bonus += 3.0
                    if last_match:
                        bonus += 5.0
                    # Heavily penalize candidate spans that overshoot past the target's last word
                    if not last_match and target_words[-1] in cand_words[:-1]:
                        bonus -= 12.0

                    # Proximity to original anchor (mild tie-breaker only)
                    span_mid = (whisper_words[i]["start"] + whisper_words[i+win_size-1]["end"]) / 2.0
                    anchor_mid = (start_sec + end_sec) / 2.0
                    time_dist = abs(span_mid - anchor_mid)
                    adjusted_score = score + bonus - (time_dist * 0.5)

                    if adjusted_score > best_score:
                        best_score = adjusted_score
                        best_span = (i, i + win_size - 1, score)

            if best_span and best_span[2] >= 60.0:
                i_s, i_e, match_ratio = best_span
                raw_w_start = whisper_words[i_s]["start"]
                raw_w_end = whisper_words[i_e]["end"]

                # Acoustic Energy Valley Snapping: find quietest pause point right before vocal onset
                snapped_start = raw_w_start
                try:
                    sr = 16000
                    # If audio_data is already in memory from fresh transcription, use it; otherwise extract quick slice
                    if "audio_data" in locals() and audio_data is not None:
                        a_buf = audio_data
                        a_base = probe_start
                    else:
                        slice_t = max(0.0, raw_w_start - 0.25)
                        pcm_valley = self.extractor.extract_slice_pcm(slice_t, 0.6, sample_rate=sr)
                        a_buf = np.frombuffer(pcm_valley, dtype=np.int16).astype(np.float32) / 32768.0
                        a_base = slice_t

                    frame_sz = int(0.03 * sr)
                    hop_sz = int(0.01 * sr)
                    min_rms = float("inf")
                    best_valley_t = raw_w_start
                    for f_i in range(0, len(a_buf) - frame_sz, hop_sz):
                        cur_t = a_base + f_i / sr
                        if raw_w_start - 0.25 <= cur_t <= raw_w_start + 0.30:
                            frm = a_buf[f_i:f_i+frame_sz]
                            rms = float(np.sqrt(np.mean(frm**2)))
                            if rms < min_rms:
                                min_rms = rms
                                best_valley_t = cur_t

                    if abs(best_valley_t - raw_w_start) <= 0.30:
                        snapped_start = best_valley_t
                except Exception:
                    pass

                snapped_end = min(self.duration, raw_w_end + 0.05)
                return max(0.0, snapped_start), snapped_end, float(match_ratio)

        return start_sec, end_sec, 80.0
