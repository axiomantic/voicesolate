import os
import json
import hashlib
import threading
from pathlib import Path
from typing import Optional, Dict, Any, List

class CacheManager:
    """
    Unified Multi-Tier Caching System for Voicesolate:
    1. STT / Word-Timestamps Cache: avoids re-transcribing context windows.
    2. Alignment Cache: avoids re-running sequence alignment for characters.
    3. Raw Slices Cache: avoids re-downloading audio slices over SSH.
    4. Demucs Stems Cache: avoids re-running heavy GPU neural separation when only tweaking mastering.
    5. Script & Subtitle Cache: avoids re-fetching text scripts or extracting subtitles.

    Granular Cache Bypass Controls:
    Can selectively bypass reading from specific tiers (--no-cache-stt, --no-cache-align,
    --no-cache-audio, --no-cache-enhance, --no-cache-script) while still writing updated
    entries back into the cache (additive/self-healing cache).
    """

    def __init__(
        self,
        cache_root: str = "cache",
        use_cache_stt: bool = True,
        use_cache_align: bool = True,
        use_cache_audio: bool = True,
        use_cache_enhance: bool = True,
        use_cache_script: bool = True,
    ):
        self.root = Path(cache_root)
        self.stt_dir = self.root / "stt"
        self.slices_dir = self.root / "audio" / "slices"
        self.stems_dir = self.root / "audio" / "stems"
        self.scripts_dir = self.root / "scripts"
        self.subtitles_dir = self.root / "subtitles"

        for d in [self.stt_dir, self.slices_dir, self.stems_dir, self.scripts_dir, self.subtitles_dir]:
            d.mkdir(parents=True, exist_ok=True)

        self.use_cache_stt = use_cache_stt
        self.use_cache_align = use_cache_align
        self.use_cache_audio = use_cache_audio
        self.use_cache_enhance = use_cache_enhance
        self.use_cache_script = use_cache_script
        self._lock = threading.Lock()

    def get_media_key(self, media_path: str) -> str:
        """Generates a consistent filesystem-safe cache key for any media file or URL."""
        basename = Path(media_path).stem
        cleaned = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in basename)
        path_hash = hashlib.sha256(media_path.encode("utf-8")).hexdigest()[:8]
        return f"{cleaned[:50]}_{path_hash}"

    # --- 1. STT & Word-Level Timestamp Caching ---

    def get_stt_cache_path(self, media_key: str) -> Path:
        return self.stt_dir / f"{media_key}_stt.json"

    def load_stt_cache(self, media_key: str) -> Dict[str, Any]:
        p = self.get_stt_cache_path(media_key)
        if p.exists():
            try:
                with open(p, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                return {}
        return {}

    def save_stt_entry(self, media_key: str, window_key: str, data: Dict[str, Any]):
        with self._lock:
            cache = self.load_stt_cache(media_key)
            if "windows" not in cache:
                cache["windows"] = {}
            cache["windows"][window_key] = data
            p = self.get_stt_cache_path(media_key)
            with open(p, "w", encoding="utf-8") as f:
                json.dump(cache, f, indent=2)

    def get_stt_entry(self, media_key: str, window_key: str) -> Optional[Dict[str, Any]]:
        if not self.use_cache_stt:
            return None
        cache = self.load_stt_cache(media_key)
        return cache.get("windows", {}).get(window_key)

    # --- 2. Character Dialogue Alignment Caching ---

    def save_alignment_cache(self, media_key: str, character: str, script_id: str, clips_data: List[Dict[str, Any]]):
        cache = self.load_stt_cache(media_key)
        if "alignments" not in cache:
            cache["alignments"] = {}
        key = f"{character.upper()}_{script_id}"
        cache["alignments"][key] = clips_data
        p = self.get_stt_cache_path(media_key)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(cache, f, indent=2)

    def get_alignment_cache(self, media_key: str, character: str, script_id: str) -> Optional[List[Dict[str, Any]]]:
        if not self.use_cache_align:
            return None
        cache = self.load_stt_cache(media_key)
        key = f"{character.upper()}_{script_id}"
        return cache.get("alignments", {}).get(key)

    # --- 3. Raw Audio Slice Caching ---

    def get_slice_path(self, media_key: str, timecode_str: str) -> Path:
        media_slice_dir = self.slices_dir / media_key
        media_slice_dir.mkdir(parents=True, exist_ok=True)
        return media_slice_dir / f"{timecode_str}.wav"

    # --- 4. Neural Demucs Vocal Stem Caching ---

    def get_stem_path(self, media_key: str, timecode_str: str) -> Path:
        media_stem_dir = self.stems_dir / media_key
        media_stem_dir.mkdir(parents=True, exist_ok=True)
        return media_stem_dir / f"{timecode_str}_vocal_stem.wav"
