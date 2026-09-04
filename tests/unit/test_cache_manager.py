import pytest
from pathlib import Path
from voicesolate.cache_manager import CacheManager

@pytest.mark.unit
class TestCacheManager:
    def test_media_key_generation(self, temp_dir: Path):
        cm = CacheManager(cache_root=str(temp_dir / "cache"))
        
        # Test standard file path
        key1 = cm.get_media_key("/path/to/my_show_s06e01.mkv")
        assert "my_show_s06e01" in key1
        assert len(key1) > 10

        # Test remote SFTP URL with spaces and symbols
        key2 = cm.get_media_key("sftp://user@flanopticon.lan/mnt/TV Shows/TNG S06E01 [1080p].mkv")
        assert "TNG_S06E01" in key2
        # Deterministic
        key2_repeat = cm.get_media_key("sftp://user@flanopticon.lan/mnt/TV Shows/TNG S06E01 [1080p].mkv")
        assert key2 == key2_repeat

    def test_stt_caching_and_bypass(self, temp_dir: Path):
        cache_dir = temp_dir / "cache"
        cm_enabled = CacheManager(cache_root=str(cache_dir), use_cache_stt=True)
        cm_disabled = CacheManager(cache_root=str(cache_dir), use_cache_stt=False)

        media_key = "test_episode_key"
        window_key = "win_00_10"
        stt_payload = {"text": "Testing dialogue", "segments": []}

        # Save STT cache entry
        cm_enabled.save_stt_entry(media_key, window_key, stt_payload)
        cache_path = cm_enabled.get_stt_cache_path(media_key)
        assert cache_path.exists()

        # Load STT cache with cache enabled
        loaded = cm_enabled.get_stt_entry(media_key, window_key)
        assert loaded == stt_payload

        # Bypass STT cache: should return None even if file exists
        bypassed = cm_disabled.get_stt_entry(media_key, window_key)
        assert bypassed is None

    def test_alignment_caching_and_bypass(self, temp_dir: Path):
        cache_dir = temp_dir / "cache"
        cm_enabled = CacheManager(cache_root=str(cache_dir), use_cache_align=True)
        cm_disabled = CacheManager(cache_root=str(cache_dir), use_cache_align=False)

        media_key = "test_episode_key"
        char_name = "DATA"
        script_id = "s06e01"
        align_data = [{"start_sec": 10.5, "end_sec": 14.2, "text": "Fascinating."}]

        cm_enabled.save_alignment_cache(media_key, char_name, script_id, align_data)
        loaded = cm_enabled.get_alignment_cache(media_key, char_name, script_id)
        assert loaded == align_data

        # Test character specificity
        assert cm_enabled.get_alignment_cache(media_key, "PICARD", script_id) is None

        # Test bypass
        bypassed = cm_disabled.get_alignment_cache(media_key, char_name, script_id)
        assert bypassed is None

    def test_slice_and_stem_paths(self, temp_dir: Path):
        cm = CacheManager(cache_root=str(temp_dir / "cache"))
        media_key = "test_media"
        timecode = "00_01_20_000-00_01_25_000"

        slice_path = cm.get_slice_path(media_key, timecode)
        assert slice_path.suffix == ".wav"
        assert slice_path.parent.exists()

        stem_path = cm.get_stem_path(media_key, timecode)
        assert "vocal_stem.wav" in stem_path.name
        assert stem_path.parent.exists()
