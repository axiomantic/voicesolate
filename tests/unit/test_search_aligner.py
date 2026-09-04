import pytest
from pathlib import Path
from rapidfuzz import fuzz
from voicesolate.search_aligner import SearchAligner, AlignedClip, SubtitleAnchor

@pytest.mark.unit
class TestSearchAligner:
    def test_format_timecode(self):
        # We can test format_timecode on Aligner instance or directly
        assert SearchAligner.format_timecode(None, 0.0) == "00_00_00_000"
        assert SearchAligner.format_timecode(None, 65.5) == "00_01_05_500"
        assert SearchAligner.format_timecode(None, 3661.123) == "01_01_01_123"

    def test_aligned_clip_dataclass(self):
        clip = AlignedClip(
            character="CLEMENS",
            text="The secret of getting ahead is getting started.",
            start_sec=120.5,
            end_sec=124.8,
            confidence=96.5,
            timecode_str="00_02_00_500-00_02_04_800"
        )
        assert clip.character == "CLEMENS"
        assert clip.confidence == 96.5
        assert clip.end_sec - clip.start_sec > 4.0

    def test_parse_srt_anchors(self, temp_dir: Path, sample_srt_content: str):
        srt_file = temp_dir / "test.srt"
        srt_file.write_text(sample_srt_content, encoding="utf-8")

        anchors = SearchAligner.parse_srt_anchors(None, str(srt_file))
        assert len(anchors) == 3
        assert isinstance(anchors[0], SubtitleAnchor)
        assert anchors[0].start_sec == pytest.approx(1.5, abs=0.1)
        assert anchors[0].end_sec == pytest.approx(4.2, abs=0.1)
        assert "sensors are detecting" in anchors[0].text

    def test_fuzzy_matching_precision(self):
        script_line = "Madam, I'd be delighted. So, this is a space ship?"
        stt_transcript = "madam id be delighted so this is a space ship"
        
        score = fuzz.ratio(script_line.lower(), stt_transcript.lower())
        assert score > 85

        partial_score = fuzz.partial_ratio("So, this is a space ship?", stt_transcript)
        assert partial_score > 90
