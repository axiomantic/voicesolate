import pytest
from pathlib import Path
from rapidfuzz import fuzz
from voicesolate.search_aligner import SearchAligner, AlignedClip, SubtitleAnchor

@pytest.mark.unit
class TestSearchAligner:
    def test_format_timecode_exact(self):
        assert SearchAligner.format_timecode(None, 0.0) == "00_00_00_000"
        assert SearchAligner.format_timecode(None, 65.5) == "00_01_05_500"
        assert SearchAligner.format_timecode(None, 3661.123) == "01_01_01_123"
        assert SearchAligner.format_timecode(None, 59.999) == "00_00_59_999"

    def test_aligned_clip_exact_dataclass_fields(self):
        clip = AlignedClip(
            character="CLEMENS",
            text="The secret of getting ahead is getting started.",
            start_sec=120.5,
            end_sec=124.8,
            confidence=96.5,
            timecode_str="00_02_00_500-00_02_04_800"
        )
        assert clip.character == "CLEMENS"
        assert clip.text == "The secret of getting ahead is getting started."
        assert clip.confidence == 96.5
        assert pytest.approx(clip.end_sec - clip.start_sec, abs=0.001) == 4.3
        assert clip.timecode_str == "00_02_00_500-00_02_04_800"

    def test_parse_srt_anchors_all_entries_consumed(self, temp_dir: Path, sample_srt_content: str):
        srt_file = temp_dir / "test.srt"
        srt_file.write_text(sample_srt_content, encoding="utf-8")

        anchors = SearchAligner.parse_srt_anchors(None, str(srt_file))
        # 1. Full length
        assert len(anchors) == 3

        # 2. Every single anchor validated for start, end, and text
        assert anchors[0].start_sec == pytest.approx(1.5, abs=0.01)
        assert anchors[0].end_sec == pytest.approx(4.2, abs=0.01)
        assert anchors[0].text == "Captain, sensors are detecting unusual spatial distortions."

        assert anchors[1].start_sec == pytest.approx(5.1, abs=0.01)
        assert anchors[1].end_sec == pytest.approx(7.8, abs=0.01)
        assert anchors[1].text == "Can you identify the source, Mister Data?"

        assert anchors[2].start_sec == pytest.approx(8.5, abs=0.01)
        assert anchors[2].end_sec == pytest.approx(12.0, abs=0.01)
        assert anchors[2].text == "It appears to be a temporal rift originating from late nineteenth century Earth."

        # Invariant: end_sec must strictly exceed start_sec
        for a in anchors:
            assert a.end_sec > a.start_sec

    def test_fuzzy_matching_precision_and_negative_controls(self):
        script_line = "Madam, I'd be delighted. So, this is a space ship?"
        stt_transcript = "madam id be delighted so this is a space ship"
        
        # Positive control
        score = fuzz.ratio(script_line.lower(), stt_transcript.lower())
        assert score > 85

        # Negative control 1: Completely unrelated dialogue must reject below match threshold (55%)
        unrelated_line = "Shields at forty percent and failing, sir!"
        unrelated_score = fuzz.ratio(script_line.lower(), unrelated_line.lower())
        assert unrelated_score < 55, f"Unrelated text should be below threshold: got {unrelated_score}"

        # Negative control 2: Empty dialogue comparison must be 0
        empty_score = fuzz.ratio("", stt_transcript)
        assert empty_score == 0
