import json
import pytest
from pathlib import Path
from voicesolate.script_parser import ScriptParser, DialogueLine

@pytest.mark.unit
class TestScriptParser:
    def test_parse_screenplay_text_full_consumption(self, temp_dir: Path, sample_script_text: str):
        script_file = temp_dir / "sample_screenplay.txt"
        script_file.write_text(sample_script_text, encoding="utf-8")

        parser = ScriptParser(cache_dir=str(temp_dir / "cache"), use_cache=False)
        dialogues, stats = parser.fetch_or_load(str(script_file))

        # 1. Exact count assertion (catching unintended duplication or skipping)
        assert len(dialogues) == 4

        # 2. Exact character sequence
        assert [d.character for d in dialogues] == ["DATA", "PICARD", "DATA", "PICARD"]

        # 3. Exact full string matching for every line
        expected_lines = [
            "Captain, sensors are detecting unusual spatial distortions.",
            "Can you identify the source, Mister Data?",
            "It appears to be a temporal rift originating from late nineteenth century Earth.",
            "Nineteenth century. We must proceed with extreme caution."
        ]
        for i, expected in enumerate(expected_lines):
            assert dialogues[i].text == expected
            assert dialogues[i].word_count == len(expected.split())

        # 4. Negative assertion: stage direction '(thoughtfully)' must be completely stripped
        for d in dialogues:
            assert "(thoughtfully)" not in d.text
            assert "(" not in d.text
            assert ")" not in d.text

    def test_parse_srt_full_consumption(self, temp_dir: Path, sample_srt_content: str):
        srt_file = temp_dir / "sample.srt"
        srt_file.write_text(sample_srt_content, encoding="utf-8")

        parser = ScriptParser(cache_dir=str(temp_dir / "cache"), use_cache=False)
        dialogues, stats = parser.fetch_or_load(str(srt_file))

        assert len(dialogues) == 3
        assert dialogues[0].text == "Captain, sensors are detecting unusual spatial distortions."
        assert dialogues[1].text == "Can you identify the source, Mister Data?"
        assert dialogues[2].text == "It appears to be a temporal rift originating from late nineteenth century Earth."

    def test_character_stats_exact_aggregation(self, temp_dir: Path, sample_script_text: str):
        script_file = temp_dir / "screenplay.txt"
        script_file.write_text(sample_script_text, encoding="utf-8")

        parser = ScriptParser(cache_dir=str(temp_dir / "cache"), use_cache=False)
        parser.fetch_or_load(str(script_file))
        sorted_chars = parser.get_characters_sorted()

        assert len(sorted_chars) == 2
        char_dict = {c.name: c.line_count for c in sorted_chars}
        assert char_dict == {"DATA": 2, "PICARD": 2}

        total_words = {c.name: c.word_count for c in sorted_chars}
        assert total_words["DATA"] == len("Captain, sensors are detecting unusual spatial distortions.".split()) + len("It appears to be a temporal rift originating from late nineteenth century Earth.".split())

    def test_caching_full_json_schema_validation(self, temp_dir: Path, sample_script_text: str):
        cache_dir = temp_dir / "cache"
        script_file = temp_dir / "screenplay.txt"
        script_file.write_text(sample_script_text, encoding="utf-8")

        parser1 = ScriptParser(cache_dir=str(cache_dir), use_cache=True)
        dialogues1, _ = parser1.fetch_or_load(str(script_file))

        # Inspect cache JSON structure
        cache_files = list(cache_dir.glob("*_dialogues.json"))
        assert len(cache_files) == 1
        with open(cache_files[0], "r", encoding="utf-8") as f:
            cached_data = json.load(f)

        assert isinstance(cached_data, list)
        assert len(cached_data) == 4
        assert {"character", "text", "word_count"}.issubset(cached_data[0].keys())

        # Second load: ensure full equality
        parser2 = ScriptParser(cache_dir=str(cache_dir), use_cache=True)
        dialogues2, _ = parser2.fetch_or_load(str(script_file))
        assert len(dialogues1) == len(dialogues2)
        for d1, d2 in zip(dialogues1, dialogues2):
            assert d1.character == d2.character
            assert d1.text == d2.text
            assert d1.word_count == d2.word_count

    def test_negative_control_corrupted_cache_recovers_cleanly(self, temp_dir: Path, sample_script_text: str):
        """Negative control: corrupted cache JSON file must not crash parser, must re-parse from source."""
        cache_dir = temp_dir / "cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        script_file = temp_dir / "screenplay.txt"
        script_file.write_text(sample_script_text, encoding="utf-8")

        # Plant corrupted JSON in cache
        cache_key = ScriptParser()._get_cache_key(str(script_file))
        corrupted_file = cache_dir / f"{cache_key}_dialogues.json"
        corrupted_file.write_text("{NOT_VALID_JSON: [corrupted", encoding="utf-8")

        parser = ScriptParser(cache_dir=str(cache_dir), use_cache=True)
        dialogues, _ = parser.fetch_or_load(str(script_file))
        assert len(dialogues) == 4
        assert dialogues[0].character == "DATA"

    def test_negative_control_empty_script_file(self, temp_dir: Path):
        """Negative control: empty text file must return empty list without crashing."""
        empty_file = temp_dir / "empty.txt"
        empty_file.write_text("", encoding="utf-8")

        parser = ScriptParser(cache_dir=str(temp_dir / "cache"), use_cache=False)
        dialogues, is_cached = parser.fetch_or_load(str(empty_file))
        assert dialogues == []
        assert is_cached is False
