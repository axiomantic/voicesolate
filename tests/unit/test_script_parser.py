import pytest
from pathlib import Path
from voicesolate.script_parser import ScriptParser, DialogueLine

@pytest.mark.unit
class TestScriptParser:
    def test_parse_screenplay_text(self, temp_dir: Path, sample_script_text: str):
        script_file = temp_dir / "sample_screenplay.txt"
        script_file.write_text(sample_script_text, encoding="utf-8")

        parser = ScriptParser(cache_dir=str(temp_dir / "cache"), use_cache=False)
        dialogues, stats = parser.fetch_or_load(str(script_file))

        assert len(dialogues) >= 4
        # Verify characters are extracted and uppercase
        characters = [d.character for d in dialogues]
        assert "DATA" in characters
        assert "PICARD" in characters

        # Check line text content doesn't retain parenthetical actions
        picard_lines = [d.text for d in dialogues if d.character == "PICARD"]
        assert any("Nineteenth century" in text for text in picard_lines)
        assert not any("(thoughtfully)" in text for text in picard_lines)

    def test_parse_srt(self, temp_dir: Path, sample_srt_content: str):
        srt_file = temp_dir / "sample.srt"
        srt_file.write_text(sample_srt_content, encoding="utf-8")

        parser = ScriptParser(cache_dir=str(temp_dir / "cache"), use_cache=False)
        dialogues, stats = parser.fetch_or_load(str(srt_file))

        assert len(dialogues) == 3
        assert "Captain, sensors are detecting" in dialogues[0].text
        assert "Can you identify the source" in dialogues[1].text

    def test_character_stats_and_sorting(self, temp_dir: Path, sample_script_text: str):
        script_file = temp_dir / "screenplay.txt"
        script_file.write_text(sample_script_text, encoding="utf-8")

        parser = ScriptParser(cache_dir=str(temp_dir / "cache"), use_cache=False)
        parser.fetch_or_load(str(script_file))
        sorted_chars = parser.get_characters_sorted()

        assert len(sorted_chars) >= 2
        char_names = [c.name for c in sorted_chars]
        assert "DATA" in char_names
        assert "PICARD" in char_names
        assert all(c.line_count > 0 for c in sorted_chars)

    def test_caching_behavior(self, temp_dir: Path, sample_script_text: str):
        cache_dir = temp_dir / "cache"
        script_file = temp_dir / "screenplay.txt"
        script_file.write_text(sample_script_text, encoding="utf-8")

        # First run: writes to cache
        parser1 = ScriptParser(cache_dir=str(cache_dir), use_cache=True)
        dialogues1, stats1 = parser1.fetch_or_load(str(script_file))

        # Check cache files exist
        cache_files = list(cache_dir.glob("*.json"))
        assert len(cache_files) > 0

        # Second run: loads from cache
        parser2 = ScriptParser(cache_dir=str(cache_dir), use_cache=True)
        dialogues2, stats2 = parser2.fetch_or_load(str(script_file))

        assert len(dialogues1) == len(dialogues2)
        assert [d.text for d in dialogues1] == [d.text for d in dialogues2]

    def test_dialogue_line_dict_access(self):
        dl = DialogueLine(index=1, character="DATA", text="Affirmative.", word_count=1)
        assert dl["character"] == "DATA"
        assert dl.get("text") == "Affirmative."
        assert dl.get("nonexistent", "default") == "default"
        assert "text" in dl
