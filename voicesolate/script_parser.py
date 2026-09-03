import os
import re
import json
import hashlib
import requests
from bs4 import BeautifulSoup
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Tuple

@dataclass
class DialogueLine:
    index: int
    character: str
    text: str
    word_count: int

@dataclass
class CharacterStats:
    name: str
    line_count: int
    word_count: int

class ScriptParser:
    """
    Parses TV & movie scripts from online sources (Chakoteya, IMSDb, URLs) or local files/subtitles.
    Features robust disk caching for raw text and parsed dialogues.
    """

    CHAKOTEYA_TNG_MAP = {
        # Season 5
        "s05e26": 226,
        "times arrow": 226,
        "times arrow part 1": 226,
        "time's arrow": 226,
        "time's arrow, part 1": 226,
        "time's arrow part i": 226,
        # Season 6
        "s06e01": 227,
        "times arrow part 2": 227,
        "time's arrow, part 2": 227,
        "time's arrow part ii": 227,
    }

    def __init__(self, cache_dir: str = "cache/scripts"):
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)
        self.dialogues: List[DialogueLine] = []
        self.character_stats: Dict[str, CharacterStats] = {}

    def _get_cache_key(self, source: str) -> str:
        """Generates a sanitized or hashed file key for caching."""
        sanitized = re.sub(r"[^a-zA-Z0-9_\-\.]", "_", source.strip().lower())
        if len(sanitized) > 40:
            md5 = hashlib.md5(source.encode("utf-8")).hexdigest()[:10]
            sanitized = sanitized[:30] + "_" + md5
        return sanitized

    def load_cached_dialogues(self, cache_key: str) -> Optional[List[DialogueLine]]:
        """Loads parsed dialogues from JSON cache if available."""
        json_cache = os.path.join(self.cache_dir, f"{cache_key}_dialogues.json")
        if os.path.exists(json_cache):
            try:
                with open(json_cache, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    dialogues = [
                        DialogueLine(
                            index=item["index"],
                            character=item["character"],
                            text=item["text"],
                            word_count=item["word_count"]
                        )
                        for item in data
                    ]
                    self.dialogues = dialogues
                    self._calculate_stats()
                    return dialogues
            except Exception:
                return None
        return None

    def save_cached_dialogues(self, cache_key: str, dialogues: List[DialogueLine]):
        """Saves parsed dialogues to JSON cache."""
        json_cache = os.path.join(self.cache_dir, f"{cache_key}_dialogues.json")
        try:
            with open(json_cache, "w", encoding="utf-8") as f:
                json.dump([asdict(d) for d in dialogues], f, indent=2)
        except Exception:
            pass

    def fetch_or_load(self, source: str) -> Tuple[List[DialogueLine], bool]:
        """
        High-level helper to fetch or load script from cache.
        Returns (dialogues, is_cache_hit).
        """
        cache_key = self._get_cache_key(source)
        cached = self.load_cached_dialogues(cache_key)
        if cached:
            return cached, True

        # Not in parsed cache; fetch and parse
        if source.startswith("http://") or source.startswith("https://"):
            raw_cache = os.path.join(self.cache_dir, f"raw_{cache_key}.html")
            if os.path.exists(raw_cache):
                with open(raw_cache, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read()
            else:
                resp = requests.get(source, timeout=15, headers={"User-Agent": "Mozilla/5.0"})
                resp.raise_for_status()
                content = resp.text
                with open(raw_cache, "w", encoding="utf-8") as f:
                    f.write(content)
            dialogues = self.parse_chakoteya_html(content)
        elif source.endswith(".srt"):
            with open(source, "r", encoding="utf-8", errors="ignore") as f:
                dialogues = self.parse_srt_file(f.read())
        elif os.path.exists(source):
            with open(source, "r", encoding="utf-8", errors="ignore") as f:
                dialogues = self.parse_raw_script_text(f.read())
        else:
            # Assume Chakoteya Star Trek episode identifier
            html = self.fetch_chakoteya_tng(source)
            dialogues = self.parse_chakoteya_html(html)

        self.save_cached_dialogues(cache_key, dialogues)
        return dialogues, False

    def fetch_chakoteya_tng(self, episode_identifier: str) -> str:
        """
        Fetches TNG script from Chakoteya.net with automatic local disk caching.
        """
        ident = episode_identifier.lower().replace("-", " ").replace("_", " ").strip()
        ep_num = self.CHAKOTEYA_TNG_MAP.get(ident)
        
        if not ep_num:
            match_se = re.search(r"s0?(\d+)e0?(\d+)", ident)
            if match_se:
                s, e = int(match_se.group(1)), int(match_se.group(2))
                season_starts = {1: 101, 2: 127, 3: 149, 4: 175, 5: 201, 6: 227, 7: 253}
                if s in season_starts:
                    ep_num = season_starts[s] + e - 1
            elif ident.isdigit():
                ep_num = int(ident)

        if not ep_num:
            raise ValueError(f"Could not map episode '{episode_identifier}' to Chakoteya episode ID. Try passing direct script URL or file.")

        cache_path = os.path.join(self.cache_dir, f"tng_{ep_num}.htm")
        if os.path.exists(cache_path):
            with open(cache_path, "r", encoding="utf-8", errors="ignore") as f:
                return f.read()

        url = f"http://www.chakoteya.net/NextGen/{ep_num}.htm"
        headers = {"User-Agent": "Mozilla/5.0 (ScriptExtractor/1.0)"}
        resp = requests.get(url, headers=headers, timeout=15)
        resp.raise_for_status()
        
        with open(cache_path, "w", encoding="utf-8") as f:
            f.write(resp.text)

        return resp.text

    def parse_chakoteya_html(self, html_content: str) -> List[DialogueLine]:
        """Parses HTML from Chakoteya scripts."""
        soup = BeautifulSoup(html_content, "html.parser")
        text = soup.get_text()
        return self.parse_raw_script_text(text)

    def parse_raw_script_text(self, text: str) -> List[DialogueLine]:
        """Parses plain text script containing standard dialogue formats."""
        lines = text.split("\n")
        dialogues: List[DialogueLine] = []
        
        dialogue_pattern = re.compile(r"^\s*(?:\[[^\]]*\]\s*)?([A-Z0-9\s'\.\-\#]+?)(?:\s*\[[^\]]*\])?:\s*(.*)$")
        all_caps_name = re.compile(r"^\s*([A-Z0-9\s'\.\-]{2,25})\s*$")

        current_speaker: Optional[str] = None
        current_text: List[str] = []
        dialogue_idx = 0

        def flush_current():
            nonlocal dialogue_idx, current_speaker, current_text
            if current_speaker and current_text:
                full_line = " ".join(" ".join(current_text).split())
                cleaned_line = re.sub(r"\([^)]*\)", "", full_line)
                cleaned_line = re.sub(r"\[[^\]]*\]", "", cleaned_line)
                cleaned_line = " ".join(cleaned_line.split())
                
                if cleaned_line:
                    words = len(cleaned_line.split())
                    dialogues.append(DialogueLine(
                        index=dialogue_idx,
                        character=current_speaker.strip().upper(),
                        text=cleaned_line,
                        word_count=words
                    ))
                    dialogue_idx += 1
            current_speaker = None
            current_text = []

        for line in lines:
            line_str = line.strip()
            if not line_str:
                continue

            if line_str.startswith("[") and line_str.endswith("]"):
                flush_current()
                continue

            m = dialogue_pattern.match(line_str)
            if m:
                flush_current()
                speaker_raw = m.group(1).strip()
                speaker = re.sub(r"\([^)]*\)", "", speaker_raw).strip()
                dialogue_part = m.group(2).strip()
                
                if speaker.upper() in ["ACT ONE", "ACT TWO", "ACT THREE", "ACT FOUR", "ACT FIVE", "TEASER", "SCENE"]:
                    continue

                current_speaker = speaker.upper()
                if dialogue_part:
                    current_text.append(dialogue_part)
            else:
                m_caps = all_caps_name.match(line_str)
                if m_caps and not any(kw in line_str.upper() for kw in ["FADE IN", "FADE OUT", "CUT TO", "CONTINUED", "THE END"]):
                    flush_current()
                    current_speaker = m_caps.group(1).strip().upper()
                elif current_speaker:
                    current_text.append(line_str)

        flush_current()
        self.dialogues = dialogues
        self._calculate_stats()
        return dialogues

    def parse_srt_file(self, srt_content: str) -> List[DialogueLine]:
        """Parses SRT subtitle content into dialogue lines."""
        blocks = srt_content.strip().split("\n\n")
        dialogues: List[DialogueLine] = []
        idx = 0

        for block in blocks:
            lines = [l.strip() for l in block.split("\n") if l.strip()]
            if len(lines) >= 3:
                text_lines = lines[2:]
                full_text = " ".join(text_lines)
                m = re.match(r"^([A-Z0-9\s'\.\-]+?):\s*(.*)$", full_text)
                if m:
                    speaker = m.group(1).strip().upper()
                    line_text = m.group(2).strip()
                else:
                    speaker = "UNKNOWN"
                    line_text = full_text

                line_text = re.sub(r"<[^>]+>", "", line_text).strip()
                if line_text:
                    dialogues.append(DialogueLine(
                        index=idx,
                        character=speaker,
                        text=line_text,
                        word_count=len(line_text.split())
                    ))
                    idx += 1

        self.dialogues = dialogues
        self._calculate_stats()
        return dialogues

    def _calculate_stats(self):
        stats: Dict[str, CharacterStats] = {}
        for d in self.dialogues:
            name = d.character
            if name not in stats:
                stats[name] = CharacterStats(name=name, line_count=0, word_count=0)
            stats[name].line_count += 1
            stats[name].word_count += d.word_count
        self.character_stats = stats

    def get_characters_sorted(self) -> List[CharacterStats]:
        """Returns character list sorted descending by word count."""
        return sorted(self.character_stats.values(), key=lambda c: c.word_count, reverse=True)
