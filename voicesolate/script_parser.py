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
    Universal Script Parser for Voicesolate.
    Supports screenplays (.txt), structured JSON, subtitle files (.srt), and web script URLs.
    Features persistent disk caching for parsed dialogues and character statistics.
    """

    def __init__(self, cache_dir: str = "cache/scripts", use_cache: bool = True):
        self.cache_dir = cache_dir
        self.use_cache = use_cache
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
        if not self.use_cache:
            return None
        json_cache = os.path.join(self.cache_dir, f"{cache_key}_dialogues.json")
        if os.path.exists(json_cache):
            try:
                with open(json_cache, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    dialogues = [DialogueLine(**d) for d in data]
                    self._compute_stats(dialogues)
                    self.dialogues = dialogues
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

    def fetch_or_load(self, source: str, provider: Optional[str] = None) -> Tuple[List[DialogueLine], bool]:
        """
        Universal script loader.
        Accepts local files (.txt, .json, .srt), URLs, or pluggable provider identifiers.
        """
        cache_key = self._get_cache_key(source)
        if self.use_cache:
            cached = self.load_cached_dialogues(cache_key)
            if cached:
                return cached, True

        # 1. Local Files
        if os.path.exists(source):
            if source.endswith(".json"):
                with open(source, "r", encoding="utf-8", errors="ignore") as f:
                    data = json.load(f)
                    dialogues = [DialogueLine(
                        index=i,
                        character=d["character"].strip().upper(),
                        text=d["text"].strip(),
                        word_count=len(d["text"].split())
                    ) for i, d in enumerate(data)]
            elif source.endswith(".srt"):
                with open(source, "r", encoding="utf-8", errors="ignore") as f:
                    dialogues = self.parse_srt_file(f.read())
            else:
                with open(source, "r", encoding="utf-8", errors="ignore") as f:
                    dialogues = self.parse_raw_script_text(f.read())

        # 2. Direct HTTP / Web URL
        elif source.startswith("http://") or source.startswith("https://"):
            raw_cache = os.path.join(self.cache_dir, f"raw_{cache_key}.html")
            if os.path.exists(raw_cache):
                with open(raw_cache, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read()
            else:
                resp = requests.get(source, timeout=15, headers={"User-Agent": "Mozilla/5.0 (Voicesolate/1.0)"})
                resp.raise_for_status()
                content = resp.text
                with open(raw_cache, "w", encoding="utf-8") as f:
                    f.write(content)
            dialogues = self.parse_web_html(content)

        # 3. Pluggable Providers (e.g. Star Trek Chakoteya mapping if requested)
        elif provider == "startrek" or "s05e26" in source.lower() or "s06e01" in source.lower():
            html = self._fetch_chakoteya_startrek(source)
            dialogues = self.parse_web_html(html)

        else:
            raise ValueError(f"Cannot resolve script source: '{source}'. Please specify a valid file path or URL.")

        self._compute_stats(dialogues)
        self.dialogues = dialogues
        self.save_cached_dialogues(cache_key, dialogues)
        return dialogues, False

    def _fetch_chakoteya_startrek(self, episode_identifier: str) -> str:
        """Specialized provider for Star Trek scripts from Chakoteya."""
        ident = episode_identifier.lower().replace("-", " ").replace("_", " ").strip()
        ep_map = {"s05e26": 226, "s06e01": 227}
        ep_num = ep_map.get(ident)
        if not ep_num:
            match_se = re.search(r"s0?(\d+)e0?(\d+)", ident)
            if match_se:
                s, e = int(match_se.group(1)), int(match_se.group(2))
                season_starts = {1: 101, 2: 127, 3: 149, 4: 175, 5: 201, 6: 227, 7: 253}
                if s in season_starts:
                    ep_num = season_starts[s] + e - 1
            if not ep_num:
                ep_num = 226

        cache_path = os.path.join(self.cache_dir, f"tng_{ep_num}.htm")
        if os.path.exists(cache_path):
            with open(cache_path, "r", encoding="utf-8", errors="ignore") as f:
                return f.read()

        url = f"http://www.chakoteya.net/NextGen/{ep_num}.htm"
        resp = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=15)
        resp.raise_for_status()
        with open(cache_path, "w", encoding="utf-8") as f:
            f.write(resp.text)
        return resp.text

    def parse_web_html(self, html_content: str) -> List[DialogueLine]:
        """Parses HTML web pages, stripping navigation links, script, style, and generic footers."""
        soup = BeautifulSoup(html_content, "html.parser")
        
        # Remove navigation links, tables, scripts, footers
        for tag in soup.find_all(["a", "footer", "script", "style", "nav", "header"]):
            tag.decompose()
            
        # Replace <br> and <p> with newlines so lines don't run together
        for br in soup.find_all(["br", "p"]):
            br.insert_after("\n")

        text = soup.get_text()
        return self.parse_raw_script_text(text)

    def parse_raw_script_text(self, text: str) -> List[DialogueLine]:
        """Parses standard script text (Screenplay format)."""
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
                # Generic copyright / disclaimer cleaner
                cleaned_line = re.sub(r"copyright\s*(?:©|\(c\)).*$", "", cleaned_line, flags=re.IGNORECASE)
                cleaned_line = re.sub(r"all rights reserved.*$", "", cleaned_line, flags=re.IGNORECASE)
                cleaned_line = re.sub(r"all other copyrights.*$", "", cleaned_line, flags=re.IGNORECASE)
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

            # Generic footer / disclaimer cutoff
            if any(term in line_str.lower() for term in [
                "back to the episode listing", "back to episodes", "terms of use",
                "copyright ©", "all rights reserved", "all other copyrights property"
            ]):
                flush_current()
                break

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
        return dialogues

    def parse_srt_file(self, srt_content: str) -> List[DialogueLine]:
        """Parses subtitles into dialogue lines, extracting speaker tags when present."""
        dialogues = []
        blocks = srt_content.strip().split("\n\n")
        speaker_pattern = re.compile(r"^(?:\[([A-Z0-9\s'\.\-]+)\]|([A-Z0-9\s'\.\-]+):)\s*(.*)$")
        
        idx = 0
        for block in blocks:
            lines = [l.strip() for l in block.split("\n") if l.strip()]
            for i, line in enumerate(lines):
                if "-->" in line and i + 1 < len(lines):
                    text = " ".join(lines[i+1:])
                    text = re.sub(r"<[^>]+>", "", text).strip()
                    m = speaker_pattern.match(text)
                    if m:
                        spk = (m.group(1) or m.group(2)).strip().upper()
                        content = m.group(3).strip()
                    else:
                        spk = "UNKNOWN"
                        content = text
                    
                    if content:
                        dialogues.append(DialogueLine(
                            index=idx,
                            character=spk,
                            text=content,
                            word_count=len(content.split())
                        ))
                        idx += 1
                    break
        return dialogues

    def _compute_stats(self, dialogues: List[DialogueLine]):
        """Aggregates line and word counts per character."""
        self.character_stats.clear()
        for d in dialogues:
            if d.character not in self.character_stats:
                self.character_stats[d.character] = CharacterStats(name=d.character, line_count=0, word_count=0)
            self.character_stats[d.character].line_count += 1
            self.character_stats[d.character].word_count += d.word_count

    def get_characters_sorted(self) -> List[CharacterStats]:
        """Returns all discovered characters sorted by word count descending."""
        return sorted(self.character_stats.values(), key=lambda c: c.word_count, reverse=True)
