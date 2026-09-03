"""
Voicesolate — Automated Character Dialogue Extraction & Studio Isolation Pipeline.
"""

from .audio_extractor import AudioExtractor
from .script_parser import ScriptParser
from .wyoming_client import WyomingSTTClient
from .search_aligner import SearchAligner
from .audio_enhancer import AudioEnhancer
from .cache_manager import CacheManager

__version__ = "0.2.0"
__all__ = [
    "AudioExtractor",
    "ScriptParser",
    "WyomingSTTClient",
    "SearchAligner",
    "AudioEnhancer",
    "CacheManager",
]
