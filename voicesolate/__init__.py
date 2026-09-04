"""
Voicesolate — Automated Character Dialogue Extraction & Studio Isolation Pipeline.
"""

from .audio_extractor import AudioExtractor
from .script_parser import ScriptParser
from .wyoming_client import WyomingSTTClient
from .search_aligner import SearchAligner
from .audio_enhancer import AudioEnhancer
from .cache_manager import CacheManager

from .dataset_builder import DatasetBuilder
from .model_trainer import ModelTrainer

__version__ = "0.3.0"
__all__ = [
    "AudioExtractor",
    "ScriptParser",
    "WyomingSTTClient",
    "SearchAligner",
    "AudioEnhancer",
    "CacheManager",
    "DatasetBuilder",
    "ModelTrainer",
]
