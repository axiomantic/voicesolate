import os
import sys
import re
import time
import json
import uuid
import shutil
import tempfile
import importlib.util
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import soundfile as sf
import numpy as np
import torch
import warnings
import urllib.parse
import logging
import wave

logger = logging.getLogger("voicesolate.engine_service")

# Suppress harmless upstream deprecation notices from PyTorch and Hugging Face transformers
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*torch.jit.script.*")
warnings.filterwarnings("ignore", message=".*GenerationMixin.*")
warnings.filterwarnings("ignore", message=".*attention mask.*")

class _SuppressFlashAttnFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return "FlashAttention" not in record.getMessage()

logging.getLogger("kanade_tokenizer").addFilter(_SuppressFlashAttnFilter())

try:
    from transformers import logging as tf_logging
    tf_logging.set_verbosity_error()
except Exception:
    pass

ENGINE_SPECS = {
    "f5-tts": {
        "id": "f5-tts",
        "name": "F5-TTS",
        "display": "F5-TTS (Flow-Matching DiT)",
        "architecture": "Flow-Matching Diffusion Transformer (24kHz)",
        "badge": "F5-TTS",
        "samplerate": 24000,
    },
    "xtts-v2": {
        "id": "xtts-v2",
        "name": "Coqui XTTS-v2",
        "display": "Coqui XTTS-v2 (Autoregressive)",
        "architecture": "Autoregressive GPT + Latent Diffusion (24kHz)",
        "badge": "XTTS-v2",
        "samplerate": 24000,
    },
    "piper": {
        "id": "piper",
        "name": "Piper VITS",
        "display": "Piper (Neural VITS / ONNX)",
        "architecture": "Fast CPU Neural VITS (22.05kHz)",
        "badge": "Piper VITS",
        "samplerate": 22050,
    },
    "kokoro": {
        "id": "kokoro",
        "name": "Kokoro-82M",
        "display": "Kokoro-82M (Deep Style TTS + Missouri Drawl)",
        "architecture": "Kokoro-82M Deep Trained Neural Voice (24kHz)",
        "badge": "Kokoro 82M",
        "samplerate": 24000,
    }
}

def resolve_engine_meta(engine_id_or_hint: str, samplerate: Optional[int] = None) -> Dict[str, Any]:
    hint = (engine_id_or_hint or "").lower()
    if "f5" in hint:
        return dict(ENGINE_SPECS["f5-tts"])
    elif "xtts" in hint or "coqui" in hint:
        return dict(ENGINE_SPECS["xtts-v2"])
    elif "piper" in hint:
        return dict(ENGINE_SPECS["piper"])
    elif "kokoro" in hint or "styletts" in hint:
        return dict(ENGINE_SPECS["kokoro"])

    if samplerate == 22050:
        return dict(ENGINE_SPECS["piper"])
    elif samplerate == 24000:
        return dict(ENGINE_SPECS["f5-tts"])

    clean_name = (engine_id_or_hint or "Neural TTS").upper()
    return {
        "id": (engine_id_or_hint or "tts").lower(),
        "name": clean_name,
        "display": f"{clean_name} Model",
        "architecture": "Neural Speech Synthesis",
        "badge": clean_name,
        "samplerate": samplerate or 24000,
    }


class EngineService:
    """
    Manages TTS synthesis engines, hardware acceleration, and model packaging.
    Supported engines:
    - F5-TTS (Diffusion Transformer Flow-Matching, 24kHz)
    - Coqui XTTS-v2 (Zero-Shot Autoregressive + Latents, 24kHz)
    - Piper VITS (Fast CPU Neural Inference, 22.05kHz)
    """

    @staticmethod
    def apply_missouri_drawl(phonemes: str) -> str:
        """
        Transforms standard English G2P phonemes into 19th-century Missouri / Upper Southern
        drawl phonetics (as portrayed by Jerry Hardin as Mark Twain):
        - Prolongs stressed monophthongs with IPA length marker 'ː'
        - Monophthongizes / drawls first-person pronoun /aɪ/ ('ˌI' -> 'ˌaː' or 'ˌIː')
        - Lengthens vowels before voiced codas
        - Inserts dramatic contemplative pauses at punctuation
        """
        p = phonemes
        # 19th-century Missouri participle reduction: -ing -> -in' (e.g. gettin', startin')
        p = re.sub(r'ɪŋ\b', r'ɪn', p)
        # Pronoun 'I' monophthongization / drawl (/aɪ/ -> /aː/)
        p = re.sub(r'(\b| )ˌI\b', r'\1ˌaː', p)
        # Natural conversational breathing pause at punctuation
        p = re.sub(r'([,;—\-])\s*', r', ', p)
        return p

    DEFAULT_QUOTES = [
        "The secret of getting ahead is getting started.",
        "Kindness is the language which the deaf can hear and the blind can see.",
        "Whenever you find yourself on the side of the majority, it is time to pause and reflect.",
        "Twenty years from now you will be more disappointed by the things you didn't do than by the ones you did do.",
        "I have long been interested in the notion of time travelers. In fact, I wrote a book about it.",
        "Madam, I'd be delighted. So, this is a space ship? You ever run into Halley's comet?",
        "If you tell the truth, you don't have to remember anything.",
        "The man who does not read has no advantage over the man who cannot read."
    ]

    def __init__(self):
        self._f5_model = None
        self._xtts_model = None
        self._piper_voice = None
        self._loaded_piper_model_path = None
        self._kokoro_model = None
        self._kokoro_pipeline = None
        self._loaded_kokoro_adapter = None
        self._kanade_model = None
        self._vocos_model = None
        self.cache_synth_dir = Path("cache/synthesized").resolve()
        self.cache_synth_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _is_module_available(module_name: str) -> bool:
        try:
            return importlib.util.find_spec(module_name) is not None
        except Exception:
            return False

    def get_system_status(self) -> Dict[str, Any]:
        """Returns hardware, OS, dependency and acceleration status."""
        has_mps = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        has_cuda = torch.cuda.is_available()
        device_str = "mps" if has_mps else ("cuda" if has_cuda else "cpu")

        ffmpeg_path = shutil.which("ffmpeg")
        ffprobe_path = shutil.which("ffprobe")

        return {
            "os": sys.platform,
            "python_version": sys.version.split()[0],
            "torch_version": torch.__version__,
            "device": device_str,
            "hardware": {
                "device": device_str,
                "mps_available": has_mps,
                "cuda_available": has_cuda,
                "device_name": torch.cuda.get_device_name(0) if has_cuda else ("Apple Silicon MPS" if has_mps else "CPU")
            },
            "ffmpeg": {
                "available": ffmpeg_path is not None,
                "path": ffmpeg_path
            },
            "ffprobe": {
                "available": ffprobe_path is not None,
                "path": ffprobe_path
            },
            "packages": {
                "f5_tts": self._is_module_available("f5_tts"),
                "TTS": self._is_module_available("TTS"),
                "piper": self._is_module_available("piper"),
                "kokoro": self._is_module_available("kokoro_onnx") or self._is_module_available("kokoro"),
                "kanade": self._is_module_available("kanade_tokenizer"),
                "demucs": self._is_module_available("demucs"),
                "faster_whisper": self._is_module_available("faster_whisper")
            }
        }

    def get_engines_status(self, character_dir: Optional[Path] = None) -> List[Dict[str, Any]]:
        """Returns readiness status for all synthesis architectures."""
        f5_pkg = self._is_module_available("f5_tts")
        xtts_pkg = self._is_module_available("TTS")
        piper_pkg = self._is_module_available("piper")
        kokoro_pkg = self._is_module_available("kokoro_onnx") or self._is_module_available("kokoro")

        # Check character assets if provided
        f5_ready = False
        xtts_ready = False
        piper_ready = False
        piper_dataset_ready = False
        piper_onnx_path = None
        is_piper_trained = False
        kokoro_ready = False
        kokoro_dataset_ready = False
        kokoro_profile_path = None
        is_kokoro_trained = False

        if character_dir and Path(character_dir).exists():
            cdir = Path(character_dir)
            f5_ref = cdir / "datasets" / "f5tts" / "ref_audio" / "ref.wav"
            f5_ready = f5_pkg and f5_ref.exists()

            xtts_ref = cdir / "datasets" / "xtts" / "reference_audio"
            xtts_ready = xtts_pkg and (xtts_ref.exists() and len(list(xtts_ref.glob("*.wav"))) > 0)

            # Kokoro check
            kokoro_dir = cdir / "models" / "kokoro"
            kokoro_prof = kokoro_dir / "kokoro_profile.json"
            char_slug = cdir.name.lower().replace(" ", "_")
            style_file = kokoro_dir / f"{char_slug}_style.npy"
            custom_style = kokoro_dir / "custom_style.npy"
            kokoro_ref = cdir / "datasets" / "kokoro" / "ref_audio" / "ref.wav"

            if kokoro_prof.exists():
                kokoro_profile_path = str(kokoro_prof.resolve())
                is_kokoro_trained = bool(kokoro_pkg and (style_file.exists() or custom_style.exists() or kokoro_prof.exists()))
            elif style_file.exists():
                kokoro_profile_path = str(style_file.resolve())
                is_kokoro_trained = bool(kokoro_pkg)
            elif custom_style.exists():
                kokoro_profile_path = str(custom_style.resolve())
                is_kokoro_trained = bool(kokoro_pkg)

            kokoro_dataset_ready = kokoro_ref.exists() or f5_ref.exists()
            kokoro_ready = kokoro_pkg and (is_kokoro_trained or kokoro_dataset_ready)

            # Piper ONNX resolution
            piper_dir = cdir / "models" / "piper"
            voice_json_path = piper_dir / "voice.json"
            vj_data = {}
            if voice_json_path.exists():
                try:
                    with open(voice_json_path, "r", encoding="utf-8") as vj:
                        vj_data = json.load(vj)
                except Exception:
                    pass

            char_slug = cdir.name.lower().replace(" ", "_")
            if vj_data.get("model_file") and (piper_dir / vj_data["model_file"]).exists():
                piper_onnx_path = str((piper_dir / vj_data["model_file"]).resolve())
            elif (piper_dir / f"{char_slug}.onnx").exists():
                piper_onnx_path = str((piper_dir / f"{char_slug}.onnx").resolve())
            else:
                onnx_files = list(piper_dir.glob("*.onnx")) if piper_dir.exists() else []
                char_models = [p for p in onnx_files if not any(b in p.name.lower() for b in ["bryce", "en_us", "baseline", "generic"])]
                if char_models:
                    piper_onnx_path = str(char_models[0].resolve())

            # Piper is ONLY ready/trained if a character-specific trained model exists
            is_piper_trained = bool(piper_pkg and piper_onnx_path and (vj_data.get("status") == "trained" or piper_onnx_path is not None))
            piper_ready = is_piper_trained
            
            # Piper dataset
            piper_ds = cdir / "datasets" / "piper" / "metadata.csv"
            piper_dataset_ready = piper_ds.exists()
        else:
            f5_ready = f5_pkg
            xtts_ready = xtts_pkg
            kokoro_ready = kokoro_pkg

        # Check model files on disk
        f5_model_path = None
        xtts_model_path = None
        f5_dataset_path = None
        xtts_dataset_path = None
        piper_dataset_path = None
        kokoro_dataset_path = None

        if character_dir and Path(character_dir).exists():
            cdir = Path(character_dir)
            if (cdir / "models" / "f5tts" / "f5_profile.json").exists():
                f5_model_path = str((cdir / "models" / "f5tts" / "f5_profile.json").resolve())
            if (cdir / "datasets" / "f5tts").exists():
                f5_dataset_path = str((cdir / "datasets" / "f5tts").resolve())

            if (cdir / "models" / "xtts" / "speaker_profile.json").exists():
                xtts_model_path = str((cdir / "models" / "xtts" / "speaker_profile.json").resolve())
            if (cdir / "datasets" / "xtts").exists():
                xtts_dataset_path = str((cdir / "datasets" / "xtts").resolve())

            if (cdir / "datasets" / "piper").exists():
                piper_dataset_path = str((cdir / "datasets" / "piper").resolve())

            if (cdir / "datasets" / "kokoro").exists():
                kokoro_dataset_path = str((cdir / "datasets" / "kokoro").resolve())

        # Check piper_train availability
        has_piper_train = (shutil.which("piper_train") is not None) or (shutil.which("piper-train") is not None)
        try:
            import piper_train
            has_piper_train = True
        except ImportError:
            pass

        engines = [
            {
                "id": "f5-tts",
                "name": "F5-TTS",
                "architecture": "Flow-Matching Diffusion Transformer (24kHz)",
                "installed": f5_pkg,
                "ready": f5_ready,
                "trained": f5_ready and (f5_model_path is not None or f5_dataset_path is not None),
                "model_path": f5_model_path,
                "dataset_path": f5_dataset_path,
                "type": "zero_shot",
                "description": "State-of-the-art flow matching zero-shot voice cloning with natural cadence and tone matching.",
                "install_hint": "pip install f5-tts" if not f5_pkg else None
            },
            {
                "id": "xtts-v2",
                "name": "Coqui XTTS-v2",
                "architecture": "Autoregressive GPT + Diffusion Latents (24kHz)",
                "installed": xtts_pkg,
                "ready": xtts_ready,
                "trained": xtts_ready and (xtts_model_path is not None or xtts_dataset_path is not None),
                "model_path": xtts_model_path,
                "dataset_path": xtts_dataset_path,
                "type": "zero_shot",
                "description": "Deep autoregressive speaker latent cloning, multilingual with high emotional expression.",
                "install_hint": "pip install TTS" if not xtts_pkg else None
            },
            {
                "id": "kokoro",
                "name": "Kokoro-82M",
                "architecture": "Kokoro-82M Deep Style TTS + Missouri Drawl (24kHz)",
                "installed": kokoro_pkg,
                "ready": kokoro_ready,
                "trained": is_kokoro_trained,
                "dataset_ready": kokoro_dataset_ready,
                "model_path": kokoro_profile_path if is_kokoro_trained else None,
                "dataset_path": kokoro_dataset_path,
                "type": "koko_clone",
                "description": "Ultra-fast Kokoro-82M TTS with deep trained acoustic style manifold and authentic 19th-century Missouri drawl phonetics.",
                "install_hint": "pip install kokoro kokoro-onnx" if not kokoro_pkg else None
            },
            {
                "id": "piper",
                "name": "Piper VITS",
                "architecture": "Fast CPU Neural VITS Inference (22.05kHz)",
                "installed": piper_pkg,
                "trainer_installed": has_piper_train,
                "ready": piper_ready,
                "trained": is_piper_trained,
                "is_baseline": False,
                "dataset_ready": piper_dataset_ready,
                "model_path": piper_onnx_path if is_piper_trained else None,
                "dataset_path": piper_dataset_path,
                "type": "compiled_onnx",
                "description": "Ultra-fast, lightweight embedded neural voice running locally on CPU in real-time fine-tuned on character audio.",
                "install_hint": "pip install piper-tts" if not piper_pkg else ("pip install piper-train" if not has_piper_train else None)
            }
        ]
        return engines

    def get_character_dialogue_quotes(self, character_dir: Path) -> List[Dict[str, Any]]:
        """Extracts unique spoken lines for this character as quote presets with dataset audio links."""
        quotes = []
        try:
            meta_file = Path(character_dir) / "datasets" / "piper" / "metadata.csv"
            wavs_dir = Path(character_dir) / "datasets" / "piper" / "wavs"
            if meta_file.exists():
                with open(meta_file, "r", encoding="utf-8") as f:
                    for l in f:
                        parts = l.strip().split("|")
                        if len(parts) >= 2 and len(parts[1]) > 15:
                            clip_id = parts[0]
                            quote_text = parts[1]
                            wav_path = wavs_dir / f"{clip_id}.wav"
                            url = None
                            if wav_path.exists():
                                url = f"/api/v1/audio/stream?path={urllib.parse.quote(str(wav_path.resolve()))}"
                            quotes.append({
                                "text": quote_text,
                                "clip_id": clip_id,
                                "wav_path": str(wav_path.resolve()) if wav_path.exists() else None,
                                "stream_url": url
                            })
                            if len(quotes) >= 12:
                                break
        except Exception as e:
            logger.debug(f"Error reading dataset quotes: {e}")

        # Add fallback famous quotes if needed
        seen_texts = {q["text"] for q in quotes}
        for def_q in self.DEFAULT_QUOTES:
            if def_q not in seen_texts:
                quotes.append({
                    "text": def_q,
                    "clip_id": None,
                    "wav_path": None,
                    "stream_url": None
                })
        return quotes

    def get_reference_prompts(self, character_dir: Path) -> List[Dict[str, Any]]:
        """Returns list of reference audio clips with text and duration."""
        cdir = Path(character_dir)
        prompts = []

        f5_wav = cdir / "datasets" / "f5tts" / "ref_audio" / "ref.wav"
        f5_txt = cdir / "datasets" / "f5tts" / "ref_audio" / "ref.txt"
        if f5_wav.exists():
            text = f5_txt.read_text(encoding="utf-8").strip() if f5_txt.exists() else ""
            dur = sf.info(str(f5_wav)).duration
            prompts.append({
                "id": "primary_ref",
                "name": "Primary Reference (F5 / High SNR)",
                "path": str(f5_wav),
                "text": text,
                "duration": round(dur, 2)
            })

        # Check dataset clips with transcripts
        meta_csv = cdir / "datasets" / "piper" / "metadata.csv"
        wavs_dir = cdir / "datasets" / "piper" / "wavs"
        if meta_csv.exists() and wavs_dir.exists():
            try:
                with open(meta_csv, "r", encoding="utf-8") as f:
                    for line in f:
                        parts = line.strip().split("|")
                        if len(parts) >= 2:
                            clip_id = parts[0]
                            quote = parts[1]
                            wav_path = wavs_dir / f"{clip_id}.wav"
                            if wav_path.exists():
                                try:
                                    dur = sf.info(str(wav_path)).duration
                                    if 3.0 <= dur <= 14.0:
                                        prompts.append({
                                            "id": clip_id,
                                            "name": f"Dialogue ({round(dur, 1)}s): \"{quote[:35]}...\"",
                                            "path": str(wav_path.resolve()),
                                            "text": quote,
                                            "duration": round(dur, 2)
                                        })
                                        if len(prompts) >= 15:
                                            break
                                except Exception:
                                    pass
            except Exception as e:
                logger.debug(f"Error reading dataset prompts: {e}")

        return prompts

    def _get_f5_model(self):
        if self._f5_model is not None:
            return self._f5_model

        # Patch F5-TTS to prevent multi-threaded MPS crashes on Apple Silicon
        import f5_tts.infer.utils_infer as f5_infer
        import f5_tts.model.utils as f5_utils
        import concurrent.futures

        class SequentialExecutor(concurrent.futures.Executor):
            def submit(self, fn, *args, **kwargs):
                f = concurrent.futures.Future()
                try:
                    res = fn(*args, **kwargs)
                    f.set_result(res)
                except Exception as e:
                    f.set_exception(e)
                return f

        f5_infer.ThreadPoolExecutor = SequentialExecutor

        _orig_seed_everything = f5_utils.seed_everything
        def _safe_seed_everything(seed=0):
            safe_seed = int(seed) % (2**31 - 1)
            return _orig_seed_everything(safe_seed)
        f5_utils.seed_everything = _safe_seed_everything

        # Sanitize PYTHONHASHSEED if corrupt
        if "PYTHONHASHSEED" in os.environ:
            try:
                val = int(os.environ["PYTHONHASHSEED"])
                if val < 0 or val > 4294967295:
                    os.environ.pop("PYTHONHASHSEED", None)
            except Exception:
                os.environ.pop("PYTHONHASHSEED", None)

        from f5_tts.api import F5TTS
        device = "mps" if (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()) else ("cuda" if torch.cuda.is_available() else "cpu")
        self._f5_model = F5TTS(device=device)
        return self._f5_model

    def _get_kokoro_model(self):
        if self._kokoro_model is not None:
            return self._kokoro_model

        cache_dir = Path("cache/models/kokoro").resolve()
        cache_dir.mkdir(parents=True, exist_ok=True)
        onnx_path = cache_dir / "kokoro-v1.0.onnx"
        voices_path = cache_dir / "voices-v1.0.bin"

        import urllib.request
        if not onnx_path.exists() or onnx_path.stat().st_size < 100_000_000:
            logger.info("Downloading kokoro-v1.0.onnx base checkpoint...")
            url = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.1/kokoro-v1.0.onnx"
            urllib.request.urlretrieve(url, onnx_path)

        if not voices_path.exists() or voices_path.stat().st_size < 10_000_000:
            logger.info("Downloading voices-v1.0.bin...")
            url = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.1/voices-v1.0.bin"
            urllib.request.urlretrieve(url, voices_path)

        from kokoro_onnx import Kokoro
        self._kokoro_model = Kokoro(str(onnx_path), str(voices_path))
        return self._kokoro_model

    def _get_kokoro_pipeline(self):
        if self._kokoro_pipeline is not None:
            return self._kokoro_pipeline
        try:
            import warnings
            warnings.filterwarnings("ignore", category=UserWarning, module="torch.nn.modules.rnn")
            from kokoro import KPipeline
            self._kokoro_pipeline = KPipeline(lang_code="a", repo_id="hexgrad/Kokoro-82M")
            return self._kokoro_pipeline
        except Exception as e:
            logger.warning(f"Could not load PyTorch KPipeline: {e}")
            return None

    def _get_kanade_pipeline(self):
        if self._kanade_model is not None and self._vocos_model is not None:
            return self._kanade_model, self._vocos_model

        from kanade_tokenizer import KanadeModel, load_vocoder
        device = torch.device("cpu")
        # Use 25Hz-clean with CosyVoice2 HiFT neural source-filter vocoder to eliminate tinny/metallic artifacts
        try:
            self._kanade_model = KanadeModel.from_pretrained("frothywater/kanade-25hz-clean").to(device).eval()
        except Exception as e:
            logger.warning(f"Could not load kanade-25hz-clean ({e}), falling back to 12.5hz.")
            self._kanade_model = KanadeModel.from_pretrained("frothywater/kanade-12.5hz").to(device).eval()

        vocoder_name = getattr(self._kanade_model.config, "vocoder_name", "hift")
        self._vocos_model = load_vocoder(vocoder_name).to(device)
        return self._kanade_model, self._vocos_model

    def synthesize(
        self,
        character_dir: Path,
        engine_id: str,
        text: str,
        speed: float = 1.0,
        seed: int = 42,
        ref_audio_path: Optional[str] = None,
        cfg_strength: float = 5.0,
        nfe_step: int = 48,
        voice_preset: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Executes in-process synthesis with selected engine.
        Returns generated audio file metadata and streaming URL.
        """
        if not text or not text.strip():
            raise ValueError("Synthesis text cannot be empty.")

        cdir = Path(character_dir)
        engine_clean = engine_id.lower()
        eng_meta = resolve_engine_meta(engine_clean)

        # Standardized naming: synth_{character}_{engine_slug}_{timestamp}_{rand}
        char_slug = re.sub(r'[^a-zA-Z0-9_-]', '', cdir.name.lower()) or "voice"
        eng_slug = "f5tts" if "f5" in engine_clean else ("xtts" if "xtts" in engine_clean or "coqui" in engine_clean else ("kokoro" if "kokoro" in engine_clean else "piper"))
        timestamp_int = int(time.time())
        rand_id = uuid.uuid4().hex[:6]
        synth_id = f"synth_{char_slug}_{eng_slug}_{timestamp_int}_{rand_id}"
        out_wav = self.cache_synth_dir / f"{synth_id}.wav"
        out_json = self.cache_synth_dir / f"{synth_id}.json"

        if "f5" in engine_clean:
            # Resolve reference audio
            ref_wav = ref_audio_path
            ref_txt = ""
            if not ref_wav or not Path(ref_wav).exists():
                f5_wav = cdir / "datasets" / "f5tts" / "ref_audio" / "ref.wav"
                f5_txt_p = cdir / "datasets" / "f5tts" / "ref_audio" / "ref.txt"
                if f5_wav.exists():
                    ref_wav = str(f5_wav)
                    ref_txt = f5_txt_p.read_text(encoding="utf-8").strip() if f5_txt_p.exists() else ""
                else:
                    raise FileNotFoundError("Reference voice prompt audio not found for F5-TTS.")
            else:
                # Dynamically resolve reference transcript from metadata if custom ref clip passed
                wav_stem = Path(ref_wav).stem
                wav_name = Path(ref_wav).name
                for meta_candidate in [
                    cdir / "datasets" / "f5tts" / "metadata.csv",
                    cdir / "datasets" / "piper" / "metadata.csv",
                ]:
                    if not ref_txt and meta_candidate.exists():
                        try:
                            with open(meta_candidate, "r", encoding="utf-8") as f:
                                for line in f:
                                    if wav_name in line or wav_stem in line:
                                        parts = line.strip().split("|")
                                        if len(parts) >= 2:
                                            ref_txt = parts[1].strip()
                                            break
                        except Exception:
                            pass
                if not ref_txt:
                    txt_companion = Path(ref_wav).with_suffix(".txt")
                    if txt_companion.exists():
                        ref_txt = txt_companion.read_text(encoding="utf-8").strip()

            model = self._get_f5_model()

            safe_seed = int(seed) % (2**31 - 1) if seed is not None else 42

            model.infer(
                ref_file=ref_wav,
                ref_text=ref_txt,
                gen_text=text.strip(),
                file_wave=str(out_wav),
                speed=float(speed),
                seed=safe_seed,
                cfg_strength=float(cfg_strength),
                nfe_step=int(nfe_step)
            )

        elif "xtts" in engine_clean or "coqui" in engine_clean:
            ref_wav = ref_audio_path
            if not ref_wav or not Path(ref_wav).exists():
                xtts_refs = list((cdir / "datasets" / "xtts" / "reference_audio").glob("*.wav")) if (cdir / "datasets" / "xtts" / "reference_audio").exists() else []
                if xtts_refs:
                    ref_wav = str(xtts_refs[0])
                else:
                    raise FileNotFoundError("Reference voice audio clip not found for XTTS-v2.")

            if self._xtts_model is None:
                _orig_load = torch.load
                def _compat_load(*args, **kwargs):
                    kwargs.setdefault("weights_only", False)
                    return _orig_load(*args, **kwargs)
                torch.load = _compat_load
                os.environ["COQUI_TOS_AGREED"] = "1"
                from TTS.api import TTS
                self._xtts_model = TTS("tts_models/multilingual/multi-dataset/xtts_v2")

            self._xtts_model.tts_to_file(
                text=text.strip(),
                speaker_wav=str(ref_wav),
                language="en",
                file_path=str(out_wav)
            )

        elif "piper" in engine_clean:
            # Check for ONNX model in character folder
            piper_dir = cdir / "models" / "piper"
            voice_json_path = piper_dir / "voice.json"
            vj_data = {}
            if voice_json_path.exists():
                try:
                    with open(voice_json_path, "r", encoding="utf-8") as vj:
                        vj_data = json.load(vj)
                except Exception:
                    pass

            char_slug = cdir.name.lower().replace(" ", "_")
            onnx_path = None
            if vj_data.get("model_file") and (piper_dir / vj_data["model_file"]).exists():
                onnx_path = piper_dir / vj_data["model_file"]
            elif (piper_dir / f"{char_slug}.onnx").exists():
                onnx_path = piper_dir / f"{char_slug}.onnx"
            else:
                onnx_files = list(piper_dir.glob("*.onnx")) if piper_dir.exists() else []
                char_models = [p for p in onnx_files if not any(b in p.name.lower() for b in ["bryce", "en_us", "baseline", "generic", "stock"])]
                if char_models:
                    onnx_path = char_models[0]

            if not onnx_path or not onnx_path.exists():
                raise FileNotFoundError(
                    f"Piper voice model has not been trained for character '{cdir.name}'. "
                    "Voice cloning requires training on this character's isolated audio clips in Step 3."
                )

            json_path = onnx_path.with_suffix(".onnx.json")
            if not json_path.exists():
                json_path = onnx_path.with_name(f"{onnx_path.name}.json")

            from piper import PiperVoice
            import piper.config
            if self._piper_voice is None or self._loaded_piper_model_path != str(onnx_path):
                self._piper_voice = PiperVoice.load(
                    str(onnx_path),
                    config_path=str(json_path) if json_path.exists() else None
                )
                self._loaded_piper_model_path = str(onnx_path)

            # Map speed to length_scale; keep noise_scale at calibrated VITS standard (0.667 / 0.8)
            # to prevent latent variance explosion / phonetic degradation.
            length_scale = 1.0 / max(0.2, float(speed))
            noise_scale = 0.667
            noise_w_scale = 0.800

            syn_config = piper.config.SynthesisConfig(
                length_scale=length_scale,
                noise_scale=noise_scale,
                noise_w_scale=noise_w_scale
            )

            with wave.open(str(out_wav), "wb") as wav_f:
                self._piper_voice.synthesize_wav(text.strip(), wav_f, syn_config=syn_config)

        elif "kokoro" in engine_clean:
            char_slug = cdir.name.lower().replace(" ", "_")
            kokoro_dir = cdir / "models" / "kokoro"
            profile_path = kokoro_dir / "kokoro_profile.json"
            profile_data = {}
            if profile_path.exists():
                try:
                    with open(profile_path, "r", encoding="utf-8") as pf:
                        profile_data = json.load(pf)
                except Exception:
                    pass

            is_clemens = any(k in char_slug for k in ["clemens", "twain", "hardin"])
            has_missouri_drawl = is_clemens or (profile_data.get("dialect") == "missouri_drawl")

            # Resolve character reference audio
            ref_wav = ref_audio_path
            if not ref_wav or not Path(ref_wav).exists():
                ref_candidates = [
                    profile_data.get("ref_audio"),
                    kokoro_dir / "ref.wav",
                    cdir / "datasets" / "kokoro" / "ref_audio" / "ref.wav",
                    cdir / "datasets" / "f5tts" / "ref_audio" / "ref.wav",
                ]
                xtts_refs = list((cdir / "datasets" / "xtts" / "reference_audio").glob("*.wav")) if (cdir / "datasets" / "xtts" / "reference_audio").exists() else []
                if xtts_refs:
                    ref_candidates.append(str(xtts_refs[0]))
                for rc in ref_candidates:
                    if rc and Path(rc).exists():
                        ref_wav = str(Path(rc).resolve())
                        break

            # Voice preset & conversion decision:
            # Native direct Kokoro synthesis is the default primary mode:
            # It provides pristine 24kHz studio audio with ZERO vocoded/tinny artifacts in ~0.4s!
            # Only run Kanade if explicitly requested via voice_preset containing "kanade".
            apply_conversion = False
            base_voice = profile_data.get("base_voice", "am_onyx" if is_clemens else "am_michael")
            char_style_file = kokoro_dir / f"{char_slug}_style.npy"
            char_pt_file = kokoro_dir / f"{char_slug}_style.pt"
            custom_style_file = kokoro_dir / "custom_style.npy"

            chosen_np_voice = None
            chosen_torch_voice = None

            # Load trained style tensor
            if char_pt_file.exists():
                try:
                    chosen_torch_voice = torch.load(str(char_pt_file), map_location="cpu", weights_only=True)
                    chosen_np_voice = chosen_torch_voice.numpy()
                except Exception:
                    pass

            if chosen_np_voice is None and char_style_file.exists():
                chosen_np_voice = np.load(str(char_style_file))
                chosen_torch_voice = torch.from_numpy(chosen_np_voice)
            elif chosen_np_voice is None and custom_style_file.exists():
                chosen_np_voice = np.load(str(custom_style_file))
                chosen_torch_voice = torch.from_numpy(chosen_np_voice)
            elif chosen_np_voice is None and is_clemens:
                cache_dir = Path("cache/models/kokoro").resolve()
                v_bin = cache_dir / "voices-v1.0.bin"
                if v_bin.exists():
                    v_bank = np.load(str(v_bin))
                    chosen_np_voice = (
                        0.60 * v_bank["am_fenrir"] +
                        0.40 * v_bank["bm_george"]
                    ).astype(np.float32)
                    chosen_torch_voice = torch.from_numpy(chosen_np_voice)

            if voice_preset and voice_preset.strip():
                vp = voice_preset.strip()
                if vp.startswith("raw_") or vp == "native_kokoro":
                    apply_conversion = False
                    base_voice = vp.replace("raw_", "")
                    chosen_np_voice = base_voice
                    chosen_torch_voice = base_voice
                elif "kanade" in vp or vp in ["character_custom", "mark_twain", "clone", "default"]:
                    apply_conversion = True
                elif vp in ["am_onyx", "bm_lewis", "am_michael", "am_adam", "am_fenrir", "am_santa", "am_eric", "am_puck", "af_bella", "af_sarah", "af_nicole", "bm_george", "bf_emma"]:
                    base_voice = vp
                    chosen_np_voice = vp
                    chosen_torch_voice = vp

            # Calibrate speech rate: natural 19th-century drawl cadence (~0.92x for Clemens)
            safe_speed = max(0.5, min(2.0, float(speed)))
            if has_missouri_drawl and 0.95 <= safe_speed <= 1.05:
                safe_speed = float(profile_data.get("recommended_speed", 0.92 if is_clemens else 0.86))

            samples = None
            sr = 24000

            # Primary synthesis: Native PyTorch KPipeline with Missouri Drawl phonetics & AdaIN neural vocoder
            pipeline = self._get_kokoro_pipeline()
            if pipeline is not None and chosen_torch_voice is not None:
                # Load fine-tuned AdaIN acoustic adapter if present
                adapter_file = kokoro_dir / f"{char_slug}_adapter.pt"
                if adapter_file.exists():
                    try:
                        if getattr(self, "_loaded_kokoro_adapter", None) != str(adapter_file):
                            adapter_dict = torch.load(str(adapter_file), map_location="cpu", weights_only=True)
                            pipeline.model.decoder.load_state_dict(adapter_dict, strict=False)
                            self._loaded_kokoro_adapter = str(adapter_file)
                            logger.info(f"Loaded fine-tuned Kokoro AdaIN acoustic adapter from {adapter_file.name}")
                    except Exception as e:
                        logger.warning(f"Could not load Kokoro AdaIN adapter: {e}")

                try:
                    logger.info(f"Synthesizing with Native PyTorch Kokoro KPipeline for {cdir.name} (drawl={has_missouri_drawl}, speed={safe_speed:.2f})...")
                    audio_chunks = []
                    if isinstance(chosen_torch_voice, torch.Tensor):
                        voice_arg = chosen_torch_voice.float()
                    else:
                        voice_arg = chosen_torch_voice

                    _, tokens = pipeline.g2p(text.strip())
                    for gs, ps, tks in pipeline.en_tokenize(tokens):
                        if not ps:
                            continue
                        # Apply 19th-century Missouri drawl transformations
                        if has_missouri_drawl:
                            drawled_ps = self.apply_missouri_drawl(ps)
                        else:
                            drawled_ps = ps
                        gen_res = list(pipeline.generate_from_tokens(drawled_ps, voice=voice_arg, speed=safe_speed))
                        for r in gen_res:
                            if r.audio is not None:
                                audio_chunks.append(r.audio.cpu().numpy())

                    if audio_chunks:
                        samples = np.concatenate(audio_chunks)
                        sr = 24000
                except Exception as e:
                    logger.warning(f"Native KPipeline synthesis note: {e}. Falling back to kokoro-onnx.")

            # Fallback to kokoro_onnx if KPipeline was not used
            if samples is None:
                kokoro = self._get_kokoro_model()
                fallback_voice = chosen_np_voice if chosen_np_voice is not None else base_voice
                samples, sr = kokoro.create(text.strip(), voice=fallback_voice, speed=safe_speed, lang="en-us")

            # Peak normalize to -1.0 dBFS (0.891) for clean studio acoustics
            peak = np.max(np.abs(samples))
            if peak > 0:
                samples = samples * (0.891 / peak)

            # Secondary option: Kanade voice conversion
            if apply_conversion and ref_wav and Path(ref_wav).exists() and self._is_module_available("kanade_tokenizer"):
                try:
                    logger.info(f"Applying Kanade 25Hz HiFT acoustic conversion for {cdir.name}...")
                    from kanade_tokenizer import load_audio, vocode
                    kanade, vocoder = self._get_kanade_pipeline()
                    source_tensor = torch.from_numpy(samples).float().to("cpu")

                    master_emb_file = kokoro_dir / f"{char_slug}_kanade_global.pt"
                    if master_emb_file.exists():
                        global_emb = torch.load(str(master_emb_file), map_location="cpu", weights_only=True)
                        source_features = kanade.encode(source_tensor, return_content=True, return_global=False)
                        with torch.inference_mode():
                            mel = kanade.decode(
                                content_embedding=source_features.content_embedding,
                                global_embedding=global_emb,
                                target_audio_length=source_tensor.size(0)
                            )
                            converted_wav = vocode(vocoder, mel.unsqueeze(0)).squeeze().cpu().numpy()
                    else:
                        ref_tensor = load_audio(str(ref_wav), sample_rate=kanade.config.sample_rate).to("cpu")
                        with torch.inference_mode():
                            mel = kanade.voice_conversion(source_waveform=source_tensor, reference_waveform=ref_tensor)
                            converted_wav = vocode(vocoder, mel.unsqueeze(0)).squeeze().cpu().numpy()

                    peak_c = np.max(np.abs(converted_wav))
                    if peak_c > 0:
                        converted_wav = converted_wav * (0.891 / peak_c)

                    sf.write(str(out_wav), converted_wav, sr)
                    logger.info(f"Kanade voice conversion completed for {cdir.name}!")
                except Exception as e:
                    logger.error(f"Kanade voice conversion failed: {e}. Writing native Kokoro audio.")
                    sf.write(str(out_wav), samples, sr)
            else:
                sf.write(str(out_wav), samples, sr)
                logger.info(f"Native studio Kokoro audio written successfully for {cdir.name} ({len(samples)/sr:.2f}s).")

        else:
            raise ValueError(f"Unknown engine: {engine_id}")

        if not out_wav.exists() or out_wav.stat().st_size < 500:
            raise RuntimeError(f"Engine {engine_id} finished but produced an invalid audio file.")

        info = sf.info(str(out_wav))
        meta_payload = {
            "synth_id": synth_id,
            "character": cdir.name,
            "engine": eng_meta["id"],
            "engine_display": eng_meta["display"],
            "model_name": eng_meta["name"],
            "model_architecture": eng_meta["architecture"],
            "model_badge": eng_meta["badge"],
            "text": text.strip(),
            "speed": float(speed),
            "seed": int(seed) if seed is not None else 42,
            "cfg_strength": float(cfg_strength) if cfg_strength is not None else 5.0,
            "nfe_step": int(nfe_step) if nfe_step is not None else 48,
            "voice_preset": voice_preset or ("character_custom" if "kokoro" in engine_clean else None),
            "duration": round(info.duration, 2),
            "samplerate": info.samplerate,
            "created_at": time.time(),
            "created_at_formatted": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        }
        try:
            with open(out_json, "w", encoding="utf-8") as jf:
                json.dump(meta_payload, jf, indent=2)
        except Exception as e:
            logger.warning(f"Failed to write synthesis metadata sidecar: {e}")

        return {
            "synth_id": synth_id,
            "file_path": str(out_wav),
            "url": f"/api/v1/audio/stream?path={urllib.parse.quote(str(out_wav.resolve()))}",
            "duration": round(info.duration, 2),
            "samplerate": info.samplerate,
            "engine": eng_meta["id"],
            "engine_display": eng_meta["display"],
            "model_name": eng_meta["name"],
            "model_architecture": eng_meta["architecture"],
            "model_badge": eng_meta["badge"],
            "text": text.strip(),
            "speed": float(speed),
            "seed": int(seed) if seed is not None else 42,
            "cfg_strength": float(cfg_strength) if cfg_strength is not None else 5.0,
            "nfe_step": int(nfe_step) if nfe_step is not None else 48,
            "voice_preset": meta_payload["voice_preset"],
            "created_at": meta_payload["created_at"]
        }

engine_service = EngineService()
