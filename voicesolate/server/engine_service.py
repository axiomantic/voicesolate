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
            "acceleration": "Apple Silicon (MPS)" if has_mps else ("NVIDIA CUDA" if has_cuda else "CPU Only"),
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
                "demucs": self._is_module_available("demucs"),
                "faster_whisper": self._is_module_available("faster_whisper")
            }
        }

    def get_engines_status(self, character_dir: Optional[Path] = None) -> List[Dict[str, Any]]:
        """Returns readiness status for all synthesis architectures."""
        f5_pkg = self._is_module_available("f5_tts")
        xtts_pkg = self._is_module_available("TTS")
        piper_pkg = self._is_module_available("piper")

        # Check character assets if provided
        f5_ready = False
        xtts_ready = False
        piper_ready = False
        piper_dataset_ready = False
        piper_onnx_path = None

        if character_dir and Path(character_dir).exists():
            cdir = Path(character_dir)
            f5_ref = cdir / "datasets" / "f5tts" / "ref_audio" / "ref.wav"
            f5_ready = f5_pkg and f5_ref.exists()

            xtts_ref = cdir / "datasets" / "xtts" / "reference_audio"
            xtts_ready = xtts_pkg and (xtts_ref.exists() and len(list(xtts_ref.glob("*.wav"))) > 0)

            # Piper ONNX
            piper_models = list((cdir / "models" / "piper").glob("*.onnx")) if (cdir / "models" / "piper").exists() else []
            if piper_models:
                piper_ready = piper_pkg
                piper_onnx_path = str(piper_models[0])
            
            # Piper dataset
            piper_ds = cdir / "datasets" / "piper" / "metadata.csv"
            piper_dataset_ready = piper_ds.exists()
        else:
            f5_ready = f5_pkg
            xtts_ready = xtts_pkg

        # Check model files on disk
        f5_model_path = None
        xtts_model_path = None
        f5_dataset_path = None
        xtts_dataset_path = None
        piper_dataset_path = None

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

        # Check piper_train availability
        has_piper_train = (shutil.which("piper_train") is not None) or (shutil.which("piper-train") is not None)
        try:
            import piper_train
            has_piper_train = True
        except ImportError:
            pass

        piper_is_baseline = False
        if piper_onnx_path:
            p_name = Path(piper_onnx_path).name.lower()
            if "bryce" in p_name or "en_us" in p_name or "baseline" in p_name:
                piper_is_baseline = True
            voice_json = Path(piper_onnx_path).parent / "voice.json"
            if voice_json.exists():
                try:
                    with open(voice_json, "r") as vj:
                        vj_data = json.load(vj)
                        if vj_data.get("status") in ["ready_to_train", "baseline"]:
                            piper_is_baseline = True
                except Exception:
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
                "id": "piper",
                "name": "Piper VITS",
                "architecture": "Fast CPU Neural VITS Inference (22.05kHz)",
                "installed": piper_pkg,
                "trainer_installed": has_piper_train,
                "ready": piper_ready,
                "trained": piper_ready and piper_onnx_path is not None and not piper_is_baseline,
                "is_baseline": piper_is_baseline,
                "dataset_ready": piper_dataset_ready,
                "model_path": piper_onnx_path if not piper_is_baseline else None,
                "dataset_path": piper_dataset_path,
                "type": "compiled_onnx",
                "description": "Ultra-fast, lightweight embedded neural voice running locally on CPU in real-time. (Requires piper-train to fine-tune on character data).",
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

    def synthesize(
        self,
        character_dir: Path,
        engine_id: str,
        text: str,
        speed: float = 1.0,
        seed: int = 42,
        ref_audio_path: Optional[str] = None,
        cfg_strength: float = 2.5,
        nfe_step: int = 32
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
        eng_slug = "f5tts" if "f5" in engine_clean else ("xtts" if "xtts" in engine_clean or "coqui" in engine_clean else "piper")
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
            piper_models = list((cdir / "models" / "piper").glob("*.onnx")) if (cdir / "models" / "piper").exists() else []
            if not piper_models:
                raise FileNotFoundError(
                    "Piper ONNX model not compiled yet for this character. "
                    "Use the Dataset & Engine Hub to compile or download a voice profile."
                )

            onnx_path = piper_models[0]
            json_path = onnx_path.with_suffix(".onnx.json")
            if not json_path.exists():
                json_path = onnx_path.with_name(f"{onnx_path.name}.json")

            from piper import PiperVoice
            if self._piper_voice is None or self._loaded_piper_model_path != str(onnx_path):
                self._piper_voice = PiperVoice.load(
                    str(onnx_path),
                    config_path=str(json_path) if json_path.exists() else None
                )
                self._loaded_piper_model_path = str(onnx_path)

            with wave.open(str(out_wav), "wb") as wav_f:
                self._piper_voice.synthesize_wav(text.strip(), wav_f)

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
            "text": text.strip()
        }

engine_service = EngineService()
