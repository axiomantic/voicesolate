import os
import sys
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

        engines = [
            {
                "id": "f5-tts",
                "name": "F5-TTS",
                "architecture": "Flow-Matching Diffusion Transformer (24kHz)",
                "installed": f5_pkg,
                "ready": f5_ready,
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
                "type": "zero_shot",
                "description": "Deep autoregressive speaker latent cloning, multilingual with high emotional expression.",
                "install_hint": "pip install TTS" if not xtts_pkg else None
            },
            {
                "id": "piper",
                "name": "Piper VITS",
                "architecture": "Fast CPU Neural VITS Inference (22.05kHz)",
                "installed": piper_pkg,
                "ready": piper_ready,
                "dataset_ready": piper_dataset_ready,
                "onnx_model": piper_onnx_path,
                "type": "compiled_onnx",
                "description": "Ultra-fast, lightweight embedded neural voice running locally on CPU in real-time.",
                "install_hint": "pip install piper-tts" if not piper_pkg else None
            }
        ]
        return engines

    def get_character_dialogue_quotes(self, character_dir: Path) -> List[str]:
        """Extracts unique spoken lines for this character as quote presets."""
        quotes = list(self.DEFAULT_QUOTES)
        try:
            meta_file = Path(character_dir) / "datasets" / "piper" / "metadata.csv"
            if meta_file.exists():
                lines = []
                with open(meta_file, "r", encoding="utf-8") as f:
                    for l in f:
                        parts = l.strip().split("|")
                        if len(parts) >= 2 and len(parts[1]) > 20:
                            lines.append(parts[1])
                if lines:
                    # Prepend top 5 character show lines
                    quotes = lines[:6] + [q for q in self.DEFAULT_QUOTES if q not in lines]
        except Exception:
            pass
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

        # Check raw or enhanced clips
        enhanced_dir = cdir / "enhanced"
        if enhanced_dir.exists():
            for f in list(enhanced_dir.glob("*.wav"))[:10]:
                try:
                    dur = sf.info(str(f)).duration
                    prompts.append({
                        "id": f.stem,
                        "name": f"Isolated Stem: {f.stem[:25]}",
                        "path": str(f),
                        "text": "",
                        "duration": round(dur, 2)
                    })
                except Exception:
                    pass

        return prompts

    def synthesize(
        self,
        character_dir: Path,
        engine_id: str,
        text: str,
        speed: float = 1.0,
        seed: int = 42,
        ref_audio_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Executes in-process synthesis with selected engine.
        Returns generated audio file metadata and streaming URL.
        """
        if not text or not text.strip():
            raise ValueError("Synthesis text cannot be empty.")

        cdir = Path(character_dir)
        synth_id = f"synth_{int(uuid.uuid4().hex[:10], 16)}_{int(speed*100)}"
        out_wav = self.cache_synth_dir / f"{synth_id}.wav"

        engine_clean = engine_id.lower()

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

            if self._f5_model is None:
                from f5_tts.api import F5TTS
                self._f5_model = F5TTS()

            safe_seed = int(seed) % (2**31 - 1) if seed else 42

            self._f5_model.infer(
                ref_file=ref_wav,
                ref_text=ref_txt,
                gen_text=text.strip(),
                file_wave=str(out_wav),
                speed=float(speed),
                seed=safe_seed
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

            with sf.SoundFile(str(out_wav), mode='w', samplerate=22050, channels=1, subtype='PCM_16') as wav_f:
                self._piper_voice.synthesize(text.strip(), wav_f)

        else:
            raise ValueError(f"Unknown engine: {engine_id}")

        if not out_wav.exists() or out_wav.stat().st_size < 500:
            raise RuntimeError(f"Engine {engine_id} finished but produced an invalid audio file.")

        info = sf.info(str(out_wav))
        return {
            "synth_id": synth_id,
            "file_path": str(out_wav),
            "url": f"/api/v1/audio/stream?path={out_wav}",
            "duration": round(info.duration, 2),
            "samplerate": info.samplerate,
            "engine": engine_id,
            "text": text
        }

engine_service = EngineService()
