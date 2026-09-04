import os
import subprocess
import tempfile
import torch
import torchaudio
import soundfile as sf
import numpy as np
import threading
from pathlib import Path
from typing import Optional, Any

_DEMUCS_LOCK = threading.RLock()

class AudioEnhancer:
    """
    High-Fidelity Neural Vocal Isolation & Natural Dynamic Mastering Pipeline.
    
    1. Demucs Hybrid Transformer (HTDemucs) on GPU/MPS with Shift-Averaging (shifts=2, overlap=0.25)
       for artifact-free, un-quantized vocal separation.
    2. Zero Gating / Zero Ducking: 100% continuous, organic speech flow with natural acoustic decay.
    3. Full-Spectrum Studio Preservation (48kHz, 24-bit PCM).
    """

    def __init__(self, device: Optional[str] = None, cache_manager: Optional[Any] = None):
        if device:
            self.device = device
        else:
            self.device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
        
        self.cache = cache_manager
        self._demucs_model = None
        self._vocal_idx = None

    def _get_demucs_model(self):
        """Lazy loads HTDemucs on GPU/MPS."""
        with _DEMUCS_LOCK:
            if self._demucs_model is None:
                try:
                    from demucs.pretrained import get_model
                    model = get_model("htdemucs")
                    model.to(self.device)
                    model.eval()
                    self._demucs_model = model
                    self._vocal_idx = model.sources.index("vocals")
                except Exception as e:
                    print(f"Warning: Failed to load Demucs model: {e}")
            return self._demucs_model

    def clean_and_enhance_file(self, input_wav: str, output_wav: str, media_key: Optional[str] = None, timecode_str: Optional[str] = None) -> str:
        """
        Runs high-fidelity neural vocal isolation and natural dynamic mastering.
        Reuses cached neural vocal stem if available to avoid repeating GPU demucs.
        """
        in_path = Path(input_wav)
        out_path = Path(output_wav)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        cached_stem = None
        if self.cache and self.cache.use_cache_enhance and media_key and timecode_str:
            stem_p = self.cache.get_stem_path(media_key, timecode_str)
            if stem_p.exists():
                cached_stem = stem_p

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_isolated = os.path.join(tmp_dir, "isolated_vocals.wav")

            if cached_stem:
                source_file = str(cached_stem)
            else:
                # Stage 1: Neural Demucs Isolation with shift averaging
                success = self.isolate_vocals_neural(str(in_path), tmp_isolated)
                source_file = tmp_isolated if success else str(in_path)
                
                # Save into cache if available
                if success and self.cache and media_key and timecode_str:
                    stem_p = self.cache.get_stem_path(media_key, timecode_str)
                    try:
                        import shutil
                        shutil.copy(tmp_isolated, str(stem_p))
                    except Exception:
                        pass

            # Stage 2: Natural Dynamic Mastering (Zero gating, transparent peak headroom)
            self.master_vocal_natural(source_file, str(out_path))

        return str(out_path)

    def isolate_vocals_neural(self, input_wav: str, output_wav: str) -> bool:
        """
        Isolates vocals using Demucs Hybrid Transformer on GPU/MPS with shift averaging.
        """
        model = self._get_demucs_model()
        if model is None:
            return False

        try:
            from demucs.apply import apply_model
            data, sr = sf.read(input_wav, dtype="float32")
            
            if data.ndim == 1:
                wav = torch.from_numpy(data).unsqueeze(0).repeat(2, 1)
            elif data.ndim == 2:
                wav = torch.from_numpy(data.T)
                if wav.shape[0] == 1:
                    wav = wav.repeat(2, 1)
                elif wav.shape[0] > 2:
                    wav = wav[:2]
            else:
                return False

            if sr != model.samplerate:
                resampler = torchaudio.transforms.Resample(sr, model.samplerate)
                wav = resampler(wav)
                sr = model.samplerate

            with _DEMUCS_LOCK:
                wav_tensor = wav.to(self.device)
                with torch.no_grad():
                    sources = apply_model(model, wav_tensor[None], device=self.device, shifts=2, overlap=0.25, progress=False)[0]
                vocal_stem = sources[self._vocal_idx].cpu()
            vocal_mono = vocal_stem.mean(dim=0, keepdim=True)

            resample_48k = torchaudio.transforms.Resample(sr, 48000)
            vocal_48k = resample_48k(vocal_mono).squeeze(0).numpy()

            # Normalize peak to -1.0dB without compression or gating
            max_val = np.abs(vocal_48k).max()
            if max_val > 0:
                vocal_48k = vocal_48k / max_val * 0.89

            sf.write(output_wav, vocal_48k, 48000, subtype="PCM_24")
            return os.path.exists(output_wav) and os.path.getsize(output_wav) > 0
        except Exception as e:
            print(f"Demucs isolation failed: {e}")
            return False

    def master_vocal_natural(self, input_wav: str, output_wav: str):
        """
        Natural Vocal Mastering:
        - Subsonic 35Hz highpass to remove DC thumps.
        - Preserves 100% of vocal fundamentals and natural chest body.
        - ZERO noise gating, ZERO compression pumping, ZERO chopped words.
        """
        filter_chain = "highpass=f=35,alimiter=limit=0.92:attack=5:release=50"
        cmd = [
            "ffmpeg", "-nostdin", "-loglevel", "error", "-y",
            "-i", input_wav,
            "-af", filter_chain,
            "-ar", "48000",
            "-ac", "1",
            "-c:a", "pcm_s24le",
            output_wav
        ]
        try:
            res = subprocess.run(cmd, capture_output=True, text=True)
            if res.returncode != 0:
                # Fallback: copy with soundfile if FFmpeg filter chain choked
                data, sr = sf.read(input_wav)
                sf.write(output_wav, data, sr, subtype="PCM_24")
        except Exception:
            data, sr = sf.read(input_wav)
            sf.write(output_wav, data, sr, subtype="PCM_24")
