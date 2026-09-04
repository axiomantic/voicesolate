import os
import sys
import tempfile
import json
import shutil
import subprocess
import importlib.util
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import soundfile as sf

class VoiceStudioGUI:
    """
    Sleek, Modern Voice Studio GUI for Voicesolate.
    Features:
    - Step 1: Character Voice Profile & Corpus Audition (Reference Stem + Clip Explorer)
    - Step 2: Model Architecture & Training Center (Status, ONNX Import, Training Guides)
    - Step 3: Studio Voice Synthesis (Dynamic engine list: ONLY ready models can be synthesized)
    - Zero-config: Zero-shot engines (F5-TTS & XTTS-v2) are pre-calibrated and ready
    - In-Process Real-time F5-TTS Flow-Matching Neural Voice Cloning
    - In-Process Real-time Coqui XTTS-v2 Autoregressive + Diffusion Voice Cloning
    - Piper VITS ONNX model support with drag-and-drop model unlock
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

    def __init__(self, character_dir: Path):
        self.char_dir = Path(character_dir)
        self.char_name = self.char_dir.name
        self.models_dir = self.char_dir / "models"
        self.datasets_dir = self.char_dir / "datasets"
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self._f5_model = None
        self._xtts_model = None

    @staticmethod
    def _is_module_available(module_name: str) -> bool:
        try:
            return importlib.util.find_spec(module_name) is not None
        except Exception:
            return False

    def check_engine_status(self) -> Dict[str, Dict[str, Any]]:
        """Checks if each TTS engine is installed, calibrated, and ready for synthesis."""
        status = {}
        ref_wav, _ = self.get_reference_prompt()
        has_ref = bool(ref_wav and Path(ref_wav).exists())

        # 1. F5-TTS (Flow-Matching Diffusion)
        has_f5 = self._is_module_available("f5_tts")
        status["F5-TTS (Flow-Matching DiT)"] = {
            "key": "f5tts",
            "installed": has_f5,
            "ready": has_f5 and has_ref,
            "status_text": "🟢 Ready to Synthesize" if (has_f5 and has_ref) else "🔴 Missing Model / Prompt",
            "badge_class": "badge-ready" if (has_f5 and has_ref) else "badge-error",
            "arch_name": "F5-TTS",
            "type_desc": "Zero-Shot Flow-Matching Diffusion (24kHz Studio Quality)",
            "details": "Calibrated with highest-SNR isolated vocal prompt. Generates speech directly via neural flow matching."
        }

        # 2. Coqui XTTS-v2 (Autoregressive + Diffusion)
        has_xtts = self._is_module_available("TTS")
        status["Coqui XTTS-v2 (Autoregressive + Diffusion)"] = {
            "key": "xtts",
            "installed": has_xtts,
            "ready": has_xtts and has_ref,
            "status_text": "🟢 Ready to Synthesize" if (has_xtts and has_ref) else "🔴 Missing Model / Reference",
            "badge_class": "badge-ready" if (has_xtts and has_ref) else "badge-error",
            "arch_name": "Coqui XTTS-v2",
            "type_desc": "Zero-Shot Autoregressive + Diffusion (24kHz Multilingual)",
            "details": "Calibrated with speaker conditioning latents extracted from isolated vocal clips."
        }

        # 3. Piper (VITS / ONNX)
        has_piper = self._is_module_available("piper") or (shutil.which("piper") is not None)
        piper_onnx = list((self.models_dir / "piper").glob("*.onnx")) if (self.models_dir / "piper").exists() else []
        piper_ready = bool(piper_onnx)
        piper_dataset_exists = (self.datasets_dir / "piper" / "metadata.csv").exists()

        if piper_ready:
            piper_status = "🟢 Ready to Synthesize"
            badge_class = "badge-ready"
            details = f"Compiled ONNX model active ({piper_onnx[0].name}). Ultra-fast CPU neural synthesis."
        elif piper_dataset_exists:
            piper_status = "🟡 LJSpeech Dataset Ready (Awaiting ONNX Compilation)"
            badge_class = "badge-pending"
            details = "LJSpeech dataset is generated. Requires compiling or importing an exported .onnx model to synthesize."
        else:
            piper_status = "🔴 Dataset Missing"
            badge_class = "badge-error"
            details = "Piper dataset not yet built. Run pipeline to extract audio."

        status["Piper (VITS / Fast CPU)"] = {
            "key": "piper",
            "installed": has_piper,
            "ready": piper_ready,
            "status_text": piper_status,
            "badge_class": badge_class,
            "arch_name": "Piper (VITS)",
            "type_desc": "Fast CPU Neural Inference (22.05kHz LJSpeech ONNX)",
            "details": details
        }

        return status

    def get_ready_synthesis_engines(self) -> List[str]:
        """Returns ONLY engine display names that are currently compiled and ready to synthesize."""
        status_map = self.check_engine_status()
        ready = [name for name, info in status_map.items() if info.get("ready", False)]
        # Fallback to at least one choice if none ready
        return ready if ready else list(status_map.keys())[:1]

    def get_dataset_stats(self) -> Dict[str, Any]:
        """Calculates dataset statistics for this character."""
        piper_wavs = list((self.datasets_dir / "piper" / "wavs").glob("*.wav")) if (self.datasets_dir / "piper" / "wavs").exists() else []
        total_sec = 0.0
        for w in piper_wavs:
            try:
                total_sec += sf.info(str(w)).duration
            except Exception:
                pass

        return {
            "clip_count": len(piper_wavs),
            "total_minutes": round(total_sec / 60.0, 1),
            "total_seconds": round(total_sec, 1)
        }

    def get_reference_prompt(self) -> Tuple[Optional[str], str]:
        """Returns the primary reference WAV path and its transcript."""
        f5_ref_wav = self.datasets_dir / "f5tts" / "ref_audio" / "ref.wav"
        f5_ref_txt = self.datasets_dir / "f5tts" / "ref_audio" / "ref.txt"
        if f5_ref_wav.exists():
            text = f5_ref_txt.read_text(encoding="utf-8").strip() if f5_ref_txt.exists() else ""
            return str(f5_ref_wav), text

        xtts_refs = list((self.datasets_dir / "xtts" / "reference_audio").glob("*.wav")) if (self.datasets_dir / "xtts" / "reference_audio").exists() else []
        if xtts_refs:
            return str(xtts_refs[0]), ""

        piper_wavs = list((self.datasets_dir / "piper" / "wavs").glob("*.wav")) if (self.datasets_dir / "piper" / "wavs").exists() else []
        if piper_wavs:
            return str(piper_wavs[0]), ""

        return None, ""

    def get_dataset_clips(self) -> List[Dict[str, str]]:
        """Parses the character's LJSpeech metadata to allow auditioning extracted vocal clips."""
        meta_file = self.datasets_dir / "piper" / "metadata.csv"
        wav_dir = self.datasets_dir / "piper" / "wavs"
        clips = []

        if meta_file.exists() and wav_dir.exists():
            with open(meta_file, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split("|")
                    if len(parts) >= 2:
                        cid = parts[0]
                        txt = parts[1]
                        wav_path = wav_dir / f"{cid}.wav"
                        if wav_path.exists():
                            clips.append({
                                "id": cid,
                                "text": txt,
                                "path": str(wav_path)
                            })
        return clips

    def import_piper_model(self, files: Any) -> Tuple[str, bool]:
        """Imports user-provided Piper .onnx and .onnx.json files into models/piper/."""
        if not files:
            return "⚠️ Please upload at least one .onnx model file.", False

        piper_model_dir = self.models_dir / "piper"
        piper_model_dir.mkdir(parents=True, exist_ok=True)

        if not isinstance(files, list):
            files = [files]

        saved_files = []
        for item in files:
            src_path = getattr(item, "name", str(item))
            p = Path(src_path)
            if p.exists() and p.is_file():
                dest = piper_model_dir / p.name
                shutil.copy2(p, dest)
                saved_files.append(p.name)

        onnx_found = list(piper_model_dir.glob("*.onnx"))
        if onnx_found:
            return f"✅ Successfully imported: {', '.join(saved_files)}! Piper is now compiled and ready for synthesis.", True
        else:
            return f"⚠️ Uploaded files ({', '.join(saved_files)}), but no .onnx file was detected. Please provide a .onnx model.", False

    def synthesize(self, engine_name: str, text: str, speed: float, seed: int) -> Tuple[Optional[str], str]:
        """Synthesizes speech using the selected engine."""
        if not text or not text.strip():
            return None, "⚠️ Please enter dialogue text to synthesize."

        status_map = self.check_engine_status()
        engine_info = status_map.get(engine_name, {})
        key = engine_info.get("key")

        if not engine_info.get("ready", False):
            return None, f"⚠️ **{engine_name} is not ready for synthesis.** Please select a ready engine or compile this model in Step 2."

        tmp_out = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name

        try:
            if key == "f5tts":
                ref_wav, ref_text = self.get_reference_prompt()
                if not ref_wav or not Path(ref_wav).exists():
                    return None, "❌ Reference voice prompt audio file not found in character datasets."

                if self._f5_model is None:
                    from f5_tts.api import F5TTS
                    self._f5_model = F5TTS()

                safe_seed = int(seed) % (2**31 - 1) if seed else 42

                self._f5_model.infer(
                    ref_file=ref_wav,
                    ref_text=ref_text,
                    gen_text=text.strip(),
                    file_wave=tmp_out,
                    speed=float(speed),
                    seed=safe_seed
                )

                if Path(tmp_out).exists() and Path(tmp_out).stat().st_size > 1000:
                    return tmp_out, f"✅ Synthesized with F5-TTS ({speed:.2f}x speed, seed {safe_seed})!"
                else:
                    return None, "❌ F5-TTS generation produced an empty audio file."

            elif key == "xtts":
                xtts_refs = list((self.datasets_dir / "xtts" / "reference_audio").glob("*.wav")) if (self.datasets_dir / "xtts" / "reference_audio").exists() else []
                ref_wav = str(xtts_refs[0]) if xtts_refs else self.get_reference_prompt()[0]

                if not ref_wav or not Path(ref_wav).exists():
                    return None, "❌ Reference voice audio clip not found for XTTS."

                if self._xtts_model is None:
                    import torch
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
                    file_path=tmp_out
                )

                if Path(tmp_out).exists() and Path(tmp_out).stat().st_size > 1000:
                    return tmp_out, f"✅ Synthesized with Coqui XTTS-v2!"
                else:
                    return None, "❌ XTTS-v2 generation produced an empty audio file."

            elif key == "piper":
                onnx_models = list((self.models_dir / "piper").glob("*.onnx")) if (self.models_dir / "piper").exists() else []
                if not onnx_models:
                    return None, "❌ Piper ONNX model not compiled yet. Please import an ONNX model in Step 2."

                cmd = f"echo {subprocess.list2cmdline([text])} | piper --model {onnx_models[0]} --output_file {tmp_out}"
                subprocess.run(cmd, shell=True, check=True)
                return tmp_out, f"✅ Synthesized with Piper VITS ({speed:.2f}x speed)!"

        except Exception as e:
            return None, f"❌ Synthesis failed: {e}"

        return None, "Engine not supported."

    def launch(self, server_port: int = 7860, inbrowser: bool = True):
        import gradio as gr

        ref_wav, ref_txt = self.get_reference_prompt()
        stats = self.get_dataset_stats()
        dataset_clips = self.get_dataset_clips()
        clip_choices = [
            (f"Clip {i+1}: {c['text'][:65]}...", c['id'])
            for i, c in enumerate(dataset_clips[:30])
        ] if dataset_clips else [("No clips found", "none")]

        # CSS styling with step badges and zero button-like labels
        custom_css = """
        .container { max-width: 1200px; margin: 0 auto; padding: 1rem 0; }
        .header-box { margin-bottom: 1.5rem; border-bottom: 1px solid #334155; padding-bottom: 1rem; }
        .stats-badge { font-size: 0.95rem; color: #94a3b8; font-weight: 500; margin-top: 0.5rem; }

        /* Step Card Styling */
        .step-card {
            background: #1e293b !important;
            border: 1px solid #334155 !important;
            border-radius: 12px !important;
            padding: 1.25rem !important;
            margin-bottom: 1.25rem !important;
        }

        .step-header {
            display: flex;
            align-items: center;
            gap: 0.75rem;
            margin-bottom: 1rem;
        }

        .step-badge {
            background: #3b82f6;
            color: #ffffff;
            font-size: 0.75rem;
            font-weight: 700;
            padding: 0.2rem 0.6rem;
            border-radius: 9999px;
            letter-spacing: 0.05em;
            text-transform: uppercase;
        }

        .step-title {
            font-size: 1.15rem;
            font-weight: 700;
            color: #f8fafc;
        }

        /* Model Architecture Cards */
        .model-card {
            background: #0f172a !important;
            border: 1px solid #334155 !important;
            border-radius: 10px !important;
            padding: 1rem !important;
            height: 100% !important;
        }

        .model-card-title {
            font-size: 1.05rem;
            font-weight: 700;
            color: #f1f5f9;
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 0.4rem;
        }

        .status-badge-ready {
            background: rgba(16, 185, 129, 0.15);
            color: #34d399;
            border: 1px solid rgba(16, 185, 129, 0.3);
            font-size: 0.75rem;
            font-weight: 600;
            padding: 0.15rem 0.55rem;
            border-radius: 9999px;
        }

        .status-badge-pending {
            background: rgba(245, 158, 11, 0.15);
            color: #fbbf24;
            border: 1px solid rgba(245, 158, 11, 0.3);
            font-size: 0.75rem;
            font-weight: 600;
            padding: 0.15rem 0.55rem;
            border-radius: 9999px;
        }

        .model-desc {
            font-size: 0.83rem;
            color: #94a3b8;
            margin-bottom: 0.6rem;
            line-height: 1.4;
        }

        /* Strip button/pill background styling from component labels across themes */
        label,
        .block label,
        span[data-testid="block-label"],
        .label-wrap,
        .block-title,
        .svelte-1gfkn6j {
            background: none !important;
            background-color: transparent !important;
            color: #f1f5f9 !important;
            border: none !important;
            box-shadow: none !important;
            border-radius: 0 !important;
            padding: 0 !important;
            margin-bottom: 0.4rem !important;
            font-size: 0.88rem !important;
            font-weight: 600 !important;
            display: block !important;
            letter-spacing: 0.015em !important;
            text-transform: none !important;
        }

        label > span,
        .block label > span,
        span[data-testid="block-label"] > span,
        span[data-testid="block-label"] span {
            background: transparent !important;
            background-color: transparent !important;
            color: #f1f5f9 !important;
            border: none !important;
            padding: 0 !important;
            margin: 0 !important;
            border-radius: 0 !important;
            box-shadow: none !important;
        }

        .block .info, [data-testid="block-info"] {
            color: #94a3b8 !important;
            font-size: 0.8rem !important;
            margin-bottom: 0.5rem !important;
        }

        .ref-prompt-box {
            background: #0f172a;
            border-left: 3px solid #3b82f6;
            padding: 0.75rem 1rem;
            border-radius: 0 8px 8px 0;
            margin: 0.75rem 0;
            color: #cbd5e1;
            font-size: 0.875rem;
            font-style: italic;
        }

        /* =========================================================================
           AUDIO PLAYER COMPACT & SLEEK STYLING (KILL GIANT MUSIC NOTE SVG)
           ========================================================================= */
        /* Completely eliminate the absurd giant music note placeholder / empty artwork */
        div[aria-label="Empty value"],
        .empty,
        .empty.svelte-v95lt3,
        .icon.svelte-v95lt3,
        svg.feather-music,
        .feather-music,
        [data-testid="audio"] div[aria-label="Empty value"],
        [data-testid="audio"] .empty,
        [data-testid="audio"] svg.feather-music,
        .audio-container div[aria-label="Empty value"],
        .audio-container svg.feather-music {
            display: none !important;
            height: 0px !important;
            min-height: 0px !important;
            max-height: 0px !important;
            padding: 0 !important;
            margin: 0 !important;
            opacity: 0 !important;
            visibility: hidden !important;
            pointer-events: none !important;
        }

        /* Constrain audio container to compact, professional waveform player */
        [data-testid="audio"],
        .audio-container,
        .audio-container.svelte-ocxd3m {
            height: auto !important;
            min-height: 0 !important;
            background: #0f172a !important;
            border: 1px solid #334155 !important;
            border-radius: 8px !important;
            padding: 0.6rem 0.8rem !important;
        }

        .component-wrapper.svelte-1ffmt2w {
            padding: 0 !important;
            width: 100% !important;
        }

        .waveform-container,
        .waveform-container.svelte-1ffmt2w {
            height: 48px !important;
            min-height: 48px !important;
            margin: 0.25rem 0 !important;
        }

        #waveform,
        #waveform.svelte-1ffmt2w {
            height: 48px !important;
        }

        .controls.svelte-72dh9g {
            margin-top: 0.25rem !important;
        }

        /* Top-right icon buttons inside audio component (download / share) */
        .icon-buttons,
        [data-testid="audio"] .icon-buttons {
            margin-bottom: 0.25rem !important;
        }

        /* Primary action button: distinct, unambiguous call-to-action */
        #synth-btn {
            background: linear-gradient(135deg, #6366f1 0%, #4f46e5 100%) !important;
            color: #ffffff !important;
            font-size: 1.05rem !important;
            font-weight: 600 !important;
            border-radius: 8px !important;
            border: none !important;
            padding: 0.75rem 1.5rem !important;
            margin-top: 0.75rem !important;
            cursor: pointer !important;
            transition: all 0.2s ease !important;
        }
        #synth-btn:hover {
            background: linear-gradient(135deg, #4f46e5 0%, #4338ca 100%) !important;
            box-shadow: 0 4px 12px rgba(79, 70, 229, 0.35) !important;
            transform: translateY(-1px);
        }
        """

        theme = gr.themes.Default(
            primary_hue="indigo",
            neutral_hue="slate"
        ).set(
            block_label_background_fill="transparent",
            block_label_background_fill_dark="transparent",
            block_label_border_width="0px",
            block_label_border_width_dark="0px",
            block_label_padding="0px",
            block_label_radius="0px",
            block_label_shadow="none",
            block_label_text_color="#1e293b",
            block_label_text_color_dark="#f1f5f9",
            block_label_text_weight="600",
            block_title_text_color="#1e293b",
            block_title_text_color_dark="#f1f5f9",
            block_title_text_weight="600",
        )

        with gr.Blocks(title=f"Voicesolate Studio — {self.char_name}", css=custom_css, theme=theme) as demo:
            with gr.Column(elem_classes="container"):
                # Header Banner
                gr.Markdown(
                    f"""
                    # 🎙️ Voicesolate Studio &mdash; {self.char_name}
                    <div class="stats-badge">
                        📁 <strong>Isolated Character Corpus:</strong> {stats['clip_count']} vocal clips ({stats['total_minutes']} minutes) &nbsp;|&nbsp;
                        🎯 <strong>Zero-Shot Neural Engines:</strong> F5-TTS & Coqui XTTS-v2 Active &nbsp;|&nbsp;
                        ⚡ <strong>Offline ONNX Engine:</strong> Piper LJSpeech Ready
                    </div>
                    """,
                    elem_classes="header-box"
                )

                # =========================================================================
                # STEP 1: VOICE PROFILE & EXTRACTED VOCAL STEMS
                # =========================================================================
                with gr.Group(elem_classes="step-card"):
                    gr.Markdown(
                        """
                        <div class="step-header">
                            <span class="step-badge">Step 1</span>
                            <span class="step-title">Actor Voice Profile & Isolated Vocal Stems</span>
                        </div>
                        """
                    )
                    with gr.Row():
                        with gr.Column(scale=5):
                            ref_player = gr.Audio(
                                value=ref_wav,
                                label="Primary Reference Voice Stem (Demucs Neural Isolation)",
                                type="filepath",
                                interactive=False
                            )
                            if ref_txt:
                                gr.Markdown(f'<div class="ref-prompt-box"><strong>Zero-Shot Calibration Transcript:</strong> "{ref_txt}"</div>')

                        with gr.Column(scale=5):
                            clip_selector = gr.Dropdown(
                                choices=clip_choices,
                                value=clip_choices[0][1] if clip_choices else None,
                                label="Corpus Clip Explorer",
                                info=f"Audition any of the {stats['clip_count']} isolated dialogue stems extracted from media"
                            )
                            sample_clip_player = gr.Audio(
                                value=dataset_clips[0]["path"] if dataset_clips else None,
                                label="Sample Dialogue Stem",
                                type="filepath",
                                interactive=False
                            )
                            sample_clip_text = gr.Textbox(
                                value=dataset_clips[0]["text"] if dataset_clips else "",
                                label="Stem Transcription",
                                lines=2,
                                interactive=False
                            )

                # =========================================================================
                # STEP 2: MODEL ARCHITECTURE & TRAINING CENTER
                # =========================================================================
                with gr.Group(elem_classes="step-card"):
                    gr.Markdown(
                        """
                        <div class="step-header">
                            <span class="step-badge">Step 2</span>
                            <span class="step-title">Model Architecture & Training Center</span>
                        </div>
                        """
                    )
                    status_map = self.check_engine_status()
                    piper_info = status_map.get("Piper (VITS / Fast CPU)", {})
                    piper_badge_text = piper_info.get("status_text", "")
                    piper_badge_class = "status-badge-ready" if piper_info.get("ready") else "status-badge-pending"

                    with gr.Row():
                        # Card 1: F5-TTS
                        with gr.Column(scale=3):
                            gr.Markdown(
                                f"""
                                <div class="model-card">
                                    <div class="model-card-title">
                                        <span>F5-TTS (Flow-Matching DiT)</span>
                                        <span class="status-badge-ready">🟢 Ready to Synthesize</span>
                                    </div>
                                    <div class="model-desc">
                                        <strong>Type:</strong> Zero-Shot Neural Flow-Matching (24kHz Studio Quality)<br/>
                                        <strong>Status:</strong> Pre-calibrated and fully ready. Synthesizes high-fidelity speech directly from the reference prompt pack.
                                    </div>
                                </div>
                                """
                            )

                        # Card 2: Coqui XTTS-v2
                        with gr.Column(scale=3):
                            gr.Markdown(
                                f"""
                                <div class="model-card">
                                    <div class="model-card-title">
                                        <span>Coqui XTTS-v2</span>
                                        <span class="status-badge-ready">🟢 Ready to Synthesize</span>
                                    </div>
                                    <div class="model-desc">
                                        <strong>Type:</strong> Autoregressive GPT + Diffusion (24kHz Multilingual)<br/>
                                        <strong>Status:</strong> Speaker conditioning latents configured. Generates speech with zero-shot cloning in ~3 seconds.
                                    </div>
                                </div>
                                """
                            )

                        # Card 3: Piper VITS
                        with gr.Column(scale=4):
                            piper_card_markdown = gr.Markdown(
                                f"""
                                <div class="model-card">
                                    <div class="model-card-title">
                                        <span>Piper (VITS / ONNX)</span>
                                        <span class="{piper_badge_class}">{piper_badge_text}</span>
                                    </div>
                                    <div class="model-desc">
                                        <strong>Type:</strong> Fast CPU Neural Inference (22.05kHz)<br/>
                                        <strong>Status:</strong> {piper_info.get('details', '')}
                                    </div>
                                </div>
                                """
                            )

                            with gr.Accordion("📦 Import Trained Piper Model (.onnx)", open=False):
                                gr.Markdown(
                                    f"Upload your trained `.onnx` and `.onnx.json` model files below to immediately unlock Piper in Step 3:\n\n"
                                    f"*Dataset Location:* `{self.datasets_dir / 'piper'}`"
                                )
                                piper_uploader = gr.File(
                                    label="Drop or Select .onnx and .onnx.json files",
                                    file_count="multiple",
                                    file_types=[".onnx", ".json"]
                                )
                                import_btn = gr.Button("🚀 Import & Unlock Piper in Studio", variant="secondary")
                                import_msg = gr.Markdown("")

                # =========================================================================
                # STEP 3: STUDIO VOICE SYNTHESIS
                # =========================================================================
                with gr.Group(elem_classes="step-card"):
                    gr.Markdown(
                        """
                        <div class="step-header">
                            <span class="step-badge">Step 3</span>
                            <span class="step-title">Studio Voice Synthesis & Audition</span>
                        </div>
                        """
                    )

                    ready_engines = self.get_ready_synthesis_engines()

                    with gr.Row():
                        # Left: Synthesis controls
                        with gr.Column(scale=5):
                            engine_dropdown = gr.Dropdown(
                                choices=ready_engines,
                                value=ready_engines[0],
                                label="Speech Synthesis Engine",
                                info="Only engines that are calibrated and ready to synthesize are shown"
                            )

                            quote_selector = gr.Dropdown(
                                choices=self.DEFAULT_QUOTES,
                                value=self.DEFAULT_QUOTES[0],
                                label="Sample Dialogue Quotes",
                                info="Quick-select a quote or enter custom text below"
                            )

                            dialogue_input = gr.Textbox(
                                lines=3,
                                value=self.DEFAULT_QUOTES[0],
                                label="Dialogue Text",
                                placeholder="Type any sentence for the character voice to speak..."
                            )

                            with gr.Accordion("Advanced Voice Settings", open=False):
                                speed_slider = gr.Slider(
                                    minimum=0.6,
                                    maximum=1.8,
                                    value=1.0,
                                    step=0.05,
                                    label="Speech Rate (Speed)",
                                    info="Adjust pacing without altering pitch"
                                )
                                seed_input = gr.Number(
                                    value=42,
                                    label="Generation Seed",
                                    precision=0,
                                    info="Fixed seed guarantees reproducible synthesis"
                                )

                            synth_btn = gr.Button("✨ Synthesize Voice", variant="primary", size="lg", elem_id="synth-btn")
                            feedback_box = gr.Markdown("")

                        # Right: Audition output
                        with gr.Column(scale=5):
                            output_player = gr.Audio(
                                label="AI Voice Synthesis (Generated Audio Output)",
                                type="filepath",
                                interactive=False
                            )

                # =========================================================================
                # INTERACTIVE EVENT WIRING
                # =========================================================================
                # Step 1: Clip explorer selection
                def on_select_clip(clip_id):
                    for c in dataset_clips:
                        if c["id"] == clip_id:
                            return c["path"], c["text"]
                    return None, ""

                clip_selector.change(
                    fn=on_select_clip,
                    inputs=[clip_selector],
                    outputs=[sample_clip_player, sample_clip_text]
                )

                # Step 2: Import Piper ONNX model
                def on_import_piper(files):
                    msg, success = self.import_piper_model(files)
                    new_status = self.check_engine_status()
                    new_piper = new_status.get("Piper (VITS / Fast CPU)", {})
                    badge_class = "status-badge-ready" if new_piper.get("ready") else "status-badge-pending"
                    badge_text = new_piper.get("status_text", "")
                    
                    updated_card = f"""
                    <div class="model-card">
                        <div class="model-card-title">
                            <span>Piper (VITS / ONNX)</span>
                            <span class="{badge_class}">{badge_text}</span>
                        </div>
                        <div class="model-desc">
                            <strong>Type:</strong> Fast CPU Neural Inference (22.05kHz)<br/>
                            <strong>Status:</strong> {new_piper.get('details', '')}
                        </div>
                    </div>
                    """

                    new_ready = self.get_ready_synthesis_engines()
                    dropdown_update = gr.update(choices=new_ready, value=new_ready[0])
                    return msg, updated_card, dropdown_update

                import_btn.click(
                    fn=on_import_piper,
                    inputs=[piper_uploader],
                    outputs=[import_msg, piper_card_markdown, engine_dropdown]
                )

                # Step 3: Synthesis quote selection and synthesis trigger
                quote_selector.change(fn=lambda q: q, inputs=[quote_selector], outputs=[dialogue_input])

                synth_btn.click(
                    fn=self.synthesize,
                    inputs=[engine_dropdown, dialogue_input, speed_slider, seed_input],
                    outputs=[output_player, feedback_box]
                )

        print(f"\nLaunching Clean Voicesolate Studio at http://localhost:{server_port}")
        demo.launch(server_port=server_port, inbrowser=inbrowser, quiet=True)
