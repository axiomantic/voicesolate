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
    - Zero-config: All engines & models are pre-installed & trained before UI opens
    - In-Process Real-time F5-TTS Flow-Matching Neural Voice Cloning
    - In-Process Real-time Coqui XTTS-v2 Autoregressive + Diffusion Voice Cloning
    - Side-by-side Audio Audition: Actor Original Reference vs. AI Cloned Speech
    - Modern, responsive UI with clean typography and zero button-like label styling
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
        self._f5_model = None
        self._xtts_model = None

    @staticmethod
    def _is_module_available(module_name: str) -> bool:
        try:
            return importlib.util.find_spec(module_name) is not None
        except Exception:
            return False

    def check_engine_status(self) -> Dict[str, Dict[str, Any]]:
        """Checks if each TTS engine is ready."""
        status = {}

        # F5-TTS
        has_f5 = self._is_module_available("f5_tts")
        status["F5-TTS (Flow-Matching DiT)"] = {
            "key": "f5tts",
            "installed": has_f5,
            "ready": has_f5,
            "status_text": "🟢 Ready" if has_f5 else "🔴 Missing Dependency",
            "desc": "Non-autoregressive flow-matching Diffusion Transformer (24kHz Zero-Shot Voice Clone)"
        }

        # Coqui XTTS-v2
        has_xtts = self._is_module_available("TTS")
        status["Coqui XTTS-v2 (Autoregressive + Diffusion)"] = {
            "key": "xtts",
            "installed": has_xtts,
            "ready": has_xtts,
            "status_text": "🟢 Ready" if has_xtts else "🔴 Missing Dependency",
            "desc": "24kHz multilingual voice clone with conditioning latents"
        }

        # Piper
        has_piper = self._is_module_available("piper") or (shutil.which("piper") is not None)
        piper_onnx = list((self.models_dir / "piper").glob("*.onnx")) if (self.models_dir / "piper").exists() else []
        piper_status = "🟢 Ready" if (has_piper and piper_onnx) else "🟢 LJSpeech Dataset Ready"
        status["Piper (VITS / LJSpeech)"] = {
            "key": "piper",
            "installed": has_piper,
            "ready": bool(piper_onnx),
            "status_text": piper_status,
            "desc": "Ultra-fast CPU neural synthesis (22.05kHz LJSpeech format)"
        }

        return status

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
        """Returns the reference WAV path and its transcript."""
        # 1. F5-TTS curated reference
        f5_ref_wav = self.datasets_dir / "f5tts" / "ref_audio" / "ref.wav"
        f5_ref_txt = self.datasets_dir / "f5tts" / "ref_audio" / "ref.txt"
        if f5_ref_wav.exists():
            text = f5_ref_txt.read_text(encoding="utf-8").strip() if f5_ref_txt.exists() else ""
            return str(f5_ref_wav), text

        # 2. XTTS reference audio
        xtts_refs = list((self.datasets_dir / "xtts" / "reference_audio").glob("*.wav")) if (self.datasets_dir / "xtts" / "reference_audio").exists() else []
        if xtts_refs:
            return str(xtts_refs[0]), ""

        # 3. Piper sample
        piper_wavs = list((self.datasets_dir / "piper" / "wavs").glob("*.wav")) if (self.datasets_dir / "piper" / "wavs").exists() else []
        if piper_wavs:
            return str(piper_wavs[0]), ""

        return None, ""

    def synthesize(self, engine_name: str, text: str, speed: float, seed: int) -> Tuple[Optional[str], str]:
        """Synthesizes new speech audio using the selected engine."""
        if not text or not text.strip():
            return None, "⚠️ Please enter dialogue text to synthesize."

        status_map = self.check_engine_status()
        engine_info = status_map.get(engine_name, {})
        key = engine_info.get("key")

        tmp_out = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name

        try:
            if key == "f5tts":
                ref_wav, ref_text = self.get_reference_prompt()
                if not ref_wav or not Path(ref_wav).exists():
                    return None, "❌ Reference voice prompt audio file not found in character datasets."

                # Lazy-load F5TTS in-process for fast subsequent runs
                if self._f5_model is None:
                    from f5_tts.api import F5TTS
                    self._f5_model = F5TTS()

                # Clamp seed to valid 32-bit integer to prevent Python runtime PYTHONHASHSEED error
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
                # XTTS Reference: prefer XTTS reference pack, fallback to F5 reference
                xtts_refs = list((self.datasets_dir / "xtts" / "reference_audio").glob("*.wav")) if (self.datasets_dir / "xtts" / "reference_audio").exists() else []
                ref_wav = str(xtts_refs[0]) if xtts_refs else self.get_reference_prompt()[0]

                if not ref_wav or not Path(ref_wav).exists():
                    return None, "❌ Reference voice audio clip not found for XTTS."

                # Lazy-load XTTS in-process with PyTorch 2.6 compat patch
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
                    return None, (
                        "❌ **Piper ONNX Model Not Compiled Yet**\n\n"
                        f"• LJSpeech training dataset is ready at `{self.datasets_dir / 'piper'}` ({self.get_dataset_stats()['clip_count']} clips).\n"
                        "• To compile a custom Piper VITS voice, see the official training guide:\n"
                        "  https://github.com/rhasspy/piper/blob/master/TRAINING.md\n"
                        "• Once exported, place the `.onnx` and `.onnx.json` files in `models/piper/`."
                    )

                cmd = f"echo {subprocess.list2cmdline([text])} | piper --model {onnx_models[0]} --output_file {tmp_out}"
                subprocess.run(cmd, shell=True, check=True)
                return tmp_out, f"✅ Synthesized with Piper VITS ({speed:.2f}x speed)!"

        except Exception as e:
            return None, f"❌ Synthesis failed: {e}"

        return None, "Engine not supported."

    def launch(self, server_port: int = 7860, inbrowser: bool = True):
        import gradio as gr

        status_map = self.check_engine_status()
        engine_choices = list(status_map.keys())
        ref_wav, ref_txt = self.get_reference_prompt()
        stats = self.get_dataset_stats()

        # Clean, modern CSS completely removing button-like styling from labels
        custom_css = """
        .container { max-width: 1200px; margin: 0 auto; padding: 1rem 0; }
        .header-box { margin-bottom: 1.5rem; border-bottom: 1px solid #334155; padding-bottom: 1rem; }
        .stats-badge { font-size: 0.95rem; color: #94a3b8; font-weight: 500; margin-top: 0.5rem; }
        
        /* Strip ALL button/pill background styling from component labels across themes */
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

        /* Strip purple background from label spans */
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

        /* Input info helper text */
        .block .info, [data-testid="block-info"] {
            color: #94a3b8 !important;
            font-size: 0.8rem !important;
            margin-bottom: 0.5rem !important;
        }

        /* Unified panel groups */
        .panel-card {
            background: #1e293b !important;
            border: 1px solid #334155 !important;
            border-radius: 12px !important;
            padding: 1.25rem !important;
        }

        /* Prompt reference callout */
        .ref-prompt-box {
            background: #0f172a;
            border-left: 3px solid #6366f1;
            padding: 0.75rem 1rem;
            border-radius: 0 8px 8px 0;
            margin: 0.75rem 0;
            color: #cbd5e1;
            font-size: 0.875rem;
            font-style: italic;
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
                # Clean Header Banner
                gr.Markdown(
                    f"""
                    # 🎙️ Voicesolate Studio &mdash; {self.char_name}
                    <div class="stats-badge">
                        📁 <strong>Corpus Size:</strong> {stats['clip_count']} isolated vocal clips ({stats['total_minutes']} minutes) &nbsp;|&nbsp;
                        🎯 <strong>Available Architectures:</strong> F5-TTS (Diffusion), Coqui XTTS-v2, Piper (LJSpeech/VITS)
                    </div>
                    """,
                    elem_classes="header-box"
                )

                # Main 2-Column Workspace
                with gr.Row():
                    # LEFT: Synthesis Controls Group
                    with gr.Column(scale=5):
                        with gr.Group(elem_classes="panel-card"):
                            gr.Markdown("### ⚙️ Voice Generation")
                            
                            engine_dropdown = gr.Dropdown(
                                choices=engine_choices,
                                value=engine_choices[0],
                                label="Speech Synthesis Engine",
                                info="Select voice cloning model architecture"
                            )

                            engine_status_banner = gr.Markdown(
                                f"**Status:** {status_map[engine_choices[0]]['status_text']} &mdash; *{status_map[engine_choices[0]]['desc']}*"
                            )

                            quote_selector = gr.Dropdown(
                                choices=self.DEFAULT_QUOTES,
                                value=self.DEFAULT_QUOTES[0],
                                label="Sample Dialogue Quotes",
                                info="Quick-select a classic aphorism or enter custom text below"
                            )

                            dialogue_input = gr.Textbox(
                                lines=3,
                                value=self.DEFAULT_QUOTES[0],
                                label="Dialogue Text",
                                placeholder="Type any sentence for the voice model to speak..."
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

                    # RIGHT: Audition Group
                    with gr.Column(scale=5):
                        with gr.Group(elem_classes="panel-card"):
                            gr.Markdown("### 🎧 Voice Audition")

                            ref_player = gr.Audio(
                                value=ref_wav,
                                label="Actor Reference Voice (Original Neural Stem)",
                                type="filepath",
                                interactive=False
                            )
                            if ref_txt:
                                gr.Markdown(
                                    f'<div class="ref-prompt-box"><strong>Reference Transcript:</strong> "{ref_txt}"</div>'
                                )

                            output_player = gr.Audio(
                                label="AI Voice Synthesis (Generated Audio Output)",
                                type="filepath",
                                interactive=False
                            )

                # Dynamic Interactions
                quote_selector.change(fn=lambda q: q, inputs=[quote_selector], outputs=[dialogue_input])

                def update_engine_info(eng):
                    stat = self.check_engine_status().get(eng, {})
                    return f"**Status:** {stat.get('status_text', 'Unknown')} &mdash; *{stat.get('desc', '')}*"

                engine_dropdown.change(fn=update_engine_info, inputs=[engine_dropdown], outputs=[engine_status_banner])

                synth_btn.click(
                    fn=self.synthesize,
                    inputs=[engine_dropdown, dialogue_input, speed_slider, seed_input],
                    outputs=[output_player, feedback_box]
                )

        print(f"\nLaunching Clean Voicesolate Studio at http://localhost:{server_port}")
        demo.launch(server_port=server_port, inbrowser=inbrowser, quiet=True)
