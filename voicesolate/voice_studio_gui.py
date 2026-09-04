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
    - Real-time Engine Installation & Status Detection (F5-TTS, Piper, Coqui XTTS)
    - One-Click Engine Installer directly from the UI
    - In-Process Real-time F5-TTS Neural Voice Cloning & Synthesis
    - Side-by-side Audio Audition: Actor Original Reference vs. AI Cloned Speech
    - Responsive, uncluttered UI without redundant boxes or button-like titles
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

    @staticmethod
    def _is_module_available(module_name: str) -> bool:
        try:
            return importlib.util.find_spec(module_name) is not None
        except Exception:
            return False

    def check_engine_status(self) -> Dict[str, Dict[str, Any]]:
        """Checks if each TTS engine is installed and ready."""
        status = {}

        # F5-TTS
        has_f5 = self._is_module_available("f5_tts") or (shutil.which("f5-tts_infer-cli") is not None)
        status["F5-TTS (Flow-Matching DiT)"] = {
            "key": "f5tts",
            "installed": has_f5,
            "ready": has_f5,
            "package": "f5-tts",
            "install_cmd": f"{sys.executable} -m pip install f5-tts",
            "status_text": "🟢 Ready" if has_f5 else "🔴 Not Installed",
            "desc": "Non-autoregressive flow-matching Diffusion Transformer (24kHz Zero-Shot Voice Clone)"
        }

        # Piper
        has_piper = self._is_module_available("piper") or (shutil.which("piper") is not None)
        piper_onnx = list((self.models_dir / "piper").glob("*.onnx")) if (self.models_dir / "piper").exists() else []
        piper_status = "🟢 Ready" if (has_piper and piper_onnx) else ("🟡 Ready for LJSpeech Training" if has_piper else "🔴 Not Installed")
        status["Piper (VITS / ONNX)"] = {
            "key": "piper",
            "installed": has_piper,
            "ready": has_piper and bool(piper_onnx),
            "package": "piper-tts",
            "install_cmd": f"{sys.executable} -m pip install piper-tts",
            "status_text": piper_status,
            "desc": "Ultra-fast CPU neural synthesis (22.05kHz LJSpeech / VITS)"
        }

        # XTTS-v2
        has_xtts = self._is_module_available("TTS") or (shutil.which("tts") is not None)
        status["Coqui XTTS-v2"] = {
            "key": "xtts",
            "installed": has_xtts,
            "ready": has_xtts,
            "package": "TTS",
            "install_cmd": f"{sys.executable} -m pip install TTS",
            "status_text": "🟢 Ready" if has_xtts else "🔴 Not Installed",
            "desc": "24kHz autoregressive + diffusion voice cloning"
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

    def install_engine(self, engine_choice: str) -> str:
        """Installs the selected TTS engine into the current environment."""
        status_map = self.check_engine_status()
        info = status_map.get(engine_choice)
        if not info:
            return f"❌ Unknown engine: {engine_choice}"

        pkg = info["package"]
        cmd = [sys.executable, "-m", "pip", "install", pkg]
        try:
            res = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            if res.returncode == 0:
                return f"✅ Successfully installed {pkg}! Refresh the engine list to begin synthesis."
            else:
                return f"❌ Installation failed:\n{res.stderr[-500:]}"
        except Exception as e:
            return f"❌ Error executing install: {e}"

    def synthesize(self, engine_name: str, text: str, speed: float, seed: int) -> Tuple[Optional[str], str]:
        """Synthesizes new speech audio using the selected engine."""
        if not text or not text.strip():
            return None, "⚠️ Please enter dialogue text to synthesize."

        status_map = self.check_engine_status()
        engine_info = status_map.get(engine_name, {})
        key = engine_info.get("key")

        if not engine_info.get("installed"):
            install_cmd = engine_info.get("install_cmd", f"pip install {engine_info.get('package', '')}")
            return None, f"⚠️ **{engine_name} is not installed.**\n\nRun `{install_cmd}` or use the Engine Manager below to install it."

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
                    return tmp_out, f"✅ Synthesized successfully with F5-TTS ({speed:.2f}x speed, seed {safe_seed})!"
                else:
                    return None, "❌ F5-TTS generation produced an empty audio file."

            elif key == "piper":
                onnx_models = list((self.models_dir / "piper").glob("*.onnx")) if (self.models_dir / "piper").exists() else []
                if not onnx_models:
                    return None, "⚠️ Piper LJSpeech training dataset is ready, but no compiled `.onnx` model was found in `models/piper/`."
                
                cmd = f"echo {subprocess.list2cmdline([text])} | piper --model {onnx_models[0]} --output_file {tmp_out}"
                subprocess.run(cmd, shell=True, check=True)
                return tmp_out, f"✅ Synthesized successfully with Piper VITS ({speed:.2f}x speed)!"

            elif key == "xtts":
                ref_wav, _ = self.get_reference_prompt()
                cmd = [
                    sys.executable, "-m", "TTS.bin.synthesize",
                    "--model_name", "tts_models/multilingual/multi-dataset/xtts_v2",
                    "--text", text,
                    "--speaker_wav", str(ref_wav),
                    "--language_idx", "en",
                    "--out_path", tmp_out
                ]
                subprocess.run(cmd, check=True)
                return tmp_out, "✅ Synthesized successfully with Coqui XTTS-v2!"

        except Exception as e:
            return None, f"❌ Synthesis failed: {e}"

        return None, "Engine not supported."

    def launch(self, server_port: int = 7860, inbrowser: bool = True):
        import gradio as gr

        status_map = self.check_engine_status()
        engine_choices = list(status_map.keys())
        ref_wav, ref_txt = self.get_reference_prompt()
        stats = self.get_dataset_stats()

        # Modern, clean CSS without card borders around titles
        custom_css = """
        .container { max-width: 1150px; margin: 0 auto; }
        .header-box { margin-bottom: 1.25rem; }
        .stats-badge { font-size: 0.95rem; color: #64748b; font-weight: 500; }
        .status-chip { display: inline-block; padding: 3px 10px; border-radius: 12px; font-size: 0.82rem; font-weight: 600; }
        """

        theme = gr.themes.Soft(
            primary_hue="indigo",
            neutral_hue="slate"
        )

        with gr.Blocks(title=f"Voicesolate Studio — {self.char_name}", css=custom_css, theme=theme) as demo:
            with gr.Column(elem_classes="container"):
                # Clean Header Banner
                gr.Markdown(
                    f"""
                    # 🎙️ Voicesolate Studio &mdash; {self.char_name}
                    <div class="stats-badge">
                        📁 <strong>Corpus Size:</strong> {stats['clip_count']} isolated vocal clips ({stats['total_minutes']} minutes) &nbsp;|&nbsp;
                        🎯 <strong>Available Architectures:</strong> F5-TTS (Diffusion), Piper (LJSpeech/VITS), XTTS-v2
                    </div>
                    """,
                    elem_classes="header-box"
                )

                # Main 2-Column Workspace
                with gr.Row():
                    # LEFT: Synthesis Controls
                    with gr.Column(scale=5):
                        engine_dropdown = gr.Dropdown(
                            choices=engine_choices,
                            value=engine_choices[0],
                            label="Speech Synthesis Engine",
                            info="Select the voice clone model architecture"
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

                        synth_btn = gr.Button("✨ Synthesize Voice", variant="primary", size="lg")
                        feedback_box = gr.Markdown("")

                    # RIGHT: Audition & Side-by-Side Comparison
                    with gr.Column(scale=5):
                        gr.Markdown("### 🎧 Audio Audition")

                        ref_player = gr.Audio(
                            value=ref_wav,
                            label="Actor Reference Voice (Original Neural Stem)",
                            type="filepath",
                            interactive=False
                        )
                        if ref_txt:
                            gr.Markdown(f"*Reference Prompt Text:* \"{ref_txt}\"")

                        output_player = gr.Audio(
                            label="AI Voice Synthesis (Generated Audio Output)",
                            type="filepath",
                            interactive=False
                        )

                # ENGINE MANAGER SECTION
                with gr.Accordion("🛠️ Engine Installation & Environment Status", open=False):
                    status_rows = []
                    for name, info in status_map.items():
                        status_rows.append(f"| **{name}** | {info['status_text']} | `{info['install_cmd']}` | {info['desc']} |")

                    engine_table = "\n".join([
                        "| Architecture | Status | Install Command | Description |",
                        "| :--- | :--- | :--- | :--- |"
                    ] + status_rows)

                    gr.Markdown(engine_table)

                    with gr.Row():
                        target_install = gr.Dropdown(
                            choices=engine_choices,
                            value=engine_choices[1],
                            label="Select Engine to Install",
                            scale=3
                        )
                        install_btn = gr.Button("📦 Install Selected Engine", variant="secondary", scale=2)

                    install_log = gr.Textbox(label="Installation Output Log", lines=2, interactive=False)

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

                install_btn.click(
                    fn=self.install_engine,
                    inputs=[target_install],
                    outputs=[install_log]
                )

        print(f"\nLaunching Clean Voicesolate Studio at http://localhost:{server_port}")
        demo.launch(server_port=server_port, inbrowser=inbrowser, quiet=True)
