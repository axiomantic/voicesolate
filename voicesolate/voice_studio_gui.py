import os
import sys
import tempfile
import json
import shutil
import subprocess
from pathlib import Path
from typing import Dict, Any, List, Optional

class VoiceStudioGUI:
    """
    Clean, Modern Voice Studio GUI for Voicesolate.
    Features:
    - Real-time Engine Installation & Status Badges
    - Clear division between Reference Audio and Synthesized Audio
    - Streamlined controls without visual clutter
    - Multi-model switcher (Piper, XTTS, F5-TTS) with parameter sliders
    """

    DEFAULT_QUOTES = [
        "The secret of getting ahead is getting started.",
        "Kindness is the language which the deaf can hear and the blind can see.",
        "Whenever you find yourself on the side of the majority, it is time to pause and reflect.",
        "Twenty years from now you will be more disappointed by the things you didn't do than by the ones you did do.",
        "I have long been interested in the notion of time travelers. In fact, I wrote a book about it.",
        "Madam, I'd be delighted. So, this is a space ship? You ever run into Halley's comet?"
    ]

    def __init__(self, character_dir: Path):
        self.char_dir = Path(character_dir)
        self.char_name = self.char_dir.name
        self.models_dir = self.char_dir / "models"
        self.datasets_dir = self.char_dir / "datasets"

    def check_engine_status(self) -> Dict[str, Dict[str, Any]]:
        """Checks if each TTS engine is installed and ready."""
        status = {}

        # Piper
        has_piper_bin = shutil.which("piper") is not None
        piper_onnx = list((self.models_dir / "piper").glob("*.onnx"))
        status["Piper (VITS / ONNX)"] = {
            "key": "piper",
            "installed": has_piper_bin or bool(piper_onnx),
            "model_ready": bool(piper_onnx),
            "install_cmd": "pip install piper-tts",
            "status_text": "🟢 Ready" if (has_piper_bin and piper_onnx) else ("🟡 Dataset Ready (Model Not Compiled)" if not piper_onnx else "🔴 Binary Missing"),
            "desc": "Ultra-low latency CPU neural synthesis (22.05kHz LJSpeech)"
        }

        # XTTS
        has_tts = shutil.which("tts") is not None
        try:
            import TTS
            has_tts = True
        except ImportError:
            pass
        status["Coqui XTTS-v2"] = {
            "key": "xtts",
            "installed": has_tts,
            "model_ready": True,
            "install_cmd": "pip install TTS",
            "status_text": "🟢 Ready" if has_tts else "🔴 Not Installed (`pip install TTS`)",
            "desc": "24kHz autoregressive + diffusion voice clone"
        }

        # F5-TTS
        has_f5 = shutil.which("f5-tts_infer-cli") is not None
        try:
            import f5_tts
            has_f5 = True
        except ImportError:
            pass
        status["F5-TTS (Flow-Matching)"] = {
            "key": "f5tts",
            "installed": has_f5,
            "model_ready": True,
            "install_cmd": "pip install f5-tts",
            "status_text": "🟢 Ready" if has_f5 else "🔴 Not Installed (`pip install f5-tts`)",
            "desc": "Non-autoregressive flow-matching Diffusion Transformer"
        }

        return status

    def get_reference_audio(self) -> Optional[str]:
        """Returns the best reference audio clip for the character."""
        # Check F5 reference first
        f5_ref = self.datasets_dir / "f5tts" / "ref_audio" / "ref.wav"
        if f5_ref.exists():
            return str(f5_ref)

        # Check XTTS reference
        xtts_refs = list((self.datasets_dir / "xtts" / "reference_audio").glob("*.wav"))
        if xtts_refs:
            return str(xtts_refs[0])

        # Check piper wavs
        piper_wavs = list((self.datasets_dir / "piper" / "wavs").glob("*.wav"))
        if piper_wavs:
            return str(piper_wavs[0])

        return None

    def synthesize(self, engine_name: str, text: str, speed: float, temp: float) -> tuple[Optional[str], str]:
        """Runs test synthesis or reports engine status."""
        status_map = self.check_engine_status()
        engine_info = status_map.get(engine_name, {})
        key = engine_info.get("key")

        if not engine_info.get("installed"):
            msg = f"⚠️ {engine_name} is not installed in the environment.\nRun: `{engine_info.get('install_cmd')}` to enable live synthesis."
            return None, msg

        tmp_out = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name

        try:
            if key == "piper":
                onnx_models = list((self.models_dir / "piper").glob("*.onnx"))
                if not onnx_models:
                    return None, "⚠️ Piper LJSpeech dataset is ready, but .onnx model has not been compiled yet."
                cmd = f"echo {subprocess.list2cmdline([text])} | piper --model {onnx_models[0]} --output_file {tmp_out}"
                subprocess.run(cmd, shell=True, check=True)
                return tmp_out, f"✓ Synthesized successfully with {engine_name} at {speed:.2f}x speed."

            elif key == "f5tts":
                ref_wav = self.get_reference_audio()
                ref_text = "Madam, I'd be delighted. So, this is a space ship? You ever run into Halley's comet?"
                cmd = [
                    "f5-tts_infer-cli",
                    "--ref_audio", str(ref_wav),
                    "--ref_text", ref_text,
                    "--gen_text", text,
                    "--output_file", tmp_out,
                    "--speed", str(speed)
                ]
                subprocess.run(cmd, check=True)
                return tmp_out, f"✓ Synthesized with F5-TTS!"

            elif key == "xtts":
                ref_wav = self.get_reference_audio()
                cmd = [
                    "tts", "--model_name", "tts_models/multilingual/multi-dataset/xtts_v2",
                    "--text", text,
                    "--speaker_wav", str(ref_wav),
                    "--language_idx", "en",
                    "--out_path", tmp_out
                ]
                subprocess.run(cmd, check=True)
                return tmp_out, f"✓ Synthesized with XTTS-v2!"

        except Exception as e:
            return None, f"❌ Error during synthesis: {e}"

        return None, "Engine not supported."

    def launch(self, server_port: int = 7860, inbrowser: bool = True):
        import gradio as gr

        status_map = self.check_engine_status()
        engine_choices = list(status_map.keys())
        ref_audio_file = self.get_reference_audio()

        # Count training clips
        num_piper = len(list((self.datasets_dir / "piper" / "wavs").glob("*.wav")))
        num_xtts = len(list((self.datasets_dir / "xtts" / "wavs").glob("*.wav")))

        custom_css = """
        .gradio-container { max-width: 1100px !important; margin: auto; }
        .status-box { padding: 12px; border-radius: 8px; background: #1e293b; margin-bottom: 12px; }
        """

        with gr.Blocks(title=f"Voicesolate Studio — {self.char_name}", css=custom_css, theme=gr.themes.Base()) as demo:
            gr.Markdown(f"""
            # 🎙️ Voicesolate — {self.char_name} Voice Studio
            **Training Dataset Size:** {num_piper} isolated vocal clips ($\ge 5.0\text{s}$) | **Formats Ready:** Piper (LJSpeech), XTTS-v2, F5-TTS
            """)

            with gr.Row():
                # LEFT COLUMN: Prompt & Settings
                with gr.Column(scale=5):
                    gr.Markdown("### 1. Select Engine")
                    engine_dropdown = gr.Dropdown(
                        choices=engine_choices,
                        value=engine_choices[0],
                        label="TTS Engine Architecture"
                    )

                    engine_badge = gr.Markdown(f"**Engine Status:** {status_map[engine_choices[0]]['status_text']}")

                    gr.Markdown("### 2. Test Dialogue")
                    quote_picker = gr.Dropdown(
                        choices=self.DEFAULT_QUOTES,
                        value=self.DEFAULT_QUOTES[0],
                        label="Pre-loaded Mark Twain Aphorisms"
                    )
                    prompt_text = gr.Textbox(
                        lines=3,
                        value=self.DEFAULT_QUOTES[0],
                        label="Custom Speech Text"
                    )

                    with gr.Accordion("⚙️ Advanced Speech Parameters", open=False):
                        speed_slider = gr.Slider(minimum=0.5, maximum=2.0, value=1.0, step=0.05, label="Speech Speed")
                        temp_slider = gr.Slider(minimum=0.1, maximum=1.0, value=0.75, step=0.05, label="Temperature (XTTS)")

                    synth_button = gr.Button("▶️ Synthesize Voice", variant="primary", size="lg")
                    status_message = gr.Markdown("")

                # RIGHT COLUMN: Audio Playback & Comparison
                with gr.Column(scale=5):
                    gr.Markdown("### 🎧 Voice Audition")
                    
                    gr.Markdown("**Actor Reference Voice Prompt** *(Cleaned & Mastered Neural Stem)*")
                    ref_player = gr.Audio(
                        value=ref_audio_file,
                        label="Pristine Voice Reference (Used to train / clone)",
                        type="filepath",
                        interactive=False
                    )

                    gr.Markdown("**Model Synthesis Output**")
                    output_player = gr.Audio(
                        label="Generated Audio Result",
                        type="filepath",
                        interactive=False
                    )

            # Wire up reactive events
            quote_picker.change(fn=lambda q: q, inputs=[quote_picker], outputs=[prompt_text])

            def on_engine_select(eng):
                stat = self.check_engine_status().get(eng, {})
                return f"**Engine Status:** {stat.get('status_text', 'Unknown')}\n*{stat.get('desc', '')}*"

            engine_dropdown.change(fn=on_engine_select, inputs=[engine_dropdown], outputs=[engine_badge])

            synth_button.click(
                fn=self.synthesize,
                inputs=[engine_dropdown, prompt_text, speed_slider, temp_slider],
                outputs=[output_player, status_message]
            )

        print(f"\nLaunching Clean Voice Studio at http://localhost:{server_port}")
        demo.launch(server_port=server_port, inbrowser=inbrowser, quiet=True)
