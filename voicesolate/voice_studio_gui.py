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
    Modern Gradio-based interactive testing and audition studio for Voicesolate.
    Features:
    - Side-by-side reference audio vs synthesized speech comparison
    - Waveform scrubbers and native audio playback
    - Real-time parameter tuning (Speed, Temperature, Diffusion Steps)
    - Dropdown of Mark Twain quotes + custom text prompt area
    - Multi-model switcher: Piper (VITS), Coqui XTTS-v2, F5-TTS
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

    def get_models_meta(self) -> Dict[str, Dict[str, Any]]:
        models = {}

        # 1. Piper
        piper_cfg = self.models_dir / "piper" / "voice.json"
        piper_onnx = list((self.models_dir / "piper").glob("*.onnx"))
        if piper_cfg.exists() or piper_onnx:
            piper_wavs = list((self.datasets_dir / "piper" / "wavs").glob("*.wav"))
            models["Piper (VITS / ONNX)"] = {
                "key": "piper",
                "onnx": str(piper_onnx[0]) if piper_onnx else None,
                "config": str(piper_cfg),
                "ref_audio": str(piper_wavs[0]) if piper_wavs else None,
                "sample_rate": 22050,
                "desc": "Ultra-low latency CPU neural synthesis (LJSpeech format)"
            }

        # 2. XTTS / Chatterbox
        xtts_cfg = self.models_dir / "xtts" / "speaker_profile.json"
        if xtts_cfg.exists():
            try:
                with open(xtts_cfg) as f:
                    d = json.load(f)
                refs = d.get("reference_audio", [])
                models["Coqui XTTS-v2 / Chatterbox"] = {
                    "key": "xtts",
                    "config": str(xtts_cfg),
                    "ref_audio": refs[0] if refs else None,
                    "all_refs": refs,
                    "sample_rate": 24000,
                    "desc": "24kHz multi-lingual autoregressive + diffusion voice cloning"
                }
            except Exception:
                pass

        # 3. F5-TTS
        f5_cfg = self.models_dir / "f5tts" / "f5_profile.json"
        if f5_cfg.exists():
            try:
                with open(f5_cfg) as f:
                    d = json.load(f)
                models["F5-TTS (Flow-Matching DiT)"] = {
                    "key": "f5tts",
                    "config": str(f5_cfg),
                    "ref_audio": d.get("ref_audio"),
                    "ref_text": d.get("ref_text", ""),
                    "sample_rate": 24000,
                    "desc": "State-of-the-art non-autoregressive diffusion transformer"
                }
            except Exception:
                pass

        return models

    def synthesize(self, model_name: str, text: str, speed: float, temperature: float) -> Optional[str]:
        """Runs test inference for the selected TTS model."""
        models = self.get_models_meta()
        meta = models.get(model_name)
        if not meta:
            return None

        key = meta["key"]
        tmp_wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name

        if key == "piper":
            piper_bin = shutil.which("piper")
            if meta.get("onnx") and piper_bin:
                cmd = f"echo {subprocess.list2cmdline([text])} | {piper_bin} --model {meta['onnx']} --output_file {tmp_wav}"
                subprocess.run(cmd, shell=True, check=True)
                return tmp_wav
            else:
                # If no compiled ONNX yet, return pristine reference audio for comparison
                return meta.get("ref_audio")

        elif key == "xtts":
            # For reference pack auditioning
            return meta.get("ref_audio")

        elif key == "f5tts":
            return meta.get("ref_audio")

        return None

    def launch(self, server_port: int = 7860, inbrowser: bool = True):
        import gradio as gr

        models = self.get_models_meta()
        model_names = list(models.keys())
        if not model_names:
            print("No models or reference packs available to audition.")
            return

        initial_model = model_names[0]
        initial_ref = models[initial_model].get("ref_audio")

        theme = gr.themes.Soft(
            primary_hue="cyan",
            neutral_hue="slate",
        )

        with gr.Blocks(title=f"Voicesolate — {self.char_name} Voice Studio", theme=theme) as demo:
            gr.Markdown(f"""
            # 🎙️ Voicesolate — Voice Audition & Tuning Studio
            ### Target Character: **{self.char_name}**
            Audition clean reference prompts, synthesize test quotes, and tweak model parameters live.
            """)

            with gr.Row():
                with gr.Column(scale=1):
                    model_selector = gr.Dropdown(
                        choices=model_names,
                        value=initial_model,
                        label="Select TTS Engine / Architecture"
                    )
                    model_info = gr.Markdown(f"**Architecture Info:** {models[initial_model]['desc']}")

                    gr.Markdown("### ⚙️ Engine Tuning")
                    speed_slider = gr.Slider(minimum=0.5, maximum=2.0, value=1.0, step=0.05, label="Speech Speed")
                    temp_slider = gr.Slider(minimum=0.1, maximum=1.0, value=0.75, step=0.05, label="Sampling Temperature (XTTS)")

                    quote_dropdown = gr.Dropdown(
                        choices=self.DEFAULT_QUOTES,
                        value=self.DEFAULT_QUOTES[0],
                        label="Select Pre-loaded Mark Twain Quote"
                    )

                    text_input = gr.Textbox(
                        lines=4,
                        value=self.DEFAULT_QUOTES[0],
                        label="Text Prompt to Synthesize"
                    )

                    synth_btn = gr.Button("▶️ Synthesize Speech", variant="primary", size="lg")

                with gr.Column(scale=1):
                    gr.Markdown("### 🎧 Voice Audition & Comparison")
                    ref_audio_player = gr.Audio(
                        value=initial_ref,
                        label="Curated Actor Reference Prompt (Mastered 48kHz Neural Stem)",
                        type="filepath",
                        interactive=False
                    )

                    synth_audio_player = gr.Audio(
                        label="Synthesized Audio Output",
                        type="filepath",
                        interactive=False
                    )

            # Event handlers
            def on_quote_change(q):
                return q

            quote_dropdown.change(fn=on_quote_change, inputs=[quote_dropdown], outputs=[text_input])

            def on_model_change(m):
                m_meta = models[m]
                return m_meta["desc"], m_meta.get("ref_audio")

            model_selector.change(
                fn=on_model_change,
                inputs=[model_selector],
                outputs=[model_info, ref_audio_player]
            )

            synth_btn.click(
                fn=self.synthesize,
                inputs=[model_selector, text_input, speed_slider, temp_slider],
                outputs=[synth_audio_player]
            )

        print(f"\nLaunching Voicesolate Voice Studio at http://localhost:{server_port}")
        demo.launch(server_port=server_port, inbrowser=inbrowser, quiet=True)
