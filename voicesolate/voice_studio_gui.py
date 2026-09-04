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
    Clean, Modern Voice Studio GUI for Voicesolate.
    Focused on immediate synthesis:
    - Zero-config: Engines are calibrated upfront
    - High-fidelity in-process F5-TTS (Flow-Matching DiT)
    - Fast in-process Coqui XTTS-v2 (Autoregressive + Diffusion)
    - Side-by-side audio audition (AI Voice vs Actor Reference Stem)
    - Built-in Gradio themes with native padding and layout
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

    THEMES = {
        "ocean": "Ocean",
        "soft": "Soft",
        "monochrome": "Monochrome",
        "citrus": "Citrus",
        "default": "Default",
        "glass": "Glass"
    }

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
        """Parses the character's extracted dialogue metadata."""
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

    def synthesize(self, engine_name: str, text: str, speed: float, seed: int) -> Tuple[Optional[str], str]:
        """Synthesizes speech using the selected engine."""
        if not text or not text.strip():
            return None, "⚠️ Please enter dialogue text to synthesize."

        tmp_out = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name

        try:
            if "f5" in engine_name.lower():
                ref_wav, ref_text = self.get_reference_prompt()
                if not ref_wav or not Path(ref_wav).exists():
                    return None, "❌ Reference voice prompt audio file not found."

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

            elif "xtts" in engine_name.lower() or "coqui" in engine_name.lower():
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
                    return tmp_out, "✅ Synthesized with Coqui XTTS-v2!"
                else:
                    return None, "❌ XTTS-v2 generation produced an empty audio file."

        except Exception as e:
            return None, f"❌ Synthesis failed: {e}"

        return None, "Engine not supported."

    def launch(self, server_port: int = 7860, inbrowser: bool = True, theme: str = "ocean"):
        import gradio as gr

        ref_wav, ref_txt = self.get_reference_prompt()
        stats = self.get_dataset_stats()
        dataset_clips = self.get_dataset_clips()
        clip_choices = [
            (f"Clip {i+1}: {c['text'][:65]}...", c['id'])
            for i, c in enumerate(dataset_clips[:25])
        ] if dataset_clips else [("No clips found", "none")]

        # Clean, minimal CSS that ONLY removes the fallback empty artwork icon
        # without altering any theme padding, margins, borders, or layout
        minimal_css = """
        div[aria-label="Empty value"] {
            display: none !important;
            height: 0px !important;
            padding: 0 !important;
            margin: 0 !important;
        }
        """

        # Resolve selected theme
        theme_map = {
            "ocean": gr.themes.Ocean(),
            "soft": gr.themes.Soft(),
            "monochrome": gr.themes.Monochrome(),
            "citrus": gr.themes.Citrus(),
            "glass": gr.themes.Glass(),
            "default": gr.themes.Default()
        }
        active_theme = theme_map.get(theme.lower(), gr.themes.Ocean())

        with gr.Blocks(title=f"Voicesolate Studio — {self.char_name}", css=minimal_css) as demo:
            # Header
            gr.Markdown(
                f"""
                # 🎙️ {self.char_name} Voice Studio
                **Isolated Corpus:** {stats['clip_count']} vocal clips ({stats['total_minutes']} minutes) &nbsp;|&nbsp;
                **Available Voice Engines:** F5-TTS (Diffusion Flow-Matching) &amp; Coqui XTTS-v2 (Autoregressive)
                """
            )

            # Main 2-Column Synthesis Workspace
            with gr.Row():
                # LEFT COLUMN: Voice Generation
                with gr.Column(scale=5):
                    engine_dropdown = gr.Dropdown(
                        choices=[
                            "F5-TTS (Flow-Matching DiT — 24kHz Studio Quality)",
                            "Coqui XTTS-v2 (Autoregressive + Diffusion — Multilingual)"
                        ],
                        value="F5-TTS (Flow-Matching DiT — 24kHz Studio Quality)",
                        label="Speech Synthesis Engine",
                        info="Select voice cloning model architecture"
                    )

                    quote_selector = gr.Dropdown(
                        choices=self.DEFAULT_QUOTES,
                        value=self.DEFAULT_QUOTES[0],
                        label="Sample Dialogue Quotes",
                        info="Quick-select an aphorism or enter custom text below"
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

                    synth_btn = gr.Button("✨ Synthesize Voice", variant="primary", size="lg")
                    feedback_box = gr.Markdown("")

                # RIGHT COLUMN: Audio Audition
                with gr.Column(scale=5):
                    output_player = gr.Audio(
                        label="AI Voice Synthesis (Generated Audio Output)",
                        type="filepath",
                        interactive=False
                    )

                    ref_player = gr.Audio(
                        value=ref_wav,
                        label="Actor Reference Voice (Original Isolated Stem)",
                        type="filepath",
                        interactive=False
                    )
                    if ref_txt:
                        gr.Markdown(f'*Reference Transcript:* "{ref_txt}"')

            # BOTTOM SECTION: Corpus Explorer
            with gr.Accordion("📁 Explore Extracted Character Dialogue Corpus", open=False):
                with gr.Row():
                    with gr.Column(scale=5):
                        clip_selector = gr.Dropdown(
                            choices=clip_choices,
                            value=clip_choices[0][1] if clip_choices else None,
                            label="Select Dialogue Clip to Audition",
                            info=f"Browse any of the {stats['clip_count']} isolated stems extracted from media"
                        )
                        sample_clip_text = gr.Textbox(
                            value=dataset_clips[0]["text"] if dataset_clips else "",
                            label="Dialogue Transcription",
                            lines=2,
                            interactive=False
                        )
                    with gr.Column(scale=5):
                        sample_clip_player = gr.Audio(
                            value=dataset_clips[0]["path"] if dataset_clips else None,
                            label="Original Isolated Dialogue Stem",
                            type="filepath",
                            interactive=False
                        )

            # Interactive Event Wiring
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

            quote_selector.change(fn=lambda q: q, inputs=[quote_selector], outputs=[dialogue_input])

            synth_btn.click(
                fn=self.synthesize,
                inputs=[engine_dropdown, dialogue_input, speed_slider, seed_input],
                outputs=[output_player, feedback_box]
            )

        print(f"\nLaunching Clean Voicesolate Studio at http://localhost:{server_port} [Theme: {theme}]")
        demo.launch(server_port=server_port, inbrowser=inbrowser, quiet=True, theme=active_theme)
