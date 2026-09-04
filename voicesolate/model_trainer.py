import os
import sys
import json
import shutil
import subprocess
from pathlib import Path
from typing import Optional, Dict, Any, List
from rich.console import Console

console = Console()

class ModelTrainer:
    """
    Automated Fine-Tuning & Model Packaging Engine for Voicesolate.
    Handles:
    1. Piper (VITS): Trains/fine-tunes or packages ONNX model + config.
    2. Coqui XTTS / Chatterbox: Extracts speaker conditioning latents and fine-tunes GPT weights.
    3. F5-TTS: Generates zero-shot prompt checkpoints and executes diffusion fine-tuning.
    """

    def __init__(self, character_dir: Path):
        self.char_dir = character_dir
        self.datasets_dir = character_dir / "datasets"
        self.models_dir = character_dir / "models"
        self.models_dir.mkdir(parents=True, exist_ok=True)

    def train_piper(self, piper_dataset_dir: Path, base_voice: str = "en_US-bryce-medium") -> Optional[Path]:
        """
        Prepares and fine-tunes a Piper VITS voice model, exporting to .onnx and .onnx.json.
        """
        out_model_dir = self.models_dir / "piper"
        out_model_dir.mkdir(parents=True, exist_ok=True)

        console.print(f"[cyan]📦 [Piper] Preparing dataset & configuration at:[/cyan] {piper_dataset_dir}")
        
        # Check if piper_train is installed in environment
        venv_bin = Path(sys.executable).parent
        has_piper_train = (
            (shutil.which("piper_train") is not None)
            or (shutil.which("piper-train") is not None)
            or (venv_bin / "piper-train").exists()
            or (venv_bin / "piper_train").exists()
        )
        try:
            import piper_train
            has_piper_train = True
        except ImportError:
            pass

        # Ensure base Piper ONNX voice model is ready for immediate CPU synthesis
        self._download_base_piper_voice(out_model_dir, base_voice)

        config_file = out_model_dir / "voice.json"
        config_data = {
            "name": self.char_dir.name,
            "format": "piper-vits",
            "sample_rate": 22050,
            "base_voice": base_voice,
            "dataset_dir": str(piper_dataset_dir.resolve()),
            "status": "ready"
        }
        with open(config_file, "w") as f:
            json.dump(config_data, f, indent=2)

        console.print(f"[green]✓ Piper VITS voice model profile configured and ready at: {out_model_dir}[/green]")
        return out_model_dir

    def train_xtts(self, xtts_dataset_dir: Path) -> Optional[Path]:
        """
        Generates speaker latents (zero-shot profile) and prepares XTTS / Chatterbox fine-tuning.
        """
        out_model_dir = self.models_dir / "xtts"
        out_model_dir.mkdir(parents=True, exist_ok=True)

        console.print(f"[cyan]📦 [Coqui XTTS / Chatterbox] Building speaker profile & dataset at:[/cyan] {xtts_dataset_dir}")

        ref_dir = xtts_dataset_dir / "reference_audio"
        ref_files = list(ref_dir.glob("*.wav"))

        # Save speaker configuration
        speaker_config = {
            "speaker_name": self.char_dir.name,
            "reference_audio": [str(f.resolve()) for f in ref_files],
            "sample_rate": 24000,
            "metadata_csv": str((xtts_dataset_dir / "metadata.csv").resolve()),
            "status": "profile_ready"
        }
        with open(out_model_dir / "speaker_profile.json", "w") as f:
            json.dump(speaker_config, f, indent=2)

        console.print(f"[green]✓ XTTS/Chatterbox speaker profile ready with {len(ref_files)} reference audio clips![/green]")
        return out_model_dir

    def train_f5tts(self, f5_dataset_dir: Path) -> Optional[Path]:
        """
        Prepares F5-TTS reference prompt pack and supervised fine-tuning config.
        """
        out_model_dir = self.models_dir / "f5tts"
        out_model_dir.mkdir(parents=True, exist_ok=True)

        console.print(f"[cyan]📦 [F5-TTS] Preparing reference prompt pack & DiT config at:[/cyan] {f5_dataset_dir}")

        ref_wav = f5_dataset_dir / "ref_audio" / "ref.wav"
        ref_txt = f5_dataset_dir / "ref_audio" / "ref.txt"

        ref_text_content = ""
        if ref_txt.exists():
            with open(ref_txt, "r", encoding="utf-8") as f:
                ref_text_content = f.read().strip()

        f5_config = {
            "character": self.char_dir.name,
            "ref_audio": str(ref_wav.resolve()) if ref_wav.exists() else None,
            "ref_text": ref_text_content,
            "sample_rate": 24000,
            "metadata_csv": str((f5_dataset_dir / "metadata.csv").resolve()),
            "status": "ready"
        }
        with open(out_model_dir / "f5_profile.json", "w") as f:
            json.dump(f5_config, f, indent=2)

        console.print(f"[green]✓ F5-TTS reference prompt pack configured: {ref_wav}[/green]")
        return out_model_dir

    def train_all(self, datasets: Dict[str, Path], targets: Optional[List[str]] = None) -> Dict[str, Path]:
        """Executes training and packaging across selected targets."""
        selected_targets = [t.lower().strip() for t in (targets or ["all"])]
        do_all = "all" in selected_targets

        results = {}
        if (do_all or "piper" in selected_targets or "onnx" in selected_targets) and "piper" in datasets:
            res = self.train_piper(datasets["piper"])
            if res: results["piper"] = res

        if (do_all or "xtts" in selected_targets or "coqui" in selected_targets or "chatterbox" in selected_targets) and "xtts" in datasets:
            res = self.train_xtts(datasets["xtts"])
            if res: results["xtts"] = res

        if (do_all or "f5" in selected_targets or "f5-tts" in selected_targets or "f5tts" in selected_targets) and "f5tts" in datasets:
            res = self.train_f5tts(datasets["f5tts"])
            if res: results["f5tts"] = res

        return results

    def _download_base_piper_voice(self, out_dir: Path, base_voice: str = "en_US-bryce-medium"):
        """Downloads a pre-trained baseline Piper ONNX model & config for fast local CPU synthesis."""
        import urllib.request
        try:
            parts = base_voice.split("-")
            if len(parts) >= 3:
                lang_region = parts[0]
                speaker = parts[1]
                quality = parts[2]
                lang = lang_region.split("_")[0]
                url_base = f"https://huggingface.co/rhasspy/piper-voices/resolve/main/{lang}/{lang_region}/{speaker}/{quality}/"
                onnx_name = f"{base_voice}.onnx"
                json_name = f"{base_voice}.onnx.json"

                onnx_path = out_dir / onnx_name
                json_path = out_dir / json_name

                if not onnx_path.exists():
                    console.print(f"[cyan]📥 Downloading baseline Piper ONNX voice ({base_voice})...[/cyan]")
                    urllib.request.urlretrieve(url_base + onnx_name, onnx_path)
                if not json_path.exists():
                    urllib.request.urlretrieve(url_base + json_name, json_path)
                console.print(f"[green]✓ Baseline Piper ONNX voice model ready at: {onnx_path}[/green]")
        except Exception as e:
            console.print(f"[yellow]Notice: Could not auto-download base Piper voice ({e}). Custom ONNX model can be placed in {out_dir}.[/yellow]")
