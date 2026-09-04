import os
import re
import io
import wave
import shlex
import subprocess
import soundfile as sf
import numpy as np
from pathlib import Path
from urllib.parse import urlparse, unquote
import shutil
from typing import Optional, Tuple, List

class AudioExtractor:
    """
    Manages FFmpeg audio extraction and chunk seeking over SSH/SFTP.
    Extracts pure discrete Front Center Channel (FC) for 5.1 surround mixes
    with zero phase cancellation or comb filtering artifacts.
    """

    def __init__(self, media_path: str, ssh_user: Optional[str] = "elijah"):
        self.raw_path = media_path
        self.is_remote = False
        self.remote_host: Optional[str] = None
        self.remote_user: Optional[str] = ssh_user
        self.remote_file_path: Optional[str] = None
        self.local_path: Optional[Path] = None

        self._parse_path(media_path)

    def _parse_path(self, path_str: str):
        """Detects if path is a remote SFTP/SSH URL or a local file path."""
        parsed = urlparse(path_str)
        if parsed.scheme in ("sftp", "ssh") or "@" in path_str.split(":")[0]:
            self.is_remote = True
            if parsed.scheme in ("sftp", "ssh"):
                self.remote_host = parsed.hostname or "flanopticon.lan"
                self.remote_user = parsed.username or self.remote_user or "elijah"
                self.remote_file_path = unquote(parsed.path)
            else:
                parts = path_str.split(":", 1)
                user_host = parts[0]
                self.remote_file_path = parts[1]
                if "@" in user_host:
                    self.remote_user, self.remote_host = user_host.split("@", 1)
                else:
                    self.remote_host = user_host
        else:
            self.local_path = Path(path_str).resolve()
            if not self.local_path.exists():
                if path_str.startswith("/mnt/") or "flanopticon" in path_str:
                    self.is_remote = True
                    self.remote_host = "flanopticon.lan"
                    self.remote_user = self.remote_user or "elijah"
                    self.remote_file_path = path_str
                else:
                    raise FileNotFoundError(f"Media file not found: {path_str}")

    def _run_cmd(self, ffmpeg_args: List[str]) -> bytes:
        """Executes FFmpeg locally or remotely via SSH."""
        if self.is_remote:
            remote_cmd_str = " ".join(shlex.quote(a) for a in ffmpeg_args)
            ssh_cmd = [
                "ssh",
                "-o", "BatchMode=yes",
                "-o", "ConnectTimeout=10",
                f"{self.remote_user}@{self.remote_host}",
                f"{remote_cmd_str} < /dev/null"
            ]
            res = subprocess.run(ssh_cmd, capture_output=True, check=True)
            return res.stdout
        else:
            exe = ffmpeg_args[0]
            if shutil.which(exe) is None:
                raise RuntimeError(
                    f"Required system binary '{exe}' is not installed or not in PATH.\n"
                    f"Installation Instructions:\n"
                    f"  • macOS: brew install ffmpeg\n"
                    f"  • Debian / Ubuntu: sudo apt-get update && sudo apt-get install -y ffmpeg\n"
                    f"  • Arch Linux: sudo pacman -S ffmpeg\n"
                    f"  • Windows: winget install Gyan.FFmpeg\n"
                    f"  • Documentation: https://ffmpeg.org/download.html"
                )
            res = subprocess.run(ffmpeg_args, capture_output=True, check=True)
            return res.stdout

    def get_duration(self) -> float:
        """Returns total media duration in seconds using ffprobe without transferring media."""
        target = self.remote_file_path if self.is_remote else str(self.local_path)
        cmd = [
            "ffprobe",
            "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            target
        ]
        out = self._run_cmd(cmd)
        return float(out.decode("utf-8").strip())

    def extract_embedded_subtitles(self, output_srt_path: str) -> bool:
        """Extracts internal subtitles from local or remote video directly."""
        target = self.remote_file_path if self.is_remote else str(self.local_path)
        try:
            if self.is_remote:
                cmd = [
                    "ffmpeg", "-nostdin", "-loglevel", "error", "-y",
                    "-i", target,
                    "-map", "0:s:0",
                    "-f", "srt",
                    "-"
                ]
                srt_bytes = self._run_cmd(cmd)
                if srt_bytes:
                    os.makedirs(os.path.dirname(output_srt_path), exist_ok=True)
                    with open(output_srt_path, "wb") as f:
                        f.write(srt_bytes)
                    return os.path.exists(output_srt_path) and os.path.getsize(output_srt_path) > 0
            else:
                cmd = [
                    "ffmpeg", "-nostdin", "-loglevel", "error", "-y",
                    "-i", target,
                    "-map", "0:s:0",
                    output_srt_path
                ]
                res = subprocess.run(cmd, capture_output=True, text=True)
                return res.returncode == 0 and os.path.exists(output_srt_path) and os.path.getsize(output_srt_path) > 0
        except Exception:
            return False
        return False

    def extract_subtitles_to_file(self, output_srt_path: str) -> bool:
        """Alias for extract_embedded_subtitles."""
        return self.extract_embedded_subtitles(output_srt_path)

    def extract_slice_pcm(self, start_sec: float, duration_sec: float, sample_rate: int = 16000) -> bytes:
        """Fast partial chunk extraction (16kHz mono) for Wyoming STT probing."""
        target = self.remote_file_path if self.is_remote else str(self.local_path)
        cmd = [
            "ffmpeg", "-nostdin", "-loglevel", "error", "-y",
            "-ss", f"{max(0.0, start_sec):.3f}",
            "-i", target,
            "-map", "0:a:0",
            "-af", "pan=mono|c0=FC",
            "-t", f"{max(0.1, duration_sec):.3f}",
            "-vn",
            "-ar", str(sample_rate),
            "-f", "s16le",
            "-"
        ]
        return self._run_cmd(cmd)

    def export_clip(self, start_sec: float, end_sec: float, output_path: str, padding_sec: float = 0.0):
        """
        Extracts discrete Front Center dialogue track at 48kHz 24-bit PCM.
        Zero comb filtering, full frequency bandwidth.
        Zero leading padding ensures no bleed from preceding actors.
        """
        duration = self.get_duration()
        padded_start = max(0.0, start_sec)
        padded_end = min(duration, end_sec + 0.05)
        clip_duration = padded_end - padded_start

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        target = self.remote_file_path if self.is_remote else str(self.local_path)

        sample_rate = 48000
        cmd = [
            "ffmpeg", "-nostdin", "-loglevel", "error", "-y",
            "-ss", f"{padded_start:.3f}",
            "-i", target,
            "-map", "0:a:0",
            "-af", "pan=mono|c0=FC",
            "-t", f"{clip_duration:.3f}",
            "-vn",
            "-ar", str(sample_rate),
            "-f", "s16le",
            "-"
        ]

        pcm_bytes = self._run_cmd(cmd)
        with wave.open(output_path, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(pcm_bytes)
