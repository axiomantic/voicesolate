import socket
import json
import struct
import io
import wave
from typing import Optional, Dict, Any

class WyomingSTTClient:
    """
    Client for the Wyoming TCP Protocol (Nabu Casa / Home Assistant standard).
    Communicates with Wyoming speech-to-text servers (e.g., faster-whisper).
    """

    def __init__(self, host: str = "10.0.2.141", port: int = 10300, timeout: float = 15.0):
        self.host = host
        self.port = port
        self.timeout = timeout

    def _send_event(self, sock: socket.socket, event_type: str, data: Optional[Dict[str, Any]] = None, payload: bytes = b""):
        msg = {"type": event_type}
        if data:
            msg["data"] = data
        if payload:
            msg["payload_length"] = len(payload)
        header = json.dumps(msg) + "\n"
        sock.sendall(header.encode("utf-8") + payload)

    def _recv_exact(self, sock: socket.socket, length: int) -> bytes:
        """Reads exactly `length` bytes from TCP socket stream."""
        buf = bytearray()
        while len(buf) < length:
            chunk = sock.recv(min(length - len(buf), 65536))
            if not chunk:
                break
            buf.extend(chunk)
        return bytes(buf)

    def _read_event(self, sock: socket.socket) -> Optional[Dict[str, Any]]:
        buf = b""
        while b"\n" not in buf:
            chunk = sock.recv(1)
            if not chunk:
                return None
            buf += chunk
        header_json = json.loads(buf.decode("utf-8"))
        data_len = header_json.get("data_length", 0)
        data = {}
        if data_len > 0:
            data_bytes = self._recv_exact(sock, data_len)
            data = json.loads(data_bytes.decode("utf-8"))
        elif "data" in header_json:
            data = header_json["data"]

        payload_len = header_json.get("payload_length", 0)
        payload = b""
        if payload_len > 0:
            payload = self._recv_exact(sock, payload_len)

        header_json["data"] = data
        header_json["payload"] = payload
        return header_json

    def check_health(self) -> Dict[str, Any]:
        """Queries the server info/capabilities."""
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(self.timeout)
            sock.connect((self.host, self.port))
            self._send_event(sock, "describe")
            info = self._read_event(sock)
            return info or {}

    def transcribe_audio_pcm(self, pcm_bytes: bytes, rate: int = 16000, width: int = 2, channels: int = 1, language: str = "en") -> str:
        """
        Sends raw PCM audio bytes to Wyoming STT and returns the transcript.
        Audio must be 16kHz, 16-bit mono PCM.
        """
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(self.timeout)
            sock.connect((self.host, self.port))

            self._send_event(sock, "transcribe", {"language": language})
            self._send_event(sock, "audio-start", {"rate": rate, "width": width, "channels": channels})

            # Stream audio in chunks of 1024 samples (2048 bytes for 16-bit mono)
            chunk_size = 2048
            for i in range(0, len(pcm_bytes), chunk_size):
                chunk = pcm_bytes[i:i + chunk_size]
                self._send_event(sock, "audio-chunk", {"rate": rate, "width": width, "channels": channels}, payload=chunk)

            self._send_event(sock, "audio-stop")

            # Collect transcript
            transcript_text = ""
            while True:
                event = self._read_event(sock)
                if not event:
                    break
                if event.get("type") == "transcript":
                    data = event.get("data", {})
                    transcript_text = data.get("text", "").strip()
                    break
                elif event.get("type") == "error":
                    data = event.get("data", {})
                    raise RuntimeError(f"Wyoming STT Error: {data.get('text', 'unknown error')}")

            return transcript_text

    def transcribe_wav_bytes(self, wav_bytes: bytes, language: str = "en") -> str:
        """Transcribes in-memory WAV file bytes."""
        with io.BytesIO(wav_bytes) as wav_file:
            with wave.open(wav_file, "rb") as wf:
                rate = wf.getframerate()
                width = wf.getsampwidth()
                channels = wf.getnchannels()
                pcm_bytes = wf.readframes(wf.getnframes())
                return self.transcribe_audio_pcm(pcm_bytes, rate=rate, width=width, channels=channels, language=language)
