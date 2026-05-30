"""FunASR transcription provider — daemon-backed SenseVoiceSmall backend.

Communicates with a long-running ``funasrd`` process over a Unix domain
socket, keeping the model pre-loaded in memory across transcriptions.
"""

from __future__ import annotations

import json
import logging
import os
import socket
import time
from typing import Any, Dict, List, Optional

from agent.transcription_provider import TranscriptionProvider

logger = logging.getLogger(__name__)

# ── Socket path (must match funasrd.py) ──────────────────────────────
_HERMES_HOME = os.environ.get("HERMES_HOME", os.path.expanduser("~/.hermes"))
SOCKET_PATH = os.path.join(_HERMES_HOME, "run", "funasrd.sock")

# ── Timeouts ─────────────────────────────────────────────────────────
_CONNECT_TIMEOUT = 10       # seconds to wait for daemon to start
_RESPONSE_TIMEOUT = 60      # seconds to wait for transcription
_RECV_SIZE = 65536


def _communicate(payload: dict, timeout: float = _RESPONSE_TIMEOUT) -> dict:
    """Send a JSON request to the daemon and return its response."""
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock.settimeout(timeout)
    try:
        sock.connect(SOCKET_PATH)
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        sock.sendall(data)

        raw = b""
        while True:
            chunk = sock.recv(_RECV_SIZE)
            if not chunk:
                break
            raw += chunk
            if b"\n" in chunk:
                break

        if not raw:
            return {"success": False, "error": "Empty response from daemon"}
        return json.loads(raw.decode("utf-8").strip())
    except socket.timeout:
        return {"success": False, "error": "Daemon response timeout"}
    except ConnectionRefusedError:
        return {"success": False, "error": "Daemon not running"}
    except Exception as exc:
        return {"success": False, "error": f"Daemon communication error: {exc}"}
    finally:
        sock.close()


def _wait_for_daemon(timeout: float = _CONNECT_TIMEOUT) -> bool:
    """Poll until the daemon socket is reachable, or timeout."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            resp = _communicate({"action": "ping"}, timeout=2)
            if resp.get("status") == "ok":
                return True
        except Exception:
            pass
        time.sleep(0.5)
    return False


# ── Provider implementation ──────────────────────────────────────────

class FunasrTranscriptionProvider(TranscriptionProvider):
    """Speech-to-text via daemonized FunASR SenseVoiceSmall.

    Delegates transcription to a pre-loaded model in a long-running
    ``funasrd`` process, eliminating per-call model loading overhead.
    """

    @property
    def name(self) -> str:
        return "funasr"

    @property
    def display_name(self) -> str:
        return "FunASR SenseVoice (daemon)"

    def is_available(self) -> bool:
        """Return True when the daemon socket is reachable."""
        try:
            resp = _communicate({"action": "ping"}, timeout=2)
            return resp.get("status") == "ok"
        except Exception:
            return False

    def list_models(self) -> List[Dict[str, Any]]:
        return [
            {
                "id": "iic/SenseVoiceSmall",
                "display": "SenseVoiceSmall",
                "languages": ["zh", "en", "ja", "ko", "yue"],
            },
        ]

    def default_model(self) -> Optional[str]:
        return "iic/SenseVoiceSmall"

    def get_setup_schema(self) -> Dict[str, Any]:
        return {
            "name": "FunASR SenseVoice",
            "badge": "local",
            "tag": "Local GPU-accelerated Chinese STT (daemonized, no API key)",
            "env_vars": [],
        }

    def transcribe(
        self,
        file_path: str,
        *,
        model: Optional[str] = None,
        language: Optional[str] = None,
        **extra: Any,
    ) -> Dict[str, Any]:
        """Transcribe audio by delegating to the funasrd daemon.

        Parameters
        ----------
        file_path : str
            Path to WAV/MP3/OGG audio file.
        model : str, optional
            Ignored — daemon uses a fixed model (SenseVoiceSmall).
        language : str, optional
            Ignored — SenseVoice auto-detects languages.
        """
        resp = _communicate({"action": "transcribe", "file_path": file_path})

        if resp.get("success"):
            transcript = resp.get("transcript", "")
            return {
                "success": True,
                "transcript": transcript,
                "provider": self.name,
            }

        error = resp.get("error", "Unknown daemon error")
        logger.error("Daemon transcription failed: %s", error)
        return {
            "success": False,
            "transcript": "",
            "error": error,
            "provider": self.name,
        }
