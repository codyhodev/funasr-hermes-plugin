#!/usr/bin/env python3
"""FunASR daemon — pre-loads SenseVoiceSmall and listens on a Unix socket.

Started as a subprocess by the FunASR Hermes plugin at startup.
Keeps the model in memory across transcription calls, eliminating
repeated ~7s model loading overhead.

Run standalone:
    python3 funasrd.py [--socket PATH] [--model MODEL_ID]
"""

from __future__ import annotations

import json
import logging
import os
import re
import signal
import socket
import sys
import time

# ── Suppress noisy output before importing FunASR ─────────────────────
os.environ.setdefault("MODELSCOPE_LOGLEVEL", "ERROR")
os.environ.setdefault("FUNASR_LOGLEVEL", "ERROR")
os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ.setdefault("OMP_NUM_THREADS", "4")

_LOG_LEVEL = os.environ.get("FUNASRD_LOGLEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, _LOG_LEVEL, logging.INFO),
    format="[funasrd] %(levelname)s %(message)s",
)
logger = logging.getLogger("funasrd")

# ── Default paths ─────────────────────────────────────────────────────
_HERMES_HOME = os.environ.get("HERMES_HOME", os.path.expanduser("~/.hermes"))
DEFAULT_SOCKET = os.path.join(_HERMES_HOME, "run", "funasrd.sock")
PID_FILE = os.path.join(_HERMES_HOME, "run", "funasrd.pid")


class _SilenceStderr:
    """Context manager to suppress stderr during FunASR loading/transcription."""

    def __enter__(self):
        self._stderr = sys.stderr
        sys.stderr = open(os.devnull, "w")
        return self

    def __exit__(self, *args):
        if self._stderr:
            sys.stderr.close()
            sys.stderr = self._stderr


# ── VAD-based punctuation inference ────────────────────────────────
_VAD_MODEL = None


def _punctuate_with_vad(text: str, audio_path: str) -> str:
    """Add punctuation by analyzing pause durations with VAD.

    Detects speech/silence segments in the audio, maps gap durations
    to punctuation marks, and inserts them at proportional positions
    in the transcript.

    Falls back to raw text if silero-vad is unavailable.
    """
    if not text.strip():
        return text

    global _VAD_MODEL
    try:
        if _VAD_MODEL is None:
            from silero_vad import load_silero_vad
            _VAD_MODEL = load_silero_vad()

        import soundfile as sf

        audio, sr = sf.read(audio_path)
        if len(audio.shape) > 1:
            audio = audio.mean(axis=1)  # mono

        stamps = _VAD_MODEL.get_speech_timestamps(audio, sr)
        if len(stamps) <= 1:
            return text  # no pauses to infer from

        # Gap durations (seconds) between consecutive speech segments
        gaps = []
        for i in range(len(stamps) - 1):
            gap = (stamps[i + 1]["start"] - stamps[i]["end"]) / sr
            gaps.append(gap)

        # Classify each gap → Chinese punctuation mark
        punct = []
        for g in gaps:
            if g < 0.25:
                punct.append(None)       # brief pause, keep flowing
            elif g < 0.6:
                punct.append("，")        # short pause → comma
            else:
                punct.append("。")        # long pause → period

        # Proportional text split by segment duration
        total_s = sum(s["end"] - s["start"] for s in stamps)
        ratios = [(s["end"] - s["start"]) / total_s for s in stamps]

        n_chars = len(text)
        pos = 0
        parts = []
        for i, r in enumerate(ratios):
            n = n_chars - pos if i == len(ratios) - 1 else max(1, int(n_chars * r))
            parts.append(text[pos : pos + n])
            pos += n
            if i < len(punct) and punct[i] is not None:
                parts.append(punct[i])

        result = "".join(parts)
        # Ensure it ends with sentence-ending punctuation
        if result and result[-1] not in "。！？；":
            result += "。"
        return result

    except ImportError:
        logger.debug("silero-vad not installed — skipping punctuation")
        return text
    except Exception as exc:
        logger.warning("VAD punctuation failed: %s", exc)
        return text


def main(socket_path: str | None = None) -> None:
    socket_path = socket_path or DEFAULT_SOCKET
    run_dir = os.path.dirname(socket_path)
    os.makedirs(run_dir, exist_ok=True)

    # Remove stale socket (from a previous unclean exit)
    if os.path.exists(socket_path):
        os.unlink(socket_path)

    # Write PID file
    with open(PID_FILE, "w") as f:
        f.write(str(os.getpid()))

    # ── Load model (the heavy part — done once) ──────────────────────
    logger.info("Loading SenseVoiceSmall model…")
    _silence_mods = {"oss2", "modelscope", "urllib3", "funasr", "tqdm"}
    for mod in _silence_mods:
        logging.getLogger(mod).setLevel(logging.ERROR)

    t0 = time.time()
    with _SilenceStderr():
        import torch
        from funasr import AutoModel

        model = AutoModel(
            model="iic/SenseVoiceSmall",
            device="cuda:0" if torch.cuda.is_available() else "cpu",
            disable_update=True,
            disable_pipeline=True,
        )
    load_time = time.time() - t0
    logger.info("Model loaded in %.1fs on %s — listening",
                load_time, "GPU" if torch.cuda.is_available() else "CPU")

    # ── Unix socket server ───────────────────────────────────────────
    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.bind(socket_path)
    server.listen(5)
    server.settimeout(1.0)  # periodic wake for shutdown check
    os.chmod(socket_path, 0o600)

    running = True

    def _handle_shutdown(signum, frame):
        nonlocal running
        running = False
        logger.info("Received signal %s, shutting down…", signum)

    signal.signal(signal.SIGTERM, _handle_shutdown)
    signal.signal(signal.SIGINT, _handle_shutdown)

    # ── Request loop ─────────────────────────────────────────────────
    while running:
        try:
            conn, _ = server.accept()
        except socket.timeout:
            continue
        except OSError:
            break

        try:
            raw = conn.recv(65536)
            if not raw:
                conn.close()
                continue

            request = json.loads(raw.decode("utf-8"))
            action = request.get("action", "transcribe")

            if action == "ping":
                response = {"status": "ok", "pid": os.getpid()}

            elif action == "quit":
                response = {"status": "ok", "message": "daemon exiting"}
                conn.sendall((json.dumps(response) + "\n").encode("utf-8"))
                conn.close()
                break

            elif action == "transcribe":
                file_path = request["file_path"]
                t1 = time.time()
                with _SilenceStderr():
                    result = model.generate(input=file_path)

                # Parse SenseVoice output
                raw_text = ""
                if isinstance(result, list) and len(result) > 0:
                    raw_text = result[0].get("text", "")
                elif isinstance(result, dict):
                    raw_text = result.get("text", "")

                # Strip special tokens: <|zh|><|NEUTRAL|><|Speech|>text
                clean_text = re.sub(r"<\|[^|]+\|>", "", raw_text).strip()

                # VAD-based punctuation inference
                clean_text = _punctuate_with_vad(clean_text, file_path)

                elapsed = time.time() - t1
                logger.info("Transcribed in %.3fs", elapsed)

                response = {
                    "success": True,
                    "transcript": clean_text,
                    "elapsed": round(elapsed, 3),
                }

            else:
                response = {"success": False, "error": f"Unknown action: {action}"}

            payload = json.dumps(response, ensure_ascii=False) + "\n"
            conn.sendall(payload.encode("utf-8"))

        except Exception as exc:
            logger.error("Request error: %s", exc)
            try:
                err = json.dumps({"success": False, "error": str(exc)}, ensure_ascii=False)
                conn.sendall((err + "\n").encode("utf-8"))
            except Exception:
                pass
        finally:
            conn.close()

    # ── Cleanup ──────────────────────────────────────────────────────
    server.close()
    for path in (socket_path, PID_FILE):
        try:
            if os.path.exists(path):
                os.unlink(path)
        except OSError:
            pass
    logger.info("Exited cleanly")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="FunASR daemon")
    parser.add_argument("--socket", default=DEFAULT_SOCKET, help="Unix socket path")
    args = parser.parse_args()
    main(socket_path=args.socket)
