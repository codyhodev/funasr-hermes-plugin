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


# ── Punctuation restoration (FunASR ct-punc model) ──────────────────
_PUNC_MODEL = None
_PUNC_MODEL_NAME = "iic/punc_ct-transformer_zh-cn-common-vocab272727-pytorch"


def _punctuate_with_model(text: str) -> str:
    """Restore punctuation using FunASR's ct-punc model.

    Unlike the old VAD-based approach that infers punctuation from
    audio pause durations, this uses a dedicated punctuation model
    that understands sentence semantics — much more accurate.
    """
    if not text.strip() or _PUNC_MODEL is None:
        return text

    try:
        result = _PUNC_MODEL.generate(input=text)
        if isinstance(result, list) and len(result) > 0:
            return result[0].get("text", text)
        return text
    except Exception as exc:
        logger.warning("Punctuation model failed: %s", exc)
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

    # ── Load punctuation restoration model ─────────────────────────
    global _PUNC_MODEL
    t1 = time.time()
    logger.info("Loading punctuation model %s…", _PUNC_MODEL_NAME)
    with _SilenceStderr():
        _PUNC_MODEL = AutoModel(
            model=_PUNC_MODEL_NAME,
            device="cuda:0" if torch.cuda.is_available() else "cpu",
            disable_update=True,
            disable_pipeline=True,
        )
    logger.info("Punctuation model loaded in %.1fs", time.time() - t1)

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

                # ── Noise reduction via ffmpeg ────────────────────────
                nr_path = file_path + ".nr.wav"
                try:
                    import subprocess as _sp
                    _sp.run(
                        ["ffmpeg", "-y",
                         "-i", file_path,
                         "-af", "highpass=f=200,lowpass=f=4000,afftdn=nr=15:nf=-30",
                         nr_path],
                        capture_output=True, timeout=30,
                    )
                    if os.path.getsize(nr_path) > 0:
                        file_path = nr_path
                except Exception as _nr_err:
                    logger.debug("Noise reduction skipped: %s", _nr_err)

                with _SilenceStderr():
                    result = model.generate(input=file_path)

                # Clean up temp file
                if nr_path != file_path and os.path.exists(nr_path):
                    try:
                        os.unlink(nr_path)
                    except OSError:
                        pass

                # Parse SenseVoice output
                raw_text = ""
                if isinstance(result, list) and len(result) > 0:
                    raw_text = result[0].get("text", "")
                elif isinstance(result, dict):
                    raw_text = result.get("text", "")

                # Strip special tokens: <|zh|><|NEUTRAL|><|Speech|>text
                clean_text = re.sub(r"<\|[^|]+\|>", "", raw_text).strip()

                # VAD-based punctuation inference
                clean_text = _punctuate_with_model(clean_text)

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
