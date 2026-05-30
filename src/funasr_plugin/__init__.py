"""FunASR STT plugin for Hermes Agent — daemon-backed.

Spawns a ``funasrd`` process at session start that pre-loads the
SenseVoiceSmall model and keeps it in memory across transcription calls.
Cleans up the daemon on session end.

Uses a reference-count file so that Gateway and TUI (or any other
component loading this plugin) share a single GPU-resident daemon
instead of duplicating it.
"""

from __future__ import annotations

import logging
import os
import signal
import subprocess
import sys
import time

from .provider import FunasrTranscriptionProvider, _wait_for_daemon

logger = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────
_DAEMON_SCRIPT = os.path.join(os.path.dirname(__file__), "funasrd.py")
_HERMES_HOME = os.environ.get("HERMES_HOME", os.path.expanduser("~/.hermes"))
_REF_COUNT_FILE = os.path.join(_HERMES_HOME, "run", "funasrd.refcount")

_daemon_process: subprocess.Popen | None = None


# ── Reference counting (cross-process) ────────────────────────────────
# Each plugin instance increments on start, decrements on stop.
# The daemon is only killed when the count reaches 0.

def _inc_refcount() -> int:
    """Increment the daemon reference count. Returns the new count."""
    try:
        count = 0
        if os.path.exists(_REF_COUNT_FILE):
            raw = open(_REF_COUNT_FILE).read().strip()
            count = int(raw) if raw else 0
        count += 1
        os.makedirs(os.path.dirname(_REF_COUNT_FILE), exist_ok=True)
        with open(_REF_COUNT_FILE, "w") as f:
            f.write(str(count))
        return count
    except Exception:
        return 1


def _dec_refcount() -> int:
    """Decrement the daemon reference count. Returns the remaining count."""
    try:
        if not os.path.exists(_REF_COUNT_FILE):
            return 0
        raw = open(_REF_COUNT_FILE).read().strip()
        count = int(raw) if raw else 0
        count = max(0, count - 1)
        if count > 0:
            with open(_REF_COUNT_FILE, "w") as f:
                f.write(str(count))
        else:
            os.unlink(_REF_COUNT_FILE)
        return count
    except Exception:
        return 1  # safe default — don't kill on error


# ── Daemon lifecycle ──────────────────────────────────────────────────

def _start_daemon() -> bool:
    """Launch the funasrd subprocess and wait for it to be ready.

    Before spawning, checks if a daemon is already listening on the
    Unix socket.  If so, reuses it — this way Gateway and TUI (or any
    other component loading the plugin) share a single GPU-resident
    model instance instead of duplicating it.

    Increments a cross-process reference count on success.
    Returns True if the daemon is reachable within the timeout.
    """
    global _daemon_process

    # ── Reuse existing daemon if socket is already alive ────────────
    if _wait_for_daemon(timeout=2):
        logger.info("FunASR daemon already running — reusing it")
        _inc_refcount()
        return True

    if _daemon_process is not None:
        ret = _daemon_process.poll()
        if ret is None:
            return True  # already running
        logger.warning("Daemon exited with code %s, restarting…", ret)

    try:
        _daemon_process = subprocess.Popen(
            [sys.executable, _DAEMON_SCRIPT],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            stdin=subprocess.DEVNULL,
        )
    except FileNotFoundError:
        logger.error("Daemon script not found: %s", _DAEMON_SCRIPT)
        return False
    except Exception as exc:
        logger.error("Failed to start daemon: %s", exc)
        return False

    # Wait for the daemon to open its socket
    ready = _wait_for_daemon(timeout=15)
    if ready:
        logger.info("FunASR daemon started (pid=%d)", _daemon_process.pid)
        _inc_refcount()
    else:
        logger.error("FunASR daemon failed to start within timeout")
    return ready


def _stop_daemon() -> None:
    """Gracefully shut down the funasrd subprocess.

    Only kills the daemon when the cross-process reference count
    reaches 0 (i.e. no other plugin instances are still using it).
    """
    global _daemon_process

    remaining = _dec_refcount()
    if remaining > 0:
        logger.info("FunASR daemon still in use (%d refs) — not killing", remaining)
        _daemon_process = None
        return

    if _daemon_process is None:
        return

    pid = _daemon_process.pid
    try:
        os.kill(pid, signal.SIGTERM)
        _daemon_process.wait(timeout=10)
        logger.info("FunASR daemon (pid=%d) stopped gracefully", pid)
    except subprocess.TimeoutExpired:
        logger.warning("Daemon (pid=%d) did not respond to SIGTERM, sending SIGKILL", pid)
        _daemon_process.kill()
        _daemon_process.wait()
    except ProcessLookupError:
        pass  # already dead
    finally:
        _daemon_process = None


def _on_session_end(**kwargs) -> None:
    """Plugin hook: clean up daemon when the agent session ends."""
    _stop_daemon()


# ── Plugin entry point ───────────────────────────────────────────────

def register(ctx) -> None:
    """Register the FunASR transcription provider with Hermes.

    Spawns the daemon process at registration time, then registers the
    provider with the plugin system.

    After registration, set ``stt.provider: funasr`` in ``config.yaml``
    to make it the active STT backend.
    """
    # Start daemon
    if not _start_daemon():
        logger.warning(
            "FunASR daemon failed to start — provider will report "
            "unavailable until the daemon is running"
        )

    # Register the provider
    provider = FunasrTranscriptionProvider()
    ctx.register_transcription_provider(provider)
    logger.info("FunASR transcription provider registered (daemon-backed)")

    # Register lifecycle hook for cleanup
    ctx.register_hook("on_session_end", _on_session_end)
