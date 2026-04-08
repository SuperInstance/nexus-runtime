"""NEXUS Lifecycle Management — graceful shutdown, signal handling, context managers.

Provides:
- Signal handlers for SIGTERM/SIGINT
- atexit registration for emergency cleanup
- Context manager support for Node
"""

from __future__ import annotations

import atexit
import logging
import signal
import threading
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

# Global registry of shutdown callbacks
_shutdown_callbacks: List[Callable[[], None]] = []
_lock = threading.Lock()
_handler_installed = False


def register_shutdown_callback(callback: Callable[[], None]) -> None:
    """Register a function to be called on graceful shutdown.
    
    Callbacks are invoked in LIFO order (last registered = first called).
    They should be idempotent and quick (avoid blocking I/O).
    """
    global _handler_installed
    with _lock:
        _shutdown_callbacks.append(callback)
        if not _handler_installed:
            _install_handlers()
            _handler_installed = True


def _install_handlers() -> None:
    """Install signal handlers for graceful shutdown."""
    try:
        signal.signal(signal.SIGTERM, _signal_handler)
    except (OSError, ValueError):
        pass
    try:
        signal.signal(signal.SIGINT, _signal_handler)
    except (OSError, ValueError):
        pass
    
    # Safety net: atexit handler
    atexit.register(_atexit_handler)


def _signal_handler(signum: int, frame: Any) -> None:
    """Handle shutdown signals."""
    logger.info("Received signal %d, initiating graceful shutdown...", signum)
    _run_callbacks()


def _atexit_handler() -> None:
    """Emergency cleanup on interpreter exit."""
    logger.debug("Running atexit cleanup...")
    _run_callbacks()


def _run_callbacks() -> None:
    """Execute all registered shutdown callbacks."""
    with _lock:
        callbacks = list(reversed(_shutdown_callbacks))
        _shutdown_callbacks.clear()
    
    for callback in callbacks:
        try:
            callback()
        except Exception:
            logger.exception("Shutdown callback failed: %s", getattr(callback, '__name__', str(callback)))


def shutdown() -> None:
    """Manually trigger graceful shutdown."""
    logger.info("Manual shutdown initiated")
    _run_callbacks()


def reset() -> None:
    """Reset the lifecycle manager (for testing)."""
    global _handler_installed
    with _lock:
        _shutdown_callbacks.clear()
        _handler_installed = False
