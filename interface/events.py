"""Utilities for streaming structured events from backend executions."""
from __future__ import annotations

import asyncio
import contextlib
import io
import logging
from datetime import datetime
from typing import Any, Callable, Iterable, List, Optional


EventEmitter = Callable[[dict], None]


class _QueueStream(io.TextIOBase):
    """File-like object that pushes newline-delimited chunks to an event emitter."""

    def __init__(self, emit: EventEmitter, stream: str) -> None:
        super().__init__()
        self._emit = emit
        self._stream = stream
        self._buffer: List[str] = []

    def write(self, data: str) -> int:  # type: ignore[override]
        if not data:
            return 0
        normalised = data.replace("\r", "\n")
        self._buffer.append(normalised)
        text = "".join(self._buffer)
        lines = text.splitlines(keepends=True)
        self._buffer = []
        remainder = ""
        for chunk in lines:
            if chunk.endswith("\n"):
                payload = chunk.rstrip("\n")
                if payload:
                    self._emit(
                        {
                            "event": "log",
                            "stream": self._stream,
                            "message": payload,
                            "ts": datetime.utcnow().isoformat() + "Z",
                        }
                    )
            else:
                remainder = chunk
        if remainder:
            self._buffer.append(remainder)
        return len(data)

    def flush(self) -> None:  # type: ignore[override]
        if not self._buffer:
            return
        text = "".join(self._buffer).strip()
        self._buffer = []
        if text:
            self._emit(
                {
                    "event": "log",
                    "stream": self._stream,
                    "message": text,
                    "ts": datetime.utcnow().isoformat() + "Z",
                }
            )


class QueueLoggingHandler(logging.Handler):
    """Logging handler that forwards log records as structured events."""

    def __init__(self, emit: EventEmitter) -> None:
        super().__init__()
        self._emit = emit
        self.setFormatter(logging.Formatter("%(message)s"))

    def emit(self, record: logging.LogRecord) -> None:  # type: ignore[override]
        msg = self.format(record)
        if not msg:
            return
        payload = {
            "event": "log",
            "stream": "logger",
            "logger": record.name,
            "level": record.levelname,
            "message": msg,
            "ts": datetime.utcnow().isoformat() + "Z",
        }
        self._emit(payload)


@contextlib.contextmanager
def capture_execution_events(emit: EventEmitter, *, logger_names: Optional[Iterable[str]] = None):
    """Capture stdout/stderr and selected loggers and forward them to ``emit``."""

    logger_names = list(logger_names or ["", "pytorch_lightning"])
    handler = QueueLoggingHandler(emit)
    attached: List[logging.Logger] = []

    try:
        for name in logger_names:
            logger = logging.getLogger(name)
            logger.addHandler(handler)
            attached.append(logger)

        stdout_proxy = _QueueStream(emit, "stdout")
        stderr_proxy = _QueueStream(emit, "stderr")

        with contextlib.redirect_stdout(stdout_proxy), contextlib.redirect_stderr(stderr_proxy):
            yield

        stdout_proxy.flush()
        stderr_proxy.flush()
    finally:
        for logger in attached:
            logger.removeHandler(handler)
        handler.close()


async def stream_function(fn: Callable[[], Any], emit: EventEmitter):
    """Run ``fn`` in a worker thread while streaming events through ``emit``."""

    loop = asyncio.get_running_loop()
    queue: asyncio.Queue[Any] = asyncio.Queue()
    sentinel = object()

    def _emit(event: dict) -> None:
        loop.call_soon_threadsafe(queue.put_nowait, event)

    def _runner() -> None:
        try:
            with capture_execution_events(_emit):
                result = fn()
            _emit({"event": "completed", "result": result, "ts": datetime.utcnow().isoformat() + "Z"})
        except Exception as exc:  # pragma: no cover - defensive
            _emit(
                {
                    "event": "error",
                    "message": str(exc),
                    "exc_type": exc.__class__.__name__,
                    "ts": datetime.utcnow().isoformat() + "Z",
                }
            )
        finally:
            loop.call_soon_threadsafe(queue.put_nowait, sentinel)

    future = loop.run_in_executor(None, _runner)

    try:
        while True:
            item = await queue.get()
            if item is sentinel:
                break
            emit(item)
    finally:
        await future
