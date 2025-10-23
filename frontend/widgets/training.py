from __future__ import annotations

import base64
import json
import shlex
from typing import Dict, Optional

from PySide6 import QtCore, QtWidgets

from ..session import RemoteSession

REMOTE_PROJECT_DIR = "~/projects/Sana"


class TrainingWorker(QtCore.QThread):
    event_received = QtCore.Signal(dict)
    finished_with_status = QtCore.Signal(int)

    def __init__(self, session: RemoteSession, command: str, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self._session = session
        self._command = command
        self._exit_status = 0

    def run(self) -> None:  # pragma: no cover - Qt thread
        try:
            for event in self._session.stream_jsonl(self._command):
                self.event_received.emit(event)
                if event.get("type") == "error":
                    self._exit_status = 1
        except Exception as exc:
            self.event_received.emit({"type": "error", "message": str(exc)})
            self._exit_status = 1
        finally:
            self.finished_with_status.emit(self._exit_status)


class TrainingWidget(QtWidgets.QWidget):
    def __init__(self, session: RemoteSession) -> None:
        super().__init__()
        self._session = session
        self._worker: Optional[TrainingWorker] = None
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QtWidgets.QVBoxLayout(self)

        form = QtWidgets.QFormLayout()
        self._config_path = QtWidgets.QLineEdit("examples/train_configs/epd_smoketest_unet.yaml")
        self._overrides = QtWidgets.QPlainTextEdit()
        self._overrides.setPlaceholderText("Optional JSON overrides to merge into the config.")

        form.addRow("Config path", self._config_path)
        form.addRow("Overrides", self._overrides)
        layout.addLayout(form)

        self._train_button = QtWidgets.QPushButton("Run Training")
        self._train_button.clicked.connect(self._on_train_clicked)
        layout.addWidget(self._train_button)

        self._progress = QtWidgets.QProgressBar()
        self._progress.setRange(0, 100)
        layout.addWidget(self._progress)

        self._log_view = QtWidgets.QPlainTextEdit()
        self._log_view.setReadOnly(True)
        layout.addWidget(self._log_view, stretch=1)

    def _on_train_clicked(self) -> None:
        overrides_text = self._overrides.toPlainText().strip()
        overrides_b64: str | None = None
        if overrides_text:
            try:
                parsed = json.loads(overrides_text)
            except json.JSONDecodeError as exc:
                self._append_log(f"Invalid overrides JSON: {exc}")
                return
            overrides_b64 = base64.b64encode(json.dumps(parsed).encode("utf-8")).decode("utf-8")

        config_path = self._config_path.text().strip()
        if not config_path:
            self._append_log("Config path is required.")
            return

        command = self._build_command(config_path, overrides_b64)
        self._append_log(f"Starting training with command: {command}")
        self._progress.setValue(0)
        self._train_button.setEnabled(False)

        self._worker = TrainingWorker(self._session, command)
        self._worker.event_received.connect(self._handle_event)
        self._worker.finished_with_status.connect(self._on_finished)
        self._worker.start()

    def _build_command(self, config_path: str, overrides_b64: str | None) -> str:
        pieces = [
            f"cd {shlex.quote(REMOTE_PROJECT_DIR)}",
            "&&",
            "python -m scripts.jsonl_train",
            f"--config {shlex.quote(config_path)}",
        ]
        if overrides_b64:
            pieces.append(f"--overrides-b64 {shlex.quote(overrides_b64)}")
        return " ".join(pieces)

    @QtCore.Slot(dict)
    def _handle_event(self, event: Dict[str, object]) -> None:
        etype = event.get("type")
        if etype == "log":
            message = str(event.get("message", ""))
            self._append_log(message)
        elif etype == "progress":
            progress = float(event.get("progress", 0.0)) * 100
            self._progress.setValue(int(progress))
        elif etype == "status":
            self._append_log(str(event))
            if event.get("event") == "completed":
                self._progress.setValue(100)
        elif etype == "error":
            self._append_log(f"Error: {event.get('message')}")
            self._progress.setValue(0)

    @QtCore.Slot(int)
    def _on_finished(self, status: int) -> None:
        if status == 0:
            self._append_log("Training finished successfully.")
        else:
            self._append_log("Training failed. Check logs above.")
        self._train_button.setEnabled(True)
        self._worker = None

    def _append_log(self, message: str) -> None:
        self._log_view.appendPlainText(message)
        cursor = self._log_view.textCursor()
        cursor.movePosition(cursor.End)
        self._log_view.setTextCursor(cursor)
