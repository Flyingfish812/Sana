from __future__ import annotations

from PySide6 import QtCore, QtWidgets

from ..session import RemoteSession, SSHCredentials


class ConnectionWorker(QtCore.QThread):
    connected = QtCore.Signal(object)
    failed = QtCore.Signal(str)

    def __init__(self, creds: SSHCredentials, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self._creds = creds

    def run(self) -> None:  # pragma: no cover - Qt thread
        try:
            session = RemoteSession()
            session.connect(self._creds)
        except Exception as exc:
            self.failed.emit(str(exc))
        else:
            self.connected.emit(session)


class LoginWidget(QtWidgets.QWidget):
    logged_in = QtCore.Signal(object)

    def __init__(self) -> None:
        super().__init__()
        self._build_ui()
        self._worker: ConnectionWorker | None = None

    def _build_ui(self) -> None:
        layout = QtWidgets.QVBoxLayout(self)

        form = QtWidgets.QFormLayout()
        self._host = QtWidgets.QLineEdit("127.0.0.1")
        self._port = QtWidgets.QSpinBox()
        self._port.setRange(1, 65535)
        self._port.setValue(22)
        self._username = QtWidgets.QLineEdit()
        self._password = QtWidgets.QLineEdit()
        self._password.setEchoMode(QtWidgets.QLineEdit.Password)

        form.addRow("Host", self._host)
        form.addRow("Port", self._port)
        form.addRow("Username", self._username)
        form.addRow("Password", self._password)

        layout.addLayout(form)

        self._status = QtWidgets.QLabel("")
        self._status.setWordWrap(True)
        layout.addWidget(self._status)

        self._button = QtWidgets.QPushButton("Connect")
        self._button.clicked.connect(self._on_connect_clicked)
        layout.addWidget(self._button)

        layout.addStretch(1)

    def _on_connect_clicked(self) -> None:
        creds = SSHCredentials(
            host=self._host.text().strip(),
            port=self._port.value(),
            username=self._username.text().strip(),
            password=self._password.text(),
        )

        if not creds.host or not creds.username:
            self._status.setText("Host and username are required.")
            return

        self._button.setEnabled(False)
        self._status.setText("Connecting…")

        self._worker = ConnectionWorker(creds)
        self._worker.connected.connect(self._on_connected)
        self._worker.failed.connect(self._on_failed)
        self._worker.finished.connect(self._cleanup_worker)
        self._worker.start()

    @QtCore.Slot(object)
    def _on_connected(self, session: RemoteSession) -> None:
        self._status.setText("Connected.")
        self.logged_in.emit(session)

    @QtCore.Slot(str)
    def _on_failed(self, message: str) -> None:
        self._status.setText(f"Connection failed: {message}")
        self._button.setEnabled(True)

    @QtCore.Slot()
    def _cleanup_worker(self) -> None:
        self._worker = None
        self._button.setEnabled(True)
