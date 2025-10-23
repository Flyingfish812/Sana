from __future__ import annotations

from PySide6 import QtCore, QtGui, QtWidgets

from .session import RemoteSession
from .widgets.login import LoginWidget
from .widgets.training import TrainingWidget


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Sana Training Client")
        self._stack = QtWidgets.QStackedWidget()
        self.setCentralWidget(self._stack)

        self._session: RemoteSession | None = None

        self._login = LoginWidget()
        self._login.logged_in.connect(self._on_logged_in)
        self._stack.addWidget(self._login)

    @QtCore.Slot(object)
    def _on_logged_in(self, session: RemoteSession) -> None:
        self._session = session
        training = TrainingWidget(session)
        self._stack.addWidget(training)
        self._stack.setCurrentWidget(training)

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:  # type: ignore[override]
        if self._session is not None:
            self._session.close()
        super().closeEvent(event)
