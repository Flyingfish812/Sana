from __future__ import annotations

from PySide6 import QtCore, QtGui, QtWidgets

from .session import RemoteSession
from .widgets.login import LoginWidget
from .widgets.training import TrainingWidget
from .widgets.data import DataWidget
from .widgets.viz import VizWidget

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

        # 构建三个页签：Training / Data / Viz
        tabs = QtWidgets.QTabWidget()
        tabs.setDocumentMode(True)

        training = TrainingWidget(session)
        data = DataWidget(session)
        viz = VizWidget(session)

        tabs.addTab(training, "Training")
        tabs.addTab(data, "Data")
        tabs.addTab(viz, "Visualization")
        tabs.setCurrentWidget(training)

        # 将页签加入到栈中并切换
        self._stack.addWidget(tabs)
        self._stack.setCurrentWidget(tabs)

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:  # type: ignore[override]
        if self._session is not None:
            self._session.close()
        super().closeEvent(event)
