from __future__ import annotations

import sys
from typing import Optional

from PySide6 import QtWidgets

from .main_window import MainWindow


def main(argv: Optional[list[str]] = None) -> int:
    app = QtWidgets.QApplication(argv or sys.argv)
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":  # pragma: no cover - script entry point
    sys.exit(main())
