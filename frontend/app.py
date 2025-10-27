from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

if __package__ is None or __package__ == "":  # pragma: no cover - script execution
    package_path = Path(__file__).resolve().parent
    sys.path.insert(0, str(package_path.parent))
    __package__ = package_path.name

from PySide6 import QtWidgets

from .main_window import MainWindow


def main(argv: Optional[list[str]] = None) -> int:
    app = QtWidgets.QApplication(argv or sys.argv)
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":  # pragma: no cover - script entry point
    sys.exit(main())
