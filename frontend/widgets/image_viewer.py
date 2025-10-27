# widgets/image_viewer.py
from __future__ import annotations
from PySide6 import QtCore, QtGui, QtWidgets
import base64

class ImageViewer(QtWidgets.QDialog):
    def __init__(self, b64_image: str, *, title: str = "可视化预览", parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        v = QtWidgets.QVBoxLayout(self)

        self._label = QtWidgets.QLabel(alignment=QtCore.Qt.AlignCenter)
        v.addWidget(self._label, 1)

        btns = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Close)
        btns.rejected.connect(self.reject)
        v.addWidget(btns)

        self._set_image(b64_image)
        self.resize(900, 640)

    def _set_image(self, b64_image: str):
        try:
            data = base64.b64decode(b64_image)
            pix = QtGui.QPixmap()
            pix.loadFromData(data)
            self._label.setPixmap(pix)
        except Exception as e:
            self._label.setText(f"图片渲染失败：{e}")
