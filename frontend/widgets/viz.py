from __future__ import annotations
from typing import Dict

from PySide6 import QtCore, QtWidgets

from ..session import RemoteSession
from .config_editor import ConfigEditor
from ..schemas.train_schema import SCHEMA
from .image_viewer import ImageViewer   # 相对导入，避免包内导入错误


class HttpVizWorker(QtCore.QThread):
    event = QtCore.Signal(dict)
    done = QtCore.Signal(int)
    def __init__(self, session: RemoteSession, payload: dict, parent=None):
        super().__init__(parent); self._s = session; self._p = payload; self._rc = 0
    def run(self):
        try:
            res = self._s.http_post_json("/viz/one-click", self._p, timeout=120.0)
            self.event.emit({"type": "viz_result", "data": res})
        except Exception as e:
            self.event.emit({"type": "error", "message": str(e)}); self._rc = 1
        finally:
            self.done.emit(self._rc)


class VizWidget(QtWidgets.QWidget):
    def __init__(self, session: RemoteSession) -> None:
        super().__init__()
        self._session = session
        self._configs_dir = "~/projects/Sana/examples/train_configs"
        self._build_ui()
        self._refresh_config_list()

    def _build_ui(self) -> None:
        layout = QtWidgets.QVBoxLayout(self)

        # 顶部：配置下拉 + 刷新 + 运行
        topbar = QtWidgets.QHBoxLayout()
        self._config_combo = QtWidgets.QComboBox()
        self._refresh_btn = QtWidgets.QPushButton("刷新列表")
        self._run_btn = QtWidgets.QPushButton("一键可视化")
        self._refresh_btn.clicked.connect(self._refresh_config_list)
        self._run_btn.clicked.connect(self._on_run_clicked)
        topbar.addWidget(QtWidgets.QLabel("Config"))
        topbar.addWidget(self._config_combo, 1)
        topbar.addWidget(self._refresh_btn)
        topbar.addWidget(self._run_btn)
        layout.addLayout(topbar)

        # 结构化编辑器（与 Training 复用 schema）
        self._editor = ConfigEditor()
        self._editor.set_schema(SCHEMA)
        layout.addWidget(self._editor, 2)

        # 日志
        self._log = QtWidgets.QPlainTextEdit()
        self._log.setReadOnly(True)
        layout.addWidget(self._log, stretch=1)

    def _append_log(self, s: str) -> None:
        self._log.appendPlainText(s)

    def _refresh_config_list(self) -> None:
        try:
            files = self._session.list_yaml_configs(self._configs_dir)
        except Exception as exc:
            self._append_log(f"[YAML 列表失败] {exc}")
            files = []
        self._config_combo.blockSignals(True)
        self._config_combo.clear()
        self._config_combo.addItems(files)
        self._config_combo.blockSignals(False)

    def _on_run_clicked(self) -> None:
        payload: Dict[str, object] = {}
        # 优先使用编辑器 cfg
        try:
            cfg = self._editor.get_cfg()
            if isinstance(cfg, dict) and cfg:
                payload["config"] = cfg
        except Exception:
            pass

        if "config" not in payload:
            filename = self._config_combo.currentText().strip()
            if not filename:
                self._append_log("请选择一个配置文件。")
                return
            payload["config_path"] = f"examples/train_configs/{filename}"

        self._append_log("[VIZ] POST /viz/one-click …")
        self._run_btn.setEnabled(False)
        self._worker = HttpVizWorker(self._session, payload, self)
        self._worker.event.connect(self._handle_event)
        self._worker.done.connect(self._on_done)
        self._worker.start()

    def _handle_event(self, ev: dict) -> None:
        t = ev.get("type")
        if t == "error":
            self._append_log(f"[ERROR] {ev.get('message')}")
            return
        if t == "viz_result":
            data = ev.get("data") or {}
            log = data.get("log", "")
            if log:
                self._append_log("[LOG]\n" + str(log).strip())
            img_b64 = data.get("image_b64")
            if img_b64:
                dlg = ImageViewer(img_b64, title="一键可视化预览", parent=self)
                dlg.exec()
            else:
                self._append_log("[VIZ] 后端未返回 image_b64（占位接口）")
            return
        self._append_log(f"[VIZ] {ev}")

    def _on_done(self, rc: int) -> None:
        self._append_log(f"[VIZ] 完成，rc={rc}")
        self._run_btn.setEnabled(True)
