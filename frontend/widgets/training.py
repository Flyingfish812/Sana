from __future__ import annotations

import base64
import json
import shlex
from typing import Dict, Optional

from PySide6 import QtCore, QtWidgets
from PySide6.QtGui import QTextCursor

from ..session import RemoteSession

import yaml
from .config_editor import ConfigEditor
from ..schemas.train_schema import SCHEMA
from widgets.image_viewer import ImageViewer

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

class HttpTrainingWorker(QtCore.QThread):
    event_received = QtCore.Signal(dict)
    finished_with_status = QtCore.Signal(int)

    def __init__(self, session: RemoteSession, payload: dict, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self._session = session
        self._payload = payload
        self._exit_status = 0

    def run(self) -> None:  # pragma: no cover - Qt thread
        try:
            # 调用 HTTP /train/run，流式读取
            for event in self._session.http_stream_ndjson("/train/run", self._payload, timeout=10.0):
                # 事件规范化：接口层可能发 {"event":"completed",...} / 也可能发日志
                if "type" not in event and "event" in event:
                    evt = event.get("event")
                    if evt == "completed":
                        self.event_received.emit({"type": "status", "event": "completed", **event})
                        continue
                    elif evt == "error":
                        self.event_received.emit({"type": "error", **event})
                        self._exit_status = 1
                        continue
                    else:
                        self.event_received.emit({"type": "status", **event})
                        continue

                # 默认原样透传（日志/进度）
                self.event_received.emit(event)
                if event.get("type") == "error":
                    self._exit_status = 1
        except Exception as exc:
            self.event_received.emit({"type": "error", "message": str(exc)})
            self._exit_status = 1
        finally:
            self.finished_with_status.emit(self._exit_status)

class HttpDataWorker(QtCore.QThread):
    event = QtCore.Signal(dict)
    done = QtCore.Signal(int)
    def __init__(self, session: RemoteSession, payload: dict, parent=None):
        super().__init__(parent); self._s = session; self._p = payload; self._rc = 0
    def run(self):
        try:
            res = self._s.http_post_json("/data/run", self._p, timeout=120.0)
            self.event.emit({"type":"status","message":"dataio finished","data":res})
        except Exception as e:
            self.event.emit({"type":"error","message":str(e)}); self._rc = 1
        finally:
            self.done.emit(self._rc)

class HttpVizWorker(QtCore.QThread):
    event = QtCore.Signal(dict)
    done = QtCore.Signal(int)
    def __init__(self, session: RemoteSession, payload: dict, parent=None):
        super().__init__(parent); self._s = session; self._p = payload; self._rc = 0
    def run(self):
        try:
            res = self._s.http_post_json("/viz/one-click", self._p, timeout=120.0)
            # 期望返回格式：{"ok":true, "image_b64":"...", "log":"..."}（占位）
            self.event.emit({"type":"viz_result","data":res})
        except Exception as e:
            self.event.emit({"type":"error","message":str(e)}); self._rc = 1
        finally:
            self.done.emit(self._rc)

class TrainingWidget(QtWidgets.QWidget):
    def __init__(self, session: RemoteSession) -> None:
        super().__init__()
        self._session = session
        self._worker: Optional[TrainingWorker] = None
        self._train_configs_dir = f"{REMOTE_PROJECT_DIR}/examples/train_configs"
        self._build_ui()
        self._refresh_config_list()

    def _build_ui(self) -> None:
        layout = QtWidgets.QVBoxLayout(self)

        # 顶部：配置下拉 + 刷新
        topbar = QtWidgets.QHBoxLayout()
        self._config_combo = QtWidgets.QComboBox()
        self._config_combo.currentIndexChanged.connect(self._on_config_selected)
        self._refresh_btn = QtWidgets.QPushButton("刷新列表")
        self._refresh_btn.clicked.connect(self._refresh_config_list)
        topbar.addWidget(QtWidgets.QLabel("Config"))
        topbar.addWidget(self._config_combo, 1)
        topbar.addWidget(self._refresh_btn)
        layout.addLayout(topbar)

        # 编辑器（结构化）
        self._editor = ConfigEditor()
        layout.addWidget(self._editor, 2)

        # 兼容旧的 Overrides（保留；可在 Batch C 再收口）
        form = QtWidgets.QFormLayout()
        self._overrides = QtWidgets.QPlainTextEdit()
        self._overrides.setPlaceholderText("可选：JSON overrides（暂保留，后续用结构化表单替代）")
        form.addRow("Overrides", self._overrides)
        layout.addLayout(form)

        # 运行区域
        self._train_button = QtWidgets.QPushButton("Run Training")
        self._train_button.clicked.connect(self._on_train_clicked)
        layout.addWidget(self._train_button)

        self._progress = QtWidgets.QProgressBar()
        self._progress.setRange(0, 100)
        layout.addWidget(self._progress)

        self._log_view = QtWidgets.QPlainTextEdit()
        self._log_view.setReadOnly(True)
        layout.addWidget(self._log_view, stretch=1)

        self._editor.set_schema(SCHEMA)
        self._editor.validityChanged.connect(self._on_editor_validity_changed)

        self._data_btn = QtWidgets.QPushButton("准备数据")
        self._viz_btn = QtWidgets.QPushButton("一键可视化")
        self._data_btn.clicked.connect(self._on_data_run_clicked)
        self._viz_btn.clicked.connect(self._on_viz_clicked)
        topbar.addWidget(self._data_btn)
        topbar.addWidget(self._viz_btn)

    def _on_editor_validity_changed(self, ok: bool):
        # 仅根据校验状态启用/禁用按钮；仍允许用户保存 YAML 预览等操作
        self._train_button.setEnabled(ok)

    def _on_train_clicked(self) -> None:
        if hasattr(self, "_editor") and self._editor is not None:
            if not self._editor.is_valid():
                errs = self._editor.get_errors()
                self._append_log("[校验未通过] " + "；".join(errs[:3]) + (" ..." if len(errs) > 3 else ""))
                QtWidgets.QMessageBox.warning(self, "配置有误", "\n".join(errs[:8]))
                return
        
        # 1) 组装 payload（优先使用结构化编辑器；否则使用 config_path）
        payload: Dict[str, object] = {}

        # 如果批次 A 已引入 ConfigEditor：
        cfg_from_editor = None
        if hasattr(self, "_editor") and self._editor is not None:
            try:
                cfg_from_editor = self._editor.get_cfg()
                if isinstance(cfg_from_editor, dict) and cfg_from_editor:
                    payload["config"] = cfg_from_editor
            except Exception:
                cfg_from_editor = None  # 容错

        # 若没有结构化 cfg，则退回到 config_path + overrides
        if "config" not in payload:
            config_path = self._config_path.text().strip()
            if not config_path:
                self._append_log("Config path is required (or provide a valid config in the editor).")
                return
            payload["config_path"] = f"{REMOTE_PROJECT_DIR}/{config_path}" if not config_path.startswith("/") and not config_path.startswith("~") else config_path

            overrides_text = self._overrides.toPlainText().strip()
            if overrides_text:
                try:
                    parsed = json.loads(overrides_text)
                except json.JSONDecodeError as exc:
                    self._append_log(f"Invalid overrides JSON: {exc}")
                    return
                # 把 overrides 合并到 config（后端也能合并，但此处直接下发完整 config 更直观）
                payload["config"] = parsed
                # 若你希望由后端合并，可改成 payload["overrides"] = parsed，并在接口侧支持

        # 2) 启动 HTTP 线程
        self._append_log(f"POST /train/run with payload keys: {list(payload.keys())}")
        self._progress.setValue(0)
        self._train_button.setEnabled(False)

        self._worker = HttpTrainingWorker(self._session, payload)
        self._worker.event_received.connect(self._handle_event)
        self._worker.finished_with_status.connect(self._on_finished)
        self._worker.start()

    def _on_data_run_clicked(self):
        # 优先用编辑器的 cfg；否则回退到下拉选中的 YAML 路径（让后端自行读取）
        payload: Dict[str, object] = {}
        if hasattr(self, "_editor") and self._editor is not None:
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

        self._append_log("[DATA] POST /data/run …")
        self._data_btn.setEnabled(False)
        self._worker_data = HttpDataWorker(self._session, payload, self)
        self._worker_data.event.connect(self._handle_data_event)
        self._worker_data.done.connect(self._on_data_done)
        self._worker_data.start()

    def _handle_data_event(self, ev: dict):
        t = ev.get("type")
        if t == "error":
            self._append_log(f"[DATA][ERROR] {ev.get('message')}")
        elif t == "status":
            self._append_log(f"[DATA] {ev.get('message')}")
            if "data" in ev:
                # 简单打印返回摘要
                self._append_log(f"[DATA][RETURN] keys={list(ev['data'].keys())}")
        else:
            self._append_log(f"[DATA] {ev}")

    def _on_data_done(self, rc: int):
        self._append_log(f"[DATA] 完成，rc={rc}")
        self._data_btn.setEnabled(True)

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
    
    def _on_viz_clicked(self):
        # 同样优先使用编辑器 cfg（可从中读取 dataset/dls 信息）
        payload: Dict[str, object] = {}
        if hasattr(self, "_editor") and self._editor is not None:
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
        self._viz_btn.setEnabled(False)
        self._worker_viz = HttpVizWorker(self._session, payload, self)
        self._worker_viz.event.connect(self._handle_viz_event)
        self._worker_viz.done.connect(self._on_viz_done)
        self._worker_viz.start()

    def _handle_viz_event(self, ev: dict):
        t = ev.get("type")
        if t == "error":
            self._append_log(f"[VIZ][ERROR] {ev.get('message')}")
            return
        if t == "viz_result":
            data = ev.get("data") or {}
            log = data.get("log", "")
            if log:
                self._append_log("[VIZ][LOG]\n" + log.strip())
            img_b64 = data.get("image_b64")
            if img_b64:
                dlg = ImageViewer(img_b64, title="一键可视化预览", parent=self)
                dlg.exec()
            else:
                self._append_log("[VIZ] 后端未返回 image_b64（占位接口）")
            return
        # 兜底
        self._append_log(f"[VIZ] {ev}")

    def _on_viz_done(self, rc: int):
        self._append_log(f"[VIZ] 完成，rc={rc}")
        self._viz_btn.setEnabled(True)

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
        cursor.movePosition(QTextCursor.MoveOperation.End)  # ✅
        self._log_view.setTextCursor(cursor)
        self._log_view.setTextCursor(cursor)

    def _refresh_config_list(self) -> None:
        """列出远端 examples/train_configs 下所有 YAML 填充下拉。"""
        try:
            files = self._session.list_yaml_configs(self._train_configs_dir)
        except Exception as exc:
            self._append_log(f"[YAML 列表失败] {exc}")
            files = []
        self._config_combo.blockSignals(True)
        self._config_combo.clear()
        self._config_combo.addItems(files)
        self._config_combo.blockSignals(False)
        if files:
            self._on_config_selected(0)

    def _on_config_selected(self, _index: int) -> None:
        self._load_selected_yaml()

    def _load_selected_yaml(self) -> None:
        filename = self._config_combo.currentText().strip()
        if not filename:
            return
        full_path = f"{self._train_configs_dir}/{filename}"
        try:
            text = self._session.read_text_file(full_path)
            cfg = yaml.safe_load(text) or {}
            if not isinstance(cfg, dict):
                self._append_log(f"[载入失败] {filename} 不是 dict 顶层，已按原文档保留。")
                cfg = {"__RAW__": text}
            self._editor.load_cfg(cfg)
            self._append_log(f"已载入配置：{filename}")
        except Exception as exc:
            self._append_log(f"[读取失败] {full_path} — {exc}")
