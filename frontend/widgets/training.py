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

        # 配置来源模式
        mode_box = QtWidgets.QGroupBox("配置来源")
        hb = QtWidgets.QHBoxLayout(mode_box)
        self._mode_form = QtWidgets.QRadioButton("使用表单")
        self._mode_yaml = QtWidgets.QRadioButton("使用 YAML（下方下拉）")
        self._mode_form.setChecked(True)
        hb.addWidget(self._mode_form); hb.addWidget(self._mode_yaml); hb.addStretch(1)
        layout.addWidget(mode_box)

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

        try:
            # 如果 RemoteSession 没有 http_get_json，可改为 http_post_json("/model/registry", {})
            registry = self._session.http_get_json("/model/registry", timeout=10.0)
            # 注入到编辑器上下文：供 widget="module" 使用
            self._editor.set_context({"model_registry": registry})
            self._append_log("[Model] 已加载模块注册表。")
        except Exception as exc:
            self._append_log(f"[Model] 加载注册表失败：{exc}")
    
    def _on_editor_validity_changed(self, ok: bool):
        # 仅根据校验状态启用/禁用按钮；仍允许用户保存 YAML 预览等操作
        self._train_button.setEnabled(ok)

    def _on_train_clicked(self) -> None:
        payload: Dict[str, object] = {}

        if self._mode_form.isChecked():
            # 表单模式：严格依赖编辑器输出
            if not self._editor.is_valid():
                errs = self._editor.get_errors()
                self._append_log("[校验未通过] " + "；".join(errs[:3]) + (" ..." if len(errs) > 3 else ""))
                QtWidgets.QMessageBox.warning(self, "配置有误", "\n".join(errs[:8]))
                return
            try:
                cfg_from_editor = self._editor.get_cfg()
                if not isinstance(cfg_from_editor, dict) or not cfg_from_editor:
                    self._append_log("表单未生成有效配置。")
                    return
                payload["config"] = cfg_from_editor
            except Exception as exc:
                self._append_log(f"表单读取失败：{exc}")
                return

        else:
            # YAML 模式：严格使用下拉 YAML（可选 overrides）
            filename = self._config_combo.currentText().strip()
            if not filename:
                self._append_log("请选择一个配置文件。")
                return
            payload["config_path"] = f"examples/train_configs/{filename}"

            overrides_text = self._overrides.toPlainText().strip()
            if overrides_text:
                try:
                    parsed = json.loads(overrides_text)
                except json.JSONDecodeError as exc:
                    self._append_log(f"Invalid overrides JSON: {exc}")
                    return
                # 若你希望后端合并，可改为 payload["overrides"]=parsed
                payload["config"] = parsed

        self._append_log(f"POST /train/run with payload keys: {list(payload.keys())}")
        self._progress.setValue(0)
        self._train_button.setEnabled(False)

        self._worker = HttpTrainingWorker(self._session, payload)
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
