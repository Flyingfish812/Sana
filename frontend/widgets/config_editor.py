# widgets/config_editor.py
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
from PySide6 import QtCore, QtWidgets
import yaml
import os

_SCALAR_TYPES = (int, float, bool, str)

class _ScalarRow(QtWidgets.QWidget):
    changed = QtCore.Signal(str, object)  # key, value

    def __init__(self, key: str, value: Any):
        super().__init__()
        self._key = key
        self._value = value
        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self._label = QtWidgets.QLabel(key)
        layout.addWidget(self._label)

        if isinstance(value, bool):
            self._editor = QtWidgets.QCheckBox()
            self._editor.setChecked(bool(value))
            self._editor.stateChanged.connect(self._on_bool_changed)
        elif isinstance(value, (int, float)):
            self._editor = QtWidgets.QLineEdit(str(value))
            self._editor.editingFinished.connect(self._on_number_changed)
        else:
            self._editor = QtWidgets.QLineEdit("" if value is None else str(value))
            self._editor.editingFinished.connect(self._on_text_changed)
        layout.addWidget(self._editor, 1)

    def _on_bool_changed(self, _):
        self._value = bool(self._editor.isChecked())
        self.changed.emit(self._key, self._value)

    def _on_number_changed(self):
        txt = self._editor.text().strip()
        if "." in txt:
            try:
                self._value = float(txt)
            except ValueError:
                pass
        else:
            try:
                self._value = int(txt)
            except ValueError:
                pass
        self.changed.emit(self._key, self._value)

    def _on_text_changed(self):
        self._value = self._editor.text()
        self.changed.emit(self._key, self._value)

    def value(self):
        return self._value


class ConfigEditor(QtWidgets.QWidget):
    """一级 key 为块；块内：标量→行编辑；复杂结构→YAML 文本"""
    cfgChanged = QtCore.Signal(dict)
    validityChanged = QtCore.Signal(bool)

    def __init__(self):
        super().__init__()
        self._cfg: Dict[str, Any] = {}
        self._blocks: Dict[str, Dict[str, Any]] = {}

        self._layout = QtWidgets.QVBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)

        # 顶部工具条：预览 YAML
        toolbar = QtWidgets.QHBoxLayout()
        self._preview_btn = QtWidgets.QPushButton("预览 YAML")
        self._preview_btn.clicked.connect(self._show_preview)
        toolbar.addStretch(1)
        toolbar.addWidget(self._preview_btn)
        self._layout.addLayout(toolbar)

        self._scroll = QtWidgets.QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._container = QtWidgets.QWidget()
        self._form = QtWidgets.QVBoxLayout(self._container)
        self._form.addStretch(1)
        self._scroll.setWidget(self._container)
        self._layout.addWidget(self._scroll, 1)
        self._schema: Optional[Dict[str, Any]] = None
        self._errors: List[str] = []

        # 错误提示条
        self._err_bar = QtWidgets.QLabel()
        self._err_bar.setStyleSheet("color:#c62828;")
        self._err_bar.setVisible(False)
        self._layout.addWidget(self._err_bar)

    def set_schema(self, schema: Dict[str, Any]):
        """注入 schema，用于排序、控件选择与校验。"""
        self._schema = schema
        # 重新渲染（若已有 cfg）
        if self._cfg:
            self.load_cfg(self._cfg)

    def load_cfg(self, cfg: Dict[str, Any]):
        self._cfg = {} if cfg is None else dict(cfg)
        # 清空
        while self._form.count() > 0:
            item = self._form.takeAt(0)
            w = item.widget()
            if w:
                w.deleteLater()

        # 重建
        covered_top_keys = set()
        if self._schema and "blocks" in self._schema:
            for block in self._schema["blocks"]:
                top_key = block.get("key")
                title = block.get("title", str(top_key))
                box = QtWidgets.QGroupBox(title)
                vbox = QtWidgets.QVBoxLayout(box)

                top_val = self._cfg.get(top_key, {})
                if not isinstance(top_val, dict):
                    top_val = {}

                grid = QtWidgets.QFormLayout()
                # 记录该 block 已覆盖的子键
                covered_sub_keys = set()

                for field in block.get("fields", []):
                    name = field["name"]
                    ftype = field.get("type", "str")
                    widget_type = field.get("widget", "text")
                    help_text = field.get("help", "")
                    default = field.get("default", None)
                    value = top_val.get(name, default)

                    editor = self._make_field_editor(widget_type, ftype, value, field)
                    # 绑定变更：更新 self._cfg[top_key][name]
                    def _bind(editor_ref=editor, tk=top_key, nk=name, ft=ftype, wtype=widget_type):
                        def _on_change():
                            new_val = self._read_editor_value(editor_ref, wtype, ft)
                            if tk not in self._cfg or not isinstance(self._cfg[tk], dict):
                                self._cfg[tk] = {}
                            self._cfg[tk][nk] = new_val
                            self._validate_and_emit()
                        return _on_change
                    self._connect_editor_signal(editor, _bind())

                    row = QtWidgets.QWidget()
                    row_layout = QtWidgets.QVBoxLayout(row)
                    row_layout.setContentsMargins(0,0,0,0)
                    row_layout.addWidget(editor)
                    if help_text:
                        hint = QtWidgets.QLabel(help_text)
                        hint.setStyleSheet("color:#666; font-size:12px;")
                        row_layout.addWidget(hint)
                    grid.addRow(name, row)

                    covered_sub_keys.add(name)

                if grid.rowCount() > 0:
                    vbox.addLayout(grid)

                # 把此 block 下未覆盖的键（复杂结构/未知键）落入 YAML 编辑区
                non_scalar_buf = {}
                for k, v in self._cfg.get(top_key, {}).items():
                    if k not in covered_sub_keys:
                        non_scalar_buf[k] = v
                if non_scalar_buf:
                    yaml_edit = QtWidgets.QPlainTextEdit(yaml.safe_dump(non_scalar_buf, sort_keys=False, allow_unicode=True))
                    yaml_edit.setLineWrapMode(QtWidgets.QPlainTextEdit.NoWrap)
                    yaml_edit.setPlaceholderText("高级字段（dict/list）以 YAML 编辑")
                    yaml_edit.textChanged.connect(lambda tk=top_key, ed=yaml_edit: self._on_yaml_blob_changed(tk, ed))
                    vbox.addWidget(yaml_edit)

                self._form.insertWidget(self._form.count() - 1, box)
                covered_top_keys.add(top_key)

            # 顶层未覆盖的 key → Advanced 区
            for top_key, top_val in self._cfg.items():
                if top_key in covered_top_keys:
                    continue
                box = QtWidgets.QGroupBox(f"Advanced: {top_key}")
                v = QtWidgets.QVBoxLayout(box)
                yaml_edit = QtWidgets.QPlainTextEdit(yaml.safe_dump(top_val, sort_keys=False, allow_unicode=True))
                yaml_edit.setLineWrapMode(QtWidgets.QPlainTextEdit.NoWrap)
                yaml_edit.textChanged.connect(lambda tk=top_key, ed=yaml_edit: self._on_top_yaml_changed(tk, ed))
                v.addWidget(yaml_edit)
                self._form.insertWidget(self._form.count() - 1, box)
        else:
            # 没有 schema：退回原来的简单渲染
            for top_key, top_val in self._cfg.items():
                box = QtWidgets.QGroupBox(str(top_key))
                v = QtWidgets.QVBoxLayout(box)
                if isinstance(top_val, dict):
                    grid = QtWidgets.QFormLayout()
                    non_scalar_buf = {}
                    for k, vval in top_val.items():
                        if isinstance(vval, _SCALAR_TYPES) or vval is None:
                            row = _ScalarRow(k, vval)
                            row.changed.connect(lambda kk, vv, tk=top_key: self._on_scalar_changed(tk, kk, vv))
                            grid.addRow(k, row)
                        else:
                            non_scalar_buf[k] = vval
                    if grid.rowCount() > 0:
                        v.addLayout(grid)
                    if non_scalar_buf:
                        yaml_edit = QtWidgets.QPlainTextEdit(yaml.safe_dump(non_scalar_buf, sort_keys=False, allow_unicode=True))
                        yaml_edit.setLineWrapMode(QtWidgets.QPlainTextEdit.NoWrap)
                        yaml_edit.setPlaceholderText("复杂结构（dict/list）以 YAML 编辑")
                        yaml_edit.textChanged.connect(lambda tk=top_key, ed=yaml_edit: self._on_yaml_blob_changed(tk, ed))
                        v.addWidget(yaml_edit)
                else:
                    yaml_edit = QtWidgets.QPlainTextEdit(yaml.safe_dump(top_val, sort_keys=False, allow_unicode=True))
                    yaml_edit.setLineWrapMode(QtWidgets.QPlainTextEdit.NoWrap)
                    yaml_edit.textChanged.connect(lambda tk=top_key, ed=yaml_edit: self._on_top_yaml_changed(tk, ed))
                    v.addWidget(yaml_edit)
                self._form.insertWidget(self._form.count() - 1, box)

        # 初次渲染后做一次校验
        self._validate_and_emit()

    def _on_scalar_changed(self, top_key: str, key: str, value: Any):
        if top_key not in self._cfg or not isinstance(self._cfg[top_key], dict):
            self._cfg[top_key] = {}
        self._cfg[top_key][key] = value
        self.cfgChanged.emit(self.get_cfg())

    def _on_yaml_blob_changed(self, top_key: str, editor: QtWidgets.QPlainTextEdit):
        txt = editor.toPlainText()
        try:
            obj = yaml.safe_load(txt) or {}
            if top_key not in self._cfg or not isinstance(self._cfg[top_key], dict):
                self._cfg[top_key] = {}
            # 用解析到的键覆盖/更新复杂区
            for k in list(self._cfg[top_key].keys()):
                pass  # 保留标量；复杂键由 obj 覆盖
            if isinstance(obj, dict):
                for k, v in obj.items():
                    self._cfg[top_key][k] = v
            else:
                # 如果用户改成了 list/标量，就直接替换顶层
                self._cfg[top_key] = obj
        except Exception:
            pass
        self.cfgChanged.emit(self.get_cfg())

    def _on_top_yaml_changed(self, top_key: str, editor: QtWidgets.QPlainTextEdit):
        txt = editor.toPlainText()
        try:
            self._cfg[top_key] = yaml.safe_load(txt)
        except Exception:
            pass
        self.cfgChanged.emit(self.get_cfg())

    def get_cfg(self) -> Dict[str, Any]:
        return dict(self._cfg)

    def _show_preview(self):
        yaml_text = yaml.safe_dump(self.get_cfg(), sort_keys=False, allow_unicode=True)
        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle("YAML 预览")
        l = QtWidgets.QVBoxLayout(dlg)
        edit = QtWidgets.QPlainTextEdit(yaml_text)
        edit.setReadOnly(True)
        edit.setLineWrapMode(QtWidgets.QPlainTextEdit.NoWrap)
        l.addWidget(edit)
        btn = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok)
        btn.accepted.connect(dlg.accept)
        l.addWidget(btn)
        dlg.resize(700, 500)
        dlg.exec()

    def _make_field_editor(self, widget_type: str, ftype: str, value: Any, field: Dict[str, Any]) -> QtWidgets.QWidget:
        if widget_type == "select":
            cb = QtWidgets.QComboBox()
            for ch in field.get("choices", []):
                cb.addItem(str(ch), ch)
            if value is not None:
                # 尝试定位初始值
                idx = cb.findData(value)
                if idx >= 0:
                    cb.setCurrentIndex(idx)
            return cb
        elif widget_type == "number":
            # 用 QLineEdit + 校验（支持 float/int）
            le = QtWidgets.QLineEdit("" if value is None else str(value))
            le.setPlaceholderText(f"{ftype} in [{field.get('min','-inf')}, {field.get('max','inf')}]")
            return le
        elif widget_type == "path":
            w = QtWidgets.QWidget()
            h = QtWidgets.QHBoxLayout(w); h.setContentsMargins(0,0,0,0)
            le = QtWidgets.QLineEdit("" if value is None else str(value))
            btn = QtWidgets.QPushButton("选择…")
            # 仅做本地对话框供参考；远端路径请仍以文本为准（不自动上传）
            def pick():
                dlg = QtWidgets.QFileDialog(self)
                dlg.setFileMode(QtWidgets.QFileDialog.Directory)
                if dlg.exec():
                    sel = dlg.selectedFiles()
                    if sel:
                        le.setText(sel[0])
                        le.editingFinished.emit()
            btn.clicked.connect(pick)
            h.addWidget(le, 1); h.addWidget(btn)
            # 存一份引用，取值时读 lineedit
            w._lineedit = le   # type: ignore
            return w
        elif widget_type == "checkbox":
            ck = QtWidgets.QCheckBox()
            ck.setChecked(bool(value))
            return ck
        else:
            # 默认文本
            le = QtWidgets.QLineEdit("" if value is None else str(value))
            return le

    def _read_editor_value(self, editor: QtWidgets.QWidget, widget_type: str, ftype: str) -> Any:
        if widget_type == "select":
            cb: QtWidgets.QComboBox = editor  # type: ignore
            return cb.currentData()
        if widget_type == "checkbox":
            ck: QtWidgets.QCheckBox = editor  # type: ignore
            return bool(ck.isChecked())
        if widget_type == "path":
            le: QtWidgets.QLineEdit = getattr(editor, "_lineedit")  # type: ignore
            return le.text().strip()
        # number / text
        le: QtWidgets.QLineEdit = editor  # type: ignore
        txt = le.text().strip()
        if ftype in ("int",):
            try:
                return int(txt)
            except Exception:
                return txt
        if ftype in ("float",):
            try:
                return float(txt)
            except Exception:
                return txt
        return txt

    def _connect_editor_signal(self, editor: QtWidgets.QWidget, slot):
        if isinstance(editor, QtWidgets.QComboBox):
            editor.currentIndexChanged.connect(lambda _i: slot())
        elif isinstance(editor, QtWidgets.QCheckBox):
            editor.stateChanged.connect(lambda _s: slot())
        elif isinstance(editor, QtWidgets.QWidget) and hasattr(editor, "_lineedit"):
            getattr(editor, "_lineedit").editingFinished.connect(slot)  # path 小组件
        else:
            editor.editingFinished.connect(slot)  # QLineEdit

    def is_valid(self) -> bool:
        return len(self._errors) == 0

    def get_errors(self) -> List[str]:
        return list(self._errors)

    def _validate_and_emit(self):
        self._errors = self._validate()
        ok = len(self._errors) == 0
        self._err_bar.setVisible(not ok)
        self._err_bar.setText("；".join(self._errors[:3]) + (" ..." if len(self._errors) > 3 else ""))
        self.cfgChanged.emit(self.get_cfg())
        self.validityChanged.emit(ok)

    def _validate(self) -> List[str]:
        errs: List[str] = []
        if not self._schema:
            return errs  # 无 schema 不做约束

        # 基础字段校验：required/type/range/choices
        for block in self._schema.get("blocks", []):
            tk = block.get("key")
            top = self._cfg.get(tk, {})
            if not isinstance(top, dict):
                errs.append(f"顶层 '{tk}' 应为 dict")
                continue
            for field in block.get("fields", []):
                name = field["name"]
                label = f"{tk}.{name}"
                required = field.get("required", False)
                ftype = field.get("type", "str")
                v = top.get(name, field.get("default"))

                if required and (v is None or v == ""):
                    errs.append(f"缺少必填项：{label}")
                    continue

                if v is None or v == "":
                    continue

                # 类型检查
                if ftype == "int" and not isinstance(v, int):
                    errs.append(f"{label} 需要为整数")
                elif ftype == "float" and not isinstance(v, (int, float)):
                    errs.append(f"{label} 需要为浮点数")
                elif ftype == "bool" and not isinstance(v, bool):
                    errs.append(f"{label} 需要为布尔值")
                elif ftype == "str" and not isinstance(v, str):
                    errs.append(f"{label} 需要为字符串")

                # 范围与 choices
                if isinstance(v, (int, float)):
                    vmin = field.get("min", None)
                    vmax = field.get("max", None)
                    if vmin is not None and v < vmin:
                        errs.append(f"{label} 不能小于 {vmin}")
                    if vmax is not None and v > vmax:
                        errs.append(f"{label} 不能大于 {vmax}")
                if "choices" in field:
                    if v not in field["choices"]:
                        errs.append(f"{label} 必须在 {field['choices']} 中")

                # 路径存在性（可选：只做形式检查，不强制本地存在）
                if field.get("widget") == "path" and isinstance(v, str):
                    # 仅简单提示：如果像 '/data' 这种绝对路径可提示“请确认远端存在”
                    if v.strip() == "":
                        errs.append(f"{label} 不能为空路径")

        # 规则校验（互斥/依赖）
        for rule in self._schema.get("rules", []):
            if "if" in rule and "then_required" in rule:
                # e.g. {"if":{"model.arch":"vit"}, "then_required":["model.patch_size"]}
                cond = rule["if"]
                ok = True
                for k, expect in cond.items():
                    gv = self._get_path_value(k)
                    if gv != expect:
                        ok = False; break
                if ok:
                    for req in rule["then_required"]:
                        if self._get_path_value(req) in (None, "", {}):
                            errs.append(f"缺少必填项（条件触发）：{req}")
            if "mutex" in rule:
                paths = rule["mutex"]
                filled = [p for p in paths if self._get_path_value(p) not in (None, "", {})]
                if len(filled) > 1:
                    errs.append(f"以下字段互斥：{paths}")

        return errs

    def _get_path_value(self, path: str):
        # "a.b.c" 形式读取
        cur = self._cfg
        parts = path.split(".")
        for p in parts:
            if not isinstance(cur, dict) or p not in cur:
                return None
            cur = cur[p]
        return cur
