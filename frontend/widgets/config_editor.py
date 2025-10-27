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

class _ListEditor(QtWidgets.QWidget):
    """
    简单列表编辑器：支持 item_type=int/float/str/bool
    schema: {"widget":"list","item_type":"int"|"float"|"str"|"bool"}
    """
    changed = QtCore.Signal(list)

    def __init__(self, values, item_type: str = "str"):
        super().__init__()
        self._item_type = item_type
        self._list: list = list(values or [])
        v = QtWidgets.QVBoxLayout(self); v.setContentsMargins(0,0,0,0)

        self._list_widget = QtWidgets.QListWidget()
        for it in self._list:
            self._list_widget.addItem(str(it))
        v.addWidget(self._list_widget, 1)

        hb = QtWidgets.QHBoxLayout()
        self._inp = QtWidgets.QLineEdit()
        self._btn_add = QtWidgets.QPushButton("添加")
        self._btn_del = QtWidgets.QPushButton("删除选中")
        hb.addWidget(self._inp, 1); hb.addWidget(self._btn_add); hb.addWidget(self._btn_del)
        v.addLayout(hb)

        self._btn_add.clicked.connect(self._on_add)
        self._btn_del.clicked.connect(self._on_del)

    def _cast(self, s: str):
        t = self._item_type
        if t == "int":
            return int(s)
        if t == "float":
            return float(s)
        if t == "bool":
            return s.lower() in ("1","true","yes","y","t")
        return s

    def _on_add(self):
        s = self._inp.text().strip()
        if not s:
            return
        try:
            val = self._cast(s)
        except Exception:
            val = s  # 容错
        self._list.append(val)
        self._list_widget.addItem(str(val))
        self._inp.clear()
        self.changed.emit(list(self._list))

    def _on_del(self):
        for it in self._list_widget.selectedItems():
            idx = self._list_widget.row(it)
            self._list_widget.takeItem(idx)
        # 重建 _list
        self._list = [self._parse_item(self._list_widget.item(i).text()) for i in range(self._list_widget.count())]
        self.changed.emit(list(self._list))

    def _parse_item(self, s: str):
        try:
            return self._cast(s)
        except Exception:
            return s

    def value(self):
        return list(self._list)


class _KVEditor(QtWidgets.QWidget):
    """
    简易字典编辑器：key->scalar
    schema: {"widget":"kv","value_type":"int"|"float"|"str"|"bool"}
    """
    changed = QtCore.Signal(dict)

    def __init__(self, mapping, value_type: str = "str"):
        super().__init__()
        self._value_type = value_type
        self._map: dict = dict(mapping or {})
        v = QtWidgets.QVBoxLayout(self); v.setContentsMargins(0,0,0,0)

        self._table = QtWidgets.QTableWidget(0, 2)
        self._table.setHorizontalHeaderLabels(["Key", "Value"])
        self._table.horizontalHeader().setStretchLastSection(True)
        v.addWidget(self._table, 1)

        for k, val in self._map.items():
            self._append_row(str(k), val)

        hb = QtWidgets.QHBoxLayout()
        self._key = QtWidgets.QLineEdit(); self._key.setPlaceholderText("key")
        self._val = QtWidgets.QLineEdit(); self._val.setPlaceholderText("value")
        self._btn_add = QtWidgets.QPushButton("添加/更新")
        self._btn_del = QtWidgets.QPushButton("删除选中")
        hb.addWidget(self._key); hb.addWidget(self._val, 1); hb.addWidget(self._btn_add); hb.addWidget(self._btn_del)
        v.addLayout(hb)

        self._btn_add.clicked.connect(self._on_add)
        self._btn_del.clicked.connect(self._on_del)

    def _cast(self, s: str):
        t = self._value_type
        if t == "int":
            return int(s)
        if t == "float":
            return float(s)
        if t == "bool":
            return s.lower() in ("1","true","yes","y","t")
        return s

    def _append_row(self, k: str, v):
        r = self._table.rowCount()
        self._table.insertRow(r)
        self._table.setItem(r, 0, QtWidgets.QTableWidgetItem(k))
        self._table.setItem(r, 1, QtWidgets.QTableWidgetItem(str(v)))

    def _on_add(self):
        k = self._key.text().strip()
        vs = self._val.text().strip()
        if not k:
            return
        try:
            v = self._cast(vs)
        except Exception:
            v = vs
        self._map[k] = v
        # 更新/追加行
        found = False
        for r in range(self._table.rowCount()):
            if self._table.item(r, 0).text() == k:
                self._table.item(r, 1).setText(str(v)); found = True; break
        if not found:
            self._append_row(k, v)
        self._key.clear(); self._val.clear()
        self.changed.emit(dict(self._map))

    def _on_del(self):
        rows = sorted({i.row() for i in self._table.selectedIndexes()}, reverse=True)
        for r in rows:
            k = self._table.item(r, 0).text()
            if k in self._map:
                self._map.pop(k, None)
            self._table.removeRow(r)
        self.changed.emit(dict(self._map))

    def value(self):
        # 从表格反读，保证同步
        m = {}
        for r in range(self._table.rowCount()):
            k = self._table.item(r, 0).text()
            vs = self._table.item(r, 1).text()
            try:
                v = self._cast(vs)
            except Exception:
                v = vs
            m[k] = v
        self._map = m
        return dict(self._map)

class DynamicModuleEditor(QtWidgets.QWidget):
    """
    动态模块编辑器：
      value 形如 {"name": "UNetEncoder", "args": {"in_channels":1, ...}}
      registry 形如 {"names":[...], "spec":{"UNetEncoder":{"params":[{"name":..., "type":..., "default":..., "required":...}, ...]}}}
    """
    changed = QtCore.Signal(dict)  # 发出 {"name":..., "args":{...}}

    def __init__(self, kind: str, registry: Dict[str, Any], value: Dict[str, Any] | None):
        super().__init__()
        self._kind = kind
        self._registry = registry or {"names": [], "spec": {}}
        self._value = value or {}
        self._name = self._value.get("name")
        self._args = dict(self._value.get("args") or {})

        v = QtWidgets.QVBoxLayout(self); v.setContentsMargins(0,0,0,0)

        # name 下拉
        hb = QtWidgets.QHBoxLayout()
        hb.addWidget(QtWidgets.QLabel(f"{kind}.name"))
        self._cb = QtWidgets.QComboBox()
        for n in self._registry.get("names", []):
            self._cb.addItem(n, n)
        if self._name:
            idx = self._cb.findData(self._name)
            if idx >= 0:
                self._cb.setCurrentIndex(idx)
        self._cb.currentIndexChanged.connect(self._on_name_changed)
        hb.addWidget(self._cb, 1)
        v.addLayout(hb)

        # 参数区容器
        self._params_box = QtWidgets.QGroupBox("args")
        self._form = QtWidgets.QFormLayout(self._params_box)
        v.addWidget(self._params_box)

        self._render_params()  # 初次

    def _on_name_changed(self, _i: int):
        self._name = self._cb.currentData()
        self._render_params()
        self._emit()

    def _emit(self):
        self.changed.emit({"name": self._name, "args": dict(self._args)})

    def _render_params(self):
        # 清空
        while self._form.count():
            it = self._form.takeAt(0)
            w = it.widget()
            if w:
                w.deleteLater()

        spec = (self._registry.get("spec") or {}).get(self._name or "", {})
        params = spec.get("params", [])

        for p in params:
            pname = p.get("name")
            ptype = (p.get("type") or "str").lower()
            default = p.get("default", None)

            cur = self._args.get(pname, default)

            editor: QtWidgets.QWidget
            if ptype in ("int",):
                editor = QtWidgets.QLineEdit("" if cur is None else str(cur))
            elif ptype in ("float",):
                editor = QtWidgets.QLineEdit("" if cur is None else str(cur))
            elif ptype in ("bool",):
                editor = QtWidgets.QCheckBox()
                editor.setChecked(bool(cur))
            else:
                editor = QtWidgets.QLineEdit("" if cur is None else str(cur))

            def _bind(pn=pname, pt=ptype, ed=editor):
                def on_change():
                    if isinstance(ed, QtWidgets.QCheckBox):
                        self._args[pn] = bool(ed.isChecked())
                    else:
                        txt = cast(QtWidgets.QLineEdit, ed).text().strip()  # type: ignore
                        if pt == "int":
                            try: self._args[pn] = int(txt)
                            except Exception: self._args[pn] = txt
                        elif pt == "float":
                            try: self._args[pn] = float(txt)
                            except Exception: self._args[pn] = txt
                        else:
                            self._args[pn] = txt
                    self._emit()
                return on_change

            # 连接信号
            if isinstance(editor, QtWidgets.QCheckBox):
                editor.stateChanged.connect(lambda _s, f=_bind(): f())
            else:
                cast(QtWidgets.QLineEdit, editor).editingFinished.connect(_bind())  # type: ignore

            self._form.addRow(pname, editor)

    def value(self) -> Dict[str, Any]:
        return {"name": self._name, "args": dict(self._args)}

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

        self._context: Dict[str, Any] = {}

    def set_schema(self, schema: Dict[str, Any]):
        """注入 schema，用于排序、控件选择与校验。"""
        self._schema = schema
        # 重新渲染（若已有 cfg）
        if self._cfg:
            self.load_cfg(self._cfg)

    def set_context(self, ctx: Dict[str, Any]):
        self._context = dict(ctx or {})
        # 有上下文后可触发重渲染（可选）
        if self._cfg and self._schema:
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
                collapsible = bool(block.get("collapsible", False))

                # 外壳：可折叠 or 普通 group
                if collapsible:
                    outer = QtWidgets.QWidget()
                    outer_v = QtWidgets.QVBoxLayout(outer); outer_v.setContentsMargins(0,0,0,0)
                    # header 按钮
                    btn = QtWidgets.QToolButton(text=title, checkable=True, checked=True)
                    btn.setToolButtonStyle(QtCore.Qt.ToolButtonTextBesideIcon)
                    btn.setArrowType(QtCore.Qt.DownArrow)
                    outer_v.addWidget(btn)

                    def _apply_header_style(expanded: bool):
                        if expanded:
                            # 展开：浅灰背景 + 黑字（可读性最佳）
                            btn.setStyleSheet(
                                "QToolButton { font-weight:600; background:#f5f5f5; color:#111; border:1px solid #e0e0e0; padding:6px 8px; }"
                            )
                        else:
                            # 收起：更浅的灰 + 次要色
                            btn.setStyleSheet(
                                "QToolButton { font-weight:600; background:#fafafa; color:#444; border:1px solid #eaeaea; padding:6px 8px; }"
                            )

                    box = QtWidgets.QWidget()
                    vbox = QtWidgets.QVBoxLayout(box); vbox.setContentsMargins(12,6,12,6)
                    def _toggle(ch: bool, ctn=box, b=btn):
                        ctn.setVisible(ch)
                        b.setArrowType(QtCore.Qt.DownArrow if ch else QtCore.Qt.RightArrow)
                        _apply_header_style(ch)
                    btn.toggled.connect(_toggle)
                    _toggle(True)
                    outer_v.addWidget(box)
                    host = outer
                else:
                    box = QtWidgets.QGroupBox(title)
                    vbox = QtWidgets.QVBoxLayout(box); host = box

                top_val = self._cfg.get(top_key, {})
                if not isinstance(top_val, dict):
                    top_val = {}

                grid = QtWidgets.QFormLayout()
                covered_sub_keys = set()

                for field in block.get("fields", []):
                    name = field["name"]
                    ftype = field.get("type", "str")
                    widget_type = field.get("widget", "text")
                    help_text = field.get("help", "")
                    default = field.get("default", None)

                    # 可见性：show_if（path->value 全满足才渲染）
                    show_if = field.get("show_if")
                    if isinstance(show_if, dict):
                        visible = True
                        for path, expect in show_if.items():
                            gv = self._get_path_value(path)
                            if gv != expect:
                                visible = False; break
                        if not visible:
                            continue

                    value = top_val.get(name, default)
                    editor = self._make_field_editor(widget_type, ftype, value, field)

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

                # 未被字段覆盖的键 → YAML 文本区
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

                self._form.insertWidget(self._form.count() - 1, host)
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
                idx = cb.findData(value)
                if idx >= 0:
                    cb.setCurrentIndex(idx)
            return cb

        elif widget_type == "number":
            le = QtWidgets.QLineEdit("" if value is None else str(value))
            le.setPlaceholderText(f"{ftype} in [{field.get('min','-inf')}, {field.get('max','inf')}]")
            return le

        elif widget_type == "path":
            w = QtWidgets.QWidget()
            h = QtWidgets.QHBoxLayout(w); h.setContentsMargins(0,0,0,0)
            le = QtWidgets.QLineEdit("" if value is None else str(value))
            btn = QtWidgets.QPushButton("选择…")
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
            w._lineedit = le   # type: ignore
            return w

        elif widget_type == "checkbox":
            ck = QtWidgets.QCheckBox()
            ck.setChecked(bool(value))
            return ck

        elif widget_type == "list":  # NEW
            item_type = field.get("item_type", "str")
            editor = _ListEditor(value or [], item_type=item_type)
            # 统一包装到 QWidget，方便连接统一的 signal
            w = QtWidgets.QWidget(); l = QtWidgets.QVBoxLayout(w); l.setContentsMargins(0,0,0,0)
            l.addWidget(editor)
            w._list_editor = editor  # type: ignore
            return w

        elif widget_type == "kv":    # NEW
            val_t = field.get("value_type", "str")
            editor = _KVEditor(value or {}, value_type=val_t)
            w = QtWidgets.QWidget(); l = QtWidgets.QVBoxLayout(w); l.setContentsMargins(0,0,0,0)
            l.addWidget(editor)
            w._kv_editor = editor  # type: ignore
            return w
        
        elif widget_type == "module":  # NEW: 四模块动态编辑器
            # 需要上下文里有 model registry
            kind = field.get("kind")
            reg_all = (self._context.get("model_registry") or {})
            registry = reg_all.get(kind) or {"names": [], "spec": {}}
            ed = DynamicModuleEditor(kind, registry, value or {})
            w = QtWidgets.QWidget(); l = QtWidgets.QVBoxLayout(w); l.setContentsMargins(0,0,0,0)
            l.addWidget(ed)
            w._module_editor = ed  # type: ignore
            return w

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

        if widget_type == "list":  # NEW
            ed: _ListEditor = getattr(editor, "_list_editor")  # type: ignore
            return ed.value()

        if widget_type == "kv":    # NEW
            ed: _KVEditor = getattr(editor, "_kv_editor")  # type: ignore
            return ed.value()
        
        if widget_type == "module":  # NEW
            ed: DynamicModuleEditor = getattr(editor, "_module_editor")  # type: ignore
            return ed.value()

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
            getattr(editor, "_lineedit").editingFinished.connect(slot)  # path
        elif isinstance(editor, QtWidgets.QWidget) and hasattr(editor, "_list_editor"):  # NEW
            getattr(editor, "_list_editor").changed.connect(lambda _v: slot())
        elif isinstance(editor, QtWidgets.QWidget) and hasattr(editor, "_kv_editor"):    # NEW
            getattr(editor, "_kv_editor").changed.connect(lambda _v: slot())
        elif isinstance(editor, QtWidgets.QWidget) and hasattr(editor, "_module_editor"):  # NEW
            getattr(editor, "_module_editor").changed.connect(lambda _v: slot())
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
            return errs

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
                widget = field.get("widget", "text")

                # show_if：不显示则不校验
                show_if = field.get("show_if")
                if isinstance(show_if, dict):
                    visible = True
                    for path, expect in show_if.items():
                        gv = self._get_path_value(path)
                        if gv != expect:
                            visible = False; break
                    if not visible:
                        continue

                v = top.get(name, field.get("default"))

                if required and (v is None or v == "" or v == [] or v == {}):
                    errs.append(f"缺少必填项：{label}")
                    continue

                if v is None or v == "" or v == [] or v == {}:
                    continue

                # 基础类型
                if widget == "list":
                    if not isinstance(v, list):
                        errs.append(f"{label} 需要为 list")
                    else:
                        item_type = field.get("item_type", "str")
                        for i, it in enumerate(v):
                            if item_type == "int" and not isinstance(it, int):
                                errs.append(f"{label}[{i}] 需要为整数")
                            if item_type == "float" and not isinstance(it, (int, float)):
                                errs.append(f"{label}[{i}] 需要为浮点数")
                            if item_type == "bool" and not isinstance(it, bool):
                                # 允许 0/1? 如需放宽可在编辑器层处理
                                errs.append(f"{label}[{i}] 需要为布尔值")
                            if item_type == "str" and not isinstance(it, str):
                                errs.append(f"{label}[{i}] 需要为字符串")

                elif widget == "kv":
                    if not isinstance(v, dict):
                        errs.append(f"{label} 需要为 dict")
                    else:
                        val_t = field.get("value_type", "str")
                        for k, it in v.items():
                            if val_t == "int" and not isinstance(it, int):
                                errs.append(f"{label}['{k}'] 需要为整数")
                            if val_t == "float" and not isinstance(it, (int, float)):
                                errs.append(f"{label}['{k}'] 需要为浮点数")
                            if val_t == "bool" and not isinstance(it, bool):
                                errs.append(f"{label}['{k}'] 需要为布尔值")
                            if val_t == "str" and not isinstance(it, str):
                                errs.append(f"{label}['{k}'] 需要为字符串")

                else:
                    # 旧有的基础 ftype 校验
                    if ftype == "int" and not isinstance(v, int):
                        errs.append(f"{label} 需要为整数")
                    elif ftype == "float" and not isinstance(v, (int, float)):
                        errs.append(f"{label} 需要为浮点数")
                    elif ftype == "bool" and not isinstance(v, bool):
                        errs.append(f"{label} 需要为布尔值")
                    elif ftype == "str" and not isinstance(v, str):
                        errs.append(f"{label} 需要为字符串")

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

                if field.get("widget") == "path" and isinstance(v, str):
                    if v.strip() == "":
                        errs.append(f"{label} 不能为空路径")

        # 规则：依赖 / 互斥
        for rule in self._schema.get("rules", []):
            if "if" in rule and "then_required" in rule:
                cond = rule["if"]; ok = True
                for k, expect in cond.items():
                    gv = self._get_path_value(k)
                    if gv != expect:
                        ok = False; break
                if ok:
                    for req in rule["then_required"]:
                        if self._get_path_value(req) in (None, "", [], {}):
                            errs.append(f"缺少必填项（条件触发）：{req}")
            if "mutex" in rule:
                paths = rule["mutex"]
                filled = [p for p in paths if self._get_path_value(p) not in (None, "", [], {})]
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
