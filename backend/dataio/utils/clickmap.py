# backend/dataio/tools/clickmap_cli.py
from __future__ import annotations
import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple, Union

import contextlib
import io
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.widgets import Button

# --------- 后端选择：优先可弹窗的 GUI 后端 ----------
def _ensure_gui_backend():
    be = (matplotlib.get_backend() or "").lower()
    if any(k in be for k in ("tkagg", "qt5agg", "qtagg")):
        return
    for gui in ("TkAgg", "Qt5Agg", "QtAgg"):
        with contextlib.suppress(Exception):
            plt.switch_backend(gui)
            return
        with contextlib.suppress(Exception):
            matplotlib.use(gui, force=True)
            return
    # 如果真的切不了，后面 show() 可能没法交互；提示用户
    print("[clickmap] Warning: failed to switch to a GUI backend. "
          "If the window does not accept clicks, try running locally with TkAgg/QtAgg.")

# --------- 背景数据加载 ----------
def _load_background(path: str, channel: Optional[int] = None) -> np.ndarray:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Background not found: {path}")
    if p.suffix.lower() in [".npy"]:
        arr = np.load(p)
    elif p.suffix.lower() in [".npz"]:
        data = np.load(p)
        # 取第一个数组
        key = list(data.keys())[0]
        arr = data[key]
    elif p.suffix.lower() in [".pt", ".pth"]:
        try:
            import torch
        except Exception as e:
            raise RuntimeError("Loading .pt/.pth requires torch installed.") from e
        arr = torch.load(p, map_location="cpu")
        if hasattr(arr, "numpy"):
            arr = arr.numpy()
    elif p.suffix.lower() in [".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"]:
        import imageio.v2 as imageio
        arr = imageio.imread(p)
    else:
        raise ValueError("Unsupported background file. Use .npy/.npz/.pt/.png/.jpg/.tif")
    arr = np.asarray(arr)

    # 选通道：若是 [H,W,C]，默认取第 0 通道
    if arr.ndim == 3 and arr.shape[-1] > 1:
        c = 0 if channel is None else int(channel)
        if c < 0 or c >= arr.shape[-1]:
            raise ValueError(f"channel out of range: got {c}, shape={arr.shape}")
        arr = arr[..., c]
    # 若是 [C,H,W]，取指定通道
    if arr.ndim == 3 and arr.shape[0] > 1 and (arr.shape[-1] != 3):
        c = 0 if channel is None else int(channel)
        if c < 0 or c >= arr.shape[0]:
            raise ValueError(f"channel out of range: got {c}, shape={arr.shape}")
        arr = arr[c]
    if arr.ndim != 2:
        # 尽力转灰度
        arr = np.mean(arr, axis=-1) if arr.ndim == 3 else np.squeeze(arr)
    arr = np.asarray(arr, dtype=float)
    return arr

# --------- 保存 ----------
def _save_points(out_path: str, H: int, W: int, pts: List[Tuple[int, int]]):
    if out_path.lower().endswith(".json"):
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump({"points": [[int(y), int(x)] for y, x in pts], "H": H, "W": W}, f, ensure_ascii=False, indent=2)
    elif out_path.lower().endswith(".npy"):
        mask = np.zeros((H, W), dtype=np.float32)
        for y, x in pts:
            if 0 <= y < H and 0 <= x < W:
                mask[y, x] = 1.0
        np.save(out_path, mask)
    else:
        raise ValueError("out_path must be .json or .npy")

@dataclass
class _State:
    H: int
    W: int
    pts: List[Tuple[int, int]]
    max_points: Optional[int] = None

# --------- 主交互 ----------
def run_click_gui(
    out_path: str,
    *,
    H: Optional[int],
    W: Optional[int],
    bg_path: Optional[str],
    bg_channel: Optional[int],
    vmax: Optional[float],
    vmin: Optional[float],
    max_points: Optional[int],
) -> dict:
    _ensure_gui_backend()

    if bg_path:
        bg = _load_background(bg_path, channel=bg_channel)
        H0, W0 = bg.shape
        H = H0; W = W0
    else:
        if H is None or W is None:
            raise ValueError("Without background, you must provide --hw H W.")
        H = int(H); W = int(W)
        bg = np.zeros((H, W), dtype=float)

    # 归一化显示范围
    show_min = np.nanmin(bg) if vmin is None else vmin
    show_max = np.nanmax(bg) if vmax is None else vmax
    if not np.isfinite(show_min): show_min = 0.0
    if not np.isfinite(show_max) or show_max == show_min: show_max = show_min + 1.0

    st = _State(H=H, W=W, pts=[], max_points=max_points)

    fig, ax = plt.subplots(figsize=(min(10, W/64), min(7, H/64)))
    plt.subplots_adjust(bottom=0.18)  # 预留按钮区域

    im = ax.imshow(bg, origin="upper", vmin=show_min, vmax=show_max)
    scat = ax.scatter([], [], s=18, c="r")
    ax.set_title("Left-click to add points • Right-click to undo nearest • Middle-click to finish")
    ax.set_xlim([-0.5, W-0.5]); ax.set_ylim([H-0.5, -0.5])

    # 按钮
    ax_finish = plt.axes([0.10, 0.05, 0.18, 0.08])
    ax_undo   = plt.axes([0.32, 0.05, 0.18, 0.08])
    ax_clear  = plt.axes([0.54, 0.05, 0.18, 0.08])
    btn_finish = Button(ax_finish, "Finish & Save [Enter]")
    btn_undo   = Button(ax_undo, "Undo [Backspace]")
    btn_clear  = Button(ax_clear, "Clear [C]")

    def _redraw():
        if st.pts:
            ys = [p[0] for p in st.pts]; xs = [p[1] for p in st.pts]
        else:
            ys = []; xs = []
        scat.set_offsets(np.array(list(zip(xs, ys))) if xs else np.zeros((0,2)))
        fig.canvas.draw_idle()
        ax.set_xlabel(f"points: {len(st.pts)}")

    def _finish_and_save(_evt=None):
        _save_points(out_path, st.H, st.W, st.pts)
        print(f"[clickmap] saved to {out_path} (points={len(st.pts)})")
        plt.close(fig)

    def _undo(_evt=None):
        if st.pts:
            st.pts.pop()
            _redraw()

    def _clear(_evt=None):
        st.pts.clear()
        _redraw()

    def _nearest_idx(xi: int, yi: int) -> Optional[int]:
        if not st.pts:
            return None
        arr = np.array(st.pts, dtype=float)  # [N,2] => (y,x)
        d = (arr[:,0] - yi)**2 + (arr[:,1] - xi)**2
        return int(np.argmin(d))

    # 事件：鼠标点击
    def onclick(event):
        if event.inaxes != ax:
            return
        if event.xdata is None or event.ydata is None:
            return
        xi, yi = int(round(event.xdata)), int(round(event.ydata))
        if xi < 0 or xi >= st.W or yi < 0 or yi >= st.H:
            return
        # 左键：添加
        if event.button == 1:
            st.pts.append((yi, xi))
            _redraw()
            if st.max_points is not None and len(st.pts) >= st.max_points:
                _finish_and_save()
        # 右键：删除最近
        elif event.button == 3:
            idx = _nearest_idx(xi, yi)
            if idx is not None:
                st.pts.pop(idx)
                _redraw()
        # 中键：结束保存
        elif event.button == 2:
            _finish_and_save()

    # 键盘：Enter 保存，Backspace 撤销，C 清空，Q 退出不保存
    def onkey(event):
        key = (event.key or "").lower()
        if key in ("enter", "return"):
            _finish_and_save()
        elif key in ("backspace",):
            _undo()
        elif key in ("c",):
            _clear()
        elif key in ("q",):
            print("[clickmap] quit without saving.")
            plt.close(fig)

    # 绑定
    fig.canvas.mpl_connect("button_press_event", onclick)
    fig.canvas.mpl_connect("key_press_event", onkey)
    btn_finish.on_clicked(_finish_and_save)
    btn_undo.on_clicked(_undo)
    btn_clear.on_clicked(_clear)

    _redraw()
    plt.show()

    # show 结束后返回结果（若未保存就返回 count=0）
    return {"count": len(st.pts), "path": str(out_path) if st.pts else ""}

# --------- CLI ---------
def main():
    ap = argparse.ArgumentParser(description="Interactive click map (console GUI).")
    ap.add_argument("--out", required=True, help="Output path (.json or .npy).")
    ap.add_argument("--hw", nargs=2, type=int, metavar=("H","W"), help="Canvas size when no background.")
    ap.add_argument("--background", type=str, help="Background array/image: .npy/.npz/.pt/.png/.jpg/.tif ...")
    ap.add_argument("--bg-channel", type=int, default=None, help="Channel index if background has channels.")
    ap.add_argument("--vmin", type=float, default=None, help="Display min for background.")
    ap.add_argument("--vmax", type=float, default=None, help="Display max for background.")
    ap.add_argument("--points", type=int, default=None, help="Auto-finish after collecting this many points.")
    args = ap.parse_args()

    H = W = None
    if args.background is None:
        if args.hw is None:
            ap.error("When --background is not set, you must provide --hw H W.")
        H, W = args.hw

    info = run_click_gui(
        out_path=args.out,
        H=H, W=W,
        bg_path=args.background,
        bg_channel=args.bg_channel,
        vmax=args.vmax, vmin=args.vmin,
        max_points=args.points,
    )
    # 额外打印，便于 VS Code 终端查看
    print(json.dumps(info, ensure_ascii=False))

if __name__ == "__main__":
    main()
