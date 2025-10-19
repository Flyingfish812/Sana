"""Helpers for turning notebook-style visualisations into API-friendly payloads."""
from __future__ import annotations

import base64
import io
from contextlib import redirect_stderr, redirect_stdout
from typing import Any, Dict, List, Tuple

import matplotlib

# Ensure a headless backend is used when the module is imported.
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt  # noqa: E402

from backend.viz.data_inspect import one_click_check


def _capture_figures(before: Tuple[int, ...]) -> List[str]:
    """Return base64-encoded PNG images for figures created after ``before``."""

    figures: List[str] = []
    current = set(plt.get_fignums())
    new_numbers = [num for num in current if num not in before]
    for num in new_numbers:
        fig = plt.figure(num)
        buffer = io.BytesIO()
        fig.savefig(buffer, format="png", bbox_inches="tight")
        buffer.seek(0)
        figures.append(base64.b64encode(buffer.read()).decode("ascii"))
        plt.close(fig)
    return figures


def run_one_click_check(
    dataset: Any,
    dataloaders: Any,
    *,
    channel: Any = None,
    n_batches: int = 2,
    with_sizecheck: bool = True,
) -> Dict[str, Any]:
    """Execute ``one_click_check`` and capture its textual and visual outputs."""

    plt.switch_backend("Agg")
    before = tuple(plt.get_fignums())

    stdout_buffer = io.StringIO()
    stderr_buffer = io.StringIO()

    with redirect_stdout(stdout_buffer), redirect_stderr(stderr_buffer):
        one_click_check(dataset, dataloaders, channel=channel, n_batches=n_batches, with_sizecheck=with_sizecheck)

    stdout = stdout_buffer.getvalue().splitlines()
    stderr = stderr_buffer.getvalue().splitlines()

    figures = _capture_figures(before)

    return {
        "stdout": stdout,
        "stderr": stderr,
        "figures": figures,
    }
