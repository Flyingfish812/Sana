# backend/eval/__init__.py
from .runtime import evaluate, render_eval_triplets
from .report import run_report

__all__ = ["evaluate", "render_eval_triplets", "run_report"]
