# backend/eval/__init__.py
from .runtime import evaluate, render_eval_triplets, ensure_eval_multiscale_vis
from .report import run_report, append_multiscale_section
from .pdfprint import export_report_pdf

__all__ = ["evaluate", 
           "render_eval_triplets", 
           "ensure_eval_multiscale_vis", 
           "run_report", 
           "append_multiscale_section", 
           "export_report_pdf",
]
