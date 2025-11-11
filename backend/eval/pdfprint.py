import subprocess, shutil, tempfile, textwrap
from pathlib import Path
from typing import Optional

def export_report_pdf(
    run_dir: Path,
    *,
    prefer: str = "pandoc",          # "pandoc" | "chromium" | "auto"
    pdf_name: str = "report.pdf",
) -> Optional[Path]:
    """
    将 run_dir/report.md 导出为 PDF。
    优先方案：
      1) Pandoc (质量最好，需要系统安装 pandoc，与 latex 引擎)
      2) Headless Chromium (将 md->html，再 print-to-pdf，免 LaTeX)
    返回生成的 PDF 路径；失败返回 None。
    """
    run_dir = Path(run_dir)
    md_path = run_dir / "report.md"
    if not md_path.exists():
        return None

    pdf_path = run_dir / pdf_name
    mode = prefer.lower()
    if mode == "auto":
        mode = "pandoc" if shutil.which("pandoc") else "chromium"

    try:
        if mode == "pandoc":
            ok = _export_md_pdf_pandoc(md_path, pdf_path)
            if ok:
                return pdf_path
            # 兜底到 chromium
            if shutil.which("chromium") or shutil.which("google-chrome") or shutil.which("chrome"):
                ok = _export_md_pdf_chromium(md_path, pdf_path)
                return pdf_path if ok else None
            return None
        elif mode == "chromium":
            ok = _export_md_pdf_chromium(md_path, pdf_path)
            return pdf_path if ok else None
        else:
            # 未知模式：尝试 auto
            return export_report_pdf(run_dir, prefer="auto", pdf_name=pdf_name)
    except Exception as e:
        print(f"[export_report_pdf] 导出 PDF 失败：{e}")
        return None

def _export_md_pdf_pandoc(md_path: Path, pdf_path: Path) -> bool:
    """
    Pandoc → PDF（tectonic / xelatex）
    - 启用表格、数学、链接属性
    - 资源路径：--resource-path 确保相对路径图片可加载
    - 字体：优先使用 <run_dir>/fonts/HarmonyOS_Sans_SC 下的 ttf（项目内字体），找不到则回退系统字体
    """
    import shutil, subprocess

    if shutil.which("pandoc") is None:
        return False

    # 选引擎：优先 tectonic，其次 xelatex
    pdf_engine = None
    if shutil.which("tectonic"):
        pdf_engine = "tectonic"
    elif shutil.which("xelatex"):
        pdf_engine = "xelatex"

    md_dir = md_path.parent

    # 资源路径（让相对图片能找到）
    resource_candidates = [md_path, md_dir]
    resource_dirs = [str(p) for p in resource_candidates if p.exists()]
    resource_arg = ":".join(resource_dirs) if resource_dirs else str(md_dir)

    # ============ 项目内 HarmonyOS 字体优先 ============
    fonts_dir = Path("~/projects/Sana/fonts").expanduser().resolve()
    family_sc_dir = fonts_dir / "HarmonyOS-Sans-SC"      # 中文/正文首选
    family_base_dir = fonts_dir / "HarmonyOS-Sans"       # 非 CJK 正文备用
    family_mono_dir = fonts_dir / "HarmonyOS-Sans"       # 无等宽家族时，临时共用

    have_sc = (family_sc_dir.exists() and any(family_sc_dir.glob("*.ttf")))
    have_base = (family_base_dir.exists() and any(family_base_dir.glob("*.ttf")))
    have_mono = (family_mono_dir.exists() and any(family_mono_dir.glob("*.ttf")))

    # 以“文件名模板”方式告诉 fontspec 查找变体：
    # UprightFont=*-Regular.ttf, BoldFont=*-Bold.ttf, ItalicFont=*-Italic.ttf
    sc_opts = [
        "Path=fonts/HarmonyOS-Sans-SC/",
        "UprightFont=*-Regular.ttf",
        "BoldFont=*-Bold.ttf",
        "ItalicFont=*-Regular.ttf",
        "AutoFakeBold=false",
        "AutoFakeSlant=true"
    ]
    base_opts = [
        "Path=fonts/HarmonyOS-Sans/",
        "UprightFont=*-Regular.ttf",
        "BoldFont=*-Bold.ttf",
        "ItalicFont=*Regular-Italic.ttf",
        "AutoFakeBold=false",
        "AutoFakeSlant=true"
    ]
    mono_opts = base_opts  # 临时共用，无等宽家族时也能正常工作

    # 4) 写一个临时 header.tex 注入 LaTeX 设置
    header_tex = textwrap.dedent(rf"""
    % ---------- auto header injected by export_report_pdf ----------
    \usepackage{{graphicx}}
    % 图片搜索路径（相对 report.md 所在目录）
    \graphicspath{{{{./}}{{./eval_vis/}}{{./eval_vis_ms/}}}}
    % 默认图像宽度不要超过列宽，保持纵横比
    \setkeys{{Gin}}{{width=\linewidth, keepaspectratio}}

    % 文段/超长单词/URL 的断行优化，缓解 Overfull \hbox
    \usepackage{{microtype}}
    \usepackage[hyphens]{{url}}
    \usepackage{{hyperref}}
    \usepackage{{ragged2e}}
    \setlength{{\emergencystretch}}{{3em}}
    \sloppy
    \Urlmuskip=0mu plus 1mu

    % Pandoc 常用表格环境：longtable 边距微调（可按需调整）
    \usepackage{{longtable}}
    \setlength\LTleft{{0pt}}
    \setlength\LTright{{0pt}}
    \setlength{{\tabcolsep}}{{5.5pt}}
    \renewcommand{{\arraystretch}}{{1.08}}
    % --------------------------------------------------------------
    """)

    # 组装 pandoc 命令
    with tempfile.TemporaryDirectory() as td:
        header_path = Path(td) / "header.tex"
        header_path.write_text(header_tex, encoding="utf-8")
        cmd = [
            "pandoc",
            str(md_path),
            "-o", str(pdf_path),
            "--from", "markdown+pipe_tables+grid_tables+table_captions+tex_math_dollars+link_attributes",
            "--toc",
            "--resource-path", resource_arg,
            "--include-in-header", str(header_path),
            "-V", "geometry:margin=16mm",
        ]
        if pdf_engine:
            cmd += ["--pdf-engine", pdf_engine]

        # 设置字体变量（仅当对应目录存在时才传，避免 fontspec 报错）
        # 主中文 + 正文字体：用 HarmonyOS_Sans_SC
        if have_sc:
            # CJKmainfont 供 xeCJK；mainfont 供 fontspec 正文；两者都指向 SC 家族以统一风格
            cmd += ["-V", 'CJKmainfont=HarmonyOS-Sans-SC']
            cmd += ["-V", f"CJKoptions={','.join(sc_opts)}"]
            cmd += ["-V", 'mainfont=HarmonyOS-Sans-SC']
            cmd += ["-V", f"mainfontoptions={','.join(sc_opts)}"]
        elif have_base:
            # 没有 SC 的话，退化到非 SC 家族
            cmd += ["-V", 'mainfont=HarmonyOS-Sans']
            cmd += ["-V", f"mainfontoptions={','.join(base_opts)}"]

        # 等宽字体：尽力从项目内找；没有就不传，留给模板默认
        if have_mono:
            cmd += ["-V", 'monofont=HarmonyOS-Sans']
            cmd += ["-V", f"monofontoptions={','.join(mono_opts)}"]

        try:
            subprocess.run(cmd, check=True)
            return pdf_path.exists()
        except subprocess.CalledProcessError:
            return False

def _export_md_pdf_chromium(md_path: Path, pdf_path: Path) -> bool:
    """
    无 Pandoc 时的兜底：将 Markdown 渲染为 HTML，再用无头 Chromium 打印 PDF。
    修复点：
      - <base href> 指向 md 所在目录，确保相对路径图片可用
      - CSS 指定包含中文的字体族
    """
    import shutil, subprocess, tempfile, textwrap

    browser = shutil.which("chromium") or shutil.which("google-chrome") or shutil.which("chrome")
    if browser is None:
        return False

    md_dir = md_path.parent

    # 1) 渲染 Markdown → HTML
    try:
        import markdown as _md
        text = md_path.read_text(encoding="utf-8")
        html_body = _md.markdown(
            text,
            extensions=["fenced_code", "tables", "toc", "codehilite", "sane_lists", "attr_list"]
        )
    except Exception:
        html_body = "<pre>" + md_path.read_text(encoding="utf-8").replace("<","&lt;").replace(">","&gt;") + "</pre>"

    # 2) CSS：包含 CJK 字体
    css = textwrap.dedent("""
    <style>
      body {
        font-family: "Noto Sans CJK SC","Microsoft YaHei","PingFang SC","Source Han Sans SC","WenQuanYi Micro Hei",
                     system-ui,-apple-system,Segoe UI,Roboto,Arial,"Apple Color Emoji","Segoe UI Emoji";
        margin: 24px; color: #111;
      }
      h1,h2,h3,h4 { font-weight: 600; }
      h1 { font-size: 1.8rem; margin-top: 1.2em; }
      h2 { font-size: 1.5rem; margin-top: 1.2em; }
      h3 { font-size: 1.25rem; margin-top: 1.0em; }
      table { border-collapse: collapse; margin: 1em 0; width: 100%; }
      th, td { border: 1px solid #ddd; padding: 6px 8px; }
      blockquote { border-left: 4px solid #ddd; padding: 0.2em 1em; color: #555; }
      code, pre { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, "Liberation Mono", monospace; }
      pre { background: #f6f8fa; padding: 12px; overflow: auto; }
      img { max-width: 100%; }
      .pagebreak { page-break-after: always; }
      @page { margin: 16mm; }
    </style>
    """)

    # 关键：<base href> 让相对链接/图片以 md 所在目录为基准解析
    base_tag = f"<base href='{md_dir.resolve().as_uri()}/'>"

    html = f"<!DOCTYPE html><html><head><meta charset='utf-8'>{base_tag}{css}</head><body>{html_body}</body></html>"

    # 3) 落地临时 HTML，再用 Chromium 打印为 PDF
    with tempfile.TemporaryDirectory() as td:
        html_path = Path(td) / "report.html"
        html_path.write_text(html, encoding="utf-8")
        cmd = [
            browser, "--headless",
            f"--print-to-pdf={str(pdf_path)}",
            "--disable-gpu",
            "--no-sandbox",
            str(html_path.resolve().as_uri()),
        ]
        try:
            subprocess.run(cmd, check=True)
            return pdf_path.exists()
        except subprocess.CalledProcessError:
            return False
