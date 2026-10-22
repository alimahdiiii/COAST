"""
Assembles the post-processing PDF report of a case.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np

# reportlab
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm, mm
from reportlab.platypus import (
    BaseDocTemplate,
    Flowable,
    Frame,
    HRFlowable,
    Image,
    KeepTogether,
    NextPageTemplate,
    PageBreak,
    PageTemplate,
    Paragraph,
    Preformatted,
    Spacer,
    Table,
    TableStyle,
)
from reportlab.platypus.tableofcontents import TableOfContents

# Palette
_DARK_BLUE  = colors.HexColor("#0E2A45")
_MID_BLUE   = colors.HexColor("#1A5276")
_ACCENT     = colors.HexColor("#2E86C1")
_LIGHT_BLUE = colors.HexColor("#D6EAF8")
_LIGHT_GREY = colors.HexColor("#F4F6F9")
_MID_GREY   = colors.HexColor("#BDC3C7")
_TEXT       = colors.HexColor("#1C2833")
_WHITE      = colors.white

_PAGE_W, _PAGE_H = A4
_MARGIN  = 2.0 * cm
_FOOTER_H = 1.1 * cm

# Figure counter (reset per report)
_fig_list: list[str] = []   # populated during story build: list of captions


def _fig_num(caption: str) -> str:
    """Register a figure and return its label, e.g. 'Figure 3'."""
    n = len(_fig_list) + 1
    _fig_list.append(f"Figure {n}  —  {caption}")
    return f"Figure {n}"


# Styles

def _make_styles() -> dict:
    base = getSampleStyleSheet()

    def _ps(name, parent_key, **kw):
        return ParagraphStyle(name, parent=base[parent_key], **kw)

    title = _ps("ReportTitle", "Title",
                fontSize=28, leading=34, textColor=_DARK_BLUE,
                alignment=TA_CENTER, spaceAfter=6, fontName="Helvetica-Bold")

    subtitle = _ps("ReportSubtitle", "Normal",
                   fontSize=14, leading=17, textColor=_MID_BLUE,
                   alignment=TA_CENTER, spaceAfter=4, fontName="Helvetica")

    toolkit = _ps("ToolkitName", "Normal",
                  fontSize=11, leading=14, textColor=_ACCENT,
                  alignment=TA_CENTER, spaceAfter=8, fontName="Helvetica-Oblique")

    h1 = _ps("H1", "Heading1",
             fontSize=14, leading=18, textColor=_DARK_BLUE,
             fontName="Helvetica-Bold", spaceBefore=16, spaceAfter=4)

    h2 = _ps("H2", "Heading2",
             fontSize=11, leading=14, textColor=_MID_BLUE,
             fontName="Helvetica-Bold", spaceBefore=10, spaceAfter=3)

    body = _ps("BodyText", "Normal",
               fontSize=9, leading=12, textColor=_TEXT,
               fontName="Helvetica", spaceAfter=4)

    intro = _ps("IntroText", "Normal",
                fontSize=9, leading=13, textColor=colors.HexColor("#3D566E"),
                fontName="Helvetica-Oblique", spaceAfter=6)

    caption = _ps("FigCaption", "Normal",
                  fontSize=8, leading=10, textColor=colors.HexColor("#555555"),
                  fontName="Helvetica-Oblique", alignment=TA_CENTER, spaceAfter=6)

    meta = _ps("MetaText", "Normal",
               fontSize=9, leading=12, textColor=colors.HexColor("#555555"),
               fontName="Helvetica", alignment=TA_CENTER, spaceAfter=3)

    contact = _ps("ContactText", "Normal",
                  fontSize=8, leading=11, textColor=colors.HexColor("#444444"),
                  fontName="Helvetica", spaceAfter=2)

    # TOC levels
    toc0 = _ps("TOC0", "Normal",
               fontSize=10, leading=13, textColor=_DARK_BLUE,
               fontName="Helvetica-Bold", spaceBefore=4, spaceAfter=2,
               leftIndent=0)
    toc1 = _ps("TOC1", "Normal",
               fontSize=9, leading=12, textColor=_MID_BLUE,
               fontName="Helvetica", spaceBefore=1, spaceAfter=1,
               leftIndent=12)

    lof = _ps("LOFEntry", "Normal",
              fontSize=9, leading=12, textColor=_TEXT,
              fontName="Helvetica", spaceBefore=1, spaceAfter=1, leftIndent=6)

    return dict(title=title, subtitle=subtitle, toolkit=toolkit,
                h1=h1, h2=h2, body=body, intro=intro, caption=caption,
                meta=meta, contact=contact, toc0=toc0, toc1=toc1, lof=lof)


# Document template with TOC support and hyperlinks

class _CoastDocTemplate(BaseDocTemplate):
    """BaseDocTemplate that feeds headings into the TOC with clickable anchors."""

    def afterFlowable(self, flowable):
        if not isinstance(flowable, Paragraph):
            return
        sname = flowable.style.name
        if sname not in ("H1", "H2"):
            return
        text = flowable.getPlainText()
        # Stable bookmark key derived from heading text
        key = "h_" + "".join(c if c.isalnum() else "_" for c in text.lower())[:50]
        self.canv.bookmarkPage(key)
        level = 0 if sname == "H1" else 1
        self.notify("TOCEntry", (level, text, self.page, key))


# Rotated flowable

class _RotatedTable(Flowable):
    """Draws a Table rotated 90° CCW within the portrait page.

    Build the inner Table at *tbl_w* ≈ page body height so columns are wide
    enough; after rotation the table's height becomes the portrait page width.
    """

    def __init__(self, tbl: Table, tbl_w: float, tbl_h: float):
        Flowable.__init__(self)
        self._tbl   = tbl
        self._tbl_w = tbl_w   # natural width  (landscape-like, wide)
        self._tbl_h = tbl_h   # natural height (short → becomes portrait width)

    def wrap(self, availW, availH):
        # Rotated 90° CCW: tbl_h becomes width, tbl_w becomes height in the flow
        return self._tbl_h, self._tbl_w

    def draw(self):
        c = self.canv
        c.saveState()
        # translate(tbl_h, 0) + rotate(90° CCW) maps natural (0,0) to page (tbl_h, 0)
        c.translate(self._tbl_h, 0)
        c.rotate(90)
        self._tbl.drawOn(c, 0, 0)
        c.restoreState()


# Image helpers

def _logo(path, max_w: float, max_h: float) -> Image | None:
    p = Path(path) if path else None
    if not p or not p.exists():
        return None
    try:
        img = Image(str(p))
        scale = min(max_w / img.imageWidth, max_h / img.imageHeight, 1.0)
        img.drawWidth  = img.imageWidth  * scale
        img.drawHeight = img.imageHeight * scale
        return img
    except Exception:
        return None


def _img_obj(path: Path, max_w: float, max_h: float) -> Image | None:
    """Return a scaled Image object without caption/spacer (used for grid layouts)."""
    if not path.exists():
        return None
    try:
        img = Image(str(path))
        scale = min(max_w / img.imageWidth, max_h / img.imageHeight, 1.0)
        img.drawWidth  = img.imageWidth  * scale
        img.drawHeight = img.imageHeight * scale
        img.hAlign = "CENTER"
        return img
    except Exception:
        return None


def _section_figure(img_path: Path, caption: str, styles: dict,
                    max_w: float | None = None) -> list:
    """Return [Image, numbered caption Paragraph, Spacer] if the image exists."""
    if not img_path.exists():
        return []
    try:
        avail = max_w or (_PAGE_W - 2 * _MARGIN)
        img = Image(str(img_path))
        scale = min(avail / img.imageWidth, (9 * cm) / img.imageHeight, 1.0)
        img.drawWidth  = img.imageWidth  * scale
        img.drawHeight = img.imageHeight * scale
        img.hAlign = "CENTER"
        label = _fig_num(caption)
        return [img, Paragraph(f"<i>{label}  —  {caption}</i>", styles["caption"]),
                Spacer(1, 4)]
    except Exception:
        return []


# Value formatter

def _fmt(val, decimals: int = 4) -> str:
    if val is None:
        return "—"
    try:
        f = float(val)
        return "—" if (f != f) else f"{f:.{decimals}f}"   # NaN check
    except (TypeError, ValueError):
        return str(val)


# Table builders

def _param_table(rows: list, col_widths: list | None = None) -> Table:
    """Two-or-three column parameter table (key / value / unit)."""
    style = TableStyle([
        ("BACKGROUND",      (0, 0), (0, -1),   _LIGHT_GREY),
        ("BACKGROUND",      (0, 0), (-1, 0),   _MID_BLUE),
        ("TEXTCOLOR",       (0, 0), (-1, 0),   _WHITE),
        ("TEXTCOLOR",       (0, 1), (-1, -1),  _TEXT),
        ("FONTNAME",        (0, 0), (-1, 0),   "Helvetica-Bold"),
        ("FONTNAME",        (0, 1), (0, -1),   "Helvetica-Bold"),
        ("FONTNAME",        (1, 1), (-1, -1),  "Helvetica"),
        ("FONTSIZE",        (0, 0), (-1, -1),  8),
        ("GRID",            (0, 0), (-1, -1),  0.4, _MID_GREY),
        ("ROWBACKGROUNDS",  (0, 1), (-1, -1),  [_WHITE, _LIGHT_GREY]),
        ("LEFTPADDING",     (0, 0), (-1, -1),  6),
        ("RIGHTPADDING",    (0, 0), (-1, -1),  6),
        ("TOPPADDING",      (0, 0), (-1, -1),  3),
        ("BOTTOMPADDING",   (0, 0), (-1, -1),  3),
    ])
    data = [[str(c) for c in row] for row in rows]
    avail = _PAGE_W - 2 * _MARGIN
    if col_widths is None:
        n = max(len(r) for r in data) if data else 2
        w0 = avail * 0.45
        rest = (avail - w0) / max(1, n - 1)
        col_widths = [w0] + [rest] * (n - 1)
    return Table(data, colWidths=col_widths, style=style, repeatRows=1)


def _df_table(df, styles: dict, max_cols_per_chunk: int = 7,
              avail_w: float | None = None) -> list:
    """
    Convert a pandas DataFrame to one or more reportlab Tables.
    Wide DataFrames (> max_cols_per_chunk) are split into column chunks,
    with the first column repeated in each chunk.
    avail_w: available width in points (default: portrait body width).
    """
    if df is None or df.empty:
        return []

    cols = list(df.columns)
    n    = len(cols)
    avail = avail_w if avail_w is not None else (_PAGE_W - 2 * _MARGIN)

    def _cell(v) -> str:
        try:
            f = float(v)
            return "—" if (f != f) else f"{f:.4g}"
        except (TypeError, ValueError):
            return str(v) if v is not None else "—"

    # If fits, render as single table
    if n <= max_cols_per_chunk:
        col_w = avail / n
        tbl_style = _wide_style(n, font_size=7.5)
        rows = [cols] + [[_cell(r[c]) for c in cols] for _, r in df.iterrows()]
        return [Table(rows, colWidths=[col_w] * n, style=tbl_style, repeatRows=1),
                Spacer(1, 4)]

    # Split into chunks: first column is repeated as index column
    idx_col   = cols[0]
    data_cols = cols[1:]
    chunk_extra = max_cols_per_chunk - 1   # number of data cols per chunk

    result = []
    for start in range(0, len(data_cols), chunk_extra):
        chunk = [idx_col] + data_cols[start: start + chunk_extra]
        nc = len(chunk)
        font_size = max(5.5, 7.5 - 0.3 * max(0, nc - 5))
        idx_w  = avail * 0.28
        rest_w = (avail - idx_w) / max(1, nc - 1)
        col_ws = [idx_w] + [rest_w] * (nc - 1)
        tbl_style = _wide_style(nc, font_size=font_size)
        rows = [chunk] + [[_cell(r[c]) for c in chunk] for _, r in df.iterrows()]
        tbl = Table(rows, colWidths=col_ws, style=tbl_style, repeatRows=1)
        result.append(tbl)
        result.append(Spacer(1, 3))
    return result


def _wide_style(n_cols: int, font_size: float = 7.5) -> TableStyle:
    padding = max(2, int(4 - (n_cols - 5) * 0.3))
    return TableStyle([
        ("BACKGROUND",      (0, 0), (-1, 0),   _MID_BLUE),
        ("TEXTCOLOR",       (0, 0), (-1, 0),   _WHITE),
        ("FONTNAME",        (0, 0), (-1, 0),   "Helvetica-Bold"),
        ("FONTNAME",        (0, 1), (-1, -1),  "Helvetica"),
        ("FONTSIZE",        (0, 0), (-1, -1),  font_size),
        ("GRID",            (0, 0), (-1, -1),  0.35, _MID_GREY),
        ("ROWBACKGROUNDS",  (0, 1), (-1, -1),  [_WHITE, _LIGHT_GREY]),
        ("ALIGN",           (1, 0), (-1, -1),  "CENTER"),
        ("LEFTPADDING",     (0, 0), (-1, -1),  padding),
        ("RIGHTPADDING",    (0, 0), (-1, -1),  padding),
        ("TOPPADDING",      (0, 0), (-1, -1),  padding),
        ("BOTTOMPADDING",   (0, 0), (-1, -1),  padding),
        ("WORDWRAP",        (0, 0), (-1, 0),   1),
    ])


# Footer (no header)

class _PageDecorator:
    """Draws footer (COAST logo + toolkit name + page number) on body pages.
    No header is drawn — removed per user request.
    """

    _FULL_NAME = "Coastal Overtopping Analysis and Simulation Toolkit"
    _logo_cache: dict = {}

    def __init__(self, case_name: str, user_name: str, coast_logo_path: Path | None):
        self._case_name  = case_name
        self._user_name  = user_name
        self._logo_path  = str(coast_logo_path) if coast_logo_path and Path(coast_logo_path).exists() else None

    def _get_logo_reader(self):
        if not self._logo_path:
            return None
        if self._logo_path not in _PageDecorator._logo_cache:
            try:
                from reportlab.lib.utils import ImageReader
                _PageDecorator._logo_cache[self._logo_path] = ImageReader(self._logo_path)
            except Exception:
                _PageDecorator._logo_cache[self._logo_path] = None
        return _PageDecorator._logo_cache[self._logo_path]

    def draw_footer(self, canvas, doc):
        # Use the current page width so this works for both portrait and landscape
        pw = doc.pagesize[0]
        canvas.saveState()
        y_line = _MARGIN - 4 * mm
        # dividing line
        canvas.setStrokeColor(_MID_GREY)
        canvas.setLineWidth(0.5)
        canvas.line(_MARGIN, y_line + 3 * mm, pw - _MARGIN, y_line + 3 * mm)

        # COAST logo (small, on the left)
        logo_reader = self._get_logo_reader()
        logo_end_x  = _MARGIN
        if logo_reader:
            try:
                lw, lh = 1.3 * cm, 0.65 * cm
                ly = y_line - lh + 3 * mm
                canvas.drawImage(logo_reader, _MARGIN, ly, width=lw, height=lh,
                                 preserveAspectRatio=True, mask="auto")
                logo_end_x = _MARGIN + lw + 3 * mm
            except Exception:
                pass

        # Toolkit name beside logo
        canvas.setFont("Helvetica-Oblique", 6.5)
        canvas.setFillColor(colors.HexColor("#666666"))
        canvas.drawString(logo_end_x, y_line + 0.5 * mm, self._FULL_NAME)

        # Page number — bottom right
        canvas.setFont("Helvetica-Bold", 8)
        canvas.setFillColor(_DARK_BLUE)
        canvas.drawRightString(pw - _MARGIN, y_line + 0.5 * mm, f"Page {doc.page}")

        canvas.restoreState()


# Cover page

def _build_cover(story: list, case_dir: Path, user_name: str, user_email: str,
                 run_info: dict, styles: dict, repo_root: Path) -> None:

    coast_logo_path = repo_root / "COAST_logo.png"
    coria_logo_path = repo_root / "python_tools" / "figures_DIR" / "coria_logo.png"
    ur_logo_path    = repo_root / "python_tools" / "figures_DIR" / "ur_logo.png"

    story.append(Spacer(1, 0.4 * cm))

    # COAST logo — centred, dominant
    logo_coast = _logo(coast_logo_path, 9 * cm, 4.5 * cm)
    if logo_coast:
        logo_coast.hAlign = "CENTER"
        story.append(logo_coast)
    story.append(Spacer(1, 0.3 * cm))

    # Full toolkit name below the logo
    story.append(Paragraph("Coastal Overtopping Analysis and Simulation Toolkit",
                            styles["toolkit"]))
    story.append(Spacer(1, 0.6 * cm))

    # Title block
    story.append(HRFlowable(width="100%", thickness=2.5, color=_DARK_BLUE))
    story.append(Spacer(1, 0.35 * cm))
    story.append(Paragraph("Analysis Report", styles["title"]))
    story.append(Paragraph(f"Case: <b>{case_dir.name}</b>", styles["subtitle"]))
    story.append(Spacer(1, 0.35 * cm))
    story.append(HRFlowable(width="100%", thickness=2.5, color=_DARK_BLUE))
    story.append(Spacer(1, 0.7 * cm))

    # Meta info
    meta = run_info.get("meta", {})
    story.append(Paragraph(f"Prepared by: <b>{user_name}</b>", styles["meta"]))
    story.append(Paragraph(f"Date: {datetime.now().strftime('%d %B %Y')}", styles["meta"]))
    if meta.get("timestamp"):
        story.append(Paragraph(f"Case generated: {meta['timestamp']}", styles["meta"]))
    if meta.get("machine"):
        story.append(Paragraph(f"Machine: {meta['machine']}", styles["meta"]))
    story.append(Spacer(1, 1.5 * cm))

    # Bottom section: software contact (left) + logos (right)
    logo_coria = _logo(coria_logo_path, 2.8 * cm, 1.5 * cm)
    logo_ur    = _logo(ur_logo_path,    2.8 * cm, 1.5 * cm)

    contact_text = (
        "<b>Software contact</b><br/>"
        "Ali MAHDI, PhD Candidate<br/>"
        "ali.mahdi@coria.fr<br/>"
        "CNRS UMR 6614 - CORIA - Univ Rouen"
    )
    contact_para = Paragraph(contact_text, styles["contact"])

    logo_right = [[logo_coria or "", logo_ur or ""]]
    logo_right_tbl = Table(
        logo_right,
        colWidths=[3.5 * cm, 3.5 * cm],
        style=TableStyle([
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("ALIGN",  (0, 0), (-1, -1), "CENTER"),
        ]),
    )
    bottom_row = [[contact_para, logo_right_tbl]]
    avail = _PAGE_W - 2 * _MARGIN
    bottom_tbl = Table(
        bottom_row,
        colWidths=[avail - 7.5 * cm, 7.5 * cm],
        style=TableStyle([
            ("VALIGN", (0, 0), (-1, -1), "BOTTOM"),
            ("BOX",    (0, 0), (-1, -1), 0.5, _MID_GREY),
            ("LEFTPADDING",  (0, 0), (-1, -1), 8),
            ("RIGHTPADDING", (0, 0), (-1, -1), 8),
            ("TOPPADDING",   (0, 0), (-1, -1), 6),
            ("BOTTOMPADDING",(0, 0), (-1, -1), 6),
        ]),
    )
    story.append(bottom_tbl)
    story.append(PageBreak())


# TOC & LOF

def _build_toc(story: list, styles: dict, toc: TableOfContents) -> None:
    story.append(Paragraph("Table of Contents", styles["h1"]))
    story.append(HRFlowable(width="100%", thickness=0.8, color=_MID_GREY))
    story.append(Spacer(1, 4))
    toc.levelStyles = [styles["toc0"], styles["toc1"]]
    story.append(toc)
    story.append(PageBreak())


def _build_lof(story: list, fig_list: list, styles: dict) -> None:
    """List of Figures — built after content sections so fig_list is fully populated."""
    story.append(Paragraph("List of Figures", styles["h1"]))
    story.append(HRFlowable(width="100%", thickness=0.8, color=_MID_GREY))
    story.append(Spacer(1, 4))
    if fig_list:
        for entry in fig_list:
            story.append(Paragraph(entry, styles["lof"]))
    else:
        story.append(Paragraph(
            "No figures found. Navigate through all analysis tabs first.",
            styles["body"]))
    story.append(PageBreak())


# Section builders

_SECTION_INTROS = {
    "parameters": (
        "This section lists all numerical and geometric parameters used to set up the "
        "OpenFOAM case. Parameters are read from <i>run_info.json</i>, which is written "
        "automatically when the case is generated."
    ),
    "geometry": (
        "The structure geometry and computational domain are illustrated below. "
        "The schematic shows the CLASH/EurOtop cross-section profile, the water depth, "
        "freeboard, and berm dimensions. The domain figure shows the wave flume extent "
        "with the STL overlay and mesh refinement zones."
    ),
    "wave_summary": (
        "This table summarises the key wave parameters computed at each free-surface gauge "
        "from both time-domain (zero-crossing) and spectral (Welch PSD) analyses, "
        "and compares them against the prescribed JONSWAP target. "
        "Rows are metrics; columns are gauge positions."
    ),
    "spectral": (
        "Per-gauge figures show the full analysis pipeline: raw signal, trimmed signal "
        "(after transient removal), zero-crossing wave detection, Welch energy spectrum "
        "overlaid with the target JONSWAP, and the wave-height PDF / CDF / PoE distributions."
    ),
    "reflection": (
        "Incident and reflected wave components are separated with the "
        "<b>Mansard &amp; Funke (1980)</b> 3-gauge frequency-domain method. "
        "For each frequency the complex incident and reflected amplitudes are obtained "
        "by least-squares fit of the three gauge records; integrating each spectrum over "
        "the analysis band gives H<sub>m0,inc</sub> and H<sub>m0,ref</sub>, and the "
        "reflection coefficient K<sub>r</sub> = H<sub>m0,ref</sub> / H<sub>m0,inc</sub> "
        "(0 = full absorption, 1 = full reflection). The signals are linearly detrended "
        "before the FFT and the analysis is restricted to a band around the peak frequency "
        "to avoid low-frequency drift and singular gauge-spacing effects."
    ),
    "overtopping": (
        "Overtopping and energy flux results are derived from three post-processing scripts: "
        "<b>OV_Test.py</b> (event detection), <b>kinetic.py</b> (kinetic + potential energy "
        "flux from probe data), and <b>merge_OV_energy.py</b> (merged event series with "
        "energy budget). Output files and figures are collected below."
    ),
    "force": (
        "Hydrodynamic forces on the structure are computed from the OpenFOAM <i>force</i> "
        "function object (pressure + viscous contributions). "
        "The horizontal force F<sub>x</sub>(t) time series and key statistics "
        "(peak force, time of occurrence) are reported below for each available force probe."
    ),
}

# Whitelist of columns to keep in the merged overtopping+energy PDF table
_MERGE_KEEP_COLS = frozenset({
    "event_id", "t_start", "t_end", "v_event", "q_peak", "e_total", "e_per_volume",
})


def _section_parameters(story: list, run_info: dict, styles: dict) -> None:
    story.append(Paragraph("1. Case Parameters", styles["h1"]))
    story.append(HRFlowable(width="100%", thickness=0.8, color=_MID_GREY))
    story.append(Paragraph(_SECTION_INTROS["parameters"], styles["intro"]))
    story.append(Spacer(1, 4))

    _UNITS = {
        "h_deep": "m",  "h": "m",   "h_t": "m", "B_t": "m", "h_b": "m",
        "B": "m",       "A_c": "m", "G_c": "m", "R_c": "m", "t_wall": "m",
        "Ly": "m",      "SWL": "m", "dx_target": "m", "dz_target": "m",
        "Hm0_deep": "m","Hm0_toe": "m",
        "Tp": "s",      "Tp_deep": "s", "Tp_toe": "s",
        "Tm_deep": "s", "Tm_toe": "s",  "Tm10_toe": "s",
        "END_TIME": "s","DELTA_T": "s", "WRITE_INTERVAL": "s",
    }
    _SECTIONS = {
        "Structure geometry": "structure_geometry",
        "Wave conditions":    "wave",
        "Domain & grid":      "domain_grid",
        "Simulation control": "control",
        "HPC settings":       "hpc",
    }
    for sec_title, key in _SECTIONS.items():
        data = run_info.get(key)
        if not data:
            continue
        story.append(Paragraph(sec_title, styles["h2"]))
        rows = [("Parameter", "Value", "Unit")]
        for k, v in data.items():
            if v is None:
                continue
            unit = _UNITS.get(k, "—")
            rows.append((k, _fmt(v, 4) if isinstance(v, float) else str(v), unit))
        story.append(_param_table(rows, col_widths=[7 * cm, 5.5 * cm, 3 * cm]))
        story.append(Spacer(1, 6))

    porous = run_info.get("porous")
    if porous and isinstance(porous, dict):
        story.append(Paragraph("Porous zones", styles["h2"]))

        rows = [("Parameter", "Value", "Unit")]
        rows.append(("USE_POROUS", str(bool(porous.get("USE_POROUS"))), "—"))
        rows.append(("POROUS_MERGED", str(bool(porous.get("POROUS_MERGED"))), "—"))
        story.append(_param_table(rows, col_widths=[7 * cm, 5.5 * cm, 3 * cm]))
        story.append(Spacer(1, 6))

        porous_zones = porous.get("POROUS_ZONES")
        if porous_zones and isinstance(porous_zones, dict):
            story.append(Paragraph("Active porous zones", styles["h2"]))
            rows = [("Zone", "Thickness", "Unit")]
            for zone, thickness in porous_zones.items():
                rows.append((str(zone), _fmt(thickness, 4), "m"))
            story.append(_param_table(rows, col_widths=[7 * cm, 5.5 * cm, 3 * cm]))
            story.append(Spacer(1, 6))

        porous_phys = porous.get("POROUS_PHYSICS_DEFAULT")
        if porous_phys and isinstance(porous_phys, dict):
            story.append(Paragraph("Porous physics parameters", styles["h2"]))
            rows = [("Parameter", "Value", "Unit")]
            porous_units = {
                "porosity": "—",
                "d50": "m",
                "KC": "—",
                "alpha": "—",
                "beta": "—",
                "gammaAddedMass": "—",
                "resistanceFormulation": "—",
            }
            for key, value in porous_phys.items():
                unit = porous_units.get(key, "—")
                rows.append((str(key), _fmt(value, 4) if isinstance(value, float) else str(value), unit))
            story.append(_param_table(rows, col_widths=[7 * cm, 5.5 * cm, 3 * cm]))
            story.append(Spacer(1, 6))


def _section_geometry(story: list, case_dir: Path, styles: dict) -> None:
    story.append(Paragraph("2. Structure Geometry & Domain", styles["h1"]))
    story.append(HRFlowable(width="100%", thickness=0.8, color=_MID_GREY))
    story.append(Paragraph(_SECTION_INTROS["geometry"], styles["intro"]))
    story.append(Spacer(1, 6))

    # Caption lookup by numeric prefix (handles any naming convention)
    _FIG_CAP = {
        "01": "CLASH/EurOtop geometry — 2D cross-section profile with structural zones and water levels",
        "02": "Computational domain — wave flume extent with STL overlay and boundary patches",
        "03": "Domain with mesh refinement zones superimposed",
        "04": "Mesh refinement bands — cell sizes dx, dz and snappyHexMesh refinement levels",
        "05": "Lines in domain — sampling probes and overtopping measurement plane",
    }
    fig_dir = case_dir / "figures"
    found = False
    if fig_dir.is_dir():
        for fpath in sorted(fig_dir.glob("*.png")):
            prefix = fpath.stem[:2]
            cap = _FIG_CAP.get(prefix, fpath.stem.replace("_", " ").replace("-", " "))
            elems = _section_figure(fpath, cap, styles)
            if elems:
                story.extend(elems)
                found = True
    if not found:
        story.append(Paragraph(
            "Case generation figures not found in this case folder.",
            styles["body"]))
    story.append(Spacer(1, 4))


def _section_wave_summary(story: list, wave_summary_df, styles: dict) -> None:
    story.append(Paragraph("3. Wave Analysis Summary", styles["h1"]))
    story.append(HRFlowable(width="100%", thickness=0.8, color=_MID_GREY))
    story.append(Paragraph(_SECTION_INTROS["wave_summary"], styles["intro"]))
    story.append(Spacer(1, 4))

    if wave_summary_df is not None and not wave_summary_df.empty:
        import pandas as pd
        # Transpose: metrics become rows, gauge positions become columns.
        # This fits all data in a single portrait-page table.
        first_col = wave_summary_df.columns[0]   # e.g. "gauge_x [m]"
        try:
            df_T = wave_summary_df.set_index(first_col).T
            df_T.index.name = "Metric"
            df_T = df_T.reset_index()
        except Exception:
            df_T = wave_summary_df   # fallback to original if transpose fails
        story.extend(_df_table(df_T, styles))
    else:
        story.append(Paragraph("No wave summary data available.", styles["body"]))
    story.append(Spacer(1, 4))


def _section_spectral_figures(story: list, case_dir: Path,
                               x_gauges: list[float], styles: dict) -> None:
    story.append(Paragraph("4. Spectral Analysis Figures", styles["h1"]))
    story.append(HRFlowable(width="100%", thickness=0.8, color=_MID_GREY))
    story.append(Paragraph(_SECTION_INTROS["spectral"], styles["intro"]))
    story.append(Spacer(1, 6))

    fig_dir = case_dir / "Wave_Analysis" / "figures"
    avail   = _PAGE_W - 2 * _MARGIN

    def _sx(xg: float) -> str:
        return f"{xg:.2f}".replace(".", "p").replace("-", "m")

    # Single-figure order (rendered individually, one per line)
    _SINGLE_ORDER = [
        ("eta_x_",            "Raw free-surface elevation η(t)"),
        ("trim_x_",           "Free-surface signal after transient trimming"),
        ("wave_detection_x_", "Zero-crossing wave detection — height and period sequence"),
        ("spectrum_x_",       "Welch power spectral density overlaid with target JONSWAP"),
    ]

    for xg in x_gauges:
        safe = _sx(xg)
        story.append(Paragraph(f"Gauge x = {xg:+.2f} m", styles["h2"]))
        story.append(Paragraph(
            f"Free-surface probe located at x = {xg:+.2f} m from the structure toe "
            "(negative values are offshore).", styles["intro"]))
        added = False

        # Single figures (eta, trim, wave_detection, spectrum)
        for prefix, cap in _SINGLE_ORDER:
            elems = _section_figure(
                fig_dir / f"{prefix}{safe}.png",
                f"{cap}  (gauge x = {xg:+.2f} m)", styles)
            if elems:
                story.extend(elems)
                added = True

        # PDF / CDF / PoE — 2 + 1 grid layout
        pdf_path = fig_dir / f"pdf_x_{safe}.png"
        cdf_path = fig_dir / f"cdf_x_{safe}.png"
        poe_path = fig_dir / f"poe_x_{safe}.png"

        have_pdf = pdf_path.exists()
        have_cdf = cdf_path.exists()
        have_poe = poe_path.exists()

        if have_pdf or have_cdf or have_poe:
            added = True
            # All 3 images at the same size
            half_w = avail / 2 - 3 * mm
            fig_h  = 6.0 * cm

            pdf_img = _img_obj(pdf_path, half_w, fig_h) if have_pdf else ""
            cdf_img = _img_obj(cdf_path, half_w, fig_h) if have_cdf else ""
            poe_img = _img_obj(poe_path, half_w, fig_h) if have_poe else ""

            # Register figures
            if have_pdf:
                _fig_num(f"PDF of wave heights (gauge x = {xg:+.2f} m)")
            if have_cdf:
                _fig_num(f"CDF of wave heights (gauge x = {xg:+.2f} m)")
            if have_poe:
                _fig_num(f"PoE of wave heights (gauge x = {xg:+.2f} m)")

            col_w = half_w + 3 * mm
            grid_data = [
                [pdf_img, cdf_img],
                [poe_img, ""],
            ]
            grid_style = TableStyle([
                ("SPAN",          (0, 1), (1, 1)),
                ("ALIGN",         (0, 0), (-1, -1), "CENTER"),
                ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
                ("TOPPADDING",    (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                ("LEFTPADDING",   (0, 0), (-1, -1), 2),
                ("RIGHTPADDING",  (0, 0), (-1, -1), 2),
            ])
            grid_tbl = Table(grid_data, colWidths=[col_w, col_w], style=grid_style)
            story.append(grid_tbl)

            # Single caption row below
            lbl = "PDF (left) · CDF (centre) · PoE (bottom)"
            story.append(Paragraph(
                f"<i>{lbl}  (gauge x = {xg:+.2f} m)</i>",
                styles["caption"]))
            story.append(Spacer(1, 4))

        if not added:
            story.append(Paragraph(
                "No figures available for this gauge. "
                "Navigate through the Streamlit analysis tabs to generate them.",
                styles["body"]))
        story.append(Spacer(1, 6))


def _section_reflection(story: list, case_dir: Path, styles: dict) -> None:
    """Section 5 — Reflection analysis (only rendered if results exist)."""
    wave_dir = case_dir / "Wave_Analysis"
    fig_dir  = wave_dir / "figures"
    json_path = wave_dir / "reflection_results.json"
    ts_fig  = fig_dir / "reflection_timeseries.png"
    sp_fig  = fig_dir / "reflection_spectra.png"

    if not (json_path.exists() or ts_fig.exists() or sp_fig.exists()):
        return

    story.append(Paragraph("5. Reflection Analysis", styles["h1"]))
    story.append(HRFlowable(width="100%", thickness=0.8, color=_MID_GREY))
    story.append(Paragraph(_SECTION_INTROS["reflection"], styles["intro"]))
    story.append(Spacer(1, 6))

    # Results table from JSON
    if json_path.exists():
        try:
            data = json.loads(json_path.read_text(encoding="utf-8"))
            gauges = data.get("gauges_x") or []
            rows = [
                ("Quantity", "Value", "Unit"),
                ("Kr (reflection coefficient)", _fmt(data.get("Kr"), 3), "—"),
                ("Hm0 incident", _fmt(data.get("Hm0_inc"), 4), "m"),
                ("Hm0 reflected", _fmt(data.get("Hm0_ref"), 4), "m"),
            ]
            if data.get("Hs_target") is not None:
                rows.append(("Hs target", _fmt(data.get("Hs_target"), 4), "m"))
            if data.get("Tp_target") is not None:
                rows.append(("Tp target", _fmt(data.get("Tp_target"), 3), "s"))
            rows.append(("Water depth h", _fmt(data.get("depth"), 4), "m"))
            if len(gauges) == 3:
                rows.append(("Gauge positions x1, x2, x3",
                             ", ".join(_fmt(g, 3) for g in gauges), "m"))
            rows.append(("Analysis band [f_min, f_max]",
                         f"{_fmt(data.get('f_min'), 3)} – {_fmt(data.get('f_max'), 3)}", "Hz"))
            rows.append(("Detrended", str(bool(data.get("detrend", True))), "—"))
            story.append(_param_table(rows, col_widths=[7 * cm, 5.5 * cm, 3 * cm]))
            story.append(Spacer(1, 6))
        except Exception:
            pass

    # Figures
    story.extend(_section_figure(
        sp_fig,
        "Incident and reflected wave spectra with JONSWAP target (Mansard & Funke 1980)",
        styles))
    story.extend(_section_figure(
        ts_fig,
        "Incident and reflected free-surface elevation at gauge 1",
        styles))
    story.append(Spacer(1, 4))


def _extract_ov_summary(text: str) -> str:
    """Extract the relevant portion of the OV_Test.py console output.

    Keeps: Window line, q_mean, Event definition, Volume budget,
           Event statistics, Largest events (top 3).
    Drops: header, Averaging window mode details, Consistency check block,
           final 'Saved event table' line.
    """
    lines = text.split("\n")
    # Find start: first line containing "Window: [t1, t2]"
    start = next((i for i, l in enumerate(lines) if "Window: [t1, t2]" in l), None)
    if start is None:
        return text.strip()
    # Find stop: first line containing "Consistency check" (exclusive)
    stop = next((i for i in range(start, len(lines)) if "Consistency check" in lines[i]), len(lines))
    extracted = lines[start:stop]
    # Strip trailing blank and separator lines
    while extracted and not extracted[-1].strip():
        extracted.pop()
    return "\n".join(extracted)


def _section_overtopping(story: list, case_dir: Path, styles: dict) -> None:
    import pandas as pd

    story.append(Paragraph("6. Overtopping & Energy Flux", styles["h1"]))
    story.append(HRFlowable(width="100%", thickness=0.8, color=_MID_GREY))
    story.append(Paragraph(_SECTION_INTROS["overtopping"], styles["intro"]))
    story.append(Spacer(1, 6))

    # OV_Test
    ov_csv = case_dir / "overtopping_events.csv"
    if ov_csv.exists():
        story.append(Paragraph("Overtopping event detection (OV_Test.py)", styles["h2"]))
        story.append(Paragraph(
            "Individual overtopping events are detected using a hysteresis threshold on the "
            "positive overtopping flux q⁺(t). Each row in the table below represents one "
            "detected event with its start/end times and integrated volume.",
            styles["intro"]))
        try:
            df = pd.read_csv(ov_csv)
            story.extend(_df_table(df, styles))
        except Exception:
            pass

    # OV_Test.py console summary
    summary_txt = case_dir / "overtopping_summary.txt"
    if summary_txt.exists():
        try:
            _raw = summary_txt.read_text(encoding="utf-8", errors="replace")
            _extracted = _extract_ov_summary(_raw)
            if _extracted:
                story.append(Spacer(1, 6))
                story.append(Paragraph("OV_Test.py analysis summary", styles["h2"]))
                _code_style = ParagraphStyle(
                    "ov_summary_code",
                    fontName="Courier",
                    fontSize=6.5,
                    leading=8.5,
                    leftIndent=6,
                    rightIndent=6,
                    spaceBefore=1,
                    spaceAfter=1,
                )
                story.append(Preformatted(_extracted, _code_style))
                story.append(Spacer(1, 6))
        except Exception:
            pass

    for fname, cap in [
        ("overtopping_fig01_cumulative_volume.png",
         "Cumulative overtopped volume V(t) — integral of the positive flux over time"),
        ("overtopping_fig02_rate_and_events.png",
         "Instantaneous overtopping rate q⁺(t) with detected event windows highlighted"),
        ("overtopping_fig03_background_samples.png",
         "Background (inter-event) overtopping samples"),
        ("overtopping_fig04_stationary_diagnostic.png",
         "Stationarity diagnostic — used by OV_Test.py to select the analysis window"),
    ]:
        story.extend(_section_figure(case_dir / fname, cap, styles))

    # kinetic.py
    ke_dat = case_dir / "energyFlux_events.dat"
    if ke_dat.exists():
        story.append(Paragraph("Energy flux (kinetic.py)", styles["h2"]))
        story.append(Paragraph(
            "Kinetic (KE) and potential (PE) energy fluxes are computed from line-probe "
            "data extracted along the overtopping measurement plane. "
            "The table below shows per-event energy statistics.",
            styles["intro"]))
        try:
            df = pd.read_csv(ke_dat, sep=r"\s+", comment="#", engine="python")
            story.extend(_df_table(df, styles))
        except Exception:
            pass

    for fname, cap in [
        ("energyFlux_components.png",
         "Energy flux components KE and PE over time"),
        ("energyFlux_cumulative.png",
         "Cumulative energy flux"),
        ("energyFlux_events_hist.png",
         "Histogram of per-event energy flux magnitudes"),
    ]:
        story.extend(_section_figure(case_dir / fname, cap, styles))

    # merge_OV_energy.py
    merge_csv = case_dir / "overtopping_energy_merged.csv"
    if merge_csv.exists():
        story.append(Paragraph("Merged overtopping + energy (merge_OV_energy.py)", styles["h2"]))
        story.append(Paragraph(
            "The merged dataset aligns overtopping event volumes with the corresponding "
            "energy flux components, enabling correlation analysis between overtopping "
            "intensity and wave energy input.",
            styles["intro"]))
        try:
            df = pd.read_csv(merge_csv)
            # Keep only the user-requested columns
            _keep = [c for c in df.columns if c.lower().strip() in _MERGE_KEEP_COLS]
            df = df[_keep] if _keep else df
            if not df.empty:
                story.extend(_df_table(df, styles))
        except Exception:
            pass

    for fname, cap in [
        ("merge_EKE_vs_V.png",
         "Kinetic energy (KE) vs. overtopped volume per event"),
        ("merge_EPE_vs_V.png",
         "Potential energy (PE) vs. overtopped volume per event"),
        ("merge_Etotal_vs_V.png",
         "Total energy (KE + PE) vs. overtopped volume per event"),
        ("merge_event_energy_components.png",
         "Energy components breakdown for each detected event"),
        ("merge_event_energy_vs_volume_median.png",
         "Event energy vs. volume — median trend"),
        ("merge_power_law_scaling.png",
         "Power-law scaling fit: E ∝ V^α"),
        ("merge_energy_per_volume_vs_rank.png",
         "Specific energy E/V ranked by event intensity"),
        ("merge_event_energy_pdf.png",
         "PDF of per-event total energy"),
        ("merge_event_energy_cdf.png",
         "CDF of per-event total energy"),
    ]:
        story.extend(_section_figure(case_dir / fname, cap, styles))


def _read_force_dat(path: Path):
    """Read OpenFOAM force.dat — handles parenthesised and plain space-separated formats.

    Returns (times, fx_pressure, fx_viscous) as lists of float.
    """
    import re
    times, fx_p, fx_v = [], [], []
    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            # Strip parentheses: "(1.2 3.4 5.6)" → "1.2 3.4 5.6"
            parts = re.sub(r"[()]", " ", line).split()
            if len(parts) < 2:
                continue
            try:
                t    = float(parts[0])
                fp_x = float(parts[1])
                fv_x = float(parts[4]) if len(parts) >= 5 else 0.0
                times.append(t)
                fx_p.append(fp_x)
                fx_v.append(fv_x)
            except (ValueError, IndexError):
                continue
    return times, fx_p, fx_v


def _section_force(story: list, case_dir: Path, styles: dict) -> None:
    """Section 6 — Force analysis (only rendered if force data exists)."""
    import re

    pp_dir = case_dir / "postProcessing"
    if not pp_dir.is_dir():
        return

    force_dirs = sorted(
        d for d in pp_dir.iterdir()
        if d.is_dir() and d.name.startswith("force")
    )
    if not force_dirs:
        return

    story.append(Paragraph("7. Force Analysis", styles["h1"]))
    story.append(HRFlowable(width="100%", thickness=0.8, color=_MID_GREY))
    story.append(Paragraph(_SECTION_INTROS["force"], styles["intro"]))
    story.append(Spacer(1, 6))

    # Impact patch segment figure (generated by coast_app.py at case build)
    _patch_fig = case_dir / "figures" / "impact_patch_segments.png"
    if _patch_fig.exists():
        story.extend(_section_figure(
            _patch_fig,
            "Structure profile — numbered segments used to define the impact/force measurement patch",
            styles,
        ))

    for fdir in force_dirs:
        # Most recent time sub-directory containing force.dat
        dat_files = sorted(
            fdir.rglob("force.dat"),
            key=lambda p: (
                float(p.parent.name) if re.match(r"^\d", p.parent.name) else 0.0
            ),
        )
        if not dat_files:
            continue
        dat_path = dat_files[-1]

        story.append(Paragraph(fdir.name, styles["h2"]))

        # Pre-saved figure (written by the Streamlit Force tab)
        safe_name = re.sub(r"[^\w]", "_", fdir.name)
        fig_path = case_dir / f"force_Fx_{safe_name}.png"
        if fig_path.exists():
            story.extend(_section_figure(
                fig_path,
                f"Horizontal force F_x(t) — {fdir.name}",
                styles,
            ))
        else:
            story.append(Paragraph(
                f"Figure not found ({fig_path.name}). "
                "Open the Force tab in the post-processing page to generate it.",
                styles["body"],
            ))

        # Summary table
        try:
            times, fx_p, fx_v = _read_force_dat(dat_path)
            if times:
                import numpy as np
                fx_total = [p + v for p, v in zip(fx_p, fx_v)]
                idx_peak = int(np.argmax(np.abs(fx_total)))
                rows = [
                    ("Parameter", "Value", "Unit"),
                    ("Records", str(len(times)), "—"),
                    ("t_start", _fmt(times[0]), "s"),
                    ("t_end",   _fmt(times[-1]), "s"),
                    ("Peak |Fx| (total)", _fmt(abs(fx_total[idx_peak]), 4), "N"),
                    ("t at peak |Fx|",   _fmt(times[idx_peak]), "s"),
                    ("Max Fx (total)",   _fmt(max(fx_total), 4), "N"),
                    ("Min Fx (total)",   _fmt(min(fx_total), 4), "N"),
                    ("Mean Fx (total)",  _fmt(float(np.mean(fx_total)), 4), "N"),
                ]
                story.append(_param_table(rows, col_widths=[7 * cm, 5.5 * cm, 2.5 * cm]))
                story.append(Spacer(1, 4))
        except Exception:
            pass

    story.append(Spacer(1, 4))


# Main entry point

def build_pdf_report(
    case_dir,
    user_name:     str  = "Unknown",
    user_email:    str  = "",
    wave_summary_df      = None,
    x_gauges:      list | None = None,
    output_name:   str  | None = None,
) -> Path:
    """
    Build a full PDF post-processing report for one EurOtop_OF case.

    Parameters
    ----------
    case_dir : str or Path
    user_name : str
    user_email : str
    wave_summary_df : pd.DataFrame or None
    x_gauges : list of float or None
    output_name : str or None
        PDF filename (default: ``<case_name>_report.pdf`` in case_dir).

    Returns
    -------
    Path to the generated PDF.
    """
    global _fig_list
    _fig_list = []               # reset figure counter for each report build

    case_dir  = Path(case_dir).resolve()
    repo_root = case_dir.parent
    x_gauges  = x_gauges or []

    # Load run_info.json
    run_info: dict = {}
    run_info_path = case_dir / "run_info.json"
    if run_info_path.exists():
        try:
            run_info = json.loads(run_info_path.read_text(encoding="utf-8"))
        except Exception:
            pass

    out_path = case_dir / (output_name or f"{case_dir.name}_report.pdf")
    styles   = _make_styles()

    # Document layout
    avail_w = _PAGE_W - 2 * _MARGIN
    avail_h = _PAGE_H - 2 * _MARGIN

    # Cover: full usable area (no header/footer)
    cover_frame = Frame(_MARGIN, _MARGIN, avail_w, avail_h,
                        leftPadding=0, rightPadding=0,
                        topPadding=0, bottomPadding=0)

    # Body: reserve room for footer only (no header)
    body_frame = Frame(
        _MARGIN,
        _MARGIN + _FOOTER_H,
        avail_w,
        avail_h - _FOOTER_H,
        leftPadding=0, rightPadding=0,
        topPadding=0, bottomPadding=0,
    )

    coast_logo_path = repo_root / "COAST_logo.png"
    _decorator = _PageDecorator(case_dir.name, user_name, coast_logo_path)

    def _on_cover(canvas, doc):
        canvas.saveState()
        # top stripe
        canvas.setFillColor(_DARK_BLUE)
        canvas.rect(0, _PAGE_H - 1.4 * cm, _PAGE_W, 1.4 * cm, fill=1, stroke=0)
        # bottom accent
        canvas.setFillColor(_ACCENT)
        canvas.rect(0, 0, _PAGE_W, 0.7 * cm, fill=1, stroke=0)
        canvas.restoreState()

    def _on_body(canvas, doc):
        # Same top/bottom accent bars as cover; no header
        canvas.saveState()
        canvas.setFillColor(_DARK_BLUE)
        canvas.rect(0, _PAGE_H - 1.4 * cm, _PAGE_W, 1.4 * cm, fill=1, stroke=0)
        canvas.setFillColor(_ACCENT)
        canvas.rect(0, 0, _PAGE_W, 0.7 * cm, fill=1, stroke=0)
        canvas.restoreState()
        _decorator.draw_footer(canvas, doc)

    doc = _CoastDocTemplate(
        str(out_path),
        pagesize=A4,
        rightMargin=_MARGIN, leftMargin=_MARGIN,
        topMargin=_MARGIN,   bottomMargin=_MARGIN,
    )
    doc.addPageTemplates([
        PageTemplate("cover", frames=[cover_frame], onPage=_on_cover),
        PageTemplate("body",  frames=[body_frame],  onPage=_on_body),
    ])

    # Build content sections first (populates _fig_list)
    content: list = []
    _section_parameters(content, run_info, styles)
    content.append(PageBreak())
    _section_geometry(content, case_dir, styles)
    content.append(PageBreak())
    _section_wave_summary(content, wave_summary_df, styles)
    content.append(PageBreak())
    _section_spectral_figures(content, case_dir, x_gauges, styles)
    content.append(PageBreak())
    _n_before_refl = len(content)
    _section_reflection(content, case_dir, styles)
    if len(content) > _n_before_refl:      # reflection section rendered
        content.append(PageBreak())
    _section_overtopping(content, case_dir, styles)
    content.append(PageBreak())
    _section_force(content, case_dir, styles)

    # Assemble full story
    # _fig_list is now fully populated by the section builders above
    toc   = TableOfContents()
    story: list = []

    # Cover (page template: cover)
    _build_cover(story, case_dir, user_name, user_email,
                 run_info, styles, repo_root)

    # All subsequent pages use the body template
    story.append(NextPageTemplate("body"))
    story.append(PageBreak())

    # TOC — page numbers filled by multiBuild second pass
    _build_toc(story, styles, toc)

    # List of Figures — now fully known
    _build_lof(story, _fig_list, styles)

    # Content
    story.extend(content)

    # Two-pass build so TOC page numbers are correct
    doc.multiBuild(story)
    return out_path
