"""
Day and night visual layer of the COAST application: colour palette, CSS and
the branded header banner.
"""

from __future__ import annotations

import base64
from pathlib import Path

import streamlit as st

# Encode the logo once when the module loads.
_FIGURES_DIR = Path(__file__).parent / "python_tools" / "figures_DIR"
_REPO_ROOT = Path(__file__).parent


def _load_logo_b64(path: Path) -> str:
    """Return a data-URI for a PNG logo, or empty string if missing."""
    if not path.exists():
        return ""
    raw = base64.b64encode(path.read_bytes()).decode()
    return f"data:image/png;base64,{raw}"


_CORIA_LOGO = _load_logo_b64(_FIGURES_DIR / "coria_logo.png")
_UR_LOGO    = _load_logo_b64(_FIGURES_DIR / "ur_logo.png")
_COAST_LOGO = _load_logo_b64(_REPO_ROOT / "COAST_logo.png")

# Color palettes
_NIGHT = dict(
    bg_primary="#0e1117",
    bg_secondary="#151a25",
    bg_card="#1a1f2e",
    bg_elevated="#212738",
    border="#2a3040",
    border_focus="#00d4aa",
    text_primary="#e6edf3",
    text_secondary="#8b949e",
    accent="#00d4aa",
    accent_hover="#00b894",
    accent2="#1fb8cd",
    warning="#f0b429",
    error="#f85149",
    # White plot backgrounds for readability in dark mode
    fig_bg="#ffffff",
    ax_bg="#f8f9fa",
    ax_text="#444444",
    ax_title="#111111",
    ax_spine="#cccccc",
    # misc
    scrollbar_hover="#3a4050",
    table_stripe="rgba(26,31,46,0.5)",
    table_hover="rgba(0,212,170,0.04)",
    accent_bg_alpha="0.08",
    tab_hover_bg="rgba(31,184,205,0.06)",
    btn_hover_bg="rgba(0,212,170,0.06)",
)

_DAY = dict(
    bg_primary="#f7f8fa",
    bg_secondary="#ffffff",
    bg_card="#ffffff",
    bg_elevated="#eef1f6",
    border="#d0d7de",
    border_focus="#0e8a6e",
    text_primary="#1f2328",
    text_secondary="#57606a",
    accent="#0e8a6e",
    accent_hover="#0b7a60",
    accent2="#1484a0",
    warning="#bf8700",
    error="#cf222e",
    # matplotlib
    fig_bg="#ffffff",
    ax_bg="#f7f8fa",
    ax_text="#57606a",
    ax_title="#1f2328",
    ax_spine="#d0d7de",
    # misc
    scrollbar_hover="#b0b8c0",
    table_stripe="rgba(0,0,0,0.025)",
    table_hover="rgba(14,138,110,0.04)",
    accent_bg_alpha="0.06",
    tab_hover_bg="rgba(20,132,160,0.06)",
    btn_hover_bg="rgba(14,138,110,0.06)",
)


def _get_palette() -> dict:
    mode = st.session_state.get("theme_mode", "day")
    return _NIGHT if mode == "night" else _DAY


def get_mpl_colors() -> dict:
    """Return dict with keys fig_bg, ax_bg, ax_text, ax_title, ax_spine."""
    p = _get_palette()
    return {k: p[k] for k in ("fig_bg", "ax_bg", "ax_text", "ax_title", "ax_spine")}


def get_fig_container_style() -> str:
    """Return inline CSS for the _show_fig container div."""
    p = _get_palette()
    return f"background:{p['bg_card']};border:1px solid {p['border']};border-radius:8px"


# CSS builder

def _build_css(p: dict) -> str:
    return f"""
/* ── Fonts ─────────────────────────────────────────────────────────────── */
/* Google Fonts import removed — use system fonts to avoid blocking on
   servers without outbound internet access. */

/* Apply IBM Plex Sans to text elements — deliberately NOT using * !important
   to avoid breaking Streamlit's Material Symbols icon font (which uses
   font-family via CSS class and would be overridden by * !important,
   causing icon names like "arrow_right" to appear as raw text). */
body,
.stApp,
[data-testid="stAppViewContainer"],
p, li, td, th, label, a,
[data-testid="stMarkdownContainer"],
[data-testid="stText"],
[data-testid="stWidgetLabel"],
.stButton > button,
.stTextInput input,
.stTextArea textarea,
[data-testid="stNumberInput"] input,
[data-baseweb="select"] span {{
    font-family: "IBM Plex Sans", "Source Sans Pro", system-ui, sans-serif;
}}
h1, h2, h3, h4, h5, h6 {{
    font-family: "IBM Plex Sans", system-ui, sans-serif !important;
}}
code, pre, [data-testid="stCode"] pre,
[data-testid="stMetricValue"],
.stCodeBlock code {{
    font-family: "JetBrains Mono", "Fira Code", "SF Mono", monospace !important;
}}

/* ── App shell ─────────────────────────────────────────────────────────── */
.stApp {{
    background: {p['bg_primary']} !important;
    color: {p['text_primary']} !important;
}}

/* ── Global text color (critical for day/night switching) ──────────────── */
[data-testid="stAppViewContainer"],
[data-testid="stMainBlockContainer"],
[data-testid="stVerticalBlock"],
[data-testid="stMarkdownContainer"],
[data-testid="stMarkdownContainer"] p,
[data-testid="stMarkdownContainer"] li,
[data-testid="stMarkdownContainer"] span,
[data-testid="stText"],
.stMarkdown, .stMarkdown p, .stMarkdown li {{
    color: {p['text_primary']} !important;
}}

/* selectbox/widget text */
[data-baseweb="select"] [data-testid="stMarkdownContainer"],
[data-baseweb="select"] span,
[data-baseweb="menu"] li {{
    color: {p['text_primary']} !important;
}}

/* ── Sidebar ───────────────────────────────────────────────────────────── */
[data-testid="stSidebar"] {{
    background: {p['bg_secondary']} !important;
    border-right: 1px solid {p['border']} !important;
}}
[data-testid="stSidebar"] [data-testid="stVerticalBlock"] {{
    padding-top: 0.5rem;
}}

/* ── Headers ───────────────────────────────────────────────────────────── */
h1, h2, h3, h4 {{
    font-family: "IBM Plex Sans", system-ui, sans-serif !important;
    color: {p['text_primary']} !important;
    font-weight: 600 !important;
}}
h1 {{ font-size: 1.6rem !important; }}
h2 {{ font-size: 1.25rem !important; }}
h3 {{ font-size: 1.1rem !important; }}

/* ── Tabs ──────────────────────────────────────────────────────────────── */
[data-baseweb="tab-list"] {{
    gap: 0 !important;
    background: transparent !important;
    border-bottom: 1px solid {p['border']} !important;
}}
[data-baseweb="tab"] {{
    color: {p['text_secondary']} !important;
    font-weight: 500 !important;
    font-size: 0.9rem !important;
    padding: 0.6rem 1.2rem !important;
    border-bottom: 2px solid transparent !important;
    transition: all 0.2s ease !important;
}}
[data-baseweb="tab"]:hover {{
    color: {p['accent2']} !important;
    background: {p['tab_hover_bg']} !important;
}}
[data-baseweb="tab"][aria-selected="true"] {{
    color: {p['accent']} !important;
    border-bottom: 2px solid {p['accent']} !important;
    background: transparent !important;
}}
[data-baseweb="tab-highlight"] {{
    display: none !important;
}}
[data-baseweb="tab-border"] {{
    display: none !important;
}}

/* ── Expanders ─────────────────────────────────────────────────────────── */
[data-testid="stExpander"] {{
    background: {p['bg_card']} !important;
    border: 1px solid {p['border']} !important;
    border-radius: 8px !important;
    margin-bottom: 0.5rem !important;
    transition: border-color 0.2s ease !important;
}}
[data-testid="stExpander"]:hover {{
    border-color: {p['accent']}40 !important;
}}
/* Fix: expander summary bg must stay light-card-bg in BOTH open and closed states */
[data-testid="stExpander"] summary,
[data-testid="stExpander"] details summary,
[data-testid="stExpander"] details[open] > summary {{
    background: {p['bg_card']} !important;
    color: {p['text_primary']} !important;
    font-weight: 500 !important;
    border-radius: 7px 7px 0 0;
}}
[data-testid="stExpander"] summary *,
[data-testid="stExpander"] details summary *,
[data-testid="stExpander"] details[open] > summary * {{
    color: {p['text_primary']} !important;
}}
[data-testid="stExpander"] details[open] > summary {{
    border-bottom: 1px solid {p['border']} !important;
    padding-bottom: 0.4rem;
    margin-bottom: 0.4rem;
}}

/* ── Inputs (number, text) ─────────────────────────────────────────────── */
[data-testid="stNumberInput"] input,
.stTextInput input,
.stTextArea textarea {{
    background: {p['bg_elevated']} !important;
    border: 1px solid {p['border']} !important;
    color: {p['text_primary']} !important;
    border-radius: 6px !important;
    transition: border-color 0.2s, box-shadow 0.2s !important;
}}
[data-testid="stNumberInput"] input {{
    font-family: "JetBrains Mono", "Fira Code", monospace !important;
}}
[data-testid="stNumberInput"] input:focus,
.stTextInput input:focus,
.stTextArea textarea:focus {{
    border-color: {p['border_focus']} !important;
    box-shadow: 0 0 0 2px {p['accent']}26 !important;
}}

/* input labels */
[data-testid="stWidgetLabel"] label,
.stTextInput label,
[data-testid="stNumberInput"] label {{
    color: {p['text_secondary']} !important;
    font-weight: 500 !important;
    font-size: 0.82rem !important;
}}

/* selectbox */
[data-baseweb="select"] {{
    background: {p['bg_elevated']} !important;
    border-radius: 6px !important;
}}
[data-baseweb="select"] [data-baseweb="tag"] {{
    background: {p['bg_card']} !important;
}}

/* ── Buttons ───────────────────────────────────────────────────────────── */
.stButton > button[kind="primary"],
button[data-testid="stBaseButton-primary"] {{
    background: linear-gradient(135deg, {p['accent']}, {p['accent_hover']}) !important;
    color: {p['bg_primary']} !important;
    font-weight: 600 !important;
    border: none !important;
    border-radius: 6px !important;
    padding: 0.45rem 1.2rem !important;
    transition: all 0.2s ease !important;
    font-family: "IBM Plex Sans", sans-serif !important;
}}
.stButton > button[kind="primary"]:hover,
button[data-testid="stBaseButton-primary"]:hover {{
    filter: brightness(1.12) !important;
    box-shadow: 0 4px 16px {p['accent']}4d !important;
}}

.stButton > button:not([kind="primary"]),
button[data-testid="stBaseButton-secondary"] {{
    background: transparent !important;
    border: 1px solid {p['border']} !important;
    color: {p['text_primary']} !important;
    border-radius: 6px !important;
    transition: all 0.2s ease !important;
}}
.stButton > button:not([kind="primary"]):hover,
button[data-testid="stBaseButton-secondary"]:hover {{
    border-color: {p['accent']} !important;
    color: {p['accent']} !important;
    background: {p['btn_hover_bg']} !important;
}}

/* ── Alerts ────────────────────────────────────────────────────────────── */
[data-testid="stAlert"] {{
    border-radius: 6px !important;
    border: none !important;
}}
div[data-testid="stAlert"]:has([data-testid="stAlertContentSuccess"]) {{
    background: {p['accent']}{int(float(p['accent_bg_alpha'])*255):02x} !important;
    border-left: 3px solid {p['accent']} !important;
}}
div[data-testid="stAlert"]:has([data-testid="stAlertContentError"]) {{
    background: {p['error']}{int(float(p['accent_bg_alpha'])*255):02x} !important;
    border-left: 3px solid {p['error']} !important;
}}
div[data-testid="stAlert"]:has([data-testid="stAlertContentWarning"]) {{
    background: {p['warning']}{int(float(p['accent_bg_alpha'])*255):02x} !important;
    border-left: 3px solid {p['warning']} !important;
}}
div[data-testid="stAlert"]:has([data-testid="stAlertContentInfo"]) {{
    background: {p['accent2']}{int(float(p['accent_bg_alpha'])*255):02x} !important;
    border-left: 3px solid {p['accent2']} !important;
}}

/* ── Metrics ───────────────────────────────────────────────────────────── */
[data-testid="stMetricValue"] {{
    color: {p['accent']} !important;
    font-family: "JetBrains Mono", monospace !important;
    font-weight: 500 !important;
}}
[data-testid="stMetricLabel"] {{
    color: {p['text_secondary']} !important;
}}

/* ── Progress bar ──────────────────────────────────────────────────────── */
[data-testid="stProgress"] > div > div {{
    background: linear-gradient(90deg, {p['accent']}, {p['accent2']}) !important;
    border-radius: 4px !important;
}}

/* ── Code blocks ───────────────────────────────────────────────────────── */
[data-testid="stCode"],
.stCodeBlock {{
    background: {p['bg_secondary']} !important;
    border: 1px solid {p['border']} !important;
    border-radius: 6px !important;
}}

/* ── Dividers ──────────────────────────────────────────────────────────── */
[data-testid="stHorizontalRule"], hr {{
    border-color: {p['border']} !important;
}}

/* ── Dataframe ─────────────────────────────────────────────────────────── */
.stDataFrame {{
    border: 1px solid {p['border']} !important;
    border-radius: 8px !important;
    overflow: hidden;
}}

/* ── st.image ──────────────────────────────────────────────────────────── */
[data-testid="stImage"] {{
    border-radius: 8px;
    overflow: hidden;
    border: 1px solid {p['border']};
}}

/* ── Scrollbar ─────────────────────────────────────────────────────────── */
::-webkit-scrollbar {{
    width: 8px;
    height: 8px;
}}
::-webkit-scrollbar-track {{
    background: transparent;
}}
::-webkit-scrollbar-thumb {{
    background: {p['border']};
    border-radius: 4px;
}}
::-webkit-scrollbar-thumb:hover {{
    background: {p['scrollbar_hover']};
}}

/* ── Checkbox ──────────────────────────────────────────────────────────── */
[data-testid="stCheckbox"] span[data-testid="stCheckbox-label"] {{
    color: {p['text_primary']} !important;
}}

/* ── Container (docs panel) ────────────────────────────────────────────── */
[data-testid="stVerticalBlockBorderWrapper"] {{
    border-color: {p['border']} !important;
    border-radius: 8px !important;
}}

/* ── Markdown tables ───────────────────────────────────────────────────── */
.stMarkdown table {{
    border-collapse: collapse !important;
    width: 100%;
}}
.stMarkdown th {{
    background: {p['bg_elevated']} !important;
    color: {p['accent']} !important;
    font-weight: 600 !important;
    border: 1px solid {p['border']} !important;
    padding: 8px 12px !important;
    font-size: 0.82rem !important;
}}
.stMarkdown td {{
    border: 1px solid {p['border']} !important;
    padding: 6px 12px !important;
    color: {p['text_primary']} !important;
    font-size: 0.82rem !important;
}}
.stMarkdown tr:nth-child(even) td {{
    background: {p['table_stripe']} !important;
}}
.stMarkdown tr:hover td {{
    background: {p['table_hover']} !important;
}}
.stMarkdown code {{
    color: {p['accent2']} !important;
    background: {p['bg_elevated']} !important;
    padding: 1px 5px !important;
    border-radius: 3px !important;
    font-size: 0.8rem !important;
}}

/* ── Captions ──────────────────────────────────────────────────────────── */
.stCaption, [data-testid="stCaptionContainer"] {{
    color: {p['text_secondary']} !important;
}}

/* ── Push content below fixed banner (72px tall) ───────────────────────── */
[data-testid="stMainBlockContainer"] {{
    padding-top: 80px !important;
}}
/* Sidebar: push content below fixed banner, remove extra Streamlit padding */
section[data-testid="stSidebar"] > div:first-child {{
    padding-top: 80px !important;
}}
[data-testid="stSidebarContent"] {{
    padding-top: 0 !important;
}}

/* ── Streamlit header: hide deploy button + decoration, keep 3-dot menu ── */
[data-testid="stDecoration"],
[data-testid="stAppDeployButton"],
.stDeployButton {{
    display: none !important;
}}
/* Make the header bar transparent so no black rectangle shows */
header[data-testid="stHeader"] {{
    background: transparent !important;
    border-bottom: none !important;
}}
/* Keep the collapsed-sidebar expand button always visible */
[data-testid="collapsedControl"] {{
    display: flex !important;
    visibility: visible !important;
}}
"""


# Header banner HTML builder

def _build_header(p: dict) -> str:
    return f"""
<style>
.coast-banner {{
    display: flex;
    align-items: center;
    justify-content: center;
    padding: 0 20px;
    background: linear-gradient(135deg, {p['bg_primary']} 0%, {p['bg_card']} 50%, {p['bg_primary']} 100%);
    border-bottom: 2px solid {p['accent']};
    position: fixed;
    top: 0;
    left: 0;
    right: 0;
    height: 72px;
    z-index: 999;
}}
.coast-center {{
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 4px;
    text-align: center;
}}
.coast-title {{
    font-family: "IBM Plex Sans", system-ui, sans-serif;
    font-size: 1.55rem;
    font-weight: 700;
    color: {p['text_primary']};
    letter-spacing: 0.01em;
    line-height: 1.2;
    margin: 0;
}}
.coast-title .coast-accent {{
    color: {p['accent']};
}}
</style>

<div class="coast-banner">
    <div class="coast-center">
        <div class="coast-title">
            <span class="coast-accent">C</span>oastal&nbsp;
            <span class="coast-accent">O</span>vertopping&nbsp;
            <span class="coast-accent">A</span>nalysis&nbsp;and&nbsp;
            <span class="coast-accent">S</span>imulation&nbsp;
            <span class="coast-accent">T</span>oolkit
        </div>
    </div>
</div>
"""


# Public API

def sidebar_logo():
    """Render a large COAST logo (or styled text) at the top of the sidebar."""
    p = _get_palette()
    if _COAST_LOGO:
        st.markdown(
            f'<div style="text-align:center;padding:8px 0 4px 0;">'
            f'<img src="{_COAST_LOGO}" style="max-width:95%;max-height:180px;width:auto;" alt="COAST"/>'
            f'</div>',
            unsafe_allow_html=True,
        )
    else:
        accent = p["accent"]
        st.markdown(
            f'<div style="text-align:center;padding:8px 0 4px 0;'
            f'font-family:\'IBM Plex Sans\',sans-serif;font-size:2rem;font-weight:800;'
            f'color:{accent};letter-spacing:0.08em;">COAST</div>',
            unsafe_allow_html=True,
        )


def sidebar_footer():
    """Render institution logos at the bottom of the sidebar."""
    p = _get_palette()
    coria_img = (
        f'<img src="{_CORIA_LOGO}" style="max-height:42px;width:auto;opacity:0.9;" alt="CORIA"/>'
        if _CORIA_LOGO else '<span style="font-weight:600;font-size:0.72rem;">CORIA</span>'
    )
    ur_img = (
        f'<img src="{_UR_LOGO}" style="max-height:42px;width:auto;opacity:0.9;" alt="Univ Rouen"/>'
        if _UR_LOGO else '<span style="font-weight:600;font-size:0.72rem;">Univ. Rouen</span>'
    )
    st.markdown(
        f"""
<div style="
    position: sticky;
    bottom: 0;
    background: {p['bg_secondary']};
    border-top: 1px solid {p['border']};
    margin-top: 24px;
    padding: 12px 4px 10px 4px;
    font-family: 'IBM Plex Sans', system-ui, sans-serif;
    font-size: 0.82rem;
    color: {p['text_secondary']};
    line-height: 1.7;
    z-index: 10;
">
  <div style="display:flex;align-items:center;gap:12px;margin-bottom:8px;">
    {coria_img}
    {ur_img}
  </div>
  CNRS UMR 6614 CORIA<br/>
  Universit&eacute; de Rouen Normandie<br/>
</div>
""",
        unsafe_allow_html=True,
    )


def inject_theme():
    """Inject theme CSS and branded header banner into the Streamlit app.

    Call once in ``main()``, immediately after ``st.set_page_config()``.
    Reads ``st.session_state.theme_mode`` ("night" or "day") to pick palette.
    """
    p = _get_palette()
    st.markdown(f"<style>{_build_css(p)}</style>", unsafe_allow_html=True)
    st.markdown(_build_header(p), unsafe_allow_html=True)
