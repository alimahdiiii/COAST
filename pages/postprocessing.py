"""
Post-processing page of the COAST application: loads a finished case and runs
the wave and overtopping analysis interactively.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

# Ensure repo root, python_tools, and Rans_OF_Main are importable
_REPO_ROOT = Path(__file__).resolve().parent.parent
for _p in [
    str(_REPO_ROOT),
    str(_REPO_ROOT / "python_tools"),
    str(_REPO_ROOT / "Rans_OF_Main"),
]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from theme import inject_theme, get_mpl_colors, sidebar_logo, sidebar_footer
import wave_analysis_py_functions as wa
from extract_from_freesurface import extract_one_gauge


# Cached analysis functions

@st.cache_data(show_spinner=False)
def _load_gauge(txt_path: str):
    t, eta = wa.read_two_columns(txt_path)
    return np.array(t, dtype=float), np.array(eta, dtype=float)


@st.cache_data(show_spinner=False)
def _trim_gauge(txt_path: str, method: str, t_cut: float):
    t, eta = _load_gauge(txt_path)
    t_out, eta_out, idx, t_start, info = wa.trim_transient(
        t, eta, method=method, t_cut=float(t_cut), reset_time=True
    )
    return (
        np.array(t_out, dtype=float),
        np.array(eta_out, dtype=float),
        float(t_start),
        info,
    )


@st.cache_data(show_spinner=False)
def _detect_waves(txt_path: str, method: str, t_cut: float):
    t, eta, *_ = _trim_gauge(txt_path, method, t_cut)
    result = wa.find_waves(t, eta)
    # result is an 8-tuple; convert inner lists to arrays for caching
    return tuple(np.array(r, dtype=float) for r in result)


@st.cache_data(show_spinner=False)
def _compute_stats(txt_path: str, method: str, t_cut: float):
    waves = _detect_waves(txt_path, method, t_cut)
    H_i, T_i = waves[0], waves[1]
    phys = wa.wave_physical_properties(list(H_i), list(T_i))
    pdf_result = wa.calculate_pdf_cdf(list(H_i), 100)
    return phys, tuple(np.array(r) for r in pdf_result)


@st.cache_data(show_spinner=False)
def _compute_spectrum(txt_path: str, method: str, t_cut: float, nBlocks: int, divisor: int = 6):
    t, eta, *_ = _trim_gauge(txt_path, method, t_cut)
    if nBlocks == 0:
        # Auto mode: nBlocks = len(t) // divisor
        effective = max(16, len(t) // max(1, divisor))
    else:
        # Fixed mode: clamp to len(t)//2 so Welch has at least 2 segments
        effective = max(16, min(nBlocks, len(t) // 2))
    freq, spec = wa.compute_wave_spectrum_welch(t, eta, effective)
    props = wa.wave_spectral_properties(freq, spec)
    return np.array(freq), np.array(spec), props, effective, len(t), float(np.std(eta))


def _detrend_linear(eta: np.ndarray) -> np.ndarray:
    """Remove mean + linear trend from a signal.

    A slowly rising/falling mean water level (wave setup, mass accumulation in
    the flume) shows up as a large spike in the very first FFT bin. Left in, it
    dominates m0 for both the incident and reflected spectra and pushes Kr
    toward 1 (a long, near-standing oscillation cannot be split into an incident
    and a reflected component from closely-spaced gauges). Removing the linear
    trend before the FFT restores a physically meaningful Kr over the wave band.
    """
    n = len(eta)
    if n < 2:
        return np.asarray(eta, dtype=float)
    idx = np.arange(n, dtype=float)
    a, b = np.polyfit(idx, eta, 1)
    return np.asarray(eta, dtype=float) - (a * idx + b)


@st.cache_data(show_spinner=False)
def _compute_reflection(
    txt1: str, txt2: str, txt3: str,
    x1: float, x2: float, x3: float,
    depth: float,
    method: str, t_cut: float,
    f_min: float, f_max: float,
    detrend: bool = True,
) -> dict:
    """Run Mansard & Funke 3-gauge reflection separation.

    The three gauges are cut with ONE trim window, common to the three signals:
    it starts at the latest of their individual trim times. The separation uses
    the phase differences between the gauges, so the signals must stay aligned in
    time. With AUTO trimming each gauge can get a different start time, and
    trimming them separately would shift one signal against the others.
    """
    t_start = max(_trim_gauge(txt, method, t_cut)[2] for txt in (txt1, txt2, txt3))
    sig = []
    for txt in (txt1, txt2, txt3):
        t, eta = _load_gauge(txt)
        sig.append(eta[np.searchsorted(t, t_start - 1e-9):])
    n = min(len(s) for s in sig)
    t1, _ = _load_gauge(txt1)
    dt = float(t1[1] - t1[0])
    e1, e2, e3 = (s[:n] for s in sig)
    if detrend:
        e1, e2, e3 = _detrend_linear(e1), _detrend_linear(e2), _detrend_linear(e3)
    result = wa.mansard_funke_separation(
        e1, e2, e3,
        dt, x1, x2, x3, depth,
        f_min=f_min, f_max=f_max if f_max > 0 else None,
    )
    result["t_start"] = float(t_start)            # start of the common trim window [s]
    result["t_rel"] = np.arange(n) * dt          # time after the trim, for the plot
    return result


@st.cache_data(show_spinner=False)
def _read_targets(case_dir_str: str):
    try:
        return wa.read_targets_from_waveProperties(case_dir_str)
    except Exception:
        return None, None, 3.3


def _read_overtopping_dat(path: Path):
    """Read postProcessing/overtopping/0/overtopping.dat → (t, q_x) arrays.

    Handles the waves2Foam parenthesised vector format:
        time  (-1.2e-07 0 0)
    as well as plain two-column format.
    """
    t_vals, q_vals = [], []
    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("Time") or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            try:
                tt = float(parts[0])
            except ValueError:
                continue
            # Strip parentheses from the rest of the line and take the first number (x-component)
            rest = " ".join(parts[1:]).replace("(", "").replace(")", "")
            nums = rest.split()
            if not nums:
                continue
            try:
                qx = float(nums[0])
            except ValueError:
                continue
            t_vals.append(tt)
            q_vals.append(qx)
    t = np.array(t_vals, dtype=float)
    q = np.array(q_vals, dtype=float)
    order = np.argsort(t)
    return t[order], q[order]


# Event figures (Rans_OF_Main/event_figures.py): case data read once
def _ef_stamps(cd: Path) -> tuple:
    """Modification times of the files read for the event figures (cache keys)."""
    def _mt(p: Path) -> float:
        try:
            return p.stat().st_mtime
        except OSError:
            return 0.0
    pm = cd / "constant" / "polyMesh"
    fields = cd / "processor0" if (cd / "processor0").is_dir() else cd
    ov = cd / "postProcessing" / "overtopping"
    return (_mt(pm / "faces") or _mt(pm / "faces.gz"), _mt(pm / "faceZones"), _mt(fields), _mt(ov),
            _mt(ov / "0" / "overtopping.dat"), _mt(cd / "overtopping_events.csv"),
            _mt(cd / "flow_depth_velocity_timeseries.csv"), _mt(cd / "flow_depth_velocity_events.csv"))


@st.cache_resource(show_spinner=False, max_entries=2)
def _ef_geometry(case_dir_str: str, mesh_stamp: tuple):
    """Mesh and overtopping section of a case (the slowest part to read)."""
    import event_figures as ef
    case = Path(case_dir_str)
    return ef.Mesh2D(case), ef.OvertoppingSection(case)


@st.cache_resource(show_spinner=False, max_entries=2)
def _ef_data(case_dir_str: str, stamps: tuple):
    """Mesh, section, output times, q+(t), h(t), u(t) and events of a case."""
    import event_figures as ef
    mesh, section = _ef_geometry(case_dir_str, stamps[:2])
    return ef.CaseData(Path(case_dir_str), mesh, section)


# Small helpers

def _gauge_label(xg: float) -> str:
    return f"x = {xg:+.2f} m"


def _gauge_txt_path(case_dir: Path, xg: float) -> Path:
    safe_x = f"{xg:.2f}".replace(".", "p")   # same encoding as extract_from_freesurface.py
    return case_dir / "Wave_Analysis" / "data" / f"eta_from_freeSurface_x_{safe_x}m.txt"


def _scan_existing_gauges(case_dir: Path) -> list[float]:
    """Return X-positions of already-extracted gauge files (sorted)."""
    data_dir = case_dir / "Wave_Analysis" / "data"
    gauges = []
    if not data_dir.is_dir():
        return gauges
    for f in sorted(data_dir.glob("eta_from_freeSurface_x_*.txt")):
        # stem example: "eta_from_freeSurface_x_-3p00"  (minus sign kept)
        stem = f.stem.replace("eta_from_freeSurface_x_", "").rstrip("m")
        try:
            gauges.append(float(stem.replace("p", ".")))
        except ValueError:
            pass
    return gauges


def _apply_theme(fig):
    """Apply current day/night theme colours to a matplotlib figure."""
    mc = get_mpl_colors()
    fig.patch.set_facecolor(mc["fig_bg"])
    for ax in fig.axes:
        ax.set_facecolor(mc["ax_bg"])
        ax.tick_params(colors=mc["ax_text"])
        ax.xaxis.label.set_color(mc["ax_text"])
        ax.yaxis.label.set_color(mc["ax_text"])
        ax.title.set_color(mc["ax_title"])
        for spine in ax.spines.values():
            spine.set_color(mc["ax_spine"])
    return fig


_WAVE_FIG_SUBDIR = "Wave_Analysis/figures"

# Overtopping / energy figures saved by OV_Test.py / kinetic.py / merge_OV_energy.py
_OV_FIGURES = [
    "overtopping_fig01_cumulative_volume.png",
    "overtopping_fig02_rate_and_events.png",
    "overtopping_fig03_background_samples.png",
    "overtopping_fig04_stationary_diagnostic.png",
]
_KE_FIGURES = [
    "energyFlux_components.png",
    "energyFlux_cumulative.png",
    "energyFlux_events_hist.png",
]
_MERGE_FIGURES = [
    "merge_EKE_vs_V.png",
    "merge_EPE_vs_V.png",
    "merge_Etotal_vs_V.png",
    "merge_event_energy_components.png",
    "merge_event_energy_vs_volume_median.png",
    "merge_power_law_scaling.png",
    "merge_energy_per_volume_vs_rank.png",
    "merge_event_energy_pdf.png",
    "merge_event_energy_cdf.png",
]

_FDV_FIGURES = [
    "fdv_fig01_timeseries.png",
    "fdv_fig02_events_vs_volume.png",
    "fdv_fig03_exceedance.png",
]


def _embed_png(path: Path) -> str:
    """Return an <img> tag with the PNG embedded as base64, or empty string."""
    import base64
    if not path.exists():
        return ""
    b64 = base64.b64encode(path.read_bytes()).decode()
    return (
        f'<div style="margin:12px 0;">'
        f'<img src="data:image/png;base64,{b64}" '
        f'style="max-width:100%;border-radius:6px;" />'
        f'<p style="font-size:0.85rem;color:#888;margin-top:4px;">{path.name}</p>'
        f'</div>'
    )


def _df_to_html(path: Path, sep: str = ",") -> str:
    """Read a CSV/DAT file and return an HTML table, or empty string on failure."""
    if not path.exists():
        return ""
    try:
        df = pd.read_csv(path, sep=sep, comment="#", engine="python")
        return df.to_html(index=False, classes="param-table", float_format=lambda x: f"{x:.4g}")
    except Exception:
        return ""


def _build_energy_section(cd: Path) -> str:
    """Build an HTML section for overtopping & energy flux results to inject into the report."""
    parts = [
        '<section id="energy" style="padding:2rem 0;">',
        '<h2>Overtopping &amp; Energy Flux (post-processing)</h2>',
        '<p style="color:#666;font-size:0.95rem;">Results from OV_Test.py, kinetic.py, and merge_OV_energy.py.</p>',
    ]

    # OV_Test results
    parts.append('<h3>Overtopping events (OV_Test.py)</h3>')
    tbl = _df_to_html(cd / "overtopping_events.csv")
    if tbl:
        parts.append(tbl)
    else:
        parts.append('<p style="color:#999;">overtopping_events.csv not found.</p>')
    for f in _OV_FIGURES:
        parts.append(_embed_png(cd / f))

    # kinetic.py results
    parts.append('<h3>Energy flux (kinetic.py)</h3>')
    tbl = _df_to_html(cd / "energyFlux_events.dat", sep=r"\s+")
    if tbl:
        parts.append(tbl)
    else:
        parts.append('<p style="color:#999;">energyFlux_events.dat not found.</p>')
    for f in _KE_FIGURES:
        parts.append(_embed_png(cd / f))

    # merge_OV_energy.py results
    parts.append('<h3>Merged overtopping + energy (merge_OV_energy.py)</h3>')
    tbl = _df_to_html(cd / "overtopping_energy_merged.csv")
    if tbl:
        parts.append(tbl)
    else:
        parts.append('<p style="color:#999;">overtopping_energy_merged.csv not found.</p>')
    for f in _MERGE_FIGURES:
        parts.append(_embed_png(cd / f))

    # flow_depth_velocity.py results
    parts.append('<h3>Flow depth and velocity per event (flow_depth_velocity.py)</h3>')
    tbl = _df_to_html(cd / "flow_depth_velocity_events.csv")
    if tbl:
        parts.append(tbl)
    else:
        parts.append('<p style="color:#999;">flow_depth_velocity_events.csv not found.</p>')
    for f in _FDV_FIGURES:
        parts.append(_embed_png(cd / f))

    parts.append('</section>')

    # Only return the section if at least one file actually existed
    has_content = any(
        (cd / f).exists()
        for f in ["overtopping_events.csv", "energyFlux_events.dat",
                  "overtopping_energy_merged.csv", "flow_depth_velocity_events.csv"]
        + _OV_FIGURES + _KE_FIGURES + _MERGE_FIGURES + _FDV_FIGURES
    )
    return "\n".join(parts) if has_content else ""


def _save_and_show(fig, filename: str, caption: str = ""):
    """Theme, save to Wave_Analysis/figures/, register for report, then display."""
    # case_dir is defined at page scope, available by the time this is called
    fig_dir = case_dir / _WAVE_FIG_SUBDIR
    fig_dir.mkdir(parents=True, exist_ok=True)
    _apply_theme(fig)
    fig.savefig(str(fig_dir / filename), dpi=200, bbox_inches="tight")
    entry = {"file": f"{_WAVE_FIG_SUBDIR}/{filename}", "caption": caption}
    figs = st.session_state.setdefault("pp_wave_figures", [])
    if not any(f["file"] == entry["file"] for f in figs):
        figs.append(entry)
    st.pyplot(fig, width="stretch")
    plt.close(fig)


def _save_and_show_current(filename: str, caption: str = ""):
    """Like _save_and_show but for plt.gcf() — used after wa.plot_pdf/cdf/poe."""
    _save_and_show(plt.gcf(), filename, caption)


# Page config & theme

st.set_page_config(
    page_title="COAST — Post-processing",
    page_icon="📊",
    layout="wide",
)

if "theme_mode" not in st.session_state:
    st.session_state.theme_mode = "day"

inject_theme()

with st.sidebar:
    sidebar_logo()
    is_night = st.session_state.theme_mode == "night"
    left, _ = st.columns([1, 3])
    with left:
        if st.button("☀️" if is_night else "🌙", key="pp_theme_toggle",
                     help="Toggle day / night theme"):
            st.session_state.theme_mode = "day" if is_night else "night"
            st.rerun()
    sidebar_footer()


# Case directory

st.title("Post-processing — Wave Analysis")

# Discover available case directories in the repo root
# Accept any sub-directory that looks like an OpenFOAM case (has run_info.json,
# TMUX_SOCKET.txt, or a constant/ sub-directory) regardless of naming convention.
_found_cases: list[Path] = sorted(
    p for p in _REPO_ROOT.iterdir()
    if p.is_dir()
    and not p.name.startswith(".")
    and (
        (p / "run_info.json").exists()
        or (p / "TMUX_SOCKET.txt").exists()
        or (p / "constant").is_dir()
    )
)

# Determine default selection from URL query param or session state
_qp_case = st.query_params.get("case_dir", "")
_ss_case  = str(st.session_state.get("case_dir") or "")
_default_str = _qp_case or _ss_case

# Build selectbox options: found cases + option to type a custom path
_case_names  = [p.name for p in _found_cases]
_case_paths  = {p.name: str(p) for p in _found_cases}
_OTHER_LABEL = "— enter path manually —"
_options     = _case_names + [_OTHER_LABEL]

# Pre-select from default if possible
_default_idx = 0
if _default_str:
    _default_name = Path(_default_str).name
    if _default_name in _case_names:
        _default_idx = _case_names.index(_default_name)
    else:
        _default_idx = len(_case_names)  # → manual entry option

_sel = st.selectbox(
    "Select case",
    _options,
    index=_default_idx,
    help="Cases found in the EurOtop_OF root folder. Select one or choose manual entry.",
)

if _sel == _OTHER_LABEL:
    case_dir_str = st.text_input(
        "Case directory (full path)",
        value=_default_str,
        placeholder="/path/to/case_001",
    )
else:
    case_dir_str = _case_paths[_sel]

if not case_dir_str:
    st.info("Select or enter a case directory to start.")
    st.stop()

case_dir = Path(case_dir_str).resolve()
if not case_dir.is_dir():
    st.warning(f"`{case_dir}` is not a valid directory.")
    st.stop()


# Gauge extraction & configuration  (above the tabs)

existing_gauges = _scan_existing_gauges(case_dir)

with st.expander("Gauge & extraction settings", expanded=not existing_gauges):
    gauge_str = st.text_input(
        "Gauge X-positions [m] (comma-separated)",
        value=", ".join(f"{x:.2f}" for x in existing_gauges) if existing_gauges
              else "-8.0, -4.0, 0.0",
        help="X-coordinate of each free-surface gauge (negative = offshore).",
    )
    col_y, col_z, col_force = st.columns([1, 1, 2])
    y_target = col_y.number_input("y_target [m]", value=0.0, format="%.3f")
    z_swl    = col_z.number_input(
        "SWL z [m]",
        value=float(st.session_state.get("SWL", 0.0)),
        format="%.4f",
    )
    force_re = col_force.checkbox("Force re-extraction (overwrite existing files)", value=False)
    do_extract = st.button("Extract free-surface gauges", type="primary")

x_gauges: list[float] = []
for tok in gauge_str.split(","):
    tok = tok.strip()
    if tok:
        try:
            x_gauges.append(float(tok))
        except ValueError:
            st.warning(f"Could not parse: '{tok}'")

if do_extract and x_gauges:
    for xg in x_gauges:
        out_path = _gauge_txt_path(case_dir, xg)
        if out_path.exists() and not force_re:
            st.info(f"x={xg:+.2f} m — already extracted ({out_path.name})")
            continue
        with st.spinner(f"Extracting x = {xg:+.2f} m …"):
            try:
                result = extract_one_gauge(case_dir, xg, y_target=y_target, z_swl=z_swl)
                st.success(f"x={xg:+.2f} m → {Path(result).name}")
            except Exception as exc:
                st.error(f"x={xg:+.2f} m: {exc}")
    _load_gauge.clear()
    _trim_gauge.clear()
    _detect_waves.clear()
    _compute_stats.clear()
    _compute_spectrum.clear()
    st.rerun()

if not x_gauges:
    st.warning("No gauge positions defined.")
    st.stop()

# Global analysis settings (used by all cached functions)
with st.expander("Analysis settings", expanded=False):
    ac1, ac2 = st.columns([1, 1])
    trim_method = ac1.radio("Trim method", ["AUTO", "MANUAL"], horizontal=True,
                            key="trim_method")
    t_cut = ac2.number_input("t_cut [s] (MANUAL)", value=50.0, step=5.0,
                             key="trim_t_cut",
                             disabled=(trim_method == "AUTO"))
    bc1, bc2, bc3 = st.columns([1, 1, 1])
    nblocks_mode = bc1.radio(
        "Welch nBlocks",
        ["Auto (len(t) // N)", "Fixed"],
        horizontal=True,
        key="nblocks_mode",
        help="Auto: nBlocks = len(t) // N (same formula as the report generator). Fixed: pick an explicit value.",
    )
    _is_auto = nblocks_mode.startswith("Auto")
    nBlocks_divisor = bc2.number_input(
        "Auto divisor N",
        min_value=2, max_value=50, value=6, step=1,
        key="nblocks_divisor",
        disabled=not _is_auto,
        help="nBlocks = len(trimmed_signal) // N.  Report generator uses N=6.",
    )
    nBlocks_fixed = bc3.select_slider(
        "Fixed nBlocks value", [512, 1024, 2048, 4096], value=2048,
        key="spec_nblocks", disabled=_is_auto,
    )
    # nBlocks == 0 signals "auto" to _compute_spectrum; divisor is passed separately
    nBlocks = 0 if _is_auto else nBlocks_fixed
    nBlocks_divisor = int(nBlocks_divisor)

_method = trim_method
_tcut   = t_cut if trim_method == "MANUAL" else 0.0

# Read target wave params once (cached)
Hs_tgt, Tp_tgt, gamma_tgt = _read_targets(str(case_dir))

# Reset figure registry when the case directory changes
if st.session_state.get("pp_figures_case") != str(case_dir):
    st.session_state["pp_wave_figures"] = []
    st.session_state["pp_figures_case"] = str(case_dir)


# Analysis tabs

tab_signals, tab_trim, tab_waves, tab_stats, tab_spec, tab_refl, tab_ov, tab_force, tab_summary, tab_compare = st.tabs([
    "1 Signals",
    "2 Trimming",
    "3 Wave Detection",
    "4 Statistics",
    "5 Spectrum",
    "6 Reflection",
    "7 Overtopping",
    "8 Force",
    "9 Summary",
    "10 Compare",
])

# Tab 1: Raw signals

with tab_signals:
    st.subheader("Raw free-surface signals")

    if Hs_tgt:
        c1, c2, c3 = st.columns(3)
        c1.metric("Hs target [m]", f"{Hs_tgt:.3f}")
        c2.metric("Tp target [s]", f"{Tp_tgt:.3f}")
        c3.metric("γ (JONSWAP)",   f"{gamma_tgt:.2f}")

    for i, xg in enumerate(x_gauges):
        txt = str(_gauge_txt_path(case_dir, xg))
        with st.expander(_gauge_label(xg), expanded=(i == 0)):
            if not Path(txt).exists():
                st.warning("File not found — run extraction above.")
                continue
            try:
                t, eta = _load_gauge(txt)
            except Exception as exc:
                st.error(str(exc))
                continue
            fig, ax = plt.subplots(figsize=(11, 3))
            ax.plot(t, eta, lw=0.8)
            ax.set_xlabel("t [s]")
            ax.set_ylabel("η [m]")
            ax.set_title(f"Raw free-surface elevation — {_gauge_label(xg)}")
            _save_and_show(fig,
                f"eta_x_{wa.safe_x_label(xg)}.png",
                f"Raw free-surface elevation — {_gauge_label(xg)}")

# Tab 2: Trimming

with tab_trim:
    st.subheader("Transient trimming")
    st.caption(
        f"Method: **{_method}**"
        + (f" | t_cut = {_tcut:.1f} s" if _method == "MANUAL" else "")
        + "  ·  Change in *Analysis settings* above."
    )

    mc = get_mpl_colors()
    for i, xg in enumerate(x_gauges):
        txt = str(_gauge_txt_path(case_dir, xg))
        with st.expander(_gauge_label(xg), expanded=(i == 0)):
            if not Path(txt).exists():
                st.warning("File not found — run extraction above.")
                continue
            try:
                t_raw, eta_raw = _load_gauge(txt)
                t_trim, eta_trim, t_start, info = _trim_gauge(txt, _method, _tcut)
            except Exception as exc:
                st.error(str(exc))
                continue
            pct = (1 - len(t_trim) / len(t_raw)) * 100 if len(t_raw) else 0
            st.info(f"Transient cut at t = **{t_start:.1f} s** — {pct:.1f}% of record removed")
            fig, ax = plt.subplots(figsize=(11, 3))
            ax.plot(t_raw, eta_raw, lw=0.5, alpha=0.3, color=mc.get("ax_spine", "gray"), label="raw")
            ax.plot(t_trim + t_start, eta_trim, lw=0.9, label="trimmed")
            ax.axvline(t_start, color="tomato", lw=1.2, ls="--", label=f"t_cut={t_start:.1f}s")
            ax.set_xlabel("t [s]")
            ax.set_ylabel("η [m]")
            ax.legend(fontsize=9)
            ax.set_title(f"Trimming — {_gauge_label(xg)}")
            _save_and_show(fig,
                f"trim_x_{wa.safe_x_label(xg)}.png",
                f"Transient trimming — {_gauge_label(xg)}")

# Tab 3: Wave detection

with tab_waves:
    st.subheader("Zero-crossing wave detection")

    for i, xg in enumerate(x_gauges):
        txt = str(_gauge_txt_path(case_dir, xg))
        with st.expander(_gauge_label(xg), expanded=(i == 0)):
            if not Path(txt).exists():
                st.warning("File not found — run extraction above.")
                continue
            try:
                t_trim, eta_trim, t_start, _ = _trim_gauge(txt, _method, _tcut)
                waves = _detect_waves(txt, _method, _tcut)
            except Exception as exc:
                st.error(str(exc))
                continue
            H_i = waves[0]
            st.info(f"{len(H_i)} waves detected")
            n_periods = st.slider("Periods to show", 3, 200, 40, key=f"np_{xg}")
            fig, ax = plt.subplots(figsize=(11, 4))
            plt.sca(ax)
            wa.plot_wave_periods(list(t_trim), list(eta_trim), n_periods)
            ax.set_title(f"Wave detection — {_gauge_label(xg)}")
            _save_and_show(fig,
                f"wave_detection_x_{wa.safe_x_label(xg)}.png",
                f"Zero-crossing wave detection — {_gauge_label(xg)}")

# Tab 4: Statistics

with tab_stats:
    st.subheader("Wave height statistics")

    for i, xg in enumerate(x_gauges):
        txt = str(_gauge_txt_path(case_dir, xg))
        with st.expander(_gauge_label(xg), expanded=(i == 0)):
            if not Path(txt).exists():
                st.warning("File not found — run extraction above.")
                continue
            try:
                phys, pdf_result = _compute_stats(txt, _method, _tcut)
            except Exception as exc:
                st.error(str(exc))
                continue

            # wave_physical_properties returns 9-tuple:
            # mean_h, global_h, max_h, rms_h, dev_h, H13, H110, mean_T, num_waves
            (mean_h, global_h, max_h, rms_h, dev_h,
             H13, H110, mean_T, num_waves) = phys

            cm1, cm2, cm3, cm4, cm5 = st.columns(5)
            cm1.metric("Hs (H₁/₃) [m]", f"{H13:.3f}")
            cm2.metric("H₁/₁₀ [m]",      f"{H110:.3f}")
            cm3.metric("Hmax [m]",         f"{max_h:.3f}")
            cm4.metric("mean T [s]",       f"{mean_T:.3f}")
            cm5.metric("# waves",          str(int(num_waves)))

            bin_c, pdf_v, cdf_v, poe_v, poe_ray = pdf_result

            # plot_pdf/cdf/poe each call plt.figure() internally → capture with plt.gcf()
            pc1, pc2, pc3 = st.columns(3)
            with pc1:
                wa.plot_pdf(bin_c, pdf_v)
                _save_and_show_current(
                    f"pdf_x_{wa.safe_x_label(xg)}.png",
                    f"PDF of wave heights — {_gauge_label(xg)}")
            with pc2:
                wa.plot_cdf(bin_c, cdf_v)
                _save_and_show_current(
                    f"cdf_x_{wa.safe_x_label(xg)}.png",
                    f"CDF of wave heights — {_gauge_label(xg)}")
            with pc3:
                wa.plot_poe(bin_c, poe_v, poe_ray)
                _save_and_show_current(
                    f"poe_x_{wa.safe_x_label(xg)}.png",
                    f"Probability of Exceedance — {_gauge_label(xg)}")

# Tab 5: Spectrum

with tab_spec:
    st.subheader("Spectral analysis")
    _nblocks_label = f"Auto (len(t) // {nBlocks_divisor})" if nBlocks == 0 else f"Fixed = {nBlocks}"
    st.caption(f"Welch nBlocks: **{_nblocks_label}**  ·  Change in *Analysis settings* above.")

    max_freq = st.number_input("Max frequency [Hz]", value=2.0, step=0.1, key="spec_maxfreq")

    for i, xg in enumerate(x_gauges):
        txt = str(_gauge_txt_path(case_dir, xg))
        with st.expander(_gauge_label(xg), expanded=(i == 0)):
            if not Path(txt).exists():
                st.warning("File not found — run extraction above.")
                continue
            try:
                freq, spec, props, eff_nblocks, data_len, eta_std = _compute_spectrum(
                    txt, _method, _tcut, nBlocks, nBlocks_divisor)
            except Exception as exc:
                st.error(str(exc))
                continue

            if nBlocks == 0:
                st.caption(
                    f"Auto nBlocks = **{eff_nblocks}** (= {data_len} pts // {nBlocks_divisor})"
                    f"  ·  σ(η) = {eta_std:.4g} m"
                )
            else:
                st.caption(
                    f"Fixed nBlocks = **{eff_nblocks}** (data: {data_len} pts)"
                    f"  ·  σ(η) = {eta_std:.4g} m"
                )

            # wave_spectral_properties returns:
            # (nth_moments, T01, T02, Hm0, Hrms, max_energy_freq, max_energy_period)
            nth_moments, T01, T02, Hm0, Hrms, fp, Tp_spec = props

            # Tm-1,0 from CFD spectrum
            _df_step = freq[1] - freq[0] if len(freq) > 1 else 1.0
            _valid_f = freq > 0
            _m_minus1 = float(np.sum(spec[_valid_f] / freq[_valid_f]) * _df_step)
            Tm_minus10 = _m_minus1 / nth_moments[0] if nth_moments[0] > 0 else 0.0

            # Target Tm-1,0 and Tm (T01) from JONSWAP
            Tm_minus10_tgt = None
            Tm01_tgt = None
            if Hs_tgt and Tp_tgt:
                _jf = wa.jonswap_spectrum(freq, Hs_tgt, Tp_tgt, gamma_tgt or 3.3)
                _m0_j = float(np.sum(_jf) * _df_step)
                _m1_j = float(np.sum(freq[_valid_f] * _jf[_valid_f]) * _df_step)
                _mm1_j = float(np.sum(_jf[_valid_f] / freq[_valid_f]) * _df_step)
                if _m0_j > 0:
                    Tm_minus10_tgt = _mm1_j / _m0_j
                    Tm01_tgt = _m0_j / _m1_j if _m1_j > 0 else None

            sm1, sm2, sm3, sm4, sm5 = st.columns(5)
            sm1.metric(
                "Hm0 [m]", f"{Hm0:.3f}",
                delta=f"{Hm0 - Hs_tgt:.3f}" if Hs_tgt else None, delta_color="off",
            )
            sm2.metric(
                "Tp [s]", f"{Tp_spec:.3f}",
                delta=f"{Tp_spec - Tp_tgt:.3f}" if Tp_tgt else None, delta_color="off",
            )
            sm3.metric(
                "Tm-1,0 [s]", f"{Tm_minus10:.3f}",
                delta=f"{Tm_minus10 - Tm_minus10_tgt:.3f}" if Tm_minus10_tgt else None,
                delta_color="off",
            )
            sm4.metric(
                "Tm (T01) [s]", f"{T01:.3f}",
                delta=f"{T01 - Tm01_tgt:.3f}" if Tm01_tgt else None, delta_color="off",
            )
            sm5.metric("T02 [s]", f"{T02:.3f}")

            mask = (freq > 0) & (freq <= max_freq)
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(freq[mask], spec[mask], lw=1.5, label="CFD spectrum")
            if Hs_tgt and Tp_tgt:
                jonswap = wa.jonswap_spectrum(freq, Hs_tgt, Tp_tgt, gamma_tgt or 3.3)
                ax.plot(
                    freq[mask], jonswap[mask], lw=1.5, ls="--",
                    label=f"JONSWAP (Hs={Hs_tgt:.2f} m, Tp={Tp_tgt:.2f} s)",
                )
            ax.set_xlabel("f [Hz]")
            ax.set_ylabel("S(f) [m²/Hz]")
            ax.set_title(f"Wave spectrum — {_gauge_label(xg)}")
            ax.legend(fontsize=9)
            _save_and_show(fig,
                f"spectrum_x_{wa.safe_x_label(xg)}.png",
                f"Wave spectrum (Welch PSD + JONSWAP) — {_gauge_label(xg)}")

# Tab 6: Reflection analysis (Mansard & Funke 1980)

with tab_refl:
    st.subheader("Reflection analysis — Mansard & Funke (1980)")
    st.caption(
        "3-gauge method: separates incident and reflected wave components "
        "in the frequency domain."
    )

    with st.expander("How the reflection coefficient works", expanded=False):
        st.markdown(
            r"""
The **Mansard & Funke (1980)** method uses three wave gauges in front of the
structure. At each frequency $f$ the recorded elevation at the three gauges is

$$\eta_j(f) = A_I\,e^{-ik(x_j-x_1)} + A_R\,e^{+ik(x_j-x_1)}$$

where $k$ solves the linear dispersion relation for the local depth $h$.
A least-squares fit of the three complex amplitudes gives the **incident** ($A_I$)
and **reflected** ($A_R$) components at every frequency. Integrating each spectrum
over the analysis band gives $H_{m0,inc}$ and $H_{m0,ref}$, and

$$K_r = \frac{H_{m0,\text{ref}}}{H_{m0,\text{inc}}}\qquad(0 \le K_r \le 1).$$

**Reading the result:** $K_r\approx0$ means the structure absorbs the waves,
$K_r\approx1$ means near-total reflection (a standing wave).

**Why a bad $K_r$ can appear:** the method is only reliable **around the peak
frequency**. Two things spoil it:
- A slowly drifting mean water level (wave setup / mass filling the flume) puts a
  huge spike in the lowest FFT bin. Because a very long wave has almost no phase
  difference between closely-spaced gauges, the fit cannot tell incident from
  reflected there and splits the energy ~50/50, inflating $K_r$ toward 1.
  → handled by the **detrend** option below.
- Frequencies where the gauge spacing $\approx n\lambda/2$ are singular.
  → handled by restricting the **analysis band** to roughly $[0.5f_p,\ 3f_p]$.

Keep the band centred on the wave energy and $K_r$ becomes physically meaningful.
"""
        )

    # Read relaxation zone end & depth from waveProperties
    _relax_end = 0.0
    _depth_refl = 0.55
    try:
        _refl_params = wa.read_relaxation_and_depth(str(case_dir))
        _relax_end = _refl_params["relax_end_x"]
        _depth_refl = _refl_params["depth"]
    except Exception as _exc_refl:
        st.warning(f"Could not read relaxation zone / depth: {_exc_refl}")

    # Auto-suggest gauge positions
    if Tp_tgt and _depth_refl > 0:
        _x1_sug, _x2_sug, _x3_sug = wa.suggest_reflection_gauges(
            _relax_end, Tp_tgt, _depth_refl
        )
    else:
        _x1_sug = _relax_end + 0.05
        _x2_sug = _x1_sug + 0.5
        _x3_sug = _x2_sug + 0.2

    with st.expander("Gauge placement", expanded=True):
        _rc1, _rc2, _rc3 = st.columns(3)
        _rx1 = _rc1.number_input("Gauge 1 x [m]", value=round(_x1_sug, 4),
                                  format="%.4f", key="refl_x1")
        _rx2 = _rc2.number_input("Gauge 2 x [m]", value=round(_x2_sug, 4),
                                  format="%.4f", key="refl_x2")
        _rx3 = _rc3.number_input("Gauge 3 x [m]", value=round(_x3_sug, 4),
                                  format="%.4f", key="refl_x3")
        _rd = st.number_input("Water depth h [m]", value=round(_depth_refl, 4),
                               format="%.4f", key="refl_depth")

        if Tp_tgt and _rd > 0:
            _omega_p = 2.0 * np.pi / Tp_tgt
            _k_p = wa.solve_dispersion(_omega_p, _rd)
            _L_p = 2.0 * np.pi / _k_p if _k_p > 0 else 0.0
            st.caption(
                f"L(Tp={Tp_tgt:.2f} s, h={_rd:.3f} m) = **{_L_p:.3f} m**  |  "
                f"\u0394\u2081\u2082 = {_rx2 - _rx1:.3f} m (L/4 = {_L_p/4:.3f})  |  "
                f"\u0394\u2082\u2083 = {_rx3 - _rx2:.3f} m (L/10 = {_L_p/10:.3f})"
            )

    # Extract missing reflection gauges
    _refl_gauges = [_rx1, _rx2, _rx3]
    _refl_missing = [xg for xg in _refl_gauges
                     if not _gauge_txt_path(case_dir, xg).exists()]

    if _refl_missing:
        st.info(
            f"Missing gauge data for: "
            f"{', '.join(f'x={xg:+.4f} m' for xg in _refl_missing)}"
        )
        if st.button("Extract reflection gauges", key="btn_extract_refl",
                      type="primary"):
            for xg in _refl_missing:
                with st.spinner(f"Extracting x = {xg:+.4f} m ..."):
                    try:
                        extract_one_gauge(case_dir, xg,
                                          y_target=y_target, z_swl=z_swl)
                        st.success(f"x={xg:+.4f} m extracted")
                    except Exception as exc:
                        st.error(f"x={xg:+.4f}: {exc}")
            _load_gauge.clear()
            _trim_gauge.clear()
            _compute_reflection.clear()
            st.rerun()

    # Run analysis if all 3 gauges exist
    _txt_r1 = str(_gauge_txt_path(case_dir, _rx1))
    _txt_r2 = str(_gauge_txt_path(case_dir, _rx2))
    _txt_r3 = str(_gauge_txt_path(case_dir, _rx3))
    _all_exist = all(Path(p).exists() for p in [_txt_r1, _txt_r2, _txt_r3])

    if not _all_exist:
        st.info("Extract all 3 reflection gauges first (button above).")
    else:
        # Default analysis band ≈ [0.5 f_p, 3 f_p] around the peak frequency.
        # This keeps Kr on the wave band and away from the singular / low-freq
        # regions that distort the separation.
        if Tp_tgt and Tp_tgt > 0:
            _fp = 1.0 / Tp_tgt
            _fmin_default = round(0.5 * _fp, 3)
            _fmax_default = round(3.0 * _fp, 3)
        else:
            _fmin_default, _fmax_default = 0.0, 2.0

        _rf_c1, _rf_c2, _rf_c3 = st.columns(3)
        _rf_fmin = _rf_c1.number_input("f_min [Hz]", value=_fmin_default,
                                        min_value=0.0, step=0.05, key="refl_fmin")
        _rf_fmax = _rf_c2.number_input("f_max [Hz]", value=_fmax_default,
                                        min_value=0.0, step=0.1, key="refl_fmax")
        _rf_detrend = _rf_c3.checkbox(
            "Detrend signals", value=True, key="refl_detrend",
            help="Remove mean + linear trend before the FFT (recommended — "
                 "kills the spurious low-frequency spike that inflates Kr).",
        )
        if Tp_tgt and Tp_tgt > 0:
            st.caption(
                f"Peak frequency f_p = 1/Tp = **{1.0/Tp_tgt:.3f} Hz** "
                f"(Tp = {Tp_tgt:.2f} s). Default band ≈ [0.5 f_p, 3 f_p]."
            )

        try:
            _refl_result = _compute_reflection(
                _txt_r1, _txt_r2, _txt_r3,
                _rx1, _rx2, _rx3, _rd,
                _method, _tcut,
                _rf_fmin, _rf_fmax,
                _rf_detrend,
            )
        except Exception as exc:
            st.error(f"Reflection analysis failed: {exc}")
            _refl_result = None

        if _refl_result:
            # Metrics
            _m1, _m2, _m3 = st.columns(3)
            _m1.metric(
                "Hm0 incident [m]", f"{_refl_result['Hm0_inc']:.4f}",
                delta=(f"{_refl_result['Hm0_inc'] - Hs_tgt:.4f}"
                       if Hs_tgt else None),
                delta_color="off",
            )
            _m2.metric("Hm0 reflected [m]",
                        f"{_refl_result['Hm0_ref']:.4f}")
            _m3.metric("Kr (reflection coeff.)",
                        f"{_refl_result['Kr']:.3f}")
            st.caption(
                f"The three gauges are analysed from t = {_refl_result['t_start']:.1f} s "
                f"(one trim window common to the three signals, so that they stay aligned in time)."
            )

            # Persist results for the PDF report
            try:
                import json as _json
                _refl_json = {
                    "method": "Mansard & Funke (1980), 3-gauge",
                    "Hm0_inc": float(_refl_result["Hm0_inc"]),
                    "Hm0_ref": float(_refl_result["Hm0_ref"]),
                    "Kr": float(_refl_result["Kr"]),
                    "gauges_x": [float(_rx1), float(_rx2), float(_rx3)],
                    "depth": float(_rd),
                    "f_min": float(_rf_fmin),
                    "f_max": float(_rf_fmax),
                    "detrend": bool(_rf_detrend),
                    "trim_start": float(_refl_result["t_start"]),
                    "Hs_target": float(Hs_tgt) if Hs_tgt else None,
                    "Tp_target": float(Tp_tgt) if Tp_tgt else None,
                }
                _refl_out = case_dir / _WAVE_FIG_SUBDIR.split("/")[0] / "reflection_results.json"
                _refl_out.parent.mkdir(parents=True, exist_ok=True)
                _refl_out.write_text(_json.dumps(_refl_json, indent=2), encoding="utf-8")
            except Exception:
                pass

            # Time-series plot: incident + reflected at gauge 1
            t_trim = _refl_result["t_rel"]
            n_plot = min(len(t_trim), len(_refl_result["eta_inc"]))
            fig_ts, ax_ts = plt.subplots(figsize=(11, 4))
            ax_ts.plot(
                t_trim[:n_plot], _refl_result["eta_inc"][:n_plot],
                lw=1.2,
                label=f"Incident (Hm0={_refl_result['Hm0_inc']:.3f} m)",
            )
            ax_ts.plot(
                t_trim[:n_plot], _refl_result["eta_ref"][:n_plot],
                lw=1.0, alpha=0.8,
                label=f"Reflected (Hm0={_refl_result['Hm0_ref']:.3f} m)",
            )
            ax_ts.set_xlabel("t [s]")
            ax_ts.set_ylabel("\u03b7 [m]")
            ax_ts.set_title(
                f"Incident & reflected elevation — gauge 1 (x = {_rx1:+.2f} m)"
            )
            ax_ts.legend(fontsize=9)
            _save_and_show(
                fig_ts, "reflection_timeseries.png",
                "Incident & reflected time series (Mansard & Funke)",
            )

            # Spectrum plot: incident + reflected + JONSWAP
            _freq_r = _refl_result["freq"]
            _mask_r = (_freq_r > 0) & (_freq_r <= _rf_fmax) if _rf_fmax > 0 else (_freq_r > 0)
            fig_sp, ax_sp = plt.subplots(figsize=(10, 4))
            ax_sp.plot(
                _freq_r[_mask_r], _refl_result["S_inc"][_mask_r],
                lw=1.5,
                label=f"Incident (Hm0={_refl_result['Hm0_inc']:.3f} m)",
            )
            ax_sp.plot(
                _freq_r[_mask_r], _refl_result["S_ref"][_mask_r],
                lw=1.5,
                label=(f"Reflected (Hm0={_refl_result['Hm0_ref']:.3f} m, "
                       f"Kr={_refl_result['Kr']:.3f})"),
            )
            if Hs_tgt and Tp_tgt:
                _jon_r = wa.jonswap_spectrum(
                    _freq_r, Hs_tgt, Tp_tgt, gamma_tgt or 3.3
                )
                ax_sp.plot(
                    _freq_r[_mask_r], _jon_r[_mask_r],
                    lw=1.5, ls="--",
                    label=f"JONSWAP target (Hs={Hs_tgt:.2f} m)",
                )
            ax_sp.set_xlabel("f [Hz]")
            ax_sp.set_ylabel("S(f) [m\u00b2/Hz]")
            ax_sp.set_title("Incident & reflected spectra (Mansard & Funke 1980)")
            ax_sp.legend(fontsize=9)
            _save_and_show(
                fig_sp, "reflection_spectra.png",
                "Incident & reflected wave spectra",
            )

# Tab 8: Force analysis

def _read_force_dat(path: Path):
    """Parse OpenFOAM force.dat — supports parenthesised and plain formats.

    Returns (times, fx_total) as numpy arrays.
    """
    import re
    times, fx_p, fx_v = [], [], []
    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = re.sub(r"[()]", " ", line).split()
            if len(parts) < 2:
                continue
            try:
                times.append(float(parts[0]))
                fx_p.append(float(parts[1]))
                fx_v.append(float(parts[4]) if len(parts) >= 5 else 0.0)
            except (ValueError, IndexError):
                continue
    t = np.array(times, dtype=float)
    fx = np.array(fx_p, dtype=float) + np.array(fx_v, dtype=float)
    return t, fx


with tab_force:
    st.subheader("Force analysis")
    st.caption(
        "Reads `postProcessing/forces*/0/force.dat` from the case directory. "
        "Figures are saved to the case folder for PDF report inclusion."
    )

    pp_dir = case_dir / "postProcessing"
    force_dirs = sorted(
        d for d in pp_dir.iterdir() if d.is_dir() and d.name.startswith("force")
    ) if pp_dir.is_dir() else []

    if not force_dirs:
        st.info(
            "No `forces*` directory found in `postProcessing/`. "
            "Enable the `forces` function object in `controlDict` before running the simulation."
        )
    else:
        for fdir in force_dirs:
            import re as _re
            st.markdown(f"**{fdir.name}**")

            # Find most recent time sub-directory with force.dat
            dat_files = sorted(
                fdir.rglob("force.dat"),
                key=lambda p: (
                    float(p.parent.name)
                    if _re.match(r"^\d", p.parent.name) else 0.0
                ),
            )
            if not dat_files:
                st.warning(f"No `force.dat` found inside `{fdir.name}/`")
                continue

            dat_path = dat_files[-1]
            try:
                t_force, fx_total = _read_force_dat(dat_path)
            except Exception as exc:
                st.error(f"Failed to read {dat_path.name}: {exc}")
                continue

            if len(t_force) == 0:
                st.warning("force.dat appears empty or could not be parsed.")
                continue

            # Metrics
            idx_peak  = int(np.argmax(np.abs(fx_total)))
            peak_fx   = fx_total[idx_peak]
            t_peak    = t_force[idx_peak]
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Peak |Fx| [N]", f"{abs(peak_fx):.4g}")
            m2.metric("t at peak [s]", f"{t_peak:.2f}")
            m3.metric("Max Fx [N]",    f"{fx_total.max():.4g}")
            m4.metric("Min Fx [N]",    f"{fx_total.min():.4g}")

            # Plot
            fig, ax = plt.subplots(figsize=(11, 3.5))
            ax.plot(t_force, fx_total, lw=0.9, label="Fx (total)")
            ax.axhline(0, color="gray", lw=0.5, ls="--")
            ax.axvline(t_peak, color="tomato", lw=1.2, ls=":",
                       label=f"peak at t={t_peak:.1f} s")
            ax.set_xlabel("t [s]")
            ax.set_ylabel("Fx [N]")
            ax.set_title(f"Horizontal force — {fdir.name}")
            ax.legend(fontsize=9)

            # Save to case directory for PDF inclusion
            safe_name = _re.sub(r"[^\w]", "_", fdir.name)
            fig_name  = f"force_Fx_{safe_name}.png"
            fig_path  = case_dir / fig_name
            _apply_theme(fig)
            fig.savefig(str(fig_path), dpi=200, bbox_inches="tight")
            st.pyplot(fig, width="stretch")
            plt.close(fig)
            st.caption(f"Figure saved → `{fig_path.name}` (included in PDF report)")

            st.divider()


# Tab 10: Multi-case comparison

with tab_compare:
    st.subheader("Multi-case comparison")
    st.caption(
        "Select 2 or more finished cases to overlay their time-series on a single figure. "
        "Each case must have the relevant post-processing output already generated."
    )

    _all_names = [p.name for p in _found_cases]
    _sel_names = st.multiselect(
        "Cases to compare", _all_names, default=[], key="cmp_cases"
    )

    if len(_sel_names) < 2:
        st.info("Select at least 2 cases above.")
    else:
        _signal_type = st.radio(
            "Signal",
            ["Cumulative overtopping volume", "Horizontal force Fx", "Energy flux (total)", "Event volume vs total energy"],
            horizontal=True,
            key="cmp_signal",
        )

        fig_cmp, ax_cmp = plt.subplots(figsize=(11, 4))
        _any_plotted = False

        for _cname in _sel_names:
            _cdir = _REPO_ROOT / _cname
            try:
                if _signal_type == "Cumulative overtopping volume":
                    _dat = _cdir / "postProcessing" / "overtopping" / "0" / "overtopping.dat"
                    _t, _q = _read_overtopping_dat(_dat)
                    _dt = np.gradient(_t)
                    _cum = np.cumsum(_q * _dt)
                    ax_cmp.plot(_t, _cum, lw=1.2, label=_cname)
                    ax_cmp.set_ylabel("Cumulative volume [m³/m]")

                elif _signal_type == "Horizontal force Fx":
                    import re as _re2
                    _pp = _cdir / "postProcessing"
                    _fdirs = sorted(
                        d for d in _pp.iterdir() if d.is_dir() and d.name.startswith("force")
                    ) if _pp.is_dir() else []
                    if _fdirs:
                        _dats = sorted(
                            _fdirs[-1].rglob("force.dat"),
                            key=lambda p: float(p.parent.name) if _re2.match(r"^\d", p.parent.name) else 0.0,
                        )
                        if _dats:
                            _t_f, _fx = _read_force_dat(_dats[-1])
                            ax_cmp.plot(_t_f, _fx, lw=0.9, label=_cname)
                            ax_cmp.set_ylabel("Fx [N]")

                elif _signal_type == "Energy flux (total)":
                    _edat = _cdir / "energyFlux_components.dat"
                    # kinetic.py writes header without '#' (comments=""), so skip row 0
                    _df_e = pd.read_csv(
                        str(_edat), sep=r"\s+", header=None, skiprows=1,
                        names=["t", "KE", "PE", "Total"], engine="python",
                    )
                    ax_cmp.plot(_df_e["t"], _df_e["Total"], lw=1.2, label=_cname)
                    ax_cmp.set_ylabel("Total energy flux [W/m]")

                elif _signal_type == "Event volume vs total energy":
                    _merged_csv = _cdir / "overtopping_energy_merged.csv"
                    _df_m = pd.read_csv(str(_merged_csv))
                    _n_events = len(_df_m)
                    ax_cmp.scatter(
                        _df_m["V_event"], _df_m["E_total"],
                        s=30, alpha=0.7,
                        label=f"{_cname} ({_n_events} events)",
                    )
                    ax_cmp.set_xlabel("Event volume [m³/m]")
                    ax_cmp.set_ylabel("Event total energy (KE+PE) [J/m]")

                _any_plotted = True
            except Exception as _exc:
                st.warning(f"**{_cname}**: could not load data — {_exc}")

        if _any_plotted:
            if _signal_type != "Event volume vs total energy":
                ax_cmp.set_xlabel("t [s]")
            ax_cmp.set_title(_signal_type)
            ax_cmp.legend(fontsize=9)
            ax_cmp.grid(True, alpha=0.25)
            _apply_theme(fig_cmp)
            st.pyplot(fig_cmp, width="stretch")
            import io as _io
            _buf_cmp = _io.BytesIO()
            fig_cmp.savefig(_buf_cmp, format="png", dpi=150, bbox_inches="tight")
            _buf_cmp.seek(0)
            _cmp_cases_str = "_vs_".join(_sel_names)
            _cmp_signal_str = _signal_type.replace(" ", "_").lower()
            st.download_button(
                "Download figure (PNG)",
                data=_buf_cmp,
                file_name=f"compare_{_cmp_cases_str}_{_cmp_signal_str}.png",
                mime="image/png",
                key="btn_dl_cmp",
            )
        plt.close(fig_cmp)


# Tab 9: Summary table

with tab_summary:
    st.subheader("Wave parameter summary")

    rows = []
    for xg in x_gauges:
        txt = str(_gauge_txt_path(case_dir, xg))
        row: dict = {"gauge_x [m]": xg}
        if not Path(txt).exists():
            rows.append(row)
            continue
        try:
            phys, _ = _compute_stats(txt, _method, _tcut)
            (mean_h, global_h, max_h, rms_h, dev_h,
             H13, H110, mean_T, num_waves) = phys
            row["Hs_time [m]"] = round(H13,   4)
            row["H_1/10 [m]"]  = round(H110,  4)
            row["Hmax [m]"]    = round(max_h,  4)
            row["mean_T [s]"]  = round(mean_T, 4)
            row["num_waves"]   = int(num_waves)
        except Exception:
            pass
        try:
            freq, spec, props, *_ = _compute_spectrum(txt, _method, _tcut, nBlocks, nBlocks_divisor)
            nth_moments, T01, T02, Hm0, Hrms, fp, Tp_spec = props
            _df_s = freq[1] - freq[0] if len(freq) > 1 else 1.0
            _vf = freq > 0
            _mm1 = float(np.sum(spec[_vf] / freq[_vf]) * _df_s)
            row["Hm0 [m]"]      = round(Hm0,    4)
            row["Tp_spec [s]"]  = round(Tp_spec, 4)
            row["Tm-1,0 [s]"]   = round(_mm1 / nth_moments[0], 4) if nth_moments[0] > 0 else None
            row["Tm (T01) [s]"] = round(T01,    4)
            row["T02 [s]"]      = round(T02,    4)
        except Exception:
            pass
        row["Hs_target [m]"] = round(Hs_tgt, 3) if Hs_tgt else None
        row["Tp_target [s]"] = round(Tp_tgt, 3) if Tp_tgt else None
        rows.append(row)

    if rows:
        df = pd.DataFrame(rows)
        st.dataframe(df, width="stretch")
        st.download_button(
            "Download summary CSV",
            data=df.to_csv(index=False).encode(),
            file_name=f"{case_dir.name}_wave_summary.csv",
            mime="text/csv",
        )

    st.divider()
    st.subheader("Generate HTML report")

    wave_figs = st.session_state.get("pp_wave_figures", [])
    st.caption(
        f"{len(wave_figs)} figure(s) collected from tabs visited so far. "
        "Navigate through all tabs to capture every plot before generating."
    )

    if st.button("Generate wave analysis report", type="primary"):
        with st.spinner("Building HTML report …"):
            try:
                def _eta_loader(xg):
                    return _load_gauge(str(_gauge_txt_path(case_dir, xg)))

                # Optional overtopping params
                q_cfd = t1_o = t2_o = None
                _ov_csv = case_dir / "overtopping_events.csv"
                if _ov_csv.exists():
                    try:
                        _df_ov = pd.read_csv(_ov_csv)
                        if "t_start" in _df_ov.columns and "t_end" in _df_ov.columns:
                            t1_o = float(_df_ov["t_start"].iloc[0])
                            t2_o = float(_df_ov["t_end"].iloc[-1])
                    except Exception:
                        pass

                Ly_o = float(st.session_state.get("Ly", 1.0))

                html_path = wa.generate_wave_analysis_report(
                    case_dir=case_dir,
                    X_GAUGES=x_gauges,
                    Hs_target=Hs_tgt,
                    Tp_target=Tp_tgt,
                    gamma_input=gamma_tgt,
                    wave_figures=wave_figs,
                    wa=wa,
                    load_eta_for_gauge=_eta_loader,
                    trimmed_data=None,
                    q_cfd=q_cfd,
                    t1_o=t1_o,
                    t2_o=t2_o,
                    Ly_o=Ly_o,
                )

                # Inject Overtopping & Energy Flux section into the report
                _energy_html = _build_energy_section(case_dir)
                if _energy_html:
                    _raw = html_path.read_text(encoding="utf-8")
                    if "</body>" in _raw:
                        _raw = _raw.replace("</body>", _energy_html + "\n</body>", 1)
                        html_path.write_text(_raw, encoding="utf-8")

                st.session_state["pp_report_path"] = str(html_path)
                st.success(f"Report saved → `{html_path}`")
            except Exception as exc:
                st.error(f"Report generation failed: {exc}")
                import traceback; st.code(traceback.format_exc())

    report_path_str = st.session_state.get("pp_report_path")
    if report_path_str and Path(report_path_str).exists():
        st.info(f"HTML report saved to: `{report_path_str}`")

    st.divider()
    st.subheader("Download PDF report")
    st.caption(
        "Generates a full PDF with logos, all case parameters, geometry figures, "
        "wave analysis results, and overtopping/energy figures."
    )

    pdf_user = st.text_input("User Name", value="", key="pdf_author",
                              placeholder="Your name")

    if st.button("Generate PDF report", key="btn_pdf", type="primary"):
        with st.spinner("Building PDF …"):
            try:
                sys.path.insert(0, str(_REPO_ROOT / "python_tools"))
                from pdf_report import build_pdf_report  # noqa: PLC0415

                # Build the summary DataFrame for inclusion
                _pdf_rows = []
                for xg in x_gauges:
                    txt = str(_gauge_txt_path(case_dir, xg))
                    row: dict = {"gauge_x [m]": xg}
                    if Path(txt).exists():
                        try:
                            phys, _ = _compute_stats(txt, _method, _tcut)
                            (_, _, max_h, _, _, H13, H110, mean_T, num_waves) = phys
                            row["Hs_time [m]"] = round(H13,   4)
                            row["H_1/10 [m]"]  = round(H110,  4)
                            row["Hmax [m]"]    = round(max_h,  4)
                            row["mean_T [s]"]  = round(mean_T, 4)
                            row["num_waves"]   = int(num_waves)
                        except Exception:
                            pass
                        try:
                            _f, _s, _p, *_ = _compute_spectrum(
                                txt, _method, _tcut, nBlocks, nBlocks_divisor)
                            _nth, T01, T02, Hm0, _, _, Tp_spec = _p
                            _df_s = _f[1] - _f[0] if len(_f) > 1 else 1.0
                            _vf = _f > 0
                            _mm1 = float(np.sum(_s[_vf] / _f[_vf]) * _df_s)
                            row["Hm0 [m]"]      = round(Hm0,    4)
                            row["Tp_spec [s]"]  = round(Tp_spec, 4)
                            row["Tm-1,0 [s]"]   = round(_mm1 / _nth[0], 4) if _nth[0] > 0 else None
                            row["Tm (T01) [s]"] = round(T01,    4)
                            row["T02 [s]"]      = round(T02,    4)
                        except Exception:
                            pass
                    row["Hs_target [m]"] = round(Hs_tgt, 3) if Hs_tgt else None
                    row["Tp_target [s]"] = round(Tp_tgt, 3) if Tp_tgt else None
                    _pdf_rows.append(row)
                _pdf_df = pd.DataFrame(_pdf_rows) if _pdf_rows else None

                pdf_path = build_pdf_report(
                    case_dir=case_dir,
                    user_name=pdf_user or "—",
                    wave_summary_df=_pdf_df,
                    x_gauges=x_gauges,
                )
                st.session_state["pp_pdf_path"] = str(pdf_path)
                st.success(f"PDF saved → `{pdf_path}`")
            except Exception as exc:
                st.error(f"PDF generation failed: {exc}")
                import traceback; st.code(traceback.format_exc())

    pdf_path_str = st.session_state.get("pp_pdf_path")
    if pdf_path_str and Path(pdf_path_str).exists():
        with open(pdf_path_str, "rb") as _pf:
            _pdf_bytes = _pf.read()
        st.download_button(
            "⬇ Download PDF Report",
            data=_pdf_bytes,
            file_name=Path(pdf_path_str).name,
            mime="application/pdf",
            type="primary",
        )

# Tab 7: Overtopping

with tab_ov:
    st.subheader("Overtopping analysis")
    st.caption("Runs the post-processing scripts from the case directory. Each step builds on the previous.")

    # Step 1: OV_Test.py with parameter controls
    st.markdown("**Step 1 — Detect overtopping events**")

    with st.expander("Detection parameters", expanded=True):
        _ov_window_mode = st.selectbox(
            "Window mode", ["FIRST_OVERTOPPING", "MANUAL"],
            key="ov_window_mode",
            help="FIRST_OVERTOPPING: window starts when cumulative volume ≥ Vthr_abs.\n"
                 "MANUAL: specify t1, t2 directly.",
        )
        # Always define all variables so they are in scope regardless of mode
        _ov_t1, _ov_t2, _ov_vthr = 10.0, 150.0, 1e-5
        _c1, _c2 = st.columns(2)
        if _ov_window_mode == "MANUAL":
            _ov_t1 = _c1.number_input("t₁ [s]",  value=10.0,  min_value=0.0, step=1.0, key="ov_t1")
            _ov_t2 = _c2.number_input("t₂ [s]",  value=150.0, min_value=0.0, step=1.0, key="ov_t2")
        else:  # FIRST_OVERTOPPING
            _ov_vthr = _c1.number_input(
                "Vthr_abs [m³/m]", value=1e-5, format="%.2e", min_value=0.0, key="ov_vthr",
                help="Cumulative volume threshold to declare start of overtopping window.",
            )
        _c3, _c4 = st.columns(2)
        _ov_q_start = _c3.number_input(
            "q_start [m³/s/m]", value=1e-3, format="%.2e", min_value=0.0, key="ov_q_start",
            help="Event triggers when q⁺ ≥ q_start.",
        )
        _ov_q_end = _c4.number_input(
            "q_end [m³/s/m]", value=3e-4, format="%.2e", min_value=0.0, key="ov_q_end",
            help="Event ends when q⁺ ≤ q_end (must be < q_start).",
        )
        _c5, _c6 = st.columns(2)
        _ov_min_dur = _c5.number_input("min_duration [s]",  value=0.02, min_value=0.0, key="ov_min_dur")
        _ov_min_vol = _c6.number_input("min_volume [m³/m]", value=1e-6, format="%.2e",
                                        min_value=0.0, key="ov_min_vol")
        if _ov_q_end >= _ov_q_start:
            st.warning("q_end must be strictly less than q_start.")

    _ov_template = _REPO_ROOT / "Rans_OF_Main" / "OV_Test.py"
    _ov_out_csv  = case_dir / "overtopping_events.csv"
    _ov_figs = [
        "overtopping_fig01_cumulative_volume.png",
        "overtopping_fig02_rate_and_events.png",
        "overtopping_fig03_background_samples.png",
        "overtopping_fig04_stationary_diagnostic.png",
    ]
    _col_btn, _col_status = st.columns([2, 3])
    if not _ov_template.exists():
        _col_status.warning("`OV_Test.py` not found in Rans_OF_Main/")

    if _col_btn.button("Run OV_Test.py", key="run_OV_Test.py",
                       disabled=not _ov_template.exists() or _ov_q_end >= _ov_q_start):
        import re as _re_ov
        _src = _ov_template.read_text()
        _src = _re_ov.sub(r'^WINDOW_MODE\s*=\s*"[^"]*"',
                          f'WINDOW_MODE = "{_ov_window_mode}"', _src, flags=_re_ov.MULTILINE)
        _src = _re_ov.sub(r'^t1_manual,\s*t2_manual\s*=.*',
                          f't1_manual, t2_manual = {_ov_t1}, {_ov_t2}', _src, flags=_re_ov.MULTILINE)
        _src = _re_ov.sub(r'Vthr_abs\s*=\s*[0-9][0-9e.+-]*', f'Vthr_abs={_ov_vthr!r}', _src)
        _src = _re_ov.sub(r'^q_start\s*=\s*[0-9][0-9e.+-]*',
                          f'q_start = {_ov_q_start!r}', _src, flags=_re_ov.MULTILINE)
        _src = _re_ov.sub(r'^q_end\s*=\s*[0-9][0-9e.+-]*',
                          f'q_end   = {_ov_q_end!r}',   _src, flags=_re_ov.MULTILINE)
        _src = _re_ov.sub(r'^min_duration\s*=\s*[0-9][0-9e.+-]*',
                          f'min_duration = {_ov_min_dur!r}', _src, flags=_re_ov.MULTILINE)
        _src = _re_ov.sub(r'^min_volume\s*=\s*[0-9][0-9e.+-]*',
                          f'min_volume   = {_ov_min_vol!r}', _src, flags=_re_ov.MULTILINE)
        (case_dir / "OV_Test.py").write_text(_src)
        with st.spinner("Running OV_Test.py …"):
            res = subprocess.run(
                [sys.executable, str(case_dir / "OV_Test.py")],
                cwd=str(case_dir), capture_output=True, text=True, timeout=600,
            )
        if res.returncode == 0:
            st.success("OV_Test.py finished (exit 0)")
            if res.stdout:
                (case_dir / "overtopping_summary.txt").write_text(res.stdout, encoding="utf-8")
        else:
            st.error(f"OV_Test.py exited with code {res.returncode}")
        combined = (res.stdout + res.stderr).strip()
        if combined:
            st.code(combined, language="text")

    if _ov_out_csv.exists():
        _col_status.success("`overtopping_events.csv` ready")
        try:
            st.dataframe(pd.read_csv(_ov_out_csv), width="stretch")
        except Exception:
            pass
    else:
        _col_status.info("`overtopping_events.csv` not yet generated")

    _existing_ov_figs = [f for f in _ov_figs if (case_dir / f).exists()]
    if _existing_ov_figs:
        figs_reg = st.session_state.setdefault("pp_wave_figures", [])
        for _fn in _existing_ov_figs:
            _entry = {"file": _fn, "caption": _fn.replace("_", " ").replace(".png", "")}
            if not any(f["file"] == _entry["file"] for f in figs_reg):
                figs_reg.append(_entry)
        _cols = st.columns(min(len(_existing_ov_figs), 2))
        for _j, _fn in enumerate(_existing_ov_figs):
            with _cols[_j % 2]:
                st.image(str(case_dir / _fn), caption=_fn, width="stretch")

    st.divider()

    # Steps 2 & 3: kinetic.py, merge_OV_energy.py
    _ov_scripts = [
        (
            "kinetic.py",
            "Step 2 — Compute energy flux",
            "energyFlux_events.dat",
            [
                "energyFlux_components.png",
                "energyFlux_cumulative.png",
                "energyFlux_events_hist.png",
            ],
        ),
        (
            "merge_OV_energy.py",
            "Step 3 — Merge events + energy",
            "overtopping_energy_merged.csv",
            [
                "merge_EKE_vs_V.png",
                "merge_EPE_vs_V.png",
                "merge_Etotal_vs_V.png",
                "merge_event_energy_components.png",
                "merge_event_energy_vs_volume_median.png",
                "merge_power_law_scaling.png",
                "merge_energy_per_volume_vs_rank.png",
                "merge_event_energy_pdf.png",
                "merge_event_energy_cdf.png",
            ],
        ),
    ]

    for script, label, output_file, fig_names in _ov_scripts:
        script_path = _REPO_ROOT / "Rans_OF_Main" / script
        out_path    = case_dir / output_file

        st.markdown(f"**{label}**")
        col_btn, col_status = st.columns([2, 3])

        if not script_path.exists():
            col_status.warning(f"`{script}` not found in Rans_OF_Main/")

        if col_btn.button(f"Run {script}", key=f"run_{script}",
                          disabled=not script_path.exists()):
            with st.spinner(f"Running {script} …"):
                res = subprocess.run(
                    [sys.executable, str(script_path)],
                    cwd=str(case_dir),
                    capture_output=True,
                    text=True,
                    timeout=600,
                )
            if res.returncode == 0:
                st.success(f"{script} finished (exit 0)")
            else:
                st.error(f"{script} exited with code {res.returncode}")
            combined = (res.stdout + res.stderr).strip()
            if combined:
                st.code(combined, language="text")

        # Output data table
        if out_path.exists():
            col_status.success(f"`{output_file}` ready")
            try:
                sep = r"\s+" if output_file.endswith(".dat") else ","
                df_ov = pd.read_csv(out_path, sep=sep, comment="#", engine="python")
                st.dataframe(df_ov, width="stretch")
            except Exception:
                pass
        else:
            col_status.info(f"`{output_file}` not yet generated")

        # Output figures — display and register for report
        existing_figs = [f for f in fig_names if (case_dir / f).exists()]
        if existing_figs:
            figs_reg = st.session_state.setdefault("pp_wave_figures", [])
            for fname in existing_figs:
                entry = {"file": fname, "caption": fname.replace("_", " ").replace(".png", "")}
                if not any(f["file"] == entry["file"] for f in figs_reg):
                    figs_reg.append(entry)
            cols = st.columns(min(len(existing_figs), 2))
            for j, fname in enumerate(existing_figs):
                with cols[j % 2]:
                    st.image(str(case_dir / fname), caption=fname, width="stretch")

        st.divider()

    # Step 4: flow depth and velocity per event
    st.markdown("**Step 4 — Flow depth and velocity per event**")
    st.caption(
        "From the vertical sampling line x01 at the overtopping section and the event windows "
        "of Step 1. Flow depth h = ∫α dz; velocity = mean horizontal velocity of the wet "
        "points (α ≥ 0.5). The velocity is left empty when the water layer is too thin to be "
        "resolved by the mesh."
    )
    with st.expander("Parameters", expanded=False):
        _f1, _f4 = st.columns(2)
        _fdv_hmin = _f1.number_input(
            "H_MIN [m]", value=0.002, format="%.4f", min_value=0.0, key="fdv_hmin",
            help="Below this flow depth the velocity is not computed (left empty): the layer "
                 "is too thin to be resolved by the mesh. It is also left empty when no point "
                 "of the line has α ≥ 0.5.",
        )
        _fdv_nwaves = _f4.number_input(
            "N_WAVES [-]", value=0, min_value=0, step=1, key="fdv_nwaves",
            help="Number of incident waves in the analysis window; if > 0, h_2% and u_2% "
                 "(exceeded by 2% of the incident waves) are also reported.",
        )

    _fdv_template = _REPO_ROOT / "Rans_OF_Main" / "flow_depth_velocity.py"
    _fdv_out_csv  = case_dir / "flow_depth_velocity_events.csv"
    _fcol_btn, _fcol_status = st.columns([2, 3])
    if not _fdv_template.exists():
        _fcol_status.warning("`flow_depth_velocity.py` not found in Rans_OF_Main/")
    if not (case_dir / "overtopping_events.csv").exists():
        _fcol_status.info("Run Step 1 first (needs `overtopping_events.csv`).")

    if _fcol_btn.button("Run flow_depth_velocity.py", key="run_flow_depth_velocity.py",
                        disabled=not _fdv_template.exists()
                        or not (case_dir / "overtopping_events.csv").exists()):
        import re as _re_fdv
        _src = _fdv_template.read_text()
        for _key, _val in (("H_MIN", float(_fdv_hmin)), ("N_WAVES", int(_fdv_nwaves))):
            _src = _re_fdv.sub(rf"^{_key}\s*=\s*[0-9][0-9e.+-]*",
                               f"{_key} = {_val!r}", _src, flags=_re_fdv.MULTILINE)
        (case_dir / "flow_depth_velocity.py").write_text(_src)
        with st.spinner("Running flow_depth_velocity.py …"):
            res = subprocess.run(
                [sys.executable, str(case_dir / "flow_depth_velocity.py")],
                cwd=str(case_dir), capture_output=True, text=True, timeout=1800,
            )
        if res.returncode == 0:
            st.success("flow_depth_velocity.py finished (exit 0)")
            if res.stdout:
                (case_dir / "flow_depth_velocity_summary.txt").write_text(res.stdout, encoding="utf-8")
        else:
            st.error(f"flow_depth_velocity.py exited with code {res.returncode}")
        combined = (res.stdout + res.stderr).strip()
        if combined:
            st.code(combined, language="text")
    elif (case_dir / "flow_depth_velocity_summary.txt").exists():
        st.code((case_dir / "flow_depth_velocity_summary.txt").read_text(), language="text")

    if _fdv_out_csv.exists():
        _fcol_status.success("`flow_depth_velocity_events.csv` ready")
        try:
            st.dataframe(pd.read_csv(_fdv_out_csv), width="stretch")
        except Exception:
            pass

    _existing_fdv = [f for f in _FDV_FIGURES if (case_dir / f).exists()]
    if _existing_fdv:
        figs_reg = st.session_state.setdefault("pp_wave_figures", [])
        for _fn in _existing_fdv:
            _entry = {"file": _fn, "caption": _fn.replace("_", " ").replace(".png", "")}
            if not any(f["file"] == _entry["file"] for f in figs_reg):
                figs_reg.append(_entry)
        _cols = st.columns(min(len(_existing_fdv), 2))
        for _j, _fn in enumerate(_existing_fdv):
            with _cols[_j % 2]:
                st.image(str(case_dir / _fn), caption=_fn, width="stretch")

    # Figures C and D: one overtopping event at the overtopping section
    st.divider()
    st.markdown("**Event figures — one overtopping event at the overtopping section (figures C and D)**")
    st.caption(
        "Figure C: the crest edge at the full-field outputs of the event (water fraction and mesh; "
        "velocity of the wet cells, α ≥ 0.5, as arrows of equal length coloured by the speed; water flux "
        "through each face of the overtopping section), with q⁺(t), h(t) and u(t) under them. "
        "Figure D: the flow over the structure and a zoom on the crest edge with the mesh and the "
        "overtopping section, at the full-field output nearest the peak discharge of the event. "
        "Files: event_figures/event<N>_snapshots and event<N>_section (PDF vector, PNG 600 dpi)."
    )
    _ef_needs = {
        "overtopping_events.csv (Step 1)": case_dir / "overtopping_events.csv",
        "flow_depth_velocity_timeseries.csv (Step 4)": case_dir / "flow_depth_velocity_timeseries.csv",
        "constant/polyMesh": case_dir / "constant" / "polyMesh",
        "postProcessing/overtopping": case_dir / "postProcessing" / "overtopping",
    }
    _ef_missing = [k for k, p in _ef_needs.items() if not p.exists()]
    _efd = None
    if _ef_missing:
        st.info("Not available yet, missing: " + ", ".join(_ef_missing) + ".")
    else:
        import event_figures as _ef
        try:
            with st.spinner("Reading the mesh and the outputs of the case …"):
                _efd = _ef_data(str(case_dir), _ef_stamps(case_dir))
        except Exception as _err:
            st.error(f"The event figures cannot read this case: {_err}")

    if _efd is not None and not _efd.times:
        st.warning("No full-field output (alpha.water and U) in the time folders of this case.")
    elif _efd is not None:
        _ev = _efd.events.copy()
        _tarr = np.array(list(_efd.times))
        _ev["n_out"] = [_ef.snapshot_count(_tarr, a, b) for a, b in zip(_ev.t_start, _ev.t_end)]
        _dtw = float(np.median(np.diff(_tarr))) if _tarr.size > 1 else float("nan")

        def _ef_label(r) -> str:
            _h = getattr(r, "h_max", np.nan)
            _u = getattr(r, "u_max", np.nan)
            return (f"Event {int(r.event_id)}  ·  t = {r.t_start:.2f}–{r.t_end:.2f} s  ·  "
                    f"V = {r.volume:.2e} m³/m  ·  "
                    f"h_max = {'–' if pd.isna(_h) else f'{100 * _h:.2f} cm'}  ·  "
                    f"u_max = {'not resolved' if pd.isna(_u) else f'{_u:.2f} m/s'}  ·  "
                    f"{int(r.n_out)} field outputs")

        _e_c1, _e_c2 = st.columns([1, 4])
        _ef_order = _e_c1.radio("Order of the events", ["by volume", "by time"], key="ef_order",
                                help="By volume: event 1 is the largest (numbering of overtopping_events.csv).")
        _ev = _ev.sort_values("event_id" if _ef_order == "by volume" else "t_start")
        _ef_labels = {int(r.event_id): _ef_label(r) for r in _ev.itertuples()}
        _ef_id = _e_c2.selectbox(
            f"Event ({len(_ev)} events; full fields written every {_dtw:.3g} s)",
            list(_ef_labels), format_func=_ef_labels.get, key=f"ef_event_{case_dir.name}",
            help="The snapshots are the full-field outputs from just before the start of the event to "
                 "just after its end. Events shorter than the write interval have few or none.",
        )
        _e = _ev.loc[_ev.event_id == _ef_id].iloc[0]

        with st.expander("Figure options", expanded=False):
            _o1, _o2 = st.columns(2)
            _ef_n = _o1.selectbox("Number of snapshots (figure C)", [3, 6, 9, 12], index=2, key="ef_n")
            _ef_vmax = _o2.number_input(
                "Top of the speed colour bar [m/s] (0 = from the event)", value=0.0, min_value=0.0,
                step=0.5, key="ef_vmax",
                help="0: rounded up from the largest speed shown. Give the same value to several events "
                     "or cases to compare their colours (e.g. 3).")
            _ef_first = _ef_step = None
            if st.checkbox("Choose the snapshot times", key="ef_manual_t"):
                _auto = _ef.select_frames(_tarr, _e.t_start, _e.t_end, int(_ef_n))
                _auto_step = float(np.median(np.diff(_auto))) if len(_auto) > 1 else _dtw
                _t1c, _t2c = st.columns(2)
                _ef_first = _t1c.number_input("First snapshot [s]", value=float(_auto[0]), format="%.3f",
                                              key=f"ef_first_{case_dir.name}_{_ef_id}")
                _ef_step = _t2c.number_input("Time between snapshots [s]", value=float(_auto_step),
                                             min_value=0.0, format="%.3f",
                                             key=f"ef_step_{case_dir.name}_{_ef_id}",
                                             help="Each snapshot is the full-field output nearest the "
                                                  "requested time.")
            _ef_windows = {}
            if st.checkbox("Set the windows of the figures", key="ef_manual_w",
                           help="By default they are set from the overtopping section: position and "
                                "length of line x01."):
                _wdef = _ef.default_windows(_efd.info, _efd.mesh)
                for _wk, _wlab in (("snapshots", "Figure C, snapshots"), ("zoom", "Figure D (b), zoom"),
                                   ("overview", "Figure D (a), structure")):
                    (_x0, _x1), (_z0, _z1) = _wdef[_wk]
                    st.caption(_wlab)
                    _wc = st.columns(4)
                    _wv = [_wc[_k].number_input(_lab, value=float(_v), format="%.4f",
                                                key=f"ef_w_{case_dir.name}_{_wk}_{_k}")
                           for _k, (_lab, _v) in enumerate((("x min [m]", _x0), ("x max [m]", _x1),
                                                            ("z min [m]", _z0), ("z max [m]", _z1)))]
                    if _wv[1] > _wv[0] and _wv[3] > _wv[2]:
                        _ef_windows[_wk] = ((_wv[0], _wv[1]), (_wv[2], _wv[3]))
                    else:
                        st.warning(f"{_wlab}: the maximum must be larger than the minimum (default used).")

        _ef_out = case_dir / _ef.OUT_DIR
        _ef_stem = f"event{int(_ef_id)}"
        if st.button(f"Make figures C and D for event {int(_ef_id)}", key="ef_run", type="primary"):
            with st.spinner(f"Reading the fields and drawing the figures of event {int(_ef_id)} …"):
                try:
                    _ef_res = _ef.make_event_figures(
                        case_dir, int(_ef_id), n_snapshots=int(_ef_n), speed_max=float(_ef_vmax) or None,
                        windows=_ef_windows or None, t_first=_ef_first, step=_ef_step, data=_efd)
                    for _note in _ef_res["notes"]:
                        st.warning(_note)
                except Exception as _err:
                    st.error(f"Event figures failed: {_err}")

        _ef_files = {k: _ef_out / f"{_ef_stem}_{k}" for k in ("snapshots", "section")}
        if all(p.with_suffix(".png").exists() for p in _ef_files.values()):
            _info = _ef_out / f"{_ef_stem}_info.txt"
            if _info.exists():
                st.code(_info.read_text(encoding="utf-8"), language="text")
            _f1, _f2 = st.columns([1, 1])
            for _col, (_k, _p), _title in zip((_f1, _f2), _ef_files.items(),
                                             ("Figure C — snapshots of the event",
                                              "Figure D — the overtopping section")):
                with _col:
                    st.image(str(_p.with_suffix(".png")), caption=f"{_title} ({_p.name})", width="stretch")
                    _d1, _d2 = st.columns(2)
                    for _dc, _ext, _mime in ((_d1, "pdf", "application/pdf"), (_d2, "png", "image/png")):
                        _fp = _p.with_suffix(f".{_ext}")
                        if _fp.exists():
                            _dc.download_button(f"⬇ {_ext.upper()}", data=_fp.read_bytes(), file_name=_fp.name,
                                                mime=_mime, key=f"ef_dl_{_k}_{_ext}")
        else:
            st.caption(f"No figures yet for event {int(_ef_id)}.")
