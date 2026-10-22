"""
Builds the HTML summary report of a finished case.
"""

from __future__ import annotations

from pathlib import Path
import re
import shutil
import math
import numpy as np
from stl import mesh

import json, socket, datetime
import inspect

import html as html_lib


def write_run_info(case_dir):
    """
    Inspect the caller's variables and write grouped parameters into
    case_dir/run_info.json, organized by categories (meta, wave, structure, ...).
    """
    frame = inspect.currentframe().f_back
    case_dir = Path(case_dir)

    info = {
        "meta": {
            "case_name": case_dir.name,
            "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
            "machine": socket.gethostname(),
        },
        "patch_logic": {},
        "structure_geometry": {},
        "wave": {},
        "domain_grid": {},
        "control": {},
        "hpc": {},
        "porous": {},
    }

    # 1) STRUCTURE / GEOMETRY
    for name in [
        "m", "h_deep", "h", "h_t", "B_t", "h_b", "B", "tan_alpha_B",
        "cot_alpha_d", "cot_alpha_u", "A_c", "G_c", "R_c",
        "ZMIN_THICKNESS", "x_toe", "x_margin", "L_back", "t_wall",
        "x_container_start", "x_container_end","dx_foreshore_ramp","dx_toe_transition","dx_berm_connector",
    ]:
        info["structure_geometry"][name] = frame.f_locals.get(name, None)

    # 2) WAVE PARAMETERS
    for name in [
        "Hm0_deep", "Tp_deep", "Tm_deep",
        "Hm0_toe", "Tp_toe", "Tm_toe", "Tm10_toe",
        "Tp", "g", "n_pre", "n_post", "top_clearance",
    ]:
        info["wave"][name] = frame.f_locals.get(name, None)

    # 3) DOMAIN / GRID
    for name in [
        "Ly", "SWL", "dx_target", "dz_target", "Ny",
        "LEVEL_INTERFACE", "LEVEL_RUNUP",
        "Vertical_distance_plus_for_refinement_interface",
        "RUNUP_TOP_ABOVE_STL", "RUNUP_X_START_OFFSET",
        "Y_LINE", "NPOINTS",
    ]:
        info["domain_grid"][name] = frame.f_locals.get(name, None)

    # 4) CONTROL DICT
    for name in [
        "END_TIME", "DELTA_T", "WRITE_INTERVAL",
        "MAX_CO", "MAX_ALPHA_CO",
    ]:
        info["control"][name] = frame.f_locals.get(name, None)

    # 5) PATCH / CASE LOGIC
    for name in [
        "new_patch", "PATCH_NAME", "SEG_ID", "FLOW_MODEL", "case_name", "USE_COLLECTOR",
    ]:
        info["patch_logic"][name] = frame.f_locals.get(name, None)

    # 6) HPC / DECOMP
    for name in [
        "N_PROCS", "DECOMP_METHOD",
    ]:
        info["hpc"][name] = frame.f_locals.get(name, None)

    # 7) POROUS ZONES
    for name in ["USE_POROUS", "POROUS_MERGED"]:
        info["porous"][name] = frame.f_locals.get(name, None)
    for name in ["POROUS_ZONES", "POROUS_PHYSICS_DEFAULT", "POROUS_PHYSICS_OVERRIDES"]:
        val = frame.f_locals.get(name, None)
        # Convert to plain dict so json.dump can serialise it
        info["porous"][name] = dict(val) if isinstance(val, dict) else val

    out = case_dir / "run_info.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        json.dump(info, f, indent=2)

    print(f"📝 run_info.json written → {out}")


def register_figure(case_dir, relpath, caption=None):
    """
    Append a figure entry to run_info.json under info['figures'].
    relpath should be relative to the case_dir (e.g. 'figures/refinement_bands.png').
    """
    case_dir = Path(case_dir)
    info_path = case_dir / "run_info.json"

    if info_path.exists():
        with info_path.open() as f:
            info = json.load(f)
    else:
        # minimal skeleton if write_run_info was not called yet
        info = {
            "meta": {
                "case_name": case_dir.name,
                "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
                "machine": socket.gethostname(),
            },
            "patch_logic": {},
            "structure_geometry": {},
            "wave": {},
            "domain_grid": {},
            "control": {},
            "hpc": {},
        }

    figs = info.setdefault("figures", [])
    figs.append({"file": relpath, "caption": caption})

    with info_path.open("w") as f:
        json.dump(info, f, indent=2)

    print(f"🖼️ registered figure {relpath} in {info_path}")


def save_and_register_figure(case_dir, fig, filename, caption=None, subdir="figures"):
    """
    Convenience helper for notebooks:
      - saves `fig` into case_dir/subdir/filename
      - registers it in run_info.json via register_figure(...)
    """
    case_dir = Path(case_dir)
    fig_dir = case_dir / subdir
    fig_dir.mkdir(parents=True, exist_ok=True)

    fig_path = fig_dir / filename
    fig.savefig(fig_path, dpi=200, bbox_inches="tight")

    relpath = str(fig_path.relative_to(case_dir))
    register_figure(case_dir, relpath, caption)

    return fig_path


def write_html_report_pro(case_dir, html_name="run_report.html", title=None):
    """
    Professional interactive HTML report (sidebar + cards + zoom modal),
    built from run_info.json + registered figures.
    """
    case_dir = Path(case_dir)
    info_path = case_dir / "run_info.json"
    if not info_path.exists():
        raise FileNotFoundError(f"{info_path} not found, run write_run_info first")

    with info_path.open() as f:
        info = json.load(f)

    meta   = info.get("meta", {})
    geom   = info.get("structure_geometry", {})
    wave   = info.get("wave", {})
    dom    = info.get("domain_grid", {})
    ctrl   = info.get("control", {})
    patch  = info.get("patch_logic", {})
    hpc    = info.get("hpc", {})
    porous = info.get("porous", {})
    figs   = list(info.get("figures", []))

    case_name  = meta.get("case_name", case_dir.name)
    flow_model = patch.get("FLOW_MODEL", "unknown")

    if title is None:
        title = f"Numerical Setup Report – {case_name}"

    # Helpers
    def fmt(v):
        if v is None:
            return ""
        if isinstance(v, float):
            return f"{v:.6g}"
        return str(v)

    def render_table(d):
        if not d or all(v is None for v in d.values()):
            return "<p class='info-text'><b>No parameters found.</b></p>"

        rows = []
        for k, v in d.items():
            vv = fmt(v)
            rows.append(f"<tr><th>{html_lib.escape(str(k))}</th><td>{html_lib.escape(vv)}</td></tr>")

        return f"""
        <div class="card">
          <div class="table-wrap">
            <table class="param-table">
              {''.join(rows)}
            </table>
          </div>
        </div>
        """

    def extract_figs_by_keywords(fig_list, keywords, max_count=None):
        keys = [k.lower() for k in keywords]
        selected, remaining = [], []
        for f in fig_list:
            cap = (f.get("caption") or "").lower()
            fname = (f.get("file") or "").lower()
            hit = any(k in cap for k in keys) or any(k in fname for k in keys)
            if hit and (max_count is None or len(selected) < max_count):
                selected.append(f)
            else:
                remaining.append(f)
        return selected, remaining

    def render_fig_gallery(fig_list):
        if not fig_list:
            return "<p class='info-text'><b>No figures registered for this section.</b></p>"
        blocks = []
        for fdict in fig_list:
            src = fdict.get("file", "")
            cap = fdict.get("caption") or src
            blocks.append(f"""
              <figure class="figure-card">
                <div class="figure-media">
                  <img class="zoomable" src="{html_lib.escape(src)}" alt="{html_lib.escape(cap)}" />
                </div>
                <figcaption class="figure-caption"><span class="fig-label"></span><span class="fig-cap">{html_lib.escape(cap)}</span></figcaption>
              </figure>
            """)
        return f"<div class='fig-grid'>{''.join(blocks)}</div>"

    def render_porous_section(d):
        if not d or d.get("USE_POROUS") is None:
            return "<p class='info-text'><b>No porous zone data recorded.</b></p>"
        if not d.get("USE_POROUS"):
            return "<p class='info-text'>USE_POROUS = False — porous zones not active for this case.</p>"
        blocks = []
        # Scalar flags
        flat = {k: v for k, v in d.items()
                if k not in ("POROUS_ZONES", "POROUS_PHYSICS_DEFAULT", "POROUS_PHYSICS_OVERRIDES")}
        blocks.append(render_table(flat))
        # Zone thicknesses
        zones = d.get("POROUS_ZONES")
        if zones:
            blocks.append("<h3 class='subsec'>Zone configuration (name → normal thickness [m])</h3>")
            blocks.append(render_table(zones))
        # Default physics
        physics = d.get("POROUS_PHYSICS_DEFAULT")
        if physics:
            blocks.append("<h3 class='subsec'>Default porous physics parameters</h3>")
            blocks.append(render_table(physics))
        # Per-zone overrides
        overrides = d.get("POROUS_PHYSICS_OVERRIDES")
        if overrides:
            blocks.append("<h3 class='subsec'>Per-zone physics overrides</h3>")
            if overrides:
                for zone_name, zone_params in overrides.items():
                    blocks.append(f"<p class='info-text'><b>{html_lib.escape(str(zone_name))}</b></p>")
                    blocks.append(render_table(zone_params) if isinstance(zone_params, dict) else f"<p class='info-text'>{html_lib.escape(str(zone_params))}</p>")
            else:
                blocks.append("<p class='info-text'>No per-zone overrides defined.</p>")
        return "".join(blocks)

    # Route figures into sections
    geom_figs, figs = extract_figs_by_keywords(figs, ["clash", "geometry", "schematic", "structure"], max_count=2)

    domain_layout_figs, figs = extract_figs_by_keywords(figs, ["domain", "stl", "layout"], max_count=None)
    refine_figs, figs = extract_figs_by_keywords(figs, ["refinement", "mesh", "band", "snappy", "levels"], max_count=None)
    sampling_figs, figs = extract_figs_by_keywords(figs, ["probe", "sampling", "line", "gauge"], max_count=None)

    patch_figs, figs = extract_figs_by_keywords(figs, ["segment", "extracted", "patch", "stl xz"], max_count=None)

    # Any figures not matched above go to "Appendix"
    appendix_figs = figs

    # Executive summary cards
    Hm0 = wave.get("Hm0_toe", wave.get("Hm0_deep", None))
    Tp  = wave.get("Tp_deep", wave.get("Tp_toe", None))
    SWL = dom.get("SWL", None)

    dx  = dom.get("dx_target", None)
    dz  = dom.get("dz_target", None)

    Ny    = dom.get("Ny", None)
    nproc = hpc.get("N_PROCS", None)

    END_TIME = ctrl.get("END_TIME", None)
    DT       = ctrl.get("DELTA_T", None)

    patch_filtered = {k: v for k, v in patch.items() if k not in ("case_name", "FLOW_MODEL")}

    # Title-page meta
    ts = str(meta.get("timestamp", ""))
    report_date = ts[:10] if len(ts) >= 10 else ts
    machine = str(meta.get("machine", ""))

   # Logos (robust): copy into case_dir so HTML always finds them
    assets_dir = case_dir / "report_assets"
    assets_dir.mkdir(parents=True, exist_ok=True)

   # case_report.py is inside .../python_tools/
    tools_dir = Path(__file__).resolve().parent
    src_coria = tools_dir / "figures_DIR" / "coria_logo.png"
    src_ur    = tools_dir / "figures_DIR" / "ur_logo.png"

    dst_coria = assets_dir / "coria_logo.png"
    dst_ur    = assets_dir / "ur_logo.png"

    if src_coria.exists():
        shutil.copyfile(src_coria, dst_coria)
    else:
        print(f"WARNING: CORIA logo not found at {src_coria}")

    if src_ur.exists():
        shutil.copyfile(src_ur, dst_ur)
    else:
        print(f"WARNING: UR logo not found at {src_ur}")

    # Use paths relative to the HTML file (which is written in case_dir)
    coria_logo = str(dst_coria.relative_to(case_dir))
    ur_logo    = str(dst_ur.relative_to(case_dir))


    patch_segment_block = ""
    if patch_figs:
        patch_segment_block = f"""
        <h3 class="subsec">6.1 Patch / segment definition</h3>
        {render_fig_gallery(patch_figs)}
        """

    case_name_safe = html_lib.escape(str(case_name))
    title_safe     = html_lib.escape(str(title))

    # In this f-string, every CSS/JS brace must be doubled: {{ and }}
    html_text = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{title_safe}</title>

  <style>
    :root {{
      /* Screen UI */
      --font-sans: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Arial, "Noto Sans", "Liberation Sans", sans-serif;
      --font-mono: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace;

      --bg: #f6f7fb;
      --bg-elev: #ffffff;
      --bg-elev-2: #fbfbfe;
      --fg: #0b1220;
      --fg-muted: #4b5563;
      --fg-soft: #6b7280;

      --border: rgba(15, 23, 42, 0.12);
      --border-strong: rgba(15, 23, 42, 0.18);

      --shadow: 0 10px 30px rgba(2, 6, 23, 0.06);
      --shadow-strong: 0 20px 60px rgba(2, 6, 23, 0.10);

      --accent: #2563eb;
      --accent-2: #0ea5e9;
      --accent-soft: rgba(37, 99, 235, 0.10);
      --accent-soft-2: rgba(14, 165, 233, 0.10);

      --radius-xl: 18px;
      --radius-lg: 14px;

      --sidebar-w: 304px;
      --topbar-h: 56px;
    }}

    [data-theme="dark"] {{
      --bg: #060812;
      --bg-elev: #0b1020;
      --bg-elev-2: #0a0f1f;
      --fg: #eef2ff;
      --fg-muted: rgba(238, 242, 255, 0.72);
      --fg-soft: rgba(238, 242, 255, 0.58);

      --border: rgba(148, 163, 184, 0.18);
      --border-strong: rgba(148, 163, 184, 0.26);

      --shadow: 0 10px 30px rgba(0, 0, 0, 0.45);
      --shadow-strong: 0 20px 70px rgba(0, 0, 0, 0.62);

      --accent: #38bdf8;
      --accent-2: #a78bfa;
      --accent-soft: rgba(56, 189, 248, 0.12);
      --accent-soft-2: rgba(167, 139, 250, 0.10);
    }}

    * {{ box-sizing: border-box; }}
    html, body {{ height: 100%; }}
    body {{
      margin: 0;
      font-family: var(--font-sans);
      color: var(--fg);
      background:
        radial-gradient(1200px 600px at 20% -10%, rgba(37, 99, 235, 0.12), transparent 55%),
        radial-gradient(900px 500px at 90% 10%, rgba(14, 165, 233, 0.10), transparent 60%),
        radial-gradient(900px 500px at 30% 120%, rgba(167, 139, 250, 0.10), transparent 65%),
        var(--bg);
      line-height: 1.55;
      letter-spacing: 0.01em;
      -webkit-font-smoothing: antialiased;
      text-rendering: optimizeLegibility;
    }}

    .page {{
      display: grid;
      grid-template-columns: var(--sidebar-w) 1fr;
      min-height: 100vh;
    }}

    /* Topbar (mobile) */
    .topbar {{
      position: sticky;
      top: 0;
      z-index: 50;
      display: none;
      height: var(--topbar-h);
      align-items: center;
      justify-content: space-between;
      gap: 0.75rem;
      padding: 0 0.9rem;
      background: color-mix(in srgb, var(--bg-elev) 86%, transparent);
      backdrop-filter: blur(10px);
      border-bottom: 1px solid var(--border);
    }}
    .topbar .left {{
      display: flex;
      align-items: center;
      gap: 0.75rem;
      min-width: 0;
    }}
    .topbar .title {{
      font-weight: 700;
      font-size: 0.95rem;
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
    }}

    .icon-btn {{
      border: 1px solid var(--border);
      background: color-mix(in srgb, var(--bg-elev) 92%, transparent);
      color: var(--fg);
      height: 38px;
      width: 38px;
      border-radius: 12px;
      display: inline-flex;
      align-items: center;
      justify-content: center;
      cursor: pointer;
      box-shadow: var(--shadow);
      transition: transform 160ms ease, border-color 160ms ease, background 160ms ease;
      user-select: none;
    }}
    .icon-btn:hover {{ transform: translateY(-1px); border-color: var(--border-strong); }}
    .icon-btn:active {{ transform: translateY(0); }}
    .icon {{ width: 18px; height: 18px; display: block; }}

    /* Sidebar */
    .sidebar {{
      position: sticky;
      top: 0;
      height: 100vh;
      overflow: auto;
      padding: 1.25rem 1.1rem;
      background: linear-gradient(180deg,
        color-mix(in srgb, var(--bg-elev) 86%, transparent),
        color-mix(in srgb, var(--bg-elev-2) 86%, transparent)
      );
      border-right: 1px solid var(--border);
    }}
    .brand {{ display: flex; align-items: flex-start; justify-content: space-between; gap: 0.75rem; margin-bottom: 1rem; }}
    .brand-title {{ margin: 0; font-size: 0.78rem; font-weight: 800; letter-spacing: 0.16em; text-transform: uppercase; color: var(--fg-soft); }}
    .case-name-side {{ margin: 0.3rem 0 0.25rem 0; font-weight: 800; font-size: 1.02rem; word-break: break-word; }}
    .brand-sub {{ margin: 0; font-size: 0.85rem; color: var(--fg-muted); }}
    .nav-label {{ margin: 1.2rem 0 0.45rem 0; font-size: 0.72rem; font-weight: 800; letter-spacing: 0.18em; text-transform: uppercase; color: var(--fg-soft); }}
    .nav {{ display: flex; flex-direction: column; gap: 0.4rem; }}

    .nav-item {{
      width: 100%;
      border: 1px solid transparent;
      background: transparent;
      color: var(--fg);
      cursor: pointer;
      text-align: left;
      padding: 0.6rem 0.7rem;
      border-radius: 14px;
      display: flex;
      align-items: center;
      gap: 0.55rem;
      transition: background 160ms ease, border-color 160ms ease, transform 160ms ease, color 160ms ease;
      user-select: none;
    }}
    .nav-item .dot {{
      width: 8px; height: 8px; border-radius: 99px;
      background: color-mix(in srgb, var(--fg-soft) 44%, transparent);
      flex: 0 0 auto;
    }}
    .nav-item:hover {{
      background: color-mix(in srgb, var(--accent-soft) 60%, transparent);
      border-color: var(--border);
      transform: translateY(-1px);
    }}
    .nav-item.active {{
      background: linear-gradient(135deg, var(--accent-soft), var(--accent-soft-2));
      border-color: color-mix(in srgb, var(--accent) 26%, var(--border));
      color: var(--accent);
    }}
    .nav-item.active .dot {{ background: var(--accent); }}

    .sidebar-actions {{ display: flex; gap: 0.5rem; margin-top: 1rem; }}
    .pill {{
      flex: 1;
      border: 1px solid var(--border);
      background: color-mix(in srgb, var(--bg-elev) 92%, transparent);
      border-radius: 999px;
      padding: 0.45rem 0.7rem;
      font-size: 0.82rem;
      color: var(--fg);
      display: inline-flex;
      align-items: center;
      justify-content: center;
      gap: 0.45rem;
      cursor: pointer;
      box-shadow: var(--shadow);
      transition: transform 160ms ease, border-color 160ms ease, background 160ms ease;
      user-select: none;
    }}
    .pill:hover {{ transform: translateY(-1px); border-color: var(--border-strong); }}
    .pill:active {{ transform: translateY(0); }}

    /* Content */
    .content {{
      min-width: 0;
      padding: 2.1rem 2.4rem 3.2rem 2.4rem;
    }}

    header.page-header {{
      display: flex;
      flex-wrap: wrap;
      align-items: flex-end;
      justify-content: space-between;
      gap: 0.9rem;
      margin-bottom: 1.0rem;
    }}
    header.page-header h1 {{
      margin: 0;
      font-size: clamp(1.35rem, 2.1vw, 2.05rem);
      letter-spacing: -0.015em;
      line-height: 1.15;
    }}
    .subtitle {{
      margin-top: 0.55rem;
      display: flex;
      flex-wrap: wrap;
      gap: 0.5rem;
      color: var(--fg-muted);
      font-size: 0.93rem;
    }}
    .case-chip {{
      border: 1px solid var(--border);
      background: color-mix(in srgb, var(--bg-elev) 92%, transparent);
      color: var(--fg);
      padding: 0.25rem 0.6rem;
      border-radius: 999px;
      font-size: 0.82rem;
      display: inline-flex;
      align-items: center;
      gap: 0.35rem;
      box-shadow: var(--shadow);
      white-space: nowrap;
    }}

    /* Title page */
    .title-page {{
      background: color-mix(in srgb, var(--bg-elev) 92%, transparent);
      border: 1px solid var(--border);
      border-radius: var(--radius-xl);
      box-shadow: var(--shadow);
      padding: 1.4rem 1.4rem 1.3rem 1.4rem;
      margin-bottom: 1.2rem;
      overflow: clip;
    }}
    .title-logos {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      align-items: center;
      gap: 1rem;
      margin-bottom: 1.2rem;
    }}
    .title-logos img {{
      max-height: 58px;
      width: auto;
      display: block;
      object-fit: contain;
    }}
    .title-logos .right {{ justify-self: end; }}
    .title-logos .left {{ justify-self: start; }}
    .title-block h1 {{
      margin: 0 0 0.35rem 0;
      font-size: 1.85rem;
      letter-spacing: -0.02em;
      line-height: 1.15;
    }}
    .title-block .case {{
      margin: 0.1rem 0 0.65rem 0;
      font-weight: 800;
      color: var(--fg);
    }}
    .title-block .affil {{
      margin: 0.2rem 0 0.2rem 0;
      color: var(--fg-muted);
    }}
    .title-meta {{
      margin-top: 0.9rem;
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
      gap: 0.7rem;
    }}
    .meta-item {{
      border: 1px solid var(--border);
      border-radius: var(--radius-lg);
      background: color-mix(in srgb, var(--bg-elev-2) 92%, transparent);
      padding: 0.65rem 0.75rem;
    }}
    .meta-item .k {{
      font-size: 0.74rem;
      font-weight: 800;
      letter-spacing: 0.18em;
      text-transform: uppercase;
      color: var(--fg-soft);
    }}
    .meta-item .v {{
      margin-top: 0.25rem;
      font-family: var(--font-mono);
      color: var(--fg-muted);
      font-size: 0.93rem;
      overflow-wrap: anywhere;
    }}

    /* Sections (interactive) */
    .content-section {{
      display: none;
      margin: 1.1rem 0 2.2rem 0;
    }}
    .content-section h2 {{
      margin: 0 0 0.6rem 0;
      font-size: 1.25rem;
      letter-spacing: -0.01em;
      padding-bottom: 0.55rem;
      position: relative;
    }}
    .content-section h2::after {{
      content: "";
      position: absolute;
      left: 0;
      bottom: 0;
      width: 72px;
      height: 3px;
      background: linear-gradient(90deg, var(--accent), var(--accent-2));
      border-radius: 99px;
      opacity: 0.85;
    }}
    .content-section h3 {{
      margin: 1.0rem 0 0.45rem 0;
      font-size: 1.02rem;
      letter-spacing: -0.01em;
      color: var(--fg);
    }}

    .info-text {{
      margin: 0.35rem 0 0.8rem 0;
      font-size: 0.95rem;
      color: var(--fg-muted);
      max-width: 84ch;
    }}

    .card {{
      background: color-mix(in srgb, var(--bg-elev) 92%, transparent);
      border: 1px solid var(--border);
      border-radius: var(--radius-xl);
      box-shadow: var(--shadow);
      padding: 1rem 1.05rem;
      margin: 0.9rem 0;
      overflow: clip;
    }}

    .summary-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(230px, 1fr));
      gap: 0.9rem;
      margin-top: 0.9rem;
    }}
    .kpi {{
      background:
        radial-gradient(600px 240px at 10% 10%, color-mix(in srgb, var(--accent) 16%, transparent), transparent 55%),
        radial-gradient(560px 220px at 100% 0%, color-mix(in srgb, var(--accent-2) 12%, transparent), transparent 60%),
        color-mix(in srgb, var(--bg-elev) 94%, transparent);
      border: 1px solid var(--border);
      border-radius: var(--radius-xl);
      box-shadow: var(--shadow);
      padding: 0.95rem 1.05rem;
      position: relative;
      overflow: hidden;
    }}
    .kpi .label {{
      font-size: 0.74rem;
      font-weight: 800;
      letter-spacing: 0.18em;
      text-transform: uppercase;
      color: var(--fg-soft);
    }}
    .kpi .value {{
      margin-top: 0.32rem;
      font-size: 1.2rem;
      font-weight: 850;
      letter-spacing: -0.01em;
      color: var(--fg);
    }}

    /* Tables */
    .table-wrap {{ overflow: auto; -webkit-overflow-scrolling: touch; border-radius: var(--radius-lg); }}
    .param-table {{
      width: 100%;
      border-collapse: separate;
      border-spacing: 0;
      font-size: 0.92rem;
      color: var(--fg);
      min-width: 520px;
    }}
    .param-table th,
    .param-table td {{
      padding: 0.55rem 0.7rem;
      border-bottom: 1px solid var(--border);
      vertical-align: top;
      white-space: nowrap;
    }}
    .param-table th {{
      font-weight: 800;
      color: var(--fg);
      background: color-mix(in srgb, var(--bg-elev-2) 92%, transparent);
      border-right: 1px solid var(--border);
      position: sticky;
      left: 0;
      z-index: 1;
    }}
    .param-table td {{
      color: var(--fg-muted);
      font-family: var(--font-mono);
      white-space: nowrap;
    }}
    .param-table tr:nth-child(even) td {{
      background: color-mix(in srgb, var(--bg-elev-2) 92%, transparent);
    }}

    /* Figures (screen) */
    .fig-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
      gap: 0.95rem;
      margin-top: 0.7rem;
    }}
    .figure-card {{
      margin: 0;
      border-radius: var(--radius-xl);
      border: 1px solid var(--border);
      background: color-mix(in srgb, var(--bg-elev) 92%, transparent);
      box-shadow: var(--shadow);
      overflow: clip;
      transition: transform 180ms ease, box-shadow 180ms ease, border-color 180ms ease;
    }}
    .figure-card:hover {{
      transform: translateY(-2px);
      box-shadow: 0 20px 60px rgba(2, 6, 23, 0.10);
      border-color: var(--border-strong);
    }}
    .figure-media {{
      position: relative;
      padding: 0.7rem;
      background:
        radial-gradient(800px 300px at 10% 0%, color-mix(in srgb, var(--accent) 10%, transparent), transparent 60%),
        radial-gradient(700px 260px at 100% 10%, color-mix(in srgb, var(--accent-2) 9%, transparent), transparent 60%),
        transparent;
    }}
    img.zoomable {{
      width: 100%;
      height: auto;
      display: block;
      border-radius: var(--radius-lg);
      cursor: zoom-in;
      background: color-mix(in srgb, var(--bg-elev-2) 92%, transparent);
      border: 1px solid var(--border);
      transition: transform 200ms ease, filter 200ms ease;
    }}
    .figure-caption {{
      margin: 0;
      padding: 0.75rem 0.9rem 0.9rem 0.9rem;
      font-size: 0.86rem;
      color: var(--fg-muted);
      border-top: 1px solid var(--border);
      line-height: 1.4;
    }}
    .fig-label {{ display: none; }}

    /* Modal */
    .img-modal {{
      display: none;
      position: fixed;
      z-index: 9999;
      inset: 0;
      background: rgba(0,0,0,0.72);
      backdrop-filter: blur(10px);
      padding: 1rem;
      align-items: center;
      justify-content: center;
    }}
    .img-modal.open {{ display: flex; }}
    .modal-content {{
      width: min(1100px, 96vw);
      height: min(90vh, 820px);
      background: color-mix(in srgb, var(--bg-elev) 92%, transparent);
      border: 1px solid var(--border-strong);
      border-radius: 18px;
      box-shadow: 0 20px 70px rgba(0, 0, 0, 0.62);
      display: grid;
      grid-template-rows: auto 1fr auto;
      overflow: hidden;
      position: relative;
    }}
    .modal-top {{
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 0.6rem;
      padding: 0.7rem 0.85rem;
      border-bottom: 1px solid var(--border);
      background: color-mix(in srgb, var(--bg-elev-2) 92%, transparent);
    }}
    .modal-title {{
      font-size: 0.9rem;
      font-weight: 800;
      color: var(--fg);
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
      margin: 0;
    }}
    .modal-actions {{
      display: inline-flex;
      gap: 0.45rem;
      align-items: center;
      flex: 0 0 auto;
    }}
    .mini-btn {{
      border: 1px solid var(--border);
      background: color-mix(in srgb, var(--bg-elev) 94%, transparent);
      color: var(--fg);
      height: 34px;
      min-width: 34px;
      padding: 0 0.55rem;
      border-radius: 12px;
      cursor: pointer;
      box-shadow: var(--shadow);
      transition: transform 160ms ease, border-color 160ms ease, background 160ms ease;
      display: inline-flex;
      align-items: center;
      justify-content: center;
      gap: 0.35rem;
      user-select: none;
      font-weight: 800;
      font-size: 0.88rem;
    }}
    .mini-btn:hover {{ transform: translateY(-1px); border-color: var(--border-strong); }}
    .mini-btn:active {{ transform: translateY(0); }}
    .modal-img-wrapper {{
      position: relative;
      overflow: auto;
      display: grid;
      place-items: center;
    }}
    #modal-img {{
      max-width: 96%;
      max-height: 96%;
      transform-origin: center center;
      border-radius: 14px;
      border: 1px solid var(--border);
      background: color-mix(in srgb, var(--bg-elev-2) 92%, transparent);
      box-shadow: var(--shadow);
    }}
    .zoom-controls {{
      display: grid;
      grid-template-columns: auto 1fr auto;
      gap: 0.7rem;
      align-items: center;
      padding: 0.7rem 0.85rem 0.85rem 0.85rem;
      border-top: 1px solid var(--border);
      background: color-mix(in srgb, var(--bg-elev-2) 92%, transparent);
      color: var(--fg);
      font-size: 0.88rem;
    }}
    .zoom-controls label {{
      font-weight: 800;
      color: var(--fg-muted);
      letter-spacing: 0.02em;
    }}
    .zoom-controls input[type=range] {{ width: 100%; accent-color: var(--accent); }}
    #zoom-value {{ font-family: var(--font-mono); color: var(--fg-muted); min-width: 60px; text-align: right; }}

    /* Floating top button */
    .fab {{
      position: fixed;
      right: 18px;
      bottom: 18px;
      z-index: 40;
      display: none;
      align-items: center;
      gap: 0.5rem;
      border: 1px solid var(--border);
      background: color-mix(in srgb, var(--bg-elev) 92%, transparent);
      color: var(--fg);
      padding: 0.65rem 0.8rem;
      border-radius: 999px;
      cursor: pointer;
      font-weight: 800;
      font-size: 0.88rem;
      user-select: none;
    }}

    /* Mobile */
    @media (max-width: 980px) {{
      .page {{ grid-template-columns: 1fr; }}
      .topbar {{ display: flex; }}
      .sidebar {{
        position: fixed;
        left: 0;
        top: var(--topbar-h);
        height: calc(100vh - var(--topbar-h));
        width: min(92vw, var(--sidebar-w));
        transform: translateX(-110%);
        transition: transform 180ms ease;
        z-index: 60;
        border-right: 1px solid var(--border);
      }}
      .sidebar.open {{ transform: translateX(0); }}
      .scrim {{
        position: fixed;
        inset: var(--topbar-h) 0 0 0;
        background: rgba(0,0,0,0.35);
        display: none;
        z-index: 55;
      }}
      .scrim.open {{ display: block; }}
      .content {{ padding: 1.25rem 1.05rem 2.4rem 1.05rem; }}
      .param-table {{ min-width: 460px; }}
    }}
    
    .print-header, .print-footer {{
      display: none;
    }}
    /* ----------------------------
       PRINT: scientific report style
       ---------------------------- */
    @media print {{
      @page {{
        size: A4;
        margin: 18mm 16mm 18mm 16mm;
      }}

      * {{
        -webkit-print-color-adjust: exact;
        print-color-adjust: exact;
      }}

      html, body {{
        background: #ffffff !important;
        color: #000000 !important;
      }}

      body {{
        font-family: "Times New Roman", Times, serif !important;
        font-size: 11pt !important;
        line-height: 1.35 !important;
        letter-spacing: 0 !important;
      }}

      .topbar, .sidebar, .scrim, .fab, #img-modal, .modal-actions, .icon-btn, .pill {{
        display: none !important;
      }}

      .page {{
        display: block !important;
        min-height: auto !important;
      }}

      .content {{
        padding: 0 !important;
      }}

      .print-header, .print-footer {{
        position: fixed;
        left: 0;
        right: 0;
        color: #000;
        font-size: 9pt;
      }}
      .print-header {{
        top: 0;
        padding-bottom: 3mm;
        border-bottom: 0.6pt solid #000;
      }}
      .print-footer {{
        bottom: 0;
        padding-top: 3mm;
        border-top: 0.6pt solid #000;
      }}
      .print-header .row,
      .print-footer .row {{
        display: grid;
        grid-template-columns: 1fr auto 1fr;
        align-items: center;
      }}
      .print-header .left,
      .print-footer .left {{ justify-self: start; }}
      .print-header .center,
      .print-footer .center {{ justify-self: center; font-style: italic; }}
      .print-header .right,
      .print-footer .right {{ justify-self: end; }}
      .page-num::after {{ content: counter(page); }}

      .print-body {{
        margin-top: 12mm;
        margin-bottom: 12mm;
        counter-reset: fig;
      }}

      .title-page {{
        border: none !important;
        box-shadow: none !important;
        border-radius: 0 !important;
        background: #fff !important;
        padding: 0 !important;
        margin: 0 !important;
        min-height: calc(100vh - 24mm);
        display: flex;
        flex-direction: column;
        justify-content: flex-start;
        page-break-after: always;
        break-after: page;
      }}
      .title-logos {{
        margin: 0 0 12mm 0 !important;
      }}
      .title-logos img {{
        max-height: 34mm !important;
      }}
      .title-block h1 {{
        font-size: 22pt !important;
        margin: 0 0 6mm 0 !important;
        font-weight: bold !important;
      }}
      .title-block .case {{
        font-size: 12pt !important;
        margin: 0 0 4mm 0 !important;
      }}
      .title-block .affil {{
        font-size: 11pt !important;
        margin: 0 0 2mm 0 !important;
      }}
      .title-meta {{
        margin-top: 8mm !important;
      }}
      .meta-item {{
        border: 0.6pt solid #000 !important;
        border-radius: 0 !important;
        background: #fff !important;
        padding: 3mm 4mm !important;
      }}
      .meta-item .k {{
        font-family: Arial, Helvetica, sans-serif !important;
        font-size: 8.5pt !important;
        letter-spacing: 0.06em !important;
      }}
      .meta-item .v {{
        font-family: "Times New Roman", Times, serif !important;
        font-size: 10.5pt !important;
        color: #000 !important;
      }}

      .content-section {{
        display: block !important;
        margin: 0 0 10mm 0 !important;
        page-break-before: always;
        break-before: page;
      }}

      .content-section h2 {{
        font-size: 14pt !important;
        margin: 0 0 4mm 0 !important;
        padding: 0 0 2mm 0 !important;
        border: none !important;
      }}
      .content-section h2::after {{
        content: "";
        display: block;
        width: 100%;
        height: 0;
        border-bottom: 0.8pt solid #000;
        margin-top: 2mm;
        border-radius: 0 !important;
        background: none !important;
        opacity: 1 !important;
      }}
      .content-section h3 {{
        font-size: 12pt !important;
        margin: 6mm 0 2.5mm 0 !important;
        font-weight: bold !important;
      }}
      .info-text {{
        font-size: 11pt !important;
        color: #000 !important;
        margin: 0 0 4mm 0 !important;
        max-width: none !important;
      }}

      .card, .kpi {{
        border: none !important;
        border-radius: 0 !important;
        box-shadow: none !important;
        background: transparent !important;
        padding: 0 !important;
        margin: 0 0 6mm 0 !important;
      }}
      .summary-grid {{
        display: block !important;
        margin: 0 0 4mm 0 !important;
      }}
      .kpi {{
        padding: 0 !important;
        margin: 0 0 2mm 0 !important;
      }}
      .kpi .label {{
        font-family: Arial, Helvetica, sans-serif !important;
        font-size: 9pt !important;
        color: #000 !important;
        letter-spacing: 0.06em !important;
      }}
      .kpi .value {{
        font-family: "Times New Roman", Times, serif !important;
        font-size: 11pt !important;
        color: #000 !important;
        font-weight: normal !important;
        margin-top: 0.5mm !important;
      }}

      .table-wrap {{
        overflow: visible !important;
        border-radius: 0 !important;
      }}
      table.param-table {{
        width: 100% !important;
        min-width: 0 !important;
        border-collapse: collapse !important;
        font-size: 10pt !important;
        margin: 2mm 0 6mm 0 !important;
      }}
      .param-table th,
      .param-table td {{
        border: 0.6pt solid #000 !important;
        padding: 2mm 2.5mm !important;
        white-space: normal !important;
        vertical-align: top !important;
      }}
      .param-table th {{
        position: static !important;
        background: #fff !important;
        font-family: Arial, Helvetica, sans-serif !important;
        font-weight: bold !important;
        width: 38% !important;
      }}
      .param-table td {{
        font-family: "Times New Roman", Times, serif !important;
      }}
      .param-table tr:nth-child(even) td {{
        background: #fff !important;
      }}

      .fig-grid {{
        display: block !important;
        margin: 0 !important;
      }}
      figure.figure-card {{
        border: none !important;
        border-radius: 0 !important;
        box-shadow: none !important;
        background: transparent !important;
        margin: 0 0 8mm 0 !important;
        page-break-inside: avoid;
        break-inside: avoid;
        counter-increment: fig;
      }}
      .figure-media {{
        padding: 0 !important;
        background: transparent !important;
      }}
      figure.figure-card img.zoomable {{
        border: none !important;
        border-radius: 0 !important;
        box-shadow: none !important;
        cursor: default !important;
        max-width: 100% !important;
        margin: 0 auto !important;
        display: block !important;
      }}
      figcaption.figure-caption {{
        border: none !important;
        padding: 2mm 0 0 0 !important;
        margin: 0 !important;
        font-size: 10pt !important;
        color: #000 !important;
        text-align: center !important;
      }}
      figcaption.figure-caption .fig-label {{
        display: inline !important;
      }}
      figcaption.figure-caption::before {{
        content: "Figure " counter(fig) ": ";
        font-weight: bold;
      }}
      figcaption.figure-caption .fig-cap {{
        display: inline;
      }}

      h2, h3, p {{ orphans: 3; widows: 3; }}
      h2, h3 {{ page-break-after: avoid; break-after: avoid; }}
      table, figure {{ page-break-inside: avoid; break-inside: avoid; }}
    }}
  </style>
</head>

<body>
<div class="print-header">
  <div class="row">
    <div class="left"><b>Case:</b> {case_name_safe}</div>
    <div class="center">Numerical Setup Report</div>
    <div class="right"><b>Date:</b> {html_lib.escape(report_date)}</div>
  </div>
</div>
<div class="print-footer">
  <div class="row">
    <div class="left">{html_lib.escape(str(flow_model))}</div>
    <div class="center">{html_lib.escape(report_date)}</div>
    <div class="right">Page <span class="page-num"></span></div>
  </div>
</div>

<div class="topbar">
  <div class="left">
    <button class="icon-btn" id="btn-menu" aria-label="Open navigation" title="Menu">
      <svg class="icon" viewBox="0 0 24 24" fill="none" aria-hidden="true">
        <path d="M4 7h16M4 12h16M4 17h16" stroke="currentColor" stroke-width="2" stroke-linecap="round"/>
      </svg>
    </button>
    <div class="title">{title_safe}</div>
  </div>
  <div class="right">
    <button class="icon-btn" id="btn-theme" aria-label="Toggle theme" title="Toggle theme">
      <svg class="icon" viewBox="0 0 24 24" fill="none" aria-hidden="true">
        <path d="M12 3v2M12 19v2M4.22 4.22l1.42 1.42M18.36 18.36l1.42 1.42M3 12h2M19 12h2M4.22 19.78l1.42-1.42M18.36 5.64l1.42-1.42"
              stroke="currentColor" stroke-width="2" stroke-linecap="round"/>
        <path d="M12 18a6 6 0 1 0 0-12 6 6 0 0 0 0 12Z" stroke="currentColor" stroke-width="2"/>
      </svg>
    </button>
  </div>
</div>

<div class="scrim" id="scrim" aria-hidden="true"></div>

<div class="page">
  <aside class="sidebar" id="sidebar">
    <div class="brand">
      <div class="brand-block">
        <p class="brand-title">Setup report</p>
        <p class="case-name-side">{case_name_safe}</p>
        <p class="brand-sub">CLASH/EurOtop-style OpenFOAM case configuration</p>
      </div>
      <button class="icon-btn" id="btn-close" aria-label="Close navigation" title="Close" style="display:none;">
        <svg class="icon" viewBox="0 0 24 24" fill="none" aria-hidden="true">
          <path d="M6 6l12 12M18 6L6 18" stroke="currentColor" stroke-width="2" stroke-linecap="round"/>
        </svg>
      </button>
    </div>

    <div class="nav-label">Sections</div>
    <nav class="nav" aria-label="Report sections">
      <button class="nav-item active" data-target="overview"><span class="dot"></span>1 Overview</button>
      <button class="nav-item" data-target="geometry"><span class="dot"></span>2 Geometry</button>
      <button class="nav-item" data-target="waves"><span class="dot"></span>3 Wave conditions</button>
      <button class="nav-item" data-target="domain"><span class="dot"></span>4 Domain and grid</button>
      <button class="nav-item" data-target="control"><span class="dot"></span>5 Numerical control and HPC</button>
      <button class="nav-item" data-target="patch"><span class="dot"></span>6 Patch logic and sampling</button>
      <button class="nav-item" data-target="porous"><span class="dot"></span>7 Porous zones</button>
      <button class="nav-item" data-target="appendix"><span class="dot"></span>Appendix A</button>
    </nav>

    <div class="sidebar-actions">
      <button class="pill" id="btn-print" title="Print / Save as PDF">
        <span style="font-family:var(--font-mono);">PDF</span>
      </button>
      <button class="pill" id="btn-theme-side" title="Toggle theme">
        Theme
      </button>
    </div>
  </aside>

  <main class="content">
    <!-- Title Page -->
    <div class="title-page">
      <div class="title-logos">
        <img class="left" src="{html_lib.escape(coria_logo)}" alt="Coria logo" />
        <img class="right" src="{html_lib.escape(ur_logo)}" alt="University logo" />
      </div>

      <div class="title-block">
        <h1>{title_safe}</h1>
        <div class="case">Reference: <span style="font-family:var(--font-mono);">{case_name_safe}</span></div>
        <div class="affil"> Ali MAHDI  </div>
        <div class="affil"> CNRS UMR 6614 CORIA </div>
        <div class="affil">Université de Rouen Normandie</div>
        <div class="affil">ali.mahdi@coria.fr - ali.mahdi@univ-rouen.fr</div>
      </div>

      <div class="title-meta">
        <div class="meta-item">
          <div class="k">Date</div>
          <div class="v">{html_lib.escape(report_date)}</div>
        </div>
        <div class="meta-item">
          <div class="k">Flow model</div>
          <div class="v">{html_lib.escape(str(flow_model))}</div>
        </div>
        <div class="meta-item">
          <div class="k">Machine</div>
          <div class="v">{html_lib.escape(machine)}</div>
        </div>
        <div class="meta-item">
          <div class="k">Case directory</div>
          <div class="v">{html_lib.escape(str(case_dir))}</div>
        </div>
      </div>
    </div>

    <div class="print-body">
      <header class="page-header">
        <div class="title-wrap">
          <h1>{title_safe}</h1>
          <div class="subtitle">
            <span class="case-chip">Flow model: {html_lib.escape(str(flow_model))}</span>
            <span class="case-chip">Case: {case_name_safe}</span>
          </div>
        </div>
      </header>

      <section id="overview" class="content-section">
        <h2>1 Overview</h2>
        <p class="info-text">
          This document provides a structured record of the numerical configuration used to generate the OpenFOAM case.
          It consolidates the governing setup choices for geometry definition, wave conditions, computational domain and grid,
          numerical control parameters, parallel decomposition, and patch logic used for post-processing.
        </p>

        <div class="summary-grid">
          <div class="kpi"><div class="label">Significant wave height Hm0 (toe/deep)</div><div class="value">{fmt(Hm0)}</div></div>
          <div class="kpi"><div class="label">Peak period Tp</div><div class="value">{fmt(Tp)}</div></div>
          <div class="kpi"><div class="label">Still water level (SWL)</div><div class="value">{fmt(SWL)}</div></div>
          <div class="kpi"><div class="label">Grid targets (dx, dz)</div><div class="value">{fmt(dx)} / {fmt(dz)}</div></div>
          <div class="kpi"><div class="label">Time control (end time, Δt)</div><div class="value">{fmt(END_TIME)} / {fmt(DT)}</div></div>
          <div class="kpi"><div class="label">MPI processes</div><div class="value">{fmt(nproc)}</div></div>
        </div>

        <div class="card">
          <p class="info-text"><b>Metadata</b></p>
          <div class="table-wrap">
            <table class="param-table">
              <tr><th>timestamp</th><td>{html_lib.escape(str(meta.get("timestamp","")))}</td></tr>
              <tr><th>machine</th><td>{html_lib.escape(str(meta.get("machine","")))}</td></tr>
            </table>
          </div>
        </div>
      </section>

      <section id="geometry" class="content-section">
        <h2>2 Geometry</h2>
        <p class="info-text">
          This section lists the geometric parameters used to construct the CLASH-style cross-section and to generate the corresponding STL representation.
        </p>
        {render_table(geom)}
        {render_fig_gallery(geom_figs)}
      </section>

      <section id="waves" class="content-section">
        <h2>3 Wave conditions</h2>
        <p class="info-text">
          Wave parameters are reported for deep water and at the toe location, consistent with CLASH/EurOtop-style input definitions.
        </p>
        {render_table(wave)}
      </section>

      <section id="domain" class="content-section">
        <h2>4 Domain and grid</h2>
        <p class="info-text">
          This section summarizes the computational domain dimensions and the mesh design targets, including refinement specifications
          for the free-surface/interface region and the run-up/impact zone. 
        </p>
        {render_table(dom)}

        <h3 class="subsec">4.1 Domain layout</h3>
        {render_fig_gallery(domain_layout_figs)}

        <h3 class="subsec">4.2 Mesh refinement</h3>
        {render_fig_gallery(refine_figs)}
      </section>

      <section id="control" class="content-section">
        <h2>5 Numerical control and HPC</h2>
        <p class="info-text">
          Time integration and output controls are provided below, together with the parallel decomposition parameters used for HPC execution.
        </p>
        {render_table(ctrl)}
        {render_table(hpc)}
      </section>

      <section id="patch" class="content-section">
        <h2>6 Patch logic and sampling</h2>
        <p class="info-text">
          Patch/segment logic is reported for configurations that enable targeted post-processing (e.g., impact patch extraction, pressure sampling, or force evaluation).
        </p>

        {render_table(patch_filtered)}

        {patch_segment_block}

        <h3 class="subsec">6.2 Sampling configuration</h3>
        {render_fig_gallery(sampling_figs)}
      </section>

      <section id="porous" class="content-section">
        <h2>7 Porous zones</h2>
        <p class="info-text">
          Configuration of porous armour layers: zone layout, normal thickness, and resistance parameters
          (Darcy–Forchheimer / van Gent formulation used in OpenFOAM fvOptions).
        </p>
        {render_porous_section(porous)}
      </section>

      <section id="appendix" class="content-section">
        <h2>Appendix A: Additional figures</h2>
        <p class="info-text">
          This appendix contains figures that were registered in the metadata but were not automatically classified into Sections 2–6.
        </p>
        {render_fig_gallery(appendix_figs)}
      </section>
    </div>
  </main>
</div>

<button class="fab" id="btn-top" aria-label="Back to top" title="Back to top">
  ↑ Top
</button>

<div id="img-modal" class="img-modal" role="dialog" aria-modal="true" aria-label="Figure viewer">
  <div class="modal-content">
    <div class="modal-top">
      <p class="modal-title" id="modal-title">Figure</p>
      <div class="modal-actions">
        <button class="mini-btn" id="btn-zoom-out" title="Zoom out" aria-label="Zoom out">−</button>
        <button class="mini-btn" id="btn-zoom-in" title="Zoom in" aria-label="Zoom in">+</button>
        <button class="mini-btn" id="btn-zoom-reset" title="Reset zoom" aria-label="Reset zoom">Reset</button>
        <button class="mini-btn" id="btn-fullscreen" title="Fullscreen" aria-label="Fullscreen">⤢</button>
        <button class="mini-btn" id="close-modal" title="Close" aria-label="Close">✕</button>
      </div>
    </div>
    <div class="modal-img-wrapper" id="modal-img-wrap" tabindex="0">
      <img id="modal-img" src="" alt="Zoomed figure" />
    </div>
    <div class="zoom-controls">
      <label for="zoom-slider">Zoom</label>
      <input id="zoom-slider" type="range" min="1" max="6" step="0.05" value="1" />
      <span id="zoom-value">100%</span>
    </div>
  </div>
</div>

<script>
document.addEventListener("DOMContentLoaded", function () {{
  const root = document.documentElement;

  // Theme
  function getPreferredTheme() {{
    const saved = localStorage.getItem("reportTheme");
    if (saved === "light" || saved === "dark") return saved;
    return "dark";
  }}
  function applyTheme(t) {{
    root.setAttribute("data-theme", t);
    localStorage.setItem("reportTheme", t);
  }}
  function toggleTheme() {{
    const t = root.getAttribute("data-theme") === "dark" ? "light" : "dark";
    applyTheme(t);
  }}
  applyTheme(getPreferredTheme());

  const btnTheme = document.getElementById("btn-theme");
  const btnThemeSide = document.getElementById("btn-theme-side");
  if (btnTheme) btnTheme.addEventListener("click", toggleTheme);
  if (btnThemeSide) btnThemeSide.addEventListener("click", toggleTheme);

  // Print
  const btnPrint = document.getElementById("btn-print");
  if (btnPrint) btnPrint.addEventListener("click", () => window.print());

  // Tabs / navigation
  const navItems = Array.from(document.querySelectorAll(".nav-item"));
  const sections = Array.from(document.querySelectorAll(".content-section"));

  function showSection(id) {{
    sections.forEach(sec => {{
      sec.style.display = (sec.id === id) ? "block" : "none";
    }});
    window.scrollTo({{ top: 0, behavior: "smooth" }});
  }}

  navItems.forEach(btn => {{
    btn.addEventListener("click", () => {{
      const target = btn.dataset.target;
      showSection(target);
      navItems.forEach(b => b.classList.toggle("active", b === btn));
      closeSidebar();
    }});
  }});

  showSection("overview");

  // Mobile sidebar
  const sidebar = document.getElementById("sidebar");
  const scrim = document.getElementById("scrim");
  const btnMenu = document.getElementById("btn-menu");
  const btnClose = document.getElementById("btn-close");

  function openSidebar() {{
    if (!sidebar) return;
    sidebar.classList.add("open");
    if (scrim) scrim.classList.add("open");
  }}
  function closeSidebar() {{
    if (!sidebar) return;
    sidebar.classList.remove("open");
    if (scrim) scrim.classList.remove("open");
  }}

  function updateCloseVisibility() {{
    const isMobile = window.matchMedia("(max-width: 980px)").matches;
    if (btnClose) btnClose.style.display = isMobile ? "inline-flex" : "none";
    if (!isMobile) closeSidebar();
  }}

  if (btnMenu) btnMenu.addEventListener("click", openSidebar);
  if (btnClose) btnClose.addEventListener("click", closeSidebar);
  if (scrim) scrim.addEventListener("click", closeSidebar);
  window.addEventListener("resize", updateCloseVisibility);
  updateCloseVisibility();

  // Back to top
  const btnTop = document.getElementById("btn-top");
  function updateFab() {{
    if (!btnTop) return;
    const show = window.scrollY > 500;
    btnTop.style.display = show ? "inline-flex" : "none";
  }}
  window.addEventListener("scroll", updateFab, {{ passive: true }});
  updateFab();
  if (btnTop) {{
    btnTop.addEventListener("click", () => {{
      window.scrollTo({{ top: 0, behavior: "smooth" }});
    }});
  }}

  // Zoom modal
  const modal = document.getElementById("img-modal");
  const modalImg = document.getElementById("modal-img");
  const modalTitle = document.getElementById("modal-title");
  const closeBtn = document.getElementById("close-modal");
  const zoomSlider = document.getElementById("zoom-slider");
  const zoomValue = document.getElementById("zoom-value");
  const btnZoomIn = document.getElementById("btn-zoom-in");
  const btnZoomOut = document.getElementById("btn-zoom-out");
  const btnZoomReset = document.getElementById("btn-zoom-reset");
  const btnFullscreen = document.getElementById("btn-fullscreen");
  const modalWrap = document.getElementById("modal-img-wrap");

  function setScale(scale) {{
    if (!modalImg || !zoomSlider || !zoomValue) return;
    const s = Math.max(parseFloat(zoomSlider.min), Math.min(parseFloat(zoomSlider.max), scale));
    zoomSlider.value = String(s);
    modalImg.style.transform = "scale(" + s + ")";
    zoomValue.textContent = Math.round(s * 100) + "%";
  }}

  function openModalWith(src, alt) {{
    if (!modal || !modalImg) return;
    modal.classList.add("open");
    modalImg.src = src;
    modalImg.alt = alt || "Zoomed figure";
    if (modalTitle) modalTitle.textContent = alt || "Figure";
    setScale(1);
    if (modalWrap) modalWrap.focus();
    document.body.style.overflow = "hidden";
  }}

  function closeModal() {{
    if (modal) modal.classList.remove("open");
    document.body.style.overflow = "";
  }}

  document.querySelectorAll("img.zoomable").forEach(img => {{
    img.addEventListener("click", () => {{
      openModalWith(img.src, img.alt);
    }});
  }});

  if (closeBtn) closeBtn.addEventListener("click", closeModal);
  if (modal) {{
    modal.addEventListener("click", (e) => {{
      if (e.target === modal) closeModal();
    }});
  }}

  if (zoomSlider && modalImg && zoomValue) {{
    zoomSlider.addEventListener("input", () => {{
      const scale = parseFloat(zoomSlider.value) || 1;
      modalImg.style.transform = "scale(" + scale + ")";
      zoomValue.textContent = Math.round(scale * 100) + "%";
    }});
  }}

  if (btnZoomIn) btnZoomIn.addEventListener("click", () => setScale((parseFloat(zoomSlider.value) || 1) + 0.15));
  if (btnZoomOut) btnZoomOut.addEventListener("click", () => setScale((parseFloat(zoomSlider.value) || 1) - 0.15));
  if (btnZoomReset) btnZoomReset.addEventListener("click", () => setScale(1));

  async function toggleFullscreen(el) {{
    try {{
      if (!document.fullscreenElement) {{
        await (el.requestFullscreen ? el.requestFullscreen() : null);
      }} else {{
        await (document.exitFullscreen ? document.exitFullscreen() : null);
      }}
    }} catch (_) {{}}
  }}
  if (btnFullscreen && modal) {{
    btnFullscreen.addEventListener("click", () => toggleFullscreen(modal));
  }}

  document.addEventListener("keydown", (e) => {{
    if (!modal || !modal.classList.contains("open")) return;
    if (e.key === "Escape") {{
      e.preventDefault();
      closeModal();
      return;
    }}
    if (e.key === "+" || e.key === "=") {{
      e.preventDefault();
      setScale((parseFloat(zoomSlider.value) || 1) + 0.15);
      return;
    }}
    if (e.key === "-" || e.key === "_") {{
      e.preventDefault();
      setScale((parseFloat(zoomSlider.value) || 1) - 0.15);
      return;
    }}
    if (e.key === "0") {{
      e.preventDefault();
      setScale(1);
      return;
    }}
    if (e.key === "ArrowUp" || e.key === "ArrowRight") {{
      e.preventDefault();
      setScale((parseFloat(zoomSlider.value) || 1) + 0.05);
      return;
    }}
    if (e.key === "ArrowDown" || e.key === "ArrowLeft") {{
      e.preventDefault();
      setScale((parseFloat(zoomSlider.value) || 1) - 0.05);
      return;
    }}
  }});
}});
</script>

</body>
</html>
"""

    html_path = case_dir / html_name
    with html_path.open("w", encoding="utf-8") as f:
        f.write(html_text)

    print(f"📄 PRO HTML report written → {html_path}")
    return html_path

