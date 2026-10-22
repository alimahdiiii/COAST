#!/usr/bin/env python3
"""
Flow depth and depth-averaged flow velocity on the vertical sampling line at the
overtopping section, per output time and per overtopping event.
"""

import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.integrate import trapezoid

# User settings
H_MIN = 0.002      # [m] below this depth the velocity is not computed (layer not resolved)
N_WAVES = 0        # number of incident waves; > 0 also gives h_2% and u_2%

CASE_DIR = Path(".")
CONTROL_DICT = CASE_DIR / "system" / "controlDict"
LINE_DIR = CASE_DIR / "postProcessing" / "line"
OV_EVENTS_FILE = CASE_DIR / "overtopping_events.csv"
RUN_INFO = CASE_DIR / "run_info.json"

VECTOR_FIELDS = {"U"}


def read_line_sets(control_dict: Path):
    """Return {set_name: (x, y, zmin, zmax)} from the 'line' function of controlDict."""
    txt = control_dict.read_text(encoding="utf-8", errors="ignore")
    m = re.search(r"^\s*line\s*\{", txt, flags=re.MULTILINE)
    if m is None:
        raise RuntimeError(f"No 'line' function found in {control_dict}")
    block = txt[m.end():]
    fields_m = re.search(r"fields\s*\(([^)]*)\)", block)
    fields = fields_m.group(1).split() if fields_m else []
    sets = {}
    pat = re.compile(
        r"(\w+)\s*\{\s*type\s+uniform;.*?start\s*\(([^)]*)\)\s*;\s*end\s*\(([^)]*)\)\s*;",
        flags=re.DOTALL,
    )
    for name, start, end in pat.findall(block):
        x0, y0, z0 = map(float, start.split())
        x1, _, z1 = map(float, end.split())
        if abs(x1 - x0) > 1e-9:
            raise RuntimeError(f"Line {name} is not vertical (x {x0} -> {x1}).")
        sets[name] = (x0, y0, min(z0, z1), max(z0, z1))
    if not sets:
        raise RuntimeError(f"No uniform line sets found in the 'line' function of {control_dict}")
    return dict(sorted(sets.items())), fields


def column_map(file_stem: str, set_name: str, known_fields):
    """Column index of alpha.water and Ux in '<set>_<f1>_<f2>....xy' (raw format)."""
    rest = file_stem[len(set_name) + 1:]
    names = sorted(known_fields, key=len, reverse=True)
    order = []
    while rest:
        for f in names:
            if rest == f or rest.startswith(f + "_"):
                order.append(f)
                rest = rest[len(f) + 1:]
                break
        else:
            raise RuntimeError(f"Cannot parse field names in {file_stem}")
    col, cols = 1, {}
    for f in order:
        cols[f] = col
        col += 3 if f in VECTOR_FIELDS else 1
    return cols["alpha.water"], cols["U"]


def list_time_dirs(line_dir: Path):
    out = []
    for p in line_dir.iterdir():
        if p.is_dir():
            try:
                out.append((float(p.name), p))
            except ValueError:
                pass
    return sorted(out)


def read_line_series(set_name, zmin, fields, tdirs):
    """Time series of h, q_l, u on one line."""
    rows = []
    cols = None
    for t, d in tdirs:
        files = list(d.glob(f"{set_name}_*.xy"))
        if not files:
            continue
        if cols is None:
            cols = column_map(files[0].stem, set_name, fields)
        a = np.loadtxt(files[0], ndmin=2)
        s = a[:, 0]
        alpha = np.clip(a[:, cols[0]], 0.0, 1.0)
        ux = a[:, cols[1]]
        h = trapezoid(alpha, s)
        ql = trapezoid(alpha * ux, s)
        wet = (alpha >= 0.5) & (s > s[0])     # first point: on the wall, U = 0
        resolved = h >= H_MIN and wet.any()
        u = float(ux[wet].mean()) if resolved else np.nan
        rows.append((t, h, ql, u))
    if not rows:
        raise RuntimeError(f"No sampled files for line {set_name} under {LINE_DIR}")
    return pd.DataFrame(rows, columns=["t", "h", "q_line", "u"])


def band_cell_size():
    """Cell size in the run-up refinement band, from run_info.json (None if unknown)."""
    try:
        import json
        g = json.loads(RUN_INFO.read_text())["domain_grid"]
        return min(float(g["dx_target"]), float(g["dz_target"])) / 2 ** int(g["LEVEL_RUNUP"])
    except Exception:
        return None


def exceedance(values, n_total=None):
    """Values in decreasing order and their probability of exceedance per event.
    n_total: number of events when the smallest ones have no value (not resolved)."""
    v = np.sort(np.asarray(values, float))[::-1]
    n = len(v) if n_total is None else n_total
    p = np.arange(1, len(v) + 1) / (n + 1)
    return v, p


def max_resolved(v):
    """Largest finite value, NaN if there is none (velocity not resolved)."""
    v = v[np.isfinite(v)]
    return float(v.max()) if v.size else np.nan


def main():
    sets, fields = read_line_sets(CONTROL_DICT)
    sec = next(iter(sets))          # x01: the overtopping section
    x_sec, _, z_bot, z_top = sets[sec]
    tdirs = list_time_dirs(LINE_DIR)
    if not tdirs:
        raise RuntimeError(f"No time folders under {LINE_DIR}")

    ts = read_line_series(sec, z_bot, fields, tdirs)
    dt_line = float(np.median(np.diff(ts["t"])))
    ts.to_csv("flow_depth_velocity_timeseries.csv", index=False)

    if not OV_EVENTS_FILE.exists():
        raise FileNotFoundError("overtopping_events.csv not found: run OV_Test.py (Step 1) first.")
    ev = pd.read_csv(OV_EVENTS_FILE).sort_values("t_start").reset_index(drop=True)

    t = ts["t"].to_numpy()
    h = ts["h"].to_numpy()
    u = ts["u"].to_numpy()
    ql = ts["q_line"].to_numpy()

    rows = []
    for _, e in ev.iterrows():
        t0, t1 = float(e["t_start"]), float(e["t_end"])
        m = (t >= t0) & (t <= t1)
        n_s = int(m.sum())
        if n_s == 0:
            # event shorter than the line output interval: use the nearest sample
            m = np.abs(t - 0.5 * (t0 + t1)) <= 0.5 * dt_line
        row = {
            "event_id": int(e["event_id"]),
            "t_start": t0,
            "t_end": t1,
            "duration": float(e["duration"]),
            "V_event": float(e["volume"]),
            "n_line_samples": n_s,
            "h_max": float(h[m].max()) if m.any() else np.nan,
            "u_max": max_resolved(u[m]),
            "t_h_max": float(t[m][np.argmax(h[m])]) if m.any() else np.nan,
        }
        mm = (t >= t0 - dt_line) & (t <= t1 + dt_line)
        row["V_line"] = float(trapezoid(np.maximum(ql[mm], 0.0), t[mm])) if mm.sum() > 1 else np.nan
        rows.append(row)

    out = pd.DataFrame(rows)
    cell = band_cell_size()
    if cell:
        out["h_max_cells"] = out["h_max"] / cell
    out.to_csv("flow_depth_velocity_events.csv", index=False)

    # Summary
    print(f"Overtopping section: line {sec} at x = {x_sec:.4f} m, z = {z_bot:.4f}-{z_top:.4f} m")
    print(f"Line output interval: {dt_line:.4g} s   |   events: {len(out)}")
    few = int((out["n_line_samples"] < 3).sum())
    if few:
        print(f"WARNING: {few} of {len(out)} events have fewer than 3 line samples; "
              f"h_max and u_max of short events are underestimated. Use a finer line output interval.")
    print(f"h_max  : median {out['h_max'].median():.4g} m, largest {out['h_max'].max():.4g} m")
    if cell:
        thin = int((out["h_max"] < cell).sum())
        print(f"Cell size in the run-up band: {cell:.4g} m; {thin} of {len(out)} events have "
              f"h_max below one cell (flow depth not resolved by the mesh).")
    unres = out["u_max"].isna()
    print(f"u_max  : median {out['u_max'].median():.4g} m/s, largest {out['u_max'].max():.4g} m/s "
          f"(depth-averaged, over the {int((~unres).sum())} events with a resolved velocity)")
    if unres.any():
        print(f"{int(unres.sum())} of {len(out)} events have no resolved velocity (layer thinner than "
              f"H_MIN = {H_MIN:g} m or no line point with alpha >= 0.5); u_max is left empty for them.")
    print(f"duration: median {out['duration'].median():.4g} s, longest {out['duration'].max():.4g} s")
    ok = out["V_line"].notna() & (out["V_event"] > 0)
    if ok.any():
        r = out.loc[ok, "V_line"].sum() / out.loc[ok, "V_event"].sum()
        print(f"Volume check: sum V_line / sum V_event = {r:.3f} "
              f"(close to 1 when the overtopping zone is a single section and the line output is fine enough)")
    if N_WAVES > 0:
        k = int(np.floor(0.02 * N_WAVES))
        for c, lab in [("h_max", "h_2%"), ("u_max", "u_2%")]:
            v = np.sort(out[c].dropna().to_numpy())[::-1]
            head = f"{lab} (exceeded by 2% of {N_WAVES} incident waves): "
            if k < 1:
                print(head + "not computed (2% of N_WAVES is less than one wave)")
            elif k > len(out):
                print(head + "0 (fewer than 2% of the waves overtop)")
            elif k > len(v):
                print(head + "not resolved (the velocity of the event of that rank is not resolved)")
            else:
                print(head + f"{v[k - 1]:.4g}")

    # Figures
    fig, ax = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
    ax[0].plot(t, h, lw=0.8)
    ax[1].plot(t, u, lw=0.8)
    ax[2].plot(t, ql, lw=0.8, color="tab:green")
    for _, e in out.iterrows():
        for a in ax:
            a.axvspan(e["t_start"], e["t_end"], color="0.85", lw=0)
    ax[0].set_ylabel("h [m]")
    ax[1].set_ylabel("u [m/s]")
    ax[2].set_ylabel(f"q at {sec} [m³/s per m]")
    ax[2].set_xlabel("Time [s]")
    ax[0].set_title(f"Flow depth and depth-averaged velocity at {sec} (grey: overtopping events)")
    for a in ax:
        a.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig("fdv_fig01_timeseries.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
    ax[0].scatter(out["V_event"], out["h_max"], s=18)
    if cell:
        ax[0].axhline(cell, color="0.5", ls="--", lw=1, label=f"one cell ({100 * cell:.2f} cm)")
        ax[0].legend()
    ax[0].set_ylabel("Maximum flow depth h_max [m]")
    res = ~unres
    ax[1].scatter(out.loc[res, "V_event"], out.loc[res, "u_max"], s=18)
    if unres.any():
        ax[1].set_title(f"{int(unres.sum())} events without a resolved velocity are not shown",
                        fontsize=9)
    ax[1].set_ylabel("Maximum depth-averaged velocity u_max [m/s]")
    for a in ax:
        a.set_xscale("log")
        a.set_xlabel("Event volume V [m³ per m]")
        a.grid(True, which="both", alpha=0.3)
    fig.suptitle(f"Per-event maxima at the overtopping section ({sec})")
    fig.tight_layout()
    fig.savefig("fdv_fig02_events_vs_volume.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
    for a, c, lab in [(ax[0], "h_max", "h_max [m]"), (ax[1], "u_max", "u_max [m/s]")]:
        v, p = exceedance(out[c].dropna(), n_total=len(out))
        a.semilogy(v, p, "o", ms=3)
        if c == "u_max" and unres.any():
            a.set_title(f"{int(unres.sum())} events without a resolved velocity are not shown",
                        fontsize=9)
        a.set_xlabel(lab)
        a.set_ylabel("Probability of exceedance (per event)")
        a.grid(True, which="both", alpha=0.3)
    fig.suptitle("Exceedance of the per-event maximum flow depth and velocity")
    fig.tight_layout()
    fig.savefig("fdv_fig03_exceedance.png", dpi=200)
    plt.close(fig)

    print("Saved: flow_depth_velocity_timeseries.csv, flow_depth_velocity_events.csv, fdv_fig01-03")


if __name__ == "__main__":
    main()
