#!/usr/bin/env python3
"""
Computes the kinetic and potential energy flux through the overtopping plane
from the sampled line data of a finished case, per output time and per event.
"""

import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import trapezoid
import pandas as pd

# User settings
CASE_DIR = Path(".")
LINE_DIR = CASE_DIR / "postProcessing" / "line"
FILE_NAME = "x01_alpha.water_p_rgh_rho_U.xy"
OV_EVENTS_FILE = CASE_DIR / "overtopping_events.csv"

CONTROL_DICT = CASE_DIR / "system" / "controlDict"
RUN_INFO = CASE_DIR / "run_info.json"

# Read Z_START and Z_END from the x01 line in system/controlDict.
# Read H_REF from SWL in run_info.json.
# Potential energy uses the height z - H_REF.

G = 9.81

# Normal for x-plane: un = Ux
POSITIVE_ONLY = True  # keep flux in the overtopping direction

# Density weighting
# True: use rho_water * alpha for water-phase energy.
# This matches the alpha weighting of overtopped volume.
# False: use the air-water mixture density without alpha weighting.
ALPHA_WEIGHTING = True
TRANSPORT_PROPERTIES = CASE_DIR / "constant" / "transportProperties"

# Energy event detection
# Detect events in KE + PE; POSITIVE_ONLY keeps positive flux.
USE_AUTO_THRESHOLDS = True

# Absolute flux thresholds used when USE_AUTO_THRESHOLDS is False
START_THRESHOLD_ABS = 50.0
END_THRESHOLD_ABS = 10.0

# Start and end thresholds as fractions of the 95th percentile of F_total
START_FRAC_OF_P95 = 0.20
END_FRAC_OF_P95 = 0.08

MIN_EVENT_DURATION = 0.05  # [s]
MIN_EVENT_ENERGY = 1e-6  # [J/m]


def read_line_z_range(control_dict: Path, set_name: str = "x01"):
    """z of the start and end points of line set 'set_name' in controlDict."""
    txt = control_dict.read_text(encoding="utf-8", errors="ignore")
    m = re.search(
        rf"\b{set_name}\s*\{{[^}}]*?start\s*\(([^)]*)\)\s*;\s*end\s*\(([^)]*)\)\s*;",
        txt, flags=re.DOTALL,
    )
    if m is None:
        raise RuntimeError(f"Line set '{set_name}' with start/end not found in {control_dict}")
    z0 = float(m.group(1).split()[2])
    z1 = float(m.group(2).split()[2])
    return z0, z1


def read_water_density(transport_properties: Path) -> float:
    """rho of the 'water' phase in constant/transportProperties (1000 if not found)."""
    try:
        txt = transport_properties.read_text(encoding="utf-8", errors="ignore")
        block = re.search(r"\bwater\s*\{([^}]*)\}", txt).group(1)
        return float(re.search(r"\brho\b[^;]*?([0-9.eE+-]+)\s*;", block).group(1))
    except Exception:
        return 1000.0


def read_swl(run_info: Path) -> float:
    """Still water level from run_info.json (0.0 if not available)."""
    try:
        return float(json.loads(run_info.read_text())["domain_grid"]["SWL"])
    except Exception:
        return 0.0


def list_time_dirs(line_dir: Path):
    if not line_dir.exists():
        raise FileNotFoundError(f"Cannot find {line_dir}")
    tdirs = [p for p in line_dir.iterdir() if p.is_dir()]
    good = []
    for p in tdirs:
        try:
            float(p.name)
            good.append(p)
        except ValueError:
            pass
    return sorted(good, key=lambda p: float(p.name))


def read_profile_xy(path_xy: Path):
    data = np.loadtxt(path_xy)
    if data.ndim != 2 or data.shape[1] < 7:
        raise ValueError(f"Unexpected format in {path_xy} (need >=7 columns). Got shape={data.shape}")
    s = data[:, 0]
    alpha = data[:, 1]
    rho = data[:, 3]
    ux = data[:, 4]
    uy = data[:, 5]
    uz = data[:, 6]
    return s, alpha, rho, ux, uy, uz


def read_overtopping_events_csv(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Missing overtopping events file: {path.resolve()}")
    df = pd.read_csv(path)
    required = ["event_id", "t_start", "t_end", "duration", "volume", "q_peak"]
    for column in required:
        if column not in df.columns:
            raise ValueError(f"{path} missing column '{column}'. Found columns: {list(df.columns)}")
    return df.sort_values("event_id").reset_index(drop=True)


def integrate_over_window(t, y, t0, t1):
    t = np.asarray(t, float)
    y = np.asarray(y, float)
    if t1 <= t0:
        return 0.0

    a = max(t0, t[0])
    b = min(t1, t[-1])
    if b <= a:
        return 0.0

    mask = (t > a) & (t < b)
    tt = np.concatenate(([a], t[mask], [b]))
    yy = np.interp(tt, t, y)
    return float(trapezoid(yy, tt))


def reconstruct_z_from_s(s, z0, z1):
    lz = abs(z1 - z0)
    if lz <= 0:
        raise ValueError("The sampling line must have different start and end heights.")
    return z0 + (s / lz) * (z1 - z0)


def integrate_flux_components(s, alpha, rho, ux, uy, uz, z0, z1, h_ref, rho_w):
    z = reconstruct_z_from_s(s, z0, z1)

    if ALPHA_WEIGHTING:
        # water phase only: density rho_w times water fraction alpha
        dens = rho_w * np.clip(alpha, 0.0, 1.0)
    else:
        # air-water mixture
        dens = rho

    un = ux
    if POSITIVE_ONLY:
        un = np.maximum(un, 0.0)

    umag2 = ux**2 + uy**2 + uz**2

    ke_integrand = 0.5 * dens * umag2 * un
    pe_integrand = dens * G * (z - h_ref) * un

    ke_flux = trapezoid(ke_integrand, z)
    pe_flux = trapezoid(pe_integrand, z)
    return ke_flux, pe_flux


def trapezoid_integral_time(t, y):
    t = np.asarray(t, float)
    y = np.asarray(y, float)
    if len(t) < 2:
        return np.zeros_like(t)
    dt = t[1:] - t[:-1]
    return np.concatenate([[0.0], np.cumsum(0.5 * (y[1:] + y[:-1]) * dt)])


def detect_events_hysteresis(t, signal, start_thr, end_thr, min_dur, min_area):
    """
    Hysteresis-based event detection on 'signal' (assumed >=0-ish).
    Event starts when signal > start_thr, ends when signal < end_thr.
    Returns list of (i0, i1) index pairs inclusive on [i0..i1].
    """
    t = np.asarray(t, float)
    x = np.asarray(signal, float)

    events = []
    in_evt = False
    i0 = None

    for i in range(len(x)):
        if not in_evt:
            if x[i] > start_thr:
                in_evt = True
                i0 = i
        elif x[i] < end_thr:
            i1 = i
            dur = t[i1] - t[i0]
            if dur >= min_dur:
                area = trapezoid(x[i0:i1 + 1], t[i0:i1 + 1])
                if area >= min_area:
                    events.append((i0, i1))
            in_evt = False
            i0 = None

    if in_evt and i0 is not None:
        i1 = len(x) - 1
        dur = t[i1] - t[i0]
        if dur >= min_dur:
            area = trapezoid(x[i0:i1 + 1], t[i0:i1 + 1])
            if area >= min_area:
                events.append((i0, i1))

    return events


def main():
    z_start, z_end = read_line_z_range(CONTROL_DICT)
    h_ref = read_swl(RUN_INFO)
    rho_w = read_water_density(TRANSPORT_PROPERTIES)
    print(f"Sampling line x01: z = {z_start:.4g} to {z_end:.4g} m; reference level (SWL) = {h_ref:.4g} m")
    print(f"Energy of the {'water phase (rho_water * alpha, rho_water = %.6g kg/m3)' % rho_w if ALPHA_WEIGHTING else 'air-water mixture (mixture density)'}")
    tdirs = list_time_dirs(LINE_DIR)
    rows = []
    for td in tdirs:
        path_xy = td / FILE_NAME
        if not path_xy.exists():
            continue
        s, alpha, rho, ux, uy, uz = read_profile_xy(path_xy)
        ke, pe = integrate_flux_components(s, alpha, rho, ux, uy, uz, z_start, z_end, h_ref, rho_w)
        total = ke + pe
        rows.append([float(td.name), ke, pe, total])

    if not rows:
        raise RuntimeError(f"No files found matching {FILE_NAME} under {LINE_DIR}/<time>/")

    data = np.array(rows, float)
    data = data[np.argsort(data[:, 0])]

    t = data[:, 0]
    ke = data[:, 1]
    pe = data[:, 2]
    total = data[:, 3]

    np.savetxt(
        "energyFlux_components.dat",
        np.column_stack([t, ke, pe, total]),
        header="time  KE_flux  PE_flux  Total_flux(KE+PE)",
        comments="",
    )

    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
    axes[0].plot(t, ke, color="tab:blue")
    axes[0].set_ylabel("KE [W per m]")
    axes[0].set_title("Kinetic energy flux")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(t, pe, color="tab:orange")
    axes[1].set_ylabel("PE [W per m]")
    axes[1].set_title("Potential energy flux")
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(t, total, color="tab:green")
    axes[2].set_xlabel("Time [s]")
    axes[2].set_ylabel("Total [W per m]")
    axes[2].set_title("Total energy flux (KE+PE)")
    axes[2].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig("energyFlux_components.png", dpi=200)
    plt.close(fig)

    cum_ke = trapezoid_integral_time(t, ke)
    cum_pe = trapezoid_integral_time(t, pe)
    cum_total = trapezoid_integral_time(t, total)

    plt.figure()
    plt.plot(t, cum_ke, label="Cumulative KE")
    plt.plot(t, cum_pe, label="Cumulative PE")
    plt.plot(t, cum_total, label="Cumulative (KE+PE)")
    plt.xlabel("Time [s]")
    plt.ylabel("Cumulative energy [J per m]")
    plt.title("Cumulative transported energy (KE+PE)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("energyFlux_cumulative.png", dpi=200)
    plt.close()

    event_source = "overtopping_events.csv"
    start_thr = np.nan
    end_thr = np.nan

    if OV_EVENTS_FILE.exists():
        ov_events = read_overtopping_events_csv(OV_EVENTS_FILE)
        evt_rows = []
        for _, row in ov_events.iterrows():
            event_id = int(row["event_id"])
            t_start = float(row["t_start"])
            t_end = float(row["t_end"])
            duration = float(row["duration"])
            e_ke = integrate_over_window(t, ke, t_start, t_end)
            e_pe = integrate_over_window(t, pe, t_start, t_end)
            e_total = e_ke + e_pe
            evt_rows.append([event_id, t_start, t_end, duration, e_total, e_ke, e_pe])
        detected_count = len(evt_rows)
    else:
        event_source = "energy hysteresis detection"
        signal = np.maximum(total, 0.0)
        if USE_AUTO_THRESHOLDS:
            scale = np.percentile(signal, 95)
            if scale <= 0:
                start_thr = np.max(signal) * 0.2
                end_thr = np.max(signal) * 0.08
            else:
                start_thr = START_FRAC_OF_P95 * scale
                end_thr = END_FRAC_OF_P95 * scale
        else:
            start_thr = START_THRESHOLD_ABS
            end_thr = END_THRESHOLD_ABS

        events = detect_events_hysteresis(
            t,
            signal,
            start_thr=start_thr,
            end_thr=end_thr,
            min_dur=MIN_EVENT_DURATION,
            min_area=MIN_EVENT_ENERGY,
        )

        evt_rows = []
        for event_id, (i0, i1) in enumerate(events, start=1):
            tt = t[i0:i1 + 1]
            e_ke = trapezoid(ke[i0:i1 + 1], tt)
            e_pe = trapezoid(pe[i0:i1 + 1], tt)
            e_total = e_ke + e_pe
            evt_rows.append([event_id, t[i0], t[i1], t[i1] - t[i0], e_total, e_ke, e_pe])
        detected_count = len(events)

    evt_data = np.array(evt_rows, float) if evt_rows else np.zeros((0, 7), float)
    np.savetxt(
        "energyFlux_events.dat",
        evt_data,
        header="event_id  t_start  t_end  duration  E_total(KE+PE)  E_KE  E_PE",
        comments="",
    )

    if len(evt_rows) > 0:
        plt.figure()
        plt.hist(evt_data[:, 4], bins=min(30, max(5, len(evt_data))))
        plt.xlabel("Event energy (KE+PE) [J per m]")
        plt.ylabel("Count [-]")
        plt.title("Histogram of overtopping-event energies")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig("energyFlux_events_hist.png", dpi=200)
        plt.close()

    print("Saved:")
    print("  energyFlux_components.dat")
    print("  energyFlux_events.dat")
    print("  energyFlux_components.*")
    print("  energyFlux_cumulative.*")
    if len(evt_rows) > 0:
        print("  energyFlux_events_hist.*")
    print()
    print(f"Event source: {event_source}")
    if np.isfinite(start_thr) and np.isfinite(end_thr):
        print(f"Event detection thresholds (on KE+PE): start={start_thr:.6g}, end={end_thr:.6g}")
    print(f"Detected {detected_count} events.")


if __name__ == "__main__":
    main()
