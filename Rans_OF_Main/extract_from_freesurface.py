#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Extracts the free-surface elevation time series at one gauge position from the
raw isoSurface output of a finished case.
"""

import os
import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt



def is_float(s: str) -> bool:
    try:
        float(s)
        return True
    except ValueError:
        return False


def extract_one_gauge(
    case_root: Path,
    x_gauge: float,
    y_target: float = 0.0,
    x_tol: float = 0.01,
    y_tol: float = 0.6,
    z_swl: float = 0.0,
    base_dir: str = "postProcessing/freeSurface",
    fs_file_name: str = "alpha.water_fs.raw",
) -> Path:
    """
    Extract η(t) at a given gauge and save it in Wave_Analysis/data.

    Returns
    -------
    out_txt : Path
        Path to the generated text file (time eta).
    """
    base_path = case_root / base_dir
    if not base_path.is_dir():
        raise FileNotFoundError(f"{base_path} not found. Run this in the case folder.")

    # Directories for Wave_Analysis
    wa_dir = case_root / "Wave_Analysis"
    data_dir = wa_dir / "data"
    fig_dir = wa_dir / "output"
    data_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    # Name files using the gauge value
    safe_x = f"{x_gauge:.2f}".replace(".", "p")   # keep the '-' sign
    out_txt = data_dir / f"eta_from_freeSurface_x_{safe_x}m.txt"
    out_fig = fig_dir / f"eta_vs_time_x_{safe_x}m.png"

    # Detect numeric time folders
    all_entries = os.listdir(base_path)
    time_folders = [d for d in all_entries if is_float(d)]
    time_folders = sorted(time_folders, key=lambda x: float(x))

    if not time_folders:
        raise RuntimeError(f"No numeric time folders found in {base_path}")

    print(f"[x={x_gauge}] Time folders: {time_folders[:5]}"
          f"{' ...' if len(time_folders) > 5 else ''}")

    times = []
    etas = []

    for tname in time_folders:
        fpath = base_path / tname / fs_file_name
        if not fpath.is_file():
            print(f"[{tname}] {fs_file_name} not found, skipping.")
            continue

        try:
            data = np.loadtxt(fpath, comments="#")
        except Exception as e:
            print(f"[{tname}] Error reading {fpath}: {e}")
            continue

        if data.ndim == 1:
            data = data[np.newaxis, :]

        x = data[:, 0]
        y = data[:, 1]
        z = data[:, 2]

        mask = (np.abs(x - x_gauge) <= x_tol) & (np.abs(y - y_target) <= y_tol)
        z_sel = z[mask]

        if z_sel.size == 0:
            print(f"[{tname}] No interface points near gauge "
                  f"(x={x_gauge}, y={y_target}).")
            continue

        z_mean = float(z_sel.mean())
        eta = z_mean - z_swl

        times.append(float(tname))
        etas.append(eta)

    if len(times) == 0:
        raise RuntimeError(f"[x={x_gauge}] No valid gauge data found.")

    times = np.asarray(times)
    etas = np.asarray(etas)
    order = np.argsort(times)
    times = times[order]
    etas = etas[order]

    # Save to text file
    with out_txt.open("w") as f:
        f.write("# time[s]   eta[m]\n")
        for t, e in zip(times, etas):
            f.write(f"{t:.6f} {e:.6f}\n")

    print(f"[x={x_gauge}] ✅ Saved {len(times)} samples to {out_txt}")

    # Plot
    plt.figure()
    plt.plot(times, etas, marker="o", linewidth=1)
    plt.xlabel("Time [s]")
    plt.ylabel("Free-surface elevation η [m]")
    plt.title(f"Free-surface at x = {x_gauge} m, y ≈ {y_target} m")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_fig, dpi=300)
    plt.close()
    print(f"[x={x_gauge}] ✅ Figure saved as {out_fig}")

    return out_txt



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--xgauge", type=float, required=True,
                        help="Gauge location x [m]")
    parser.add_argument("--ytarget", type=float, default=0.0,
                        help="Target y position [m]")
    parser.add_argument("--xtol", type=float, default=0.01,
                        help="Tolerance in x [m]")
    parser.add_argument("--ytol", type=float, default=0.6,
                        help="Tolerance in y [m]")
    parser.add_argument("--zswl", type=float, default=0.0,
                        help="Still water level Z_SWL [m]")
    args = parser.parse_args()

    case_root = Path(".").resolve()
    extract_one_gauge(
        case_root=case_root,
        x_gauge=args.xgauge,
        y_target=args.ytarget,
        x_tol=args.xtol,
        y_tol=args.ytol,
        z_swl=args.zswl,
    )


if __name__ == "__main__":
    main()

