#!/usr/bin/env python3
from io import StringIO
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import trapezoid
import pandas as pd
from scipy.optimize import linprog


# User settings
ENERGY_FILE = Path("energyFlux_components.dat")
OV_EVENTS = Path("overtopping_events.csv")
OUT_CSV = Path("overtopping_energy_merged.csv")
PDF_BINS = 20
MIN_EVENT_VOLUME = 5e-5


def _is_float_token(tok: str) -> bool:
    try:
        float(tok)
        return True
    except Exception:
        return False


def read_energyflux_dat(path: Path) -> pd.DataFrame:
    """
    Robust numeric reader for energyFlux_components.dat even if:
      - first line is a plain header (not starting with '#')
      - there are commented lines
      - scientific notation

    Expected numeric columns:
      0 time
      1 KE_flux
      2 PE_flux
      3 Total_flux(KE+PE)
    """
    if not path.exists():
        raise FileNotFoundError(f"Missing energy file: {path.resolve()}")

    lines = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if (not stripped) or stripped.startswith("#"):
                continue
            lines.append(stripped)

    if not lines:
        raise ValueError(f"{path} is empty after removing comments/blank lines.")

    first_tokens = lines[0].split()
    if len(first_tokens) == 0:
        raise ValueError(f"{path}: first usable line is empty?")
    if not _is_float_token(first_tokens[0]):
        lines = lines[1:]

    if not lines:
        raise ValueError(f"{path}: no numeric lines found after skipping header.")

    raw = np.loadtxt(StringIO("\n".join(lines)))
    if raw.ndim == 1:
        raw = raw.reshape(1, -1)

    if raw.shape[1] != 4:
        raise ValueError(f"{path}: expected exactly 4 numeric columns, got {raw.shape[1]}")

    df = pd.DataFrame(raw, columns=["time", "KE_flux", "PE_flux", "Total_flux"])
    df = df.sort_values("time").drop_duplicates("time").reset_index(drop=True)
    return df


def read_overtopping_events_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing overtopping events file: {path.resolve()}")
    df = pd.read_csv(path)
    required = ["event_id", "t_start", "t_end", "duration", "volume", "q_peak"]
    for column in required:
        if column not in df.columns:
            raise ValueError(f"{path} missing column '{column}'. Found columns: {list(df.columns)}")
    return df


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


def max_over_window(t, y, t0, t1):
    t = np.asarray(t, float)
    y = np.asarray(y, float)
    if t1 <= t0:
        return float("nan")

    a = max(t0, t[0])
    b = min(t1, t[-1])
    if b <= a:
        return float("nan")

    ya = float(np.interp(a, t, y))
    yb = float(np.interp(b, t, y))
    mask = (t >= a) & (t <= b)
    yy = y[mask]
    if yy.size == 0:
        return max(ya, yb)
    return float(np.max(np.concatenate([yy, [ya, yb]])))


def compute_pdf_cdf(values, bins):
    values = np.asarray(values, float)
    if values.size == 0:
        raise ValueError("Cannot compute PDF/CDF of an empty dataset.")

    if values.size == 1:
        center = values.copy()
        pdf = np.array([1.0], float)
        cdf = np.array([1.0], float)
        widths = np.array([1.0], float)
        return center, pdf, cdf, widths

    pdf, edges = np.histogram(values, bins=min(bins, values.size), density=True)
    widths = np.diff(edges)
    centers = 0.5 * (edges[:-1] + edges[1:])
    cdf = np.cumsum(pdf * widths)
    return centers, pdf, cdf, widths


def fit_median_regression_line(volumes, energies):
    volumes = np.asarray(volumes, float)
    energies = np.asarray(energies, float)
    if volumes.size == 0:
        raise ValueError("Cannot fit a median regression line on an empty dataset.")
    if volumes.size == 1 or np.allclose(volumes, volumes[0]):
        slope = float(np.median(energies / np.maximum(volumes, 1e-12)))
        return slope

    n = volumes.size
    # Variables are [c, u_1..u_n, v_1..v_n] where residual = u_i - v_i.
    # For q=0.5, minimizing the pinball loss is equivalent to minimizing |residual|.
    c = np.concatenate([np.zeros(1), np.ones(n), np.ones(n)])

    a_eq = np.zeros((n, 1 + 2 * n))
    a_eq[:, 0] = volumes
    a_eq[np.arange(n), 1 + np.arange(n)] = 1.0
    a_eq[np.arange(n), 1 + n + np.arange(n)] = -1.0
    b_eq = energies

    bounds = [(None, None)] + [(0.0, None)] * (2 * n)
    result = linprog(c=c, A_eq=a_eq, b_eq=b_eq, bounds=bounds, method="highs")
    if not result.success:
        raise RuntimeError(f"Median regression failed: {result.message}")

    slope = result.x[0]
    return float(slope)


def fit_power_law(volumes, energies):
    volumes = np.asarray(volumes, float)
    energies = np.asarray(energies, float)
    mask = (volumes > 0.0) & (energies > 0.0)
    if np.count_nonzero(mask) < 2:
        return np.nan, np.nan, mask

    log_v = np.log(volumes[mask])
    log_e = np.log(energies[mask])
    alpha, log_a = np.polyfit(log_v, log_e, 1)
    a = np.exp(log_a)
    return float(a), float(alpha), mask


def main():
    ef = read_energyflux_dat(ENERGY_FILE)
    ev = read_overtopping_events_csv(OV_EVENTS)

    t = ef["time"].to_numpy()
    f_ke = ef["KE_flux"].to_numpy()
    f_pe = ef["PE_flux"].to_numpy()
    f_total = ef["Total_flux"].to_numpy()

    merged_rows = []
    for _, row in ev.iterrows():
        event_id = int(row["event_id"])
        t_start = float(row["t_start"])
        t_end = float(row["t_end"])
        duration = float(row["duration"])
        volume = float(row["volume"])
        q_peak = float(row["q_peak"])

        e_ke = integrate_over_window(t, f_ke, t_start, t_end)
        e_pe = integrate_over_window(t, f_pe, t_start, t_end)
        e_total = e_ke + e_pe
        f_total_max = max_over_window(t, f_total, t_start, t_end)
        f_total_mean = e_total / duration if duration > 0 else np.nan

        merged_rows.append(
            {
                "event_id": event_id,
                "t_start": t_start,
                "t_end": t_end,
                "duration": duration,
                "V_event": volume,
                "q_peak": q_peak,
                "E_KE": e_ke,
                "E_PE": e_pe,
                "E_total": e_total,
                "E_per_volume": e_total / volume if volume > 0 else np.nan,
                "F_total_mean": f_total_mean,
                "F_total_max": f_total_max,
            }
        )

    out = pd.DataFrame(merged_rows).sort_values("event_id").reset_index(drop=True)
    out = out[out["V_event"] > MIN_EVENT_VOLUME].reset_index(drop=True)
    if out.empty:
        raise RuntimeError(
            f"No overtopping events remain after applying the volume filter V_event > {MIN_EVENT_VOLUME:.6g}."
        )

    energy_median = float(out["E_total"].median())
    energy_per_volume_median = float(out["E_per_volume"].dropna().median()) if out["E_per_volume"].notna().any() else np.nan
    out["median_energy_threshold"] = energy_median
    out["is_strong_event"] = out["E_total"] > energy_median
    out["strength_class"] = np.where(out["is_strong_event"], "strong", "not_strong")
    median_regression_slope = fit_median_regression_line(
        out["V_event"].to_numpy(),
        out["E_total"].to_numpy(),
    )
    power_law_prefactor, power_law_exponent, power_law_mask = fit_power_law(
        out["V_event"].to_numpy(),
        out["E_total"].to_numpy(),
    )
    out["median_regression_slope"] = median_regression_slope
    out["median_regression_energy"] = median_regression_slope * out["V_event"]
    out["above_median_regression"] = out["E_total"] > out["median_regression_energy"]
    out["median_energy_per_volume_threshold"] = energy_per_volume_median
    out["above_median_energy_per_volume"] = out["E_per_volume"] > energy_per_volume_median
    out["power_law_prefactor"] = power_law_prefactor
    out["power_law_exponent"] = power_law_exponent

    out.to_csv(OUT_CSV, index=False)
    print(f"Wrote: {OUT_CSV.resolve()}  ({len(out)} events)")

    strong_mask = out["is_strong_event"].to_numpy()
    strong_count = int(strong_mask.sum())
    weak_count = int((~strong_mask).sum())

    print("Event energy summary (KE+PE):")
    print(f"  volume filter : V_event > {MIN_EVENT_VOLUME:.6g} m^3/m")
    print(f"  count         : {len(out)}")
    print(f"  median        : {energy_median:.6g} J/m")
    print(f"  min           : {out['E_total'].min():.6g} J/m")
    print(f"  max           : {out['E_total'].max():.6g} J/m")
    print(f"  strong events : {strong_count}")
    print(f"  not strong    : {weak_count}")
    print(f"  median line   : E_total = {median_regression_slope:.6g} * V")
    if np.isfinite(power_law_prefactor) and np.isfinite(power_law_exponent):
        print(f"  power law     : E_total = {power_law_prefactor:.6g} * V^{power_law_exponent:.6g}")
    if np.isfinite(energy_per_volume_median):
        print(f"  median E/V    : {energy_per_volume_median:.6g} J/m^3")

    plt.figure()
    plt.scatter(out["V_event"], out["E_KE"])
    plt.xlabel("Event volume V_i [m$^3$ per m]")
    plt.ylabel("Kinetic energy E_KE [J per m]")
    plt.title("Overtopping event volume vs kinetic energy")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("merge_EKE_vs_V.png", dpi=200)
    plt.close()

    plt.figure()
    plt.scatter(out["V_event"], out["E_PE"])
    plt.xlabel("Event volume V_i [m$^3$ per m]")
    plt.ylabel("Potential energy E_PE [J per m]")
    plt.title("Overtopping event volume vs potential energy")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("merge_EPE_vs_V.png", dpi=200)
    plt.close()

    plt.figure()
    plt.scatter(out["V_event"], out["E_total"])
    plt.xlabel("Event volume V_i [m$^3$ per m]")
    plt.ylabel("Event energy E_total = E_KE + E_PE [J per m]")
    plt.title("Overtopping event volume vs total energy (KE+PE)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("merge_Etotal_vs_V.png", dpi=200)
    plt.close()

    x_event = out["event_id"].to_numpy()
    plt.figure(figsize=(10, 5))
    plt.bar(x_event, out["E_KE"], label="E_KE")
    plt.bar(x_event, out["E_PE"], bottom=out["E_KE"], label="E_PE")
    plt.xlabel("Overtopping event id")
    plt.ylabel("Event energy [J per m]")
    plt.title("Energy of every overtopping event (KE and PE)")
    plt.grid(True, axis="y", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig("merge_event_energy_components.png", dpi=200)
    plt.close()

    colors = np.where(out["above_median_regression"], "tab:red", "tab:blue")
    v_min = float(out["V_event"].min())
    v_max = float(out["V_event"].max())
    line_x = np.array([v_min, v_max], float)
    line_y = median_regression_slope * line_x
    plt.figure(figsize=(10, 5))
    plt.scatter(out["V_event"], out["E_total"], c=colors, s=40)
    plt.plot(line_x, line_y, color="black", linestyle="--", linewidth=2, label="Median regression (q=0.5)")
    plt.xlabel("Event volume V_i [m$^3$ per m]")
    plt.ylabel("Event energy E_total [J per m]")
    plt.title("Overtopping-event energy vs volume with median regression")
    plt.grid(True, alpha=0.3)
    plt.scatter([], [], c="tab:red", s=40, label="Above median line")
    plt.scatter([], [], c="tab:blue", s=40, label="Below median line")
    plt.legend()
    plt.tight_layout()
    plt.savefig("merge_event_energy_vs_volume_median.png", dpi=200)
    plt.close()

    power_law_data = out.loc[power_law_mask].copy()
    if len(power_law_data) >= 2:
        power_colors = np.where(power_law_data["above_median_regression"], "tab:red", "tab:blue")
        x_fit = np.linspace(power_law_data["V_event"].min(), power_law_data["V_event"].max(), 200)
        y_fit = power_law_prefactor * x_fit**power_law_exponent
        plt.figure(figsize=(10, 5))
        plt.scatter(power_law_data["V_event"], power_law_data["E_total"], c=power_colors, s=40)
        plt.plot(x_fit, y_fit, color="black", linestyle="--", linewidth=2, label=rf"Power law: $E=aV^{{\alpha}}$, $\alpha={power_law_exponent:.3f}$")
        plt.xscale("log")
        plt.yscale("log")
        plt.xlabel("Event volume V_i [m$^3$ per m]")
        plt.ylabel("Event energy E_total [J per m]")
        plt.title("Power-law scaling of overtopping-event energy vs volume")
        plt.grid(True, which="both", alpha=0.3)
        plt.scatter([], [], c="tab:red", s=40, label="Above median line")
        plt.scatter([], [], c="tab:blue", s=40, label="Below median line")
        plt.legend()
        plt.tight_layout()
        plt.savefig("merge_power_law_scaling.png", dpi=200)
        plt.close()

    ranked_density = out.dropna(subset=["E_per_volume"]).sort_values("E_per_volume", ascending=False).reset_index(drop=True)
    if not ranked_density.empty:
        density_colors = np.where(ranked_density["above_median_energy_per_volume"], "tab:red", "tab:blue")
        plt.figure(figsize=(10, 5))
        x_rank = np.arange(1, len(ranked_density) + 1)
        plt.plot(x_rank, ranked_density["E_per_volume"], color="0.75", linewidth=1)
        plt.scatter(x_rank, ranked_density["E_per_volume"], c=density_colors, s=40)
        if np.isfinite(energy_per_volume_median):
            plt.axhline(energy_per_volume_median, color="black", linestyle="--", linewidth=2, label="Median $E_i/V_i$")
        plt.xlabel(r"Event rank by decreasing $E_i/V_i$")
        plt.ylabel(r"Energy per unit volume $E_i/V_i$ [J/m$^3$]")
        plt.title(r"Energy per unit volume vs event rank (sorted by decreasing $E_i/V_i$)")
        plt.grid(True, alpha=0.3)
        plt.scatter([], [], c="tab:red", s=40, label="Above median")
        plt.scatter([], [], c="tab:blue", s=40, label="Below median")
        plt.legend()
        plt.tight_layout()
        plt.savefig("merge_energy_per_volume_vs_rank.png", dpi=200)
    plt.close()

    centers, pdf, cdf, widths = compute_pdf_cdf(out["E_total"].to_numpy(), PDF_BINS)

    plt.figure()
    if len(centers) == 1:
        plt.bar(centers, pdf, width=widths, align="center")
    else:
        plt.bar(centers, pdf, width=0.9 * widths, align="center")
    plt.xlabel("Event energy E_total [J per m]")
    plt.ylabel("PDF [-]")
    plt.title("PDF of overtopping-event energies (KE+PE)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("merge_event_energy_pdf.png", dpi=200)
    plt.close()

    plt.figure()
    if len(centers) == 1:
        plt.bar(centers, cdf, width=widths, align="center")
    else:
        plt.bar(centers, cdf, width=0.9 * widths, align="center")
    plt.xlabel("Event energy E_total [J per m]")
    plt.ylabel("CDF [-]")
    plt.title("CDF of overtopping-event energies (KE+PE)")
    plt.ylim(0.0, 1.05)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("merge_event_energy_cdf.png", dpi=200)
    plt.close()

    print("Saved plots:")
    print("  merge_EKE_vs_V.*")
    print("  merge_EPE_vs_V.*")
    print("  merge_Etotal_vs_V.*")
    print("  merge_event_energy_components.*")
    print("  merge_event_energy_vs_volume_median.*")
    if len(power_law_data) >= 2:
        print("  merge_power_law_scaling.*")
    if not ranked_density.empty:
        print("  merge_energy_per_volume_vs_rank.*")
    print("  merge_event_energy_pdf.*")
    print("  merge_event_energy_cdf.*")


if __name__ == "__main__":
    main()
