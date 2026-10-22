# surface_on_stl.py
# Plot the free surface over the domain and projected STL triangles.
# Read surfaces from postProcessing/freeSurface/<time>/alpha.water_fs.raw.
#
# Select one y-plane to avoid duplicate points.
# Crop the surface to the computed domain limits.
#
# Output:
#   <case_root>/Wave_Analysis/figures/freeSurface_on_stl_t_<time>.png
# Optionally registers each image in run_info.json.

from __future__ import annotations
from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt

from .domain_tools import (
    compute_domain_from_stl_and_Tp,
    plot_domain_and_stl,
)

try:
    from foam_case_tools import register_figure  # optional
except Exception:
    register_figure = None


def _read_run_info(case_root: Path) -> dict:
    p = case_root / "run_info.json"
    if not p.exists():
        raise FileNotFoundError(f"Missing run_info.json in {case_root}")
    return json.loads(p.read_text(encoding="utf-8"))


def _find_single_stl(case_root: Path) -> Path:
    tri = case_root / "constant" / "triSurface"
    if not tri.exists():
        raise FileNotFoundError(f"Missing folder: {tri}")
    stls = sorted(tri.glob("*.stl"))
    if len(stls) == 0:
        raise FileNotFoundError(f"No STL found in {tri}")
    if len(stls) > 1:
        raise RuntimeError(
            f"Expected 1 STL in {tri}, found {len(stls)}: {[s.name for s in stls]}"
        )
    return stls[0]


def _list_free_surface_files(case_root: Path, surface_file: str):
    base = case_root / "postProcessing" / "freeSurface"
    if not base.exists():
        raise FileNotFoundError(f"Missing folder: {base}")

    out: list[tuple[float, Path]] = []
    for tdir in base.iterdir():
        if not tdir.is_dir():
            continue
        try:
            t = float(tdir.name)
        except ValueError:
            continue
        f = tdir / surface_file
        if f.exists():
            out.append((t, f))

    if not out:
        raise FileNotFoundError(f"No {surface_file} found in {base}/<time>/")

    out.sort(key=lambda x: x[0])
    return out


def _safe_time_tag(t: float) -> str:
    s = f"{t:.6f}".rstrip("0").rstrip(".")
    return s.replace(".", "p")


def plot_domain_and_free_surface_all_times(
    case_root: str | Path,
    *,
    surface_file: str = "alpha.water_fs.raw",
    y_plane: float = 0.0,                 # y-plane to plot
    out_subdir: str = "Wave_Analysis/figures",
    fs_linewidth: float = 2.2,
    register: bool = True,
):
    case_root = Path(case_root).resolve()
    info = _read_run_info(case_root)

    # read wave inputs from run_info.json
    wave = info["wave"]

    # Use the deep-water peak period when available.
    Tp = float(wave.get("Tp_deep", wave["Tp"]))
    g = float(wave["g"])
    n_pre = float(wave["n_pre"])
    n_post = float(wave["n_post"])
    top_clearance = float(wave["top_clearance"])

    # STL path (from the case)
    stl_path = _find_single_stl(case_root)

    # Compute the domain from the STL and wave period.
    dom, stl_mesh, stl_bbox, L0 = compute_domain_from_stl_and_Tp(
        stl_file=stl_path,
        Tp=Tp,
        g=g,
        n_pre=n_pre,
        n_post=n_post,
        top_clearance=top_clearance,
    )

    # Use the computed domain limits.
    Dxmin = float(dom["Dxmin"])
    Dxmax = float(dom["Dxmax"])

    # output folder
    fig_dir = case_root / out_subdir
    fig_dir.mkdir(parents=True, exist_ok=True)

    # list all freeSurface files
    fs_list = _list_free_surface_files(case_root, surface_file=surface_file)

    saved: list[Path] = []
    for t_sel, fs_path in fs_list:
        data = np.loadtxt(str(fs_path), comments=("#", "%"))
        if data.ndim == 1:
            data = data.reshape(1, -1)
        if data.shape[1] < 3:
            raise RuntimeError(f"Unexpected freeSurface format: {data.shape} in {fs_path}")

        x_fs = data[:, 0].astype(float)
        y_fs = data[:, 1].astype(float)
        z_fs = data[:, 2].astype(float)

        # Select one y-plane to avoid duplicate surface points.
        mY = (y_fs == float(y_plane))
        if not np.any(mY):
            # Use the nearest y-plane if there is no exact match.
            y_unique = np.unique(y_fs)
            y_pick = y_unique[np.argmin(np.abs(y_unique - float(y_plane)))]
            mY = (y_fs == y_pick)

        x_fs = x_fs[mY]
        z_fs = z_fs[mY]

        # Crop to domain x-range
        mX = (x_fs >= Dxmin) & (x_fs <= Dxmax)
        x_fs = x_fs[mX]
        z_fs = z_fs[mX]
        if x_fs.size < 2:
            continue

        # Sort points by x for the surface line.
        o = np.argsort(x_fs)
        x_fs = x_fs[o]
        z_fs = z_fs[o]

        # background plot (domain + STL)
        fig, ax = plot_domain_and_stl(dom, stl_mesh, show=False)

        # Plot the free-surface line.
        ax.plot(
            x_fs, z_fs, "-",
            color="blue",
            lw=fs_linewidth,
            alpha=0.95,
            label=f"freeSurface @ y={float(y_plane):g}, t={t_sel:.3f} s",
        )

        ax.set_xlim(Dxmin, Dxmax)
        ax.set_title(f" free-surface | t = {t_sel:.3f} s")
        ax.legend(loc="upper left", bbox_to_anchor=(0.01, 0.99), framealpha=1.0)

        tag = _safe_time_tag(t_sel)
        png_name = f"freeSurface_on_stl_t_{tag}.png"
        out_path = fig_dir / png_name

        plt.tight_layout()
        plt.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close(fig)

        saved.append(out_path)

        if register and (register_figure is not None):
            rel = str(out_path.relative_to(case_root))
            caption = f"Domain + STL with free-surface (y={float(y_plane):g}) at t={t_sel:.3f} s."
            register_figure(case_root, rel, caption)

    return saved


