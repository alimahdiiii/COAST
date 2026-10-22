"""
Sizes the 2D computational domain around the structure from the peak period and
the extent of its STL, and plots the domain box over the cross-section.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from stl import mesh
from matplotlib.collections import PolyCollection

def deepwater_L0(Tp: float, g: float = 9.81) -> float:
    return g * Tp**2 / (2.0*np.pi)

def load_stl_bbox(stl_file: str):
    m = mesh.Mesh.from_file(stl_file)
    X = m.x.flatten()
    Y = m.y.flatten()
    Z = m.z.flatten()
    bbox = {
        "xmin": float(X.min()), "xmax": float(X.max()),
        "ymin": float(Y.min()), "ymax": float(Y.max()),
        "zmin": float(Z.min()), "zmax": float(Z.max()),
    }
    return m, bbox

def compute_domain_from_stl_and_Tp(
    stl_file: str,
    Tp: float,
    g: float = 9.81,
    n_pre: float = 4.0,
    n_post: float = 0.0,
    top_clearance: float = 0.30
):
    """
    Returns: (domain_dict, stl_mesh, stl_bbox, L0)
    domain_dict keys: Dxmin, Dxmax, Dymin, Dymax, Dzmin, Dzmax
    """
    L0 = deepwater_L0(Tp, g=g)
    stl_mesh, bb = load_stl_bbox(stl_file)

    inlet_length = n_pre * L0
    outlet_length = n_post * L0

    Dxmin = bb["xmin"] - inlet_length
    Dxmax = bb["xmax"] + outlet_length
    Dzmin = bb["zmin"]
    Dzmax = bb["zmax"] + top_clearance
    Dymin = bb["ymin"]
    Dymax = bb["ymax"]

    dom = dict(Dxmin=Dxmin, Dxmax=Dxmax, Dymin=Dymin, Dymax=Dymax, Dzmin=Dzmin, Dzmax=Dzmax)
    return dom, stl_mesh, bb, L0

def plot_domain_and_stl(dom: dict, stl_mesh, title: str = "2D Domain and STL (real scale)", show=True):
    """
    Plots the domain box over the filled STL projection in X-Z.
    """
    Dxmin, Dxmax = dom["Dxmin"], dom["Dxmax"]
    Dymin, Dymax = dom["Dymin"], dom["Dymax"]
    Dzmin, Dzmax = dom["Dzmin"], dom["Dzmax"]

    tri_xz = stl_mesh.vectors[:, :, [0, 2]]  # (ntri, 3, 2)

    poly = PolyCollection(tri_xz, closed=True, alpha=0.75)
    poly.set_facecolor("lightblue")
    poly.set_edgecolor("none")

    Lx = Dxmax - Dxmin
    Lz = Dzmax - Dzmin
    aspect = Lx / max(Lz, 1e-9)

    fig_h = 5.0
    fig_w = float(np.clip(fig_h * aspect, 8.0, 20.0))

    fig, ax = plt.subplots(1, 1, figsize=(fig_w, fig_h))

    ax.plot([Dxmin, Dxmax, Dxmax, Dxmin, Dxmin],
            [Dzmin, Dzmin, Dzmax, Dzmax, Dzmin],
            'k--', linewidth=1.5)

    ax.add_collection(poly)

    ax.set_xlim(Dxmin, Dxmax)
    ax.set_ylim(Dzmin, Dzmax)
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Z [m]")
    ax.set_title(title)

    ax.set_aspect('equal', adjustable='box')

    plt.tight_layout()
    if show:
        plt.show()
    return fig, ax
    
    
    
    
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
from matplotlib.collections import PolyCollection

def plot_domain_stl_with_refinement(
    dom: dict,
    stl_mesh,
    stl_bbox: dict,
    *,
    SWL: float,
    Hs_plus: float,
    Hs_extra: float,
    x_ht: float,
    runup_top_above_stl: float,
    runup_bottom_below_stl: float = 0.0,
    runup_x_end=None,
    title: str = "Domain + STL + refinement boxes ( --- SWL )"
):
    """
    Overlay refinement boxes on the same X–Z plot:
      - interfaceBand: x=[Dxmin,Dxmax], z=[SWL-(Hs_plus+Hs_extra), SWL+(Hs_plus+Hs_extra)]
      - runupBand (STL-only): x=[clamped x_ht, stl_xmax], z=[stl_zmin-bottom, stl_zmax+top]
    """

    Dxmin, Dxmax = dom["Dxmin"], dom["Dxmax"]
    Dzmin, Dzmax = dom["Dzmin"], dom["Dzmax"]

    stl_xmin, stl_xmax = stl_bbox["xmin"], stl_bbox["xmax"]
    stl_zmin, stl_zmax = stl_bbox["zmin"], stl_bbox["zmax"]
    stl_ymin, stl_ymax = stl_bbox["ymin"], stl_bbox["ymax"]  # not plotted, but kept consistent

    # Compute the mesh refinement boxes (same logic as the snappyHexMeshDict writer)
    Hs_plus_eff = float(Hs_plus) + float(Hs_extra)

    zmin_interface = float(SWL) - Hs_plus_eff
    zmax_interface = float(SWL) + Hs_plus_eff

    x_ht_clamped = max(float(stl_xmin), min(float(x_ht), float(stl_xmax)))

    zmin_runup = float(stl_zmin) - float(runup_bottom_below_stl)
    zmax_runup = float(stl_zmax) + float(runup_top_above_stl)

    # STL filled projection in X–Z (same as plot_domain_and_stl)
    tri_xz = stl_mesh.vectors[:, :, [0, 2]]  # (ntri, 3, 2)
    poly = PolyCollection(tri_xz, closed=True, alpha=0.75)
    poly.set_facecolor("lightblue")
    poly.set_edgecolor("none")

    # Size the figure to match the domain aspect ratio.
    Lx = Dxmax - Dxmin
    Lz = Dzmax - Dzmin
    aspect = Lx / max(Lz, 1e-9)

    fig_h = 5.0
    fig_w = float(np.clip(fig_h * aspect, 8.0, 20.0))

    fig, ax = plt.subplots(1, 1, figsize=(fig_w, fig_h))

    # domain box
    ax.plot([Dxmin, Dxmax, Dxmax, Dxmin, Dxmin],
            [Dzmin, Dzmin, Dzmax, Dzmax, Dzmin],
            'k--', linewidth=1.5)

    ax.add_collection(poly)

    # SWL line
    ax.axhline(float(SWL), color="k", linestyle="--", linewidth=1)

    # interfaceBand rectangle (full x, band in z)
    ax.add_patch(
        Rectangle(
            (Dxmin, zmin_interface),
            Dxmax - Dxmin,
            zmax_interface - zmin_interface,
            fill=False,
            linewidth=2.0,
            edgecolor="blue"
        )
    )
    ax.text(Dxmin, zmax_interface, "interfaceBand", color="blue", va="bottom")

    # runupBand rectangle (STL-only z, local x)
    if runup_x_end is None:
        x_runup_end = stl_xmax
    else:
        x_runup_end = float(runup_x_end)

    if x_runup_end < x_ht_clamped:
        raise ValueError(
            f"runup_x_end ({x_runup_end}) must be >= x_ht_clamped ({x_ht_clamped})"
        )

    ax.add_patch(
        Rectangle(
            (x_ht_clamped, zmin_runup),
            x_runup_end - x_ht_clamped,
            zmax_runup - zmin_runup,
            fill=False,
            linewidth=2.0,
            edgecolor="red"
        )
    )
    ax.text(x_ht_clamped, zmax_runup, "runupBand", color="red", va="bottom")

    # cosmetics
    ax.set_xlim(Dxmin, Dxmax)
    ax.set_ylim(Dzmin, Dzmax)
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Z [m]")
    ax.set_title(title)
    ax.set_aspect('equal', adjustable='box')
    ax.grid(False)

    plt.tight_layout()
    return fig, ax




import matplotlib.pyplot as plt

def plot_domain_stl_with_lines(dom, stl_mesh, x_lines, Dzmin=None, Dzmax=None,
                               title="Domain + STL + probe lines"):
    fig, ax = plot_domain_and_stl(dom, stl_mesh, title=title, show=False)

    if Dzmin is None: Dzmin = dom["Dzmin"]
    if Dzmax is None: Dzmax = dom["Dzmax"]

    for i, x in enumerate(x_lines, start=1):
        name = f"x{i:02d}"
        ax.vlines(x, Dzmin, Dzmax,
                  colors="red", linestyles="--",
                  linewidth=2.5, zorder=50)
        ax.text(x, Dzmax, name,
                rotation=90, va="bottom", ha="center",
                color="red", fontsize=9, zorder=50)

    plt.show()
    return fig, ax


