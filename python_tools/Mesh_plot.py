import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


def _edges_with_boundary(L: float, d: float) -> np.ndarray:
    """Cell-edge positions from 0 to L with step d, always include L."""
    if d <= 0:
        return np.array([0.0, L], dtype=float)

    edges = list(np.arange(0.0, L, d, dtype=float))
    if len(edges) == 0 or abs(edges[0] - 0.0) > 1e-12:
        edges.insert(0, 0.0)
    if abs(edges[-1] - L) > 1e-12:
        edges.append(L)
    return np.array(edges, dtype=float)



def plot_refinement_bands(
    dx_target: float,
    dz_target: float,
    level_interface: int,
    level_runup: int,
    x_length: float = 1.0,
):
    """
    Schematic with 3 strips in x:
      - Top:    blockMesh (black)
      - Middle: interfaceBand refined (blue)  with level_interface
      - Bottom: runupBand refined (red)       with level_runup

    Notes:
      - A 1D-in-x schematic.
      - Shows the different Δx produced by LEVEL_INTERFACE vs LEVEL_RUNUP.
    """

    # refined spacing
    dx_int = dx_target / (2 ** int(level_interface))
    dz_int = dz_target / (2 ** int(level_interface))

    dx_run = dx_target / (2 ** int(level_runup))
    dz_run = dz_target / (2 ** int(level_runup))

    # x-edges
    x_edges_base = _edges_with_boundary(x_length, dx_target)
    x_edges_int  = _edges_with_boundary(x_length, dx_int)
    x_edges_run  = _edges_with_boundary(x_length, dx_run)

    # header text
    cells_per_m_base = 1.0 / dx_target if dx_target > 0 else np.nan
    cells_per_m_int  = 1.0 / dx_int if dx_int > 0 else np.nan
    cells_per_m_run  = 1.0 / dx_run if dx_run > 0 else np.nan

    header = (
        f"{x_length:g} m in x | "
        f"blockMesh: Δx={dx_target:.3f} m ({cells_per_m_base:.1f} cells/m), Δz={dz_target:.3f} m | "
        f"interfaceBand (L={level_interface}): Δx={dx_int:.4f} m ({cells_per_m_int:.1f} cells/m), Δz={dz_int:.4f} m | "
        f"runupBand (L={level_runup}): Δx={dx_run:.4f} m ({cells_per_m_run:.1f} cells/m), Δz={dz_run:.4f} m"
    )

    # figure
    fig, ax = plt.subplots(figsize=(12, 4.2))
    fig.subplots_adjust(top=0.80)

    fig.text(
        0.5, 0.965, header,
        ha="center", va="top", fontsize=10,
        bbox=dict(facecolor="white", edgecolor="0.7", boxstyle="round,pad=0.35", alpha=0.98)
    )

    # strip geometry (top to bottom)
    y0_base, h_base = 0.68, 0.22
    y0_int,  h_int  = 0.38, 0.22
    y0_run,  h_run  = 0.08, 0.22

    # strip borders
    ax.add_patch(Rectangle((0, y0_base), x_length, h_base, fill=False, edgecolor="black", linewidth=2.2))
    ax.add_patch(Rectangle((0, y0_int ), x_length, h_int,  fill=False, edgecolor="blue",  linewidth=1.8))
    ax.add_patch(Rectangle((0, y0_run ), x_length, h_run,  fill=False, edgecolor="red",   linewidth=1.8))

    # base vertical edges
    for x in x_edges_base:
        ax.plot([x, x], [y0_base, y0_base + h_base], color="black", lw=2.2)

    # interface vertical edges
    for x in x_edges_int:
        ax.plot([x, x], [y0_int, y0_int + h_int], color="blue", lw=0.9, alpha=0.9)

    # runup vertical edges
    for x in x_edges_run:
        ax.plot([x, x], [y0_run, y0_run + h_run], color="red", lw=0.9, alpha=0.9)

    # axes formatting
    ax.set_xlim(0, x_length)
    ax.set_ylim(0, 1.0)
    ax.set_yticks([y0_run + h_run/2, y0_int + h_int/2, y0_base + h_base/2])
    ax.set_yticklabels(["runupBand", "interfaceBand", "blockMesh"])
    ax.set_xlabel("x [m]")
    ax.grid(False)

    plt.tight_layout()
    return fig, ax

