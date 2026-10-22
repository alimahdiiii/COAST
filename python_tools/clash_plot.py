"""
Draws the annotated structure cross-section: filled profile, still water level,
and the dimension and angle annotations of the geometry parameters.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Arc

# Helpers (labels attached to arrows)
def dx_from_cot(dz, cot):
    if abs(dz) < 1e-15:
        return 0.0
    cot = float(cot)
    if abs(cot) < 1e-15:
        return 0.0
    return float(dz) * cot

def v_dim(ax, x, z0, z1, label, dx=8, dy=0, fs=11):
    """Vertical arrow with label attached to arrow midpoint."""
    if z0 is None or z1 is None:
        return
    if abs(z1 - z0) < 1e-12:
        return

    ax.annotate(
        "",
        xy=(x, z1),
        xytext=(x, z0),
        arrowprops=dict(arrowstyle="<->", linewidth=1),
    )

    ax.annotate(
        label,
        xy=(x, 0.5 * (z0 + z1)),
        xytext=(dx, dy),
        textcoords="offset points",
        ha="left",
        va="center",
        fontsize=fs,
    )

def h_dim(ax, x0, x1, z, label, dx=0, dy=6, fs=11):
    """Horizontal arrow with label attached to arrow midpoint."""
    if abs(x1 - x0) < 1e-12:
        return

    ax.annotate(
        "",
        xy=(x1, z),
        xytext=(x0, z),
        arrowprops=dict(arrowstyle="<->", linewidth=1),
    )

    ax.annotate(
        label,
        xy=(0.5 * (x0 + x1), z),
        xytext=(dx, dy),
        textcoords="offset points",
        ha="center",
        va="bottom",
        fontsize=fs,
    )

def seg_dim(ax, p0, p1, label, frac=1.0, off_arrow=0.0, off_text=0.02, fs=10):
    """
    Dimension arrow along the segment p0->p1.

    frac=1.0  -> arrow spans the FULL segment (start to end)
    frac<1.0  -> arrow spans a centered fraction of the segment
    """
    x0, z0 = p0
    x1, z1 = p1

    vx, vz = (x1 - x0), (z1 - z0)
    L = (vx*vx + vz*vz) ** 0.5
    if L < 1e-12:
        return

    # unit direction along segment
    ux, uz = vx / L, vz / L

    # unit normal (rotate +90°)
    nx, nz = -uz, ux

    if frac >= 1.0:
        # full segment
        xa, za = x0, z0
        xb, zb = x1, z1
    else:
        # centered fraction of segment
        xm, zm = 0.5*(x0 + x1), 0.5*(z0 + z1)
        half = 0.5 * frac * L
        xa = xm - half*ux
        za = zm - half*uz
        xb = xm + half*ux
        zb = zm + half*uz

    # arrow offset (0 = on the slope)
    da = off_arrow * L
    xa += da*nx; za += da*nz
    xb += da*nx; zb += da*nz

    ax.annotate(
        "",
        xy=(xb, zb),
        xytext=(xa, za),
        arrowprops=dict(arrowstyle="<->", linewidth=1),
    )

    # text offset (tiny) to keep it readable
    dt = off_text * L
    ax.text(
        0.5*(xa + xb) + dt*nx,
        0.5*(za + zb) + dt*nz,
        label,
        fontsize=fs,
        ha="center",
        va="center",
        rotation=np.degrees(np.arctan2(vz, vx)),
        rotation_mode="anchor",
    )


def angle_arc(ax, p0, p1, label, arc_r, fs=11):
    """Angle arc with attached label."""
    x0, z0 = p0
    x1, z1 = p1
    ang = np.degrees(np.arctan2(z1 - z0, x1 - x0))

    theta1, theta2 = (0, ang) if ang >= 0 else (ang, 0)

    arc = Arc(
        (x0, z0),
        width=2 * arc_r,
        height=2 * arc_r,
        angle=0,
        theta1=theta1,
        theta2=theta2,
        linewidth=1,
    )
    ax.add_patch(arc)

    ax.annotate(
        label,
        xy=(x0 + arc_r * np.cos(np.radians(0.5 * ang)),
            z0 + arc_r * np.sin(np.radians(0.5 * ang))),
        xytext=(4, 4),
        textcoords="offset points",
        fontsize=fs,
    )

def plot_porous_zones_on_ax(ax, porous_data: dict, alpha: float = 0.55):
    """
    Overlay porous zone polygons on an existing geometry axis (returned by
    plot_clash_schematic).  Each zone is filled in a distinct blue shade with
    a navy edge and a legend entry.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
    porous_data : dict  — output of clash_cad.build_porous_zone_data()
    alpha : float       — fill transparency (0–1)
    """
    from matplotlib.patches import Polygon as MplPolygon

    porous_polys = porous_data.get('porous_polys', {})
    n = max(1, len(porous_polys))
    colors = [plt.cm.Blues(v) for v in np.linspace(0.45, 0.85, n)]

    for i, (zone, poly) in enumerate(porous_polys.items()):
        patch = MplPolygon(
            poly,
            closed=True,
            facecolor=colors[i],
            edgecolor='navy',
            linewidth=0.9,
            alpha=alpha,
            label=f"porous: {zone}",
            zorder=3,
        )
        ax.add_patch(patch)

    if porous_polys:
        ax.legend(loc='upper left', fontsize=9, framealpha=0.7)


def plot_layered_porous_on_ax(ax, layered_data: dict, alpha: float = 0.55):
    """Overlay multi-layer porous zone polygons with color-coded layers."""
    from matplotlib.patches import Polygon as MplPolygon

    layer_polys = layered_data.get("layer_polys", {})
    if not layer_polys:
        return

    cmaps = [plt.cm.Blues, plt.cm.Greens, plt.cm.Oranges, plt.cm.Purples, plt.cm.Reds]
    layer_configs = layered_data.get("layer_configs", [])

    # Group by layer index
    by_layer: dict[int, list] = {}
    for (li, zone), poly in layer_polys.items():
        by_layer.setdefault(li, []).append((zone, poly))

    for li, items in sorted(by_layer.items()):
        cmap = cmaps[li % len(cmaps)]
        n = max(1, len(items))
        colors = [cmap(v) for v in np.linspace(0.4, 0.85, n)]
        lname = layer_configs[li]["name"] if li < len(layer_configs) else f"layer{li}"
        for i, (zone, poly) in enumerate(items):
            patch = MplPolygon(
                poly, closed=True,
                facecolor=colors[i], edgecolor="navy",
                linewidth=0.9, alpha=alpha,
                label=f"{lname}: {zone}",
                zorder=3 + li,
            )
            ax.add_patch(patch)

    ax.legend(loc="upper left", fontsize=8, framealpha=0.7)


def plot_obstacles_on_ax(ax, abs_verts_list: list, obstacles: list = None, alpha: float = 0.6):
    """Overlay obstacle polygons on a schematic axes.

    Parameters
    ----------
    abs_verts_list : list[list[(x, z)]]
        Pre-computed absolute XZ vertices for each obstacle.
    obstacles : list[dict] | None
        Original obstacle defs (used for labels).
    """
    from matplotlib.patches import Polygon as MplPolygon

    for i, verts in enumerate(abs_verts_list):
        if len(verts) < 3:
            continue
        poly_pts = np.array(verts, dtype=float)
        label = "obstacle"
        if obstacles and i < len(obstacles):
            label = obstacles[i].get("name", f"obstacle_{i}")
        patch = MplPolygon(
            poly_pts, closed=True,
            facecolor="0.65", edgecolor="k",
            linewidth=1.2, alpha=alpha,
            label=label,
            zorder=5,
        )
        ax.add_patch(patch)

    if abs_verts_list:
        ax.legend(loc="upper left", fontsize=8, framealpha=0.7)


def render_schematic_for_canvas(
    params: dict,
    poly: np.ndarray,
    canvas_width: int = 800,
    canvas_height: int = 400,
    porous_data: dict | None = None,
    xlim_override: tuple | None = None,
    ylim_override: tuple | None = None,
):
    """Render the structure cross-section as a clean PIL Image for the drawing canvas.

    No matplotlib margins, ticks, or labels — the entire image maps directly
    to (xlim, ylim) for pixel ↔ meters coordinate mapping.

    xlim_override / ylim_override allow the caller to zoom into a sub-region
    (e.g. the seaward face only) while still rendering the full polygon.

    Returns
    -------
    (PIL.Image, xlim, ylim, canvas_width, canvas_height)
    """
    import io
    from PIL import Image as PILImage
    from python_tools.canvas_tools import compute_canvas_limits

    if xlim_override is not None and ylim_override is not None:
        xlim, ylim = xlim_override, ylim_override
    elif xlim_override is not None:
        _, ylim = compute_canvas_limits(poly)
        xlim = xlim_override
    else:
        xlim, ylim = compute_canvas_limits(poly)

    # Dynamic canvas height from aspect ratio (keep caller's width)
    aspect = (xlim[1] - xlim[0]) / (ylim[1] - ylim[0])
    canvas_height = max(300, min(int(canvas_width / aspect), 600))

    dpi = 100
    fig_w = canvas_width / dpi
    fig_h = canvas_height / dpi

    fig = plt.figure(figsize=(fig_w, fig_h), dpi=dpi, facecolor="white")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_facecolor("#e8f4f8")  # light blue-grey for water/air area
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("auto")

    # Grid with coordinate labels for reference
    ax.grid(True, alpha=0.4, linestyle="-", color="#aaa", linewidth=0.6)
    ax.xaxis.set_major_locator(plt.MaxNLocator(nbins=10, steps=[1, 2, 2.5, 5, 10]))
    ax.yaxis.set_major_locator(plt.MaxNLocator(nbins=8,  steps=[1, 2, 2.5, 5, 10]))
    ax.tick_params(labelsize=9, colors="#222", length=3, pad=2)
    for spine in ax.spines.values():
        spine.set_visible(False)

    # Structure fill with visible edge
    ax.fill(poly[:, 0], poly[:, 1], color="#c49a6c", alpha=0.95,
            edgecolor="#4a3728", linewidth=2, zorder=3)

    # SWL line
    SWL = float(params["SWL"])
    ax.axhline(SWL, color="#1565c0", linestyle="--", linewidth=1.5,
               alpha=0.7, zorder=4, label=f"SWL = {SWL:.2f} m")
    # Water fill below SWL
    ax.axhspan(ylim[0], SWL, color="#bbdefb", alpha=0.3, zorder=1)

    # Porous overlays
    if porous_data is not None:
        try:
            plot_layered_porous_on_ax(ax, porous_data)
        except Exception:
            pass

    # Scale annotation (bottom-right)
    Lx = xlim[1] - xlim[0]
    scale_len = round(Lx * 0.2, 2)  # ~20% of x-range
    sx = xlim[1] - Lx * 0.05 - scale_len
    sy = ylim[0] + (ylim[1] - ylim[0]) * 0.05
    ax.plot([sx, sx + scale_len], [sy, sy], color="k", linewidth=2, zorder=5)
    ax.text(sx + scale_len / 2, sy + (ylim[1] - ylim[0]) * 0.03,
            f"{scale_len:.2f} m", ha="center", va="bottom", fontsize=7,
            color="#333", zorder=5)

    # Render to PIL Image
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, pad_inches=0, facecolor="white")
    plt.close(fig)
    buf.seek(0)
    img = PILImage.open(buf).copy()
    buf.close()

    return img, xlim, ylim, canvas_width, canvas_height


def plot_clash_schematic(params: dict, poly: np.ndarray, title: str = "Geometry schematic"):
    """
    Recomputes the key points of the profile and plots the schematic.
    Returns (fig, ax).
    """
    # Recompute key points from params (self-contained)
    SWL = float(params["SWL"])
    m = float(params["m"])
    h_deep = float(params["h_deep"])
    h = float(params["h"])
    h_t = float(params["h_t"])
    B_t = float(params["B_t"])
    h_b = float(params["h_b"])
    B = float(params["B"])
    tan_alpha_B = float(params.get("tan_alpha_B", 0.0))
    cot_alpha_d = float(params["cot_alpha_d"])
    cot_alpha_u = float(params["cot_alpha_u"])
    A_c = float(params["A_c"])
    G_c = float(params["G_c"])
    R_c = params["R_c"]
    x_toe = float(params["x_toe"])
    x_margin = float(params["x_margin"])
    L_back = float(params["L_back"])
    t_wall = float(params["t_wall"])
    zmin = float(params["zmin"])

    # CAD constants (must match clash_cad.py)
    M_FLUME = 1000.0
    M_TOL = 1e-9
    dx_foreshore_ramp = float(params.get("dx_foreshore_ramp", 0.20))
    dx_toe_transition = float(params.get("dx_toe_transition", 0.02))
    dx_berm_connector = float(params.get("dx_berm_connector", 0.05))
    ENABLE_COLLECTOR = True
    COLLECTOR_ZMIN_MARGIN = 0.05
    # Levels
    z_deep = SWL - h_deep
    z_toe_water = SWL - h
    z_toe_berm  = SWL - h_t
    z_berm      = SWL - h_b            # level at the middle of the berm
    z_berm_start = z_berm - 0.5 * B * tan_alpha_B
    z_berm_end   = z_berm + 0.5 * B * tan_alpha_B
    z_crest     = SWL + A_c

    # Crown wall top (may be None)
    z_wall_top = None
    if (R_c is not None) and (not np.isnan(R_c)):
        z_wall_top = SWL + float(R_c)

    # Foreshore x positions
    dz_fs = (z_toe_water - z_deep)
    is_flume = abs(m - M_FLUME) < M_TOL
    dx_fs = (dx_foreshore_ramp if (is_flume and abs(dz_fs) > 1e-14) else dz_fs * m)

    x_deep = x_toe - dx_fs
    x_left = x_deep - x_margin

    # Toe transition start
    x_toe0 = x_toe
    if abs(z_toe_berm - z_toe_water) > 1e-14:
        if is_flume:
            # flume mode: use existing HT ramp
            x_toe_berm_start = x_toe0 + dx_toe_transition
        else:
            # real slope mode: create a small connector ramp before Bt
            x_toe_berm_start = x_toe0 + dx_berm_connector
    else:
        x_toe_berm_start = x_toe0


    # Toe berm end
    x_toe1 = x_toe_berm_start + B_t

    # Lower slope end
    dz_d = (z_berm_start - z_toe_berm)
    x_d_end = x_toe1 + dx_from_cot(dz_d, cot_alpha_d)

    # Berm end
    x_berm_end = x_d_end + B

    # Upper slope end
    dz_u = (z_crest - z_berm_end)
    x_u_end = x_berm_end + dx_from_cot(dz_u, cot_alpha_u)

    # Crest end
    x_crest_end = x_u_end + G_c
    
    # Wall geometry (for plotting t_wall)
    x_wall0 = None
    x_wall1 = None

    if (R_c is not None) and (not np.isnan(R_c)) and (z_wall_top is not None) and (z_wall_top > z_crest + 1e-15):
        x_wall0 = x_crest_end
        x_wall1 = x_wall0 + t_wall

    # Container (collector) geometry for plotting L_back
    x_after = x_crest_end  # right after Gc

    # Wall thickness affects where the container starts
    if (R_c is not None) and (not np.isnan(R_c)) and (z_wall_top is not None) and (z_wall_top > z_crest + 1e-15):
        x_after = x_crest_end + t_wall

    x_container_start = None
    x_container_end   = None
    z_collector_floor = None

    if ENABLE_COLLECTOR and L_back > 0.0:
        z_collector_floor = zmin + COLLECTOR_ZMIN_MARGIN

        # keep the floor safely below crest and above zmin
        if z_collector_floor >= z_crest - 1e-12:
            z_collector_floor = z_crest - 0.01
        if z_collector_floor <= zmin + 1e-12:
            z_collector_floor = zmin + 1e-3

        x_container_start = x_after
        x_container_end   = x_after + L_back


    # Adaptive figure sizing & limits (per case)
    xmin, xmax = float(np.min(poly[:,0])), float(np.max(poly[:,0]))
    zmin_poly, zmax_poly = float(np.min(poly[:,1])), float(np.max(poly[:,1]))

    Lx = max(xmax - xmin, 1e-9)
    Lz = max(zmax_poly - zmin_poly, 1e-9)

    mx = 0.05 * Lx
    mz = 0.10 * Lz

    xlim = (xmin - mx, xmax + mx)
    ylim = (zmin_poly - mz, zmax_poly + mz)

    aspect = (xlim[1] - xlim[0]) / (ylim[1] - ylim[0])
    fig_w = np.clip(10 * aspect, 10, 22)
    fig_h = np.clip(6, 5, 10)

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    fig.set_dpi(150)

    # Plot ground + SWL
    ax.fill(poly[:, 0], poly[:, 1], color="#c49a6c", alpha=0.9)

    ax.axhline(SWL, color="k", linestyle="--", linewidth=1)
    ax.text(xlim[0], SWL + 0.02*(ylim[1]-ylim[0]), "SWL", fontsize=11, va="bottom")
    



    # Foreshore slope m (only if m != 1000)
    if abs(m - M_FLUME) > M_TOL:
        ax.text(
            (x_deep + x_toe)/2,
            (z_deep + z_toe_water)/2,
            r"Foreshore slope $m$",
            fontsize=11,
            rotation=np.degrees(np.arctan2(z_toe_water - z_deep, x_toe - x_deep)),
            ha="center", va="center"
        )

    # Vertical arrows: h_deep, h, h_t, h_b, A_c, R_c
    tol = 1e-12

    # h_deep arrow at the seaward end of the profile (not poly[-1]: the vertex
    # order changes when obstacles are union-merged into the polygon)
    x_hdeep = float(x_left)

    # h at toe
    x_h = x_toe 

    # ht to the right of B_t end
    x_ht = x_toe1 

    # hb near berm start (slightly right of lower slope end)
    x_hb = x_d_end + 0.02*(xlim[1] - xlim[0])
    if abs(tan_alpha_B) > 1e-12:
        x_hb = 0.5 * (x_d_end + x_berm_end)   # h_b is taken at mid-berm

    # Place Ac / Rc arrows near the wall (left side of container)
    dx_ac = 0.0001 * (xlim[1] - xlim[0])

    # By construction in clash_cad: wall starts exactly at x_crest_end (if it exists)
    x_wall0 = x_crest_end

    # Default: put labels just LEFT of the wall (so they don't sit on the slope)
    x_Ac = x_wall0 - dx_ac
    x_Rc = x_wall0 + dx_ac

    
    
    all_equal = (abs(h_deep - h) < tol) and (abs(h - h_t) < tol)

    if all_equal:
        # draw ONLY ONE arrow to avoid unreadable overlapping labels
        v_dim(ax, x_h, z_toe_water, SWL, r"$h_{deep} = h = h_t$", dx=12)
    else:
        # h_deep
        if abs(h_deep - h) < tol:
            v_dim(ax, x_hdeep, z_deep, SWL, r"$h_{deep}=h$", dx=12)
        else:
            v_dim(ax, x_hdeep, z_deep, SWL, r"$h_{deep}$", dx=12)

        # h / h_t
        if abs(h_t - h) < tol:
            v_dim(ax, x_h, z_toe_water, SWL, r"$h = h_t$", dx=12)
        else:
            v_dim(ax, x_h,  z_toe_water, SWL, r"$h$", dx=12)
            v_dim(ax, x_ht, z_toe_berm,  SWL, r"$h_t$", dx=12)


    # hb arrow only if hb differs from ht (avoid clutter)
    if abs(h_b - h_t) > tol:
        v_dim(ax, x_hb, SWL , z_berm, r"$h_b$")

    # A_c / R_c logic (single arrow if equal)
    Ac_ok = (A_c > 0)
    Rc_ok = (R_c is not None) and (not np.isnan(R_c)) and (z_wall_top is not None) and (R_c > 0)

    # A_c / R_c arrows:
    # - If A_c == R_c: show a single arrow at the wall
    # - If R_c > A_c: put A_c at the wall start (crest level) and R_c on the wall
    if Ac_ok and Rc_ok and abs(A_c - float(R_c)) < tol:
        # One combined arrow: SWL -> wall top
        v_dim(ax, x_Ac, SWL, z_wall_top, r"$A_c = R_c$", dx=12)
    else:
        if Ac_ok:
            if Rc_ok and (z_wall_top is not None) and (z_wall_top > z_crest + tol):
                # Rc higher than Ac -> draw Ac at the wall start (SWL -> crest)
                v_dim(ax, x_Ac, SWL, z_crest, r"$A_c$", dx=12)
            else:
                # No higher wall -> standard Ac
                v_dim(ax, x_Ac, SWL, z_crest, r"$A_c$", dx=12)

        if Rc_ok and (z_wall_top is not None) and (z_wall_top > z_crest + tol):
            # Wall height above crest: crest -> wall top
            v_dim(ax, x_Rc, z_crest, z_wall_top, r"$R_c$", dx=12)


    # Horizontal arrows: Gc (crest), Bt, B
    if G_c > 0:
        z_gc = z_crest + 0.01*(ylim[1] - ylim[0])  # attached to crest (tiny lift)
        h_dim(ax, x_u_end, x_crest_end, z_gc, r"$G_c$")

    if B_t > 0:
        z_bt = z_toe_berm - 0.01*(ylim[1] - ylim[0])
        h_dim(ax, x_toe_berm_start, x_toe1, z_bt, r"$B_t$")

    if B > 0:
        z_b = max(z_berm_start, z_berm_end) + 0.01*(ylim[1] - ylim[0])
        h_dim(ax, x_d_end, x_berm_end, z_b, r"$B$")
        
        
    # t_wall arrow fixed at the top of the crown wall
    if (x_wall0 is not None) and (x_wall1 is not None):
        z_tw = z_wall_top + 0.01*(ylim[1] - ylim[0])  # slight lift for readability
        h_dim(
            ax,
            x_wall0,
            x_wall1,
            z_tw,
            r"$t_{wall}$",
            dy=4,
            fs=10
        )

        
    # L_back arrow fixed to the container bottom (collector floor)
    if (x_container_start is not None) and (x_container_end is not None) and (z_collector_floor is not None):
        h_dim(
            ax,
            x_container_start,
            x_container_end,
            z_collector_floor,
            r"$L_{back}$",
            dy=2,   # label slightly below the arrow
            fs=10
        )

    # CAD ramps (small inclined arrows along the ramps)
    if is_flume and abs(dz_fs) > tol and dx_foreshore_ramp > tol:
        seg_dim(
            ax,
            (x_deep, z_deep),
            (x_toe0, z_toe_water),
            r"$Ramp 1$",
            frac=1.0,        # FULL length
            off_arrow=0.0,   # on the slope
            off_text=0.03,
            fs=6
        )

    if is_flume and abs(z_toe_berm - z_toe_water) > tol and dx_toe_transition > tol:
        seg_dim(
            ax,
            (x_toe0, z_toe_water),
            (x_toe_berm_start, z_toe_berm),
            r"$Ramp 2$",
            frac=1.0,        # FULL length
            off_arrow=0.0,   # on the slope
            off_text=0.06,
            fs=6
        )

    if (not is_flume) and abs(z_toe_berm - z_toe_water) > tol and (x_toe_berm_start - x_toe0) > tol:
        seg_dim(
            ax,
            (x_toe0, z_toe_water),
            (x_toe_berm_start, z_toe_berm),
            r"$RAMP\_{BT}$",
            frac=1.0,
            off_arrow=0.0,
            off_text=0.06,
            fs=5
        )



    # Slope angles: only if cot != 0 (attach cot to alpha)
    arc_r = 0.02 * (xlim[1] - xlim[0])

    if abs(cot_alpha_d) > tol:
        angle_arc(
            ax,
            (x_toe1, z_toe_berm),
            (x_d_end, z_berm_start),
            r"$\alpha_d$",
            arc_r
        )

    if abs(cot_alpha_u) > tol:
        angle_arc(
            ax,
            (x_berm_end, z_berm_end),
            (x_u_end, z_crest),
            r"$\alpha_u$",
            arc_r
        )

    if B > 0 and abs(tan_alpha_B) > tol:
        angle_arc(
            ax,
            (x_d_end, z_berm_start),
            (x_berm_end, z_berm_end),
            r"$\alpha_B$",
            arc_r
        )

    # Final formatting
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect('equal', adjustable='datalim')
    ax.set_xlabel("x [m]")
    ax.set_ylabel("z [m]")
    ax.set_title(title)
    ax.grid(False)

    return fig, ax
