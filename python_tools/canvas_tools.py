"""
Coordinate mapping, shape conversion, path simplification and STL outline
extraction for the interactive obstacle drawing canvas.
"""

from __future__ import annotations

import math
import numpy as np
from pathlib import Path


# Coordinate mapping

def px_to_meters(
    px_x: float, px_y: float,
    canvas_w: int, canvas_h: int,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
) -> tuple[float, float]:
    """Convert canvas pixel coordinates to real-world (x, z) in meters."""
    x_m = xlim[0] + (px_x / canvas_w) * (xlim[1] - xlim[0])
    z_m = ylim[1] - (px_y / canvas_h) * (ylim[1] - ylim[0])  # y inverted
    return (x_m, z_m)


def meters_to_px(
    x_m: float, z_m: float,
    canvas_w: int, canvas_h: int,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
) -> tuple[float, float]:
    """Convert real-world (x, z) meters to canvas pixel coordinates."""
    px_x = (x_m - xlim[0]) / (xlim[1] - xlim[0]) * canvas_w
    px_y = (ylim[1] - z_m) / (ylim[1] - ylim[0]) * canvas_h
    return (px_x, px_y)


def compute_canvas_limits(
    poly: np.ndarray,
    margin_frac_x: float = 0.08,
    margin_frac_z: float = 0.15,
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Compute (xlim, ylim) from structure polygon with small margins so the structure
    fills most of the canvas.  Small margins (8 % / 15 %) leave just enough room to
    place obstacles at the edges while ensuring the full structure is visible even in
    narrow Streamlit containers."""
    xmin, xmax = float(poly[:, 0].min()), float(poly[:, 0].max())
    zmin, zmax = float(poly[:, 1].min()), float(poly[:, 1].max())
    Lx = max(xmax - xmin, 1e-9)
    Lz = max(zmax - zmin, 1e-9)
    mx = margin_frac_x * Lx
    mz = margin_frac_z * Lz
    return ((xmin - mx, xmax + mx), (zmin - mz, zmax + mz))


# RDP path simplification

def _point_line_dist(p, a, b):
    """Perpendicular distance from point p to line segment a–b."""
    dx, dz = b[0] - a[0], b[1] - a[1]
    L2 = dx * dx + dz * dz
    if L2 < 1e-30:
        return math.hypot(p[0] - a[0], p[1] - a[1])
    return abs(dz * p[0] - dx * p[1] + b[0] * a[1] - b[1] * a[0]) / math.sqrt(L2)


def simplify_path_rdp(
    points: list[tuple[float, float]],
    epsilon: float = 0.002,
) -> list[tuple[float, float]]:
    """Ramer–Douglas–Peucker simplification. *epsilon* is in the same units as points."""
    if len(points) <= 2:
        return list(points)

    dmax = 0.0
    idx = 0
    for i in range(1, len(points) - 1):
        d = _point_line_dist(points[i], points[0], points[-1])
        if d > dmax:
            dmax = d
            idx = i

    if dmax > epsilon:
        left = simplify_path_rdp(points[: idx + 1], epsilon)
        right = simplify_path_rdp(points[idx:], epsilon)
        return left[:-1] + right
    return [points[0], points[-1]]


# Fabric.js → obstacle conversion

def _extract_path_points(path_cmds: list) -> list[tuple[float, float]]:
    """Extract (x, y) points from fabric.js SVG path commands."""
    pts = []
    for cmd in path_cmds:
        if not cmd:
            continue
        op = cmd[0]
        if op in ("M", "L") and len(cmd) >= 3:
            pts.append((float(cmd[1]), float(cmd[2])))
        elif op == "Q" and len(cmd) >= 5:
            # Quadratic bezier — sample the endpoint
            pts.append((float(cmd[3]), float(cmd[4])))
        elif op == "C" and len(cmd) >= 7:
            # Cubic bezier — sample the endpoint
            pts.append((float(cmd[5]), float(cmd[6])))
    return pts


def _apply_fabric_transform(
    points_px: list[tuple[float, float]],
    obj: dict,
) -> list[tuple[float, float]]:
    """Apply fabric.js object transform (left, top, scaleX, scaleY) to points.

    fabric.js path coordinates are relative to the object bounding box.
    The object's left/top is the position of the bounding-box top-left corner.
    """
    left = float(obj.get("left", 0))
    top = float(obj.get("top", 0))
    sx = float(obj.get("scaleX", 1))
    sy = float(obj.get("scaleY", 1))

    if not points_px:
        return points_px

    # Path coordinates are relative to the path's own origin:
    # place the path's min x / min y at the object's left / top.
    xs = [p[0] for p in points_px]
    ys = [p[1] for p in points_px]
    ox, oy = min(xs), min(ys)

    return [
        (left + (px - ox) * sx, top + (py - oy) * sy)
        for px, py in points_px
    ]


def _local_pts_to_canvas_px(
    local_pts: list[tuple[float, float]],
    obj: dict,
) -> list[tuple[float, float]]:
    """Convert a list of LOCAL fabric.js coordinates to canvas pixel coordinates.

    Unified rule for rect / polygon / path types:
      canvas_x = left + (local_x - min_local_x) * scaleX
      canvas_y = top  + (local_y - min_local_y) * scaleY

    This works because fabric.js always stores `left`/`top` as the canvas position of
    the bounding-box TOP-LEFT corner (for default originX/Y='left'/'top'), regardless
    of whether the coordinates are absolute (fabric v4), normalised to [0,W] (v5), or
    centred at (0,0) (polygon type after move/resize).  Subtracting the local min shifts
    the origin to the bounding-box corner, which is exactly where `left`/`top` sit.
    """
    if not local_pts:
        return []
    left = float(obj.get("left", 0))
    top  = float(obj.get("top",  0))
    sx   = float(obj.get("scaleX", 1))
    sy   = float(obj.get("scaleY", 1))
    xs   = [p[0] for p in local_pts]
    ys   = [p[1] for p in local_pts]
    ox, oy = min(xs), min(ys)
    return [(left + (px - ox) * sx, top + (py - oy) * sy) for px, py in local_pts]


def fabric_objects_to_obstacles(
    json_data: dict,
    canvas_w: int, canvas_h: int,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    rdp_epsilon: float = 0.002,
) -> list[list[tuple[float, float]]]:
    """Convert fabric.js JSON → list of obstacle vertex lists in meters.

    Each element is a list of (x, z) tuples forming a closed polygon.
    Handles rect, polygon (polygon-mode clicks), and path (freehand) objects.
    """
    obstacles = []
    for obj in json_data.get("objects", []):
        obj_type = obj.get("type", "")

        if obj_type == "rect":
            left = float(obj.get("left", 0))
            top  = float(obj.get("top",  0))
            w    = float(obj.get("width",  0)) * float(obj.get("scaleX", 1))
            h    = float(obj.get("height", 0)) * float(obj.get("scaleY", 1))
            corners_px = [
                (left, top), (left + w, top),
                (left + w, top + h), (left, top + h),
            ]
            verts = [
                px_to_meters(px, py, canvas_w, canvas_h, xlim, ylim)
                for px, py in corners_px
            ]
            obstacles.append(verts)

        elif obj_type == "polygon":
            pts = obj.get("points", [])
            local_pts = [(float(p["x"]), float(p["y"])) for p in pts if "x" in p and "y" in p]
            if len(local_pts) < 3:
                continue
            canvas_pts = _local_pts_to_canvas_px(local_pts, obj)
            verts_m = [
                px_to_meters(px, py, canvas_w, canvas_h, xlim, ylim)
                for px, py in canvas_pts
            ]
            if len(verts_m) >= 3:
                obstacles.append(verts_m)

        elif obj_type == "path":
            local_pts = _extract_path_points(obj.get("path", []))
            if len(local_pts) < 3:
                continue
            canvas_pts = _local_pts_to_canvas_px(local_pts, obj)
            verts_m = [
                px_to_meters(px, py, canvas_w, canvas_h, xlim, ylim)
                for px, py in canvas_pts
            ]
            if len(verts_m) > 20:
                verts_m = simplify_path_rdp(verts_m, epsilon=rdp_epsilon)
            if len(verts_m) >= 3:
                obstacles.append(verts_m)

    return obstacles


# STL cross-section extraction

def extract_xz_outline_from_stl(
    stl_path: str | Path,
    y_plane: float = 0.0,
    atol: float = 1e-6,
) -> list[tuple[float, float]]:
    """Extract 2D XZ boundary outline from a 3D STL on y=y_plane.

    Returns a closed polygon as list of (x, z) tuples.
    """
    from stl import mesh as stlmesh

    m = stlmesh.Mesh.from_file(str(stl_path))
    V = m.vectors  # (ntri, 3, 3)

    # Triangles on y_plane
    on_plane = np.all(np.isclose(V[:, :, 1], y_plane, atol=atol), axis=1)
    Vp = V[on_plane]
    if Vp.shape[0] == 0:
        raise ValueError(
            f"No triangles on y={y_plane}. "
            f"Try y_plane={float(V[:,:,1].max()):.3f} or increase atol."
        )

    # Collect XZ edges, find boundary (edges appearing once)
    edge_count: dict[tuple, int] = {}
    nd = 10
    for tri in Vp:
        pts = [(round(float(tri[i, 0]), nd), round(float(tri[i, 2]), nd)) for i in range(3)]
        for i in range(3):
            a, b = pts[i], pts[(i + 1) % 3]
            key = (a, b) if a <= b else (b, a)
            edge_count[key] = edge_count.get(key, 0) + 1

    boundary = [(a, b) for (a, b), c in edge_count.items() if c == 1]
    if len(boundary) < 3:
        raise RuntimeError("Not enough boundary edges for outline.")

    # Build adjacency and traverse loop
    adj: dict[tuple, list] = {}
    for a, b in boundary:
        adj.setdefault(a, []).append(b)
        adj.setdefault(b, []).append(a)

    start = min(adj.keys(), key=lambda p: (p[0], p[1]))
    loop = [start]
    prev = None
    cur = start
    for _ in range(len(adj) + 10):
        nbrs = adj.get(cur, [])
        if not nbrs:
            break
        if prev is None:
            nxt = nbrs[0]
        else:
            nxt = nbrs[0] if nbrs[0] != prev else (nbrs[1] if len(nbrs) > 1 else None)
        if nxt is None or nxt == start:
            break
        loop.append(nxt)
        prev, cur = cur, nxt

    return loop


def outline_to_initial_drawing(
    outline_xz: list[tuple[float, float]],
    canvas_w: int, canvas_h: int,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
) -> dict:
    """Convert XZ outline to fabric.js initial_drawing JSON for st_canvas."""
    px_pts = [
        meters_to_px(x, z, canvas_w, canvas_h, xlim, ylim)
        for x, z in outline_xz
    ]
    if len(px_pts) < 3:
        return {"version": "4.4.0", "objects": []}

    # Build SVG path commands
    path_cmds = [["M", px_pts[0][0], px_pts[0][1]]]
    for px, py in px_pts[1:]:
        path_cmds.append(["L", px, py])
    path_cmds.append(["L", px_pts[0][0], px_pts[0][1]])  # close

    return {
        "version": "4.4.0",
        "objects": [{
            "type": "path",
            "path": path_cmds,
            "left": min(p[0] for p in px_pts),
            "top": min(p[1] for p in px_pts),
            "fill": "rgba(100, 100, 200, 0.3)",
            "stroke": "#1d3557",
            "strokeWidth": 2,
            "selectable": True,
        }],
    }
