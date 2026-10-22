"""
Builds the 2D structure profile from the geometry parameters, then triangulates
and extrudes it into a 3D STL.
"""

from __future__ import annotations

import numpy as np
from stl import mesh

# CAD-only constants
M_FLUME = 1000.0
M_TOL = 1e-9

dx_foreshore_ramp = 0.20
dx_toe_transition = 0.02
dx_berm_connector = 0.05
ENABLE_COLLECTOR = True
COLLECTOR_ZMIN_MARGIN = 0.05


# INTERNAL HELPERS (triangulation + extrusion)

def polygon_area(poly: np.ndarray) -> float:
    x = poly[:, 0]
    y = poly[:, 1]
    return 0.5 * np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y)

def ensure_ccw(poly: np.ndarray) -> np.ndarray:
    poly = np.asarray(poly, float)
    if polygon_area(poly) < 0:
        poly = poly[::-1].copy()
    return poly

def remove_consecutive_duplicates(pts, tol: float = 1e-12):
    if not pts:
        return pts
    out = [pts[0]]
    for p in pts[1:]:
        if abs(p[0] - out[-1][0]) > tol or abs(p[1] - out[-1][1]) > tol:
            out.append(p)
    return out

def _clip_poly_zmin(poly: np.ndarray, zmin: float) -> np.ndarray:
    """
    Clip a 2D polygon to the half-plane z >= zmin (Sutherland-Hodgman).
    Returns the clipped polygon or an empty (0,2) array if fully below.
    """
    pts = list(poly)
    n = len(pts)
    if n == 0:
        return np.empty((0, 2))
    output = []
    for i in range(n):
        curr = pts[i]
        prev = pts[(i - 1) % n]
        in_curr = curr[1] >= zmin - 1e-12
        in_prev = prev[1] >= zmin - 1e-12
        if in_curr:
            if not in_prev:
                dz = curr[1] - prev[1]
                t = (zmin - prev[1]) / dz if abs(dz) > 1e-14 else 0.0
                output.append(np.array([prev[0] + t * (curr[0] - prev[0]), zmin]))
            output.append(np.array(curr, float))
        elif in_prev:
            dz = curr[1] - prev[1]
            t = (zmin - prev[1]) / dz if abs(dz) > 1e-14 else 0.0
            output.append(np.array([prev[0] + t * (curr[0] - prev[0]), zmin]))
    # Remove consecutive duplicate vertices (can appear when a vertex sits on zmin)
    output = remove_consecutive_duplicates(output)
    if len(output) < 3:
        return np.empty((0, 2))
    return ensure_ccw(np.array(output, float))


def point_in_tri(p, a, b, c) -> bool:
    v0 = c - a
    v1 = b - a
    v2 = p - a
    den = v0[0] * v1[1] - v1[0] * v0[1]
    if abs(den) < 1e-16:
        return False
    u = (v2[0] * v1[1] - v1[0] * v2[1]) / den
    v = (v0[0] * v2[1] - v2[0] * v0[1]) / den
    return (u >= -1e-14) and (v >= -1e-14) and (u + v <= 1 + 1e-14)

def is_convex(prev, cur, nxt) -> bool:
    a = cur - prev
    b = nxt - cur
    return (a[0] * b[1] - a[1] * b[0]) > 1e-14

def triangulate(poly: np.ndarray):
    """
    Ear-clipping triangulation of a simple polygon (CCW).
    Returns: (poly_ccw, list_of_triangle_index_triples)
    """
    poly = ensure_ccw(poly)
    # Remove consecutive duplicate vertices (zero-length edges break ear-clipping)
    poly = np.array(remove_consecutive_duplicates(list(poly)), float)
    idx = list(range(len(poly)))
    tris = []
    guard = 0
    while len(idx) > 3 and guard < 20000:
        guard += 1
        ear = False
        for k in range(len(idx)):
            i0 = idx[(k - 1) % len(idx)]
            i1 = idx[k]
            i2 = idx[(k + 1) % len(idx)]
            a, b, c = poly[i0], poly[i1], poly[i2]

            if not is_convex(a, b, c):
                continue

            ok = True
            for j in idx:
                if j in (i0, i1, i2):
                    continue
                if point_in_tri(poly[j], a, b, c):
                    ok = False
                    break

            if ok:
                tris.append((i0, i1, i2))
                del idx[k]
                ear = True
                break

        if not ear:
            raise ValueError("Triangulation failed (polygon self-intersecting/backtracking).")

    tris.append((idx[0], idx[1], idx[2]))
    return poly, tris

def extrude_to_stl(poly_xz: np.ndarray, Ly: float, out_name: str):
    """
    Extrude 2D X–Z polygon to 3D STL with thickness Ly in Y.
    """
    poly_xz = ensure_ccw(poly_xz)
    poly, tris = triangulate(poly_xz)

    V0 = np.column_stack([poly[:, 0], np.zeros(len(poly)),           poly[:, 1]])
    V1 = np.column_stack([poly[:, 0], np.full(len(poly), float(Ly)), poly[:, 1]])

    faces = []
    for (a, b, c) in tris:
        faces.append((V0[a], V0[b], V0[c]))       # front
    for (a, b, c) in tris:
        faces.append((V1[c], V1[b], V1[a]))       # back

    n = len(poly)
    for i in range(n):
        j = (i + 1) % n
        faces.append((V0[i], V0[j], V1[j]))
        faces.append((V0[i], V1[j], V1[i]))

    mmesh = mesh.Mesh(np.zeros(len(faces), dtype=mesh.Mesh.dtype))
    for k, tri in enumerate(faces):
        mmesh.vectors[k] = np.array(tri)

    mmesh.save(out_name)


# CLASH GEOMETRY BUILDER

def dx_from_cot(dz: float, cot: float) -> float:
    if abs(dz) < 1e-15:
        return 0.0
    if cot is None or np.isnan(cot):
        raise ValueError("Missing cot value.")
    cot = float(cot)
    if abs(cot) < 1e-15:
        return 0.0
    return float(dz) * cot

def build_clash_profile_polygon_xz(params: dict, *, return_meta: bool = False):
    """
    Build the CLASH-style profile polygon in X–Z, closed by bottom segment at zmin.
    Returns:
      - poly (N,2) in CCW order
      - OR (poly, meta) if return_meta=True
    """
    SWL = params["SWL"]
    m = float(params["m"])
    h_deep = float(params["h_deep"])
    h = float(params["h"])
    h_t = float(params["h_t"])
    B_t = float(params["B_t"])

    h_b = float(params["h_b"])
    B = float(params["B"])
    tan_B = float(params.get("tan_alpha_B", 0.0))

    cot_d = float(params["cot_alpha_d"])
    cot_u = float(params["cot_alpha_u"])

    A_c = float(params["A_c"])
    G_c = float(params["G_c"])
    R_c = params["R_c"]
    x_toe = float(params["x_toe"])
    x_margin = float(params["x_margin"])
    L_back = float(params["L_back"])
    use_collector = bool(params.get("use_collector", ENABLE_COLLECTOR))
    t_wall = float(params["t_wall"])
    zmin = float(params["zmin"])

    z_deep = SWL - h_deep
    z_toe_water = SWL - h
    z_toe_berm = SWL - h_t
    z_berm = SWL - h_b
    # Sloping berm (CLASH tanαB): h_b is the depth at the middle of the berm,
    # the berm rises landward for tan_alpha_B > 0.
    z_berm_start = z_berm - 0.5 * B * tan_B
    z_berm_end = z_berm + 0.5 * B * tan_B
    z_crest = SWL + A_c
    dx_foreshore_ramp_local = float(params.get("dx_foreshore_ramp", dx_foreshore_ramp))
    dx_toe_transition_local = float(params.get("dx_toe_transition", dx_toe_transition))
    dx_berm_connector_local = float(params.get("dx_berm_connector", dx_berm_connector))

    z_wall_top = None
    if R_c is not None and not np.isnan(R_c):
        z_wall_top = SWL + float(R_c)

    if z_berm_start < z_toe_berm - 1e-12 or z_berm_end > z_crest + 1e-12:
        raise ValueError("Invalid: sloping berm leaves the toe-to-crest range "
                         "(check h_b, B, tan_alpha_B vs h_t and A_c).")

    z_low = min(z_deep, z_toe_water, z_toe_berm, z_berm_start, z_berm_end, z_crest)
    if zmin >= z_low:
        raise ValueError(f"zmin={zmin:.3f} must be < lowest level {z_low:.3f}")

    dz_fs = (z_toe_water - z_deep)
    if dz_fs < -1e-12:
        raise ValueError("Invalid: toe water depth deeper than deep bed (check h_deep vs h).")

    is_flume = abs(m - M_FLUME) < M_TOL
    if is_flume:
        dx_fs = (dx_foreshore_ramp_local if abs(dz_fs) > 1e-14 else 0.0)
    else:
        dx_fs = dz_fs * m

    x_deep = x_toe - dx_fs
    x_left = x_deep - x_margin

    x_toe0 = x_toe

    if abs(z_toe_berm - z_toe_water) > 1e-14:
        if is_flume:
            # flume-style tiny ramp (legacy behavior)
            x_toe_berm_start = x_toe0 + dx_toe_transition_local
        else:
            # real-slope case: add a short connector ramp before Bt
            x_toe_berm_start = x_toe0 + dx_berm_connector_local
    else:
        x_toe_berm_start = x_toe0


    x_toe1 = x_toe_berm_start + B_t

    dz_d = (z_berm_start - z_toe_berm)
    if dz_d < -1e-12:
        raise ValueError("Invalid: berm deeper than toe berm (check h_b, tan_alpha_B vs h_t).")
    dx_d = dx_from_cot(dz_d, cot_d)
    x_d_end = x_toe1 + dx_d

    x_berm_end = x_d_end + B

    dz_u = (z_crest - z_berm_end)
    if dz_u < -1e-12:
        raise ValueError("Invalid: crest below berm (check h_b, tan_alpha_B vs A_c).")
    dx_u = dx_from_cot(dz_u, cot_u)
    x_u_end = x_berm_end + dx_u

    x_crest_end = x_u_end + G_c

    # Key x positions returned with the polygon
    meta = {
        "x_u_end": float(x_u_end),
        "x_berm_end": float(x_berm_end),
        "x_crest_end": float(x_crest_end),
        "z_crest": float(z_crest),
        "z_wall_top": float(z_wall_top) if z_wall_top is not None else None,
        "x_wall0": None,
        "x_wall1": None,
        "x_back_start": None,
        "x_container_start": None,
        "x_container_end": None,
        "x_end": None,
    }

    # Wall immediately after Gc, then box
    wall_pts = []
    collector_pts = []

    x_after = x_crest_end  # right after Gc

    # 1) Wall starts at x_crest_end (directly after Gc)
    if z_wall_top is not None and z_wall_top > z_crest + 1e-15:
        x_wall0 = x_crest_end
        x_wall1 = x_wall0 + t_wall

        meta["x_wall0"] = float(x_wall0)
        meta["x_wall1"] = float(x_wall1)

        wall_pts = [
            (x_wall0, z_wall_top),
            (x_wall1, z_wall_top),
            (x_wall1, z_crest),
        ]
        x_after = x_wall1  # landward face of wall   
        
       
    
    meta["x_back_start"] = float(x_after)
    
    
    # 2) Container starts AFTER the wall
    if use_collector and L_back > 0.0:
        z_collector_floor = zmin + COLLECTOR_ZMIN_MARGIN
        if z_collector_floor >= z_crest - 1e-12:
            z_collector_floor = z_crest - 0.01
        if z_collector_floor <= zmin + 1e-12:
            z_collector_floor = zmin + 1e-3

        x_collector_end = x_after + L_back

        meta["x_container_start"] = float(x_after)
        meta["x_container_end"] = float(x_collector_end)

        collector_pts = [
            (x_after,         z_collector_floor),
            (x_collector_end, z_collector_floor),
            (x_collector_end, z_crest),
        ]
        x_after = x_collector_end

    # landward extension (optional tail)
    if use_collector:
        x_end = x_after + 0.02  # short tail after the collector
    else:
        x_end = x_after + L_back
    meta["x_end"] = float(x_end)

    # Build the top boundary in order: crest end -> wall -> collector
    top = [
        (x_left, z_deep),
        (x_deep, z_deep),
        (x_toe0, z_toe_water),
    ]

    if abs(z_toe_berm - z_toe_water) > 1e-14:
        top.append((x_toe_berm_start, z_toe_berm))

    top.append((x_toe1, z_toe_berm))

    top += [
        (x_d_end, z_berm_start),
        (x_berm_end, z_berm_end),
        (x_u_end, z_crest),
        (x_crest_end, z_crest),
    ]

    # wall first
    if wall_pts:
        top += wall_pts

    # then the collector box
    if collector_pts:
        top += collector_pts

    if use_collector:
        # container case: close the box on the right at crest level
        top.append((x_end, z_crest))
    else:
        # open-outlet case: end on the bed (no right vertical wall)
        top.append((x_end, zmin))

    poly = top + [(x_end, zmin), (x_left, zmin)]
    poly = ensure_ccw(np.array(remove_consecutive_duplicates(poly), float))

    if return_meta:
        return poly, meta
    return poly


def build_and_save_stl(
    params: dict, Ly: float, out_name: str,
    *, return_meta: bool = False,
):
    """Build CLASH polygon and save STL.

    Returns
    -------
    poly or (poly, meta) depending on *return_meta*.
    """
    poly, meta = build_clash_profile_polygon_xz(params, return_meta=True)
    extrude_to_stl(poly, Ly, out_name)
    if return_meta:
        return poly, meta
    return poly


# POROUS ZONE GEOMETRY

def _compute_clash_coords(params: dict) -> dict:
    """
    Compute all key x/z coordinates from CLASH params.
    Returns a flat dict of coordinate values used by porous-zone helpers.
    """
    SWL = float(params["SWL"])
    m = float(params["m"])
    h_deep = float(params["h_deep"])
    h = float(params["h"])
    h_t = float(params["h_t"])
    B_t = float(params["B_t"])
    h_b = float(params["h_b"])
    B = float(params["B"])
    tan_B = float(params.get("tan_alpha_B", 0.0))
    cot_d = float(params["cot_alpha_d"])
    cot_u = float(params["cot_alpha_u"])
    A_c = float(params["A_c"])
    G_c = float(params["G_c"])
    x_toe = float(params["x_toe"])
    x_margin = float(params["x_margin"])
    dx_foreshore_ramp_local = float(params.get("dx_foreshore_ramp", dx_foreshore_ramp))
    dx_toe_transition_local = float(params.get("dx_toe_transition", dx_toe_transition))
    dx_berm_connector_local = float(params.get("dx_berm_connector", dx_berm_connector))

    z_deep = SWL - h_deep
    z_toe_water = SWL - h
    z_toe_berm = SWL - h_t
    z_berm = SWL - h_b
    z_berm_start = z_berm - 0.5 * B * tan_B
    z_berm_end = z_berm + 0.5 * B * tan_B
    z_crest = SWL + A_c

    is_flume = abs(m - M_FLUME) < M_TOL
    dz_fs = z_toe_water - z_deep
    dx_fs = (dx_foreshore_ramp_local if (is_flume and abs(dz_fs) > 1e-14) else dz_fs * m)

    x_deep = x_toe - dx_fs
    x_left = x_deep - x_margin
    x_toe0 = x_toe

    if abs(z_toe_berm - z_toe_water) > 1e-14:
        x_toe_berm_start = x_toe0 + (dx_toe_transition_local if is_flume else dx_berm_connector_local)
    else:
        x_toe_berm_start = x_toe0

    x_toe1 = x_toe_berm_start + B_t
    dz_d = z_berm_start - z_toe_berm
    dx_d = dx_from_cot(dz_d, cot_d)
    x_d_end = x_toe1 + dx_d
    x_berm_end = x_d_end + B
    dz_u = z_crest - z_berm_end
    dx_u = dx_from_cot(dz_u, cot_u)
    x_u_end = x_berm_end + dx_u
    x_crest_end = x_u_end + G_c

    return dict(
        SWL=SWL, is_flume=is_flume,
        z_deep=z_deep, z_toe_water=z_toe_water, z_toe_berm=z_toe_berm,
        z_berm=z_berm, z_berm_start=z_berm_start, z_berm_end=z_berm_end,
        z_crest=z_crest,
        x_left=x_left, x_deep=x_deep, x_toe0=x_toe0,
        x_toe_berm_start=x_toe_berm_start, x_toe1=x_toe1,
        x_d_end=x_d_end, x_berm_end=x_berm_end,
        x_u_end=x_u_end, x_crest_end=x_crest_end,
        B_t=B_t, B=B,
    )


def _compute_surface_segments(params: dict) -> list:
    """
    Return ordered list of surface segment descriptors for the exposed CLASH profile.
    Each dict: name, P_start, P_end, tangent, normal (inward, into solid), length.
    Only segments with non-zero length are included.

    Valid segment names (in order, left to right):
        'foreshore', 'toe_slope', 'toe_berm', 'lower_slope',
        'berm', 'upper_slope', 'crest', 'back_slope'
    """
    c = _compute_clash_coords(params)

    def _seg(name, p0, p1):
        dx = p1[0] - p0[0]
        dz = p1[1] - p0[1]
        L = np.hypot(dx, dz)
        if L < 1e-12:
            return None
        t = np.array([dx / L, dz / L])
        n = np.array([dz / L, -dx / L])   # inward normal = rotate tangent -90°
        return dict(
            name=name,
            P_start=np.array(p0, float),
            P_end=np.array(p1, float),
            tangent=t,
            normal=n,
            length=L,
        )

    segs = []

    s = _seg('foreshore',
             (c['x_deep'], c['z_deep']),
             (c['x_toe0'], c['z_toe_water']))
    if s: segs.append(s)

    if abs(c['z_toe_berm'] - c['z_toe_water']) > 1e-14:
        s = _seg('toe_slope',
                 (c['x_toe0'], c['z_toe_water']),
                 (c['x_toe_berm_start'], c['z_toe_berm']))
        if s: segs.append(s)

    if c['B_t'] > 1e-12:
        s = _seg('toe_berm',
                 (c['x_toe_berm_start'], c['z_toe_berm']),
                 (c['x_toe1'], c['z_toe_berm']))
        if s: segs.append(s)

    s = _seg('lower_slope',
             (c['x_toe1'], c['z_toe_berm']),
             (c['x_d_end'], c['z_berm_start']))
    if s: segs.append(s)

    if c['B'] > 1e-12:
        s = _seg('berm',
                 (c['x_d_end'], c['z_berm_start']),
                 (c['x_berm_end'], c['z_berm_end']))
        if s: segs.append(s)

    s = _seg('upper_slope',
             (c['x_berm_end'], c['z_berm_end']),
             (c['x_u_end'], c['z_crest']))
    if s: segs.append(s)

    s = _seg('crest',
             (c['x_u_end'], c['z_crest']),
             (c['x_crest_end'], c['z_crest']))
    if s: segs.append(s)

    # Back slope: from crest end down to bed level.
    # Only when there is no collector and no crown wall (wall changes the
    # geometry after the crest significantly).
    L_back = float(params.get("L_back", 0))
    zmin = float(params.get("zmin", c['z_deep']))
    use_collector = bool(params.get("USE_COLLECTOR", False))
    R_c = params.get("R_c")
    A_c = float(params.get("A_c", 0))
    has_wall = (R_c is not None and not np.isnan(float(R_c)) and float(R_c) > A_c + 1e-15)
    if not use_collector and not has_wall and L_back > 1e-12:
        x_end = c['x_crest_end'] + L_back
        s = _seg('back_slope',
                 (c['x_crest_end'], c['z_crest']),
                 (x_end, zmin))
        if s: segs.append(s)

    return segs


def _inner_corner_point(P, n_a, t_a, d_a, n_b, t_b, d_b) -> np.ndarray:
    """
    Intersection of inner offset lines from two adjacent surface segments.

    At the shared junction point P:
      Line A  (end of segment A):  P + d_a*n_a + λ * t_a
      Line B  (start of segment B): P + d_b*n_b + μ * t_b

    Solves: d_a*n_a + λ*t_a  =  d_b*n_b + μ*t_b
    Returns the corner point P + d_a*n_a + λ*t_a.
    Falls back to average offset when lines are parallel.
    """
    A_mat = np.array([[t_a[0], -t_b[0]],
                      [t_a[1], -t_b[1]]])
    b_vec = d_b * n_b - d_a * n_a
    det = A_mat[0, 0] * A_mat[1, 1] - A_mat[0, 1] * A_mat[1, 0]
    if abs(det) < 1e-14:
        # Parallel (same slope) — use simple offset from either side
        return P + d_a * n_a
    lam = float(np.linalg.solve(A_mat, b_vec)[0])
    return P + d_a * n_a + lam * t_a


def _clipped_inner_pts(P_start, P_end, d, zmin):
    """
    Inner boundary of a porous zone with vertical downward offset d, clipped at zmin.

    Returns a list of (x, z) tuples (left-to-right) that trace the inner boundary
    at exactly d below the surface everywhere possible, then flat at zmin where the
    full offset would go below the domain floor.

    Case table (P_si = P_start - [0,d], P_ei = P_end - [0,d]):
      both above zmin  → [P_si, P_ei]
      si below, ei above → [(x_start, zmin), (x_cross, zmin), P_ei]
      si above, ei below → [P_si, (x_cross, zmin), (x_end, zmin)]
      both below zmin  → [(x_start, zmin), (x_end, zmin)]
    """
    P_si = (float(P_start[0]), float(P_start[1]) - d)
    P_ei = (float(P_end[0]),   float(P_end[1])   - d)
    si_below = P_si[1] < zmin - 1e-14
    ei_below = P_ei[1] < zmin - 1e-14

    if not si_below and not ei_below:
        return [P_si, P_ei]

    dz_inner = P_ei[1] - P_si[1]

    if si_below and not ei_below:
        t = (zmin - P_si[1]) / dz_inner if abs(dz_inner) > 1e-14 else 0.0
        x_cross = P_si[0] + t * (P_ei[0] - P_si[0])
        return [(float(P_start[0]), zmin), (x_cross, zmin), P_ei]

    if not si_below and ei_below:
        t = (zmin - P_si[1]) / dz_inner if abs(dz_inner) > 1e-14 else 1.0
        x_cross = P_si[0] + t * (P_ei[0] - P_si[0])
        return [P_si, (x_cross, zmin), (float(P_end[0]), zmin)]

    # Both below zmin
    return [(float(P_start[0]), zmin), (float(P_end[0]), zmin)]


def _remove_spikes(pts, tol: float = 1e-12):
    """
    Remove the vertices where a closed ring turns back on itself (zero-width
    spikes: the edges before and after the vertex are collinear and opposite),
    e.g. the solid going up to the surface at x_left and straight back down
    when the first segment is porous.
    """
    pts = [tuple(p) for p in pts]
    changed = True
    while changed and len(pts) > 3:
        changed = False
        n = len(pts)
        for k in range(n):
            a, b, c = np.array(pts[k - 1]), np.array(pts[k]), np.array(pts[(k + 1) % n])
            u, v = b - a, c - b
            nu, nv = np.hypot(*u), np.hypot(*v)
            if nu < tol or nv < tol or (abs(u[0] * v[1] - u[1] * v[0]) <= tol * nu * nv
                                        and float(u @ v) < 0):
                pts.pop(k)
                changed = True
                break
    return pts


def _clean_ring(poly, tol: float = 1e-12):
    """
    Closed polygon (N,2) without repeated points (also first == last), or None
    if fewer than 3 points or no area remain.
    """
    pts = remove_consecutive_duplicates([tuple(p) for p in np.asarray(poly, float)], tol)
    while len(pts) > 1 and abs(pts[0][0] - pts[-1][0]) <= tol and abs(pts[0][1] - pts[-1][1]) <= tol:
        pts.pop()
    if len(pts) < 3:
        return None
    arr = np.array(pts, float)
    if abs(polygon_area(arr)) < 1e-12:
        return None
    return arr


def _wedge_inner_pts(P_start, P_end, d, zmin):
    """
    Inner boundary of a back-slope layer limited to the crest wedge.

    The boundary is horizontal at z = z_start - d (clipped at zmin), from below
    the crest end to the point where the back slope reaches that level. Below
    it the back slope stays solid. With the crest layer of the same thickness,
    the two layers meet at (x_crest_end, z_crest - d).

    Returns [(x_start, z_lvl), (x_w, z_lvl)]; both points coincide when d = 0.
    """
    x0, z0 = float(P_start[0]), float(P_start[1])
    x1, z1 = float(P_end[0]), float(P_end[1])
    z_lvl = max(z0 - d, zmin)
    dz = z0 - z1
    t = (z0 - z_lvl) / dz if dz > 1e-14 else 1.0
    t = min(max(t, 0.0), 1.0)
    return [(x0, z_lvl), (x0 + t * (x1 - x0), z_lvl)]


def solid_top_at_x(poly, x: float, tol: float = 1e-9):
    """
    Highest z of the boundary of a polygon (N,2) at abscissa x, or None if the
    polygon does not reach x. Vertical edges at x count with both end points.
    """
    P = np.asarray(poly, float)
    z_top = None
    for (xa, za), (xb, zb) in zip(P, np.roll(P, -1, axis=0)):
        if min(xa, xb) - tol <= x <= max(xa, xb) + tol:
            if abs(xb - xa) < tol:
                z = max(za, zb)
            else:
                z = za + (x - xa) / (xb - xa) * (zb - za)
            z_top = z if z_top is None else max(z_top, z)
    return z_top


def build_porous_zone_data(params: dict, porous_zones_config: dict) -> dict:
    """
    Compute porous zone polygons and the modified solid polygon.

    Parameters
    ----------
    params : dict
        CLASH parameter dict (same as build_and_save_stl).
    porous_zones_config : dict
        {zone_name: d_porous_float}
        Valid names: 'foreshore', 'toe_slope', 'toe_berm',
                     'lower_slope', 'berm', 'upper_slope', 'crest', 'back_slope'
        d_porous is the layer thickness measured normal to the surface [m].

    Returns
    -------
    dict with keys:
        'porous_polys' : {zone_name: np.ndarray (N,2)}  — one polygon per zone
        'solid_poly'   : np.ndarray (N,2)  — structure minus porous zones
        'merged_poly'  : np.ndarray (N,2) or None  — all porous zones merged
        'segments'     : list[dict]  — all surface segment metadata
        'active_zones' : list[str]   — zone names that were applied
    """
    import warnings

    zmin = float(params["zmin"])
    segs = _compute_surface_segments(params)
    seg_dict = {s['name']: s for s in segs}
    ordered_names = [s['name'] for s in segs]

    # Validate requested zones
    active = {}
    for zone, d in porous_zones_config.items():
        if zone not in seg_dict:
            warnings.warn(
                f"Porous zone '{zone}' is not present in this geometry "
                f"(available: {ordered_names}). Skipped."
            )
        else:
            if d <= 0:
                raise ValueError(f"d_porous for zone '{zone}' must be > 0, got {d}.")
            active[zone] = float(d)

    if not active:
        raise ValueError(
            f"No valid porous zones found. "
            f"Requested: {list(porous_zones_config)}, "
            f"available: {ordered_names}."
        )

    # Step A: inner points — vertical downward offset
    # Each inner point sits directly below its surface point by d metres.
    # This gives a porous layer of vertical height = d everywhere, matching
    # the user-specified value regardless of slope steepness.
    inner = {}
    for zone, d in active.items():
        seg = seg_dict[zone]
        inner[zone] = {
            'P_start_inner': seg['P_start'] + np.array([0.0, -d]),
            'P_end_inner':   seg['P_end']   + np.array([0.0, -d]),
        }

    # Step C: per-zone porous polygons
    # Use the unclamped inner points and clip the resulting polygon at zmin.
    # _clip_poly_zmin (Sutherland-Hodgman) correctly handles crossing the domain
    # floor, preserving the full vertical height d wherever the structure allows it.
    porous_polys = {}
    for zone in active:
        seg = seg_dict[zone]
        d = active[zone]
        poly_pts = np.array([
            seg['P_start'],
            seg['P_end'],
            inner[zone]['P_end_inner'],
            inner[zone]['P_start_inner'],
        ], float)
        clipped = _clip_poly_zmin(ensure_ccw(poly_pts), zmin)
        if len(clipped) == 0:
            warnings.warn(
                f"Porous zone '{zone}' is entirely below z_min={zmin}; skipped."
            )
        else:
            porous_polys[zone] = clipped

    # Step D: modified solid polygon
    # Derive the wall/collector tail from params (same as build_clash_profile_polygon_xz)
    R_c = params["R_c"]
    L_back = float(params["L_back"])
    use_collector = bool(params.get("use_collector", ENABLE_COLLECTOR))
    t_wall = float(params["t_wall"])
    c = _compute_clash_coords(params)

    z_wall_top = None
    if R_c is not None and not np.isnan(R_c):
        z_wall_top = float(params["SWL"]) + float(R_c)

    # Build solid top boundary by walking segments and substituting porous zones.
    # For porous zones, the inner boundary is computed via _clipped_inner_pts so
    # that the full vertical thickness d is preserved everywhere, with a flat
    # section at zmin only where the structure does not extend deep enough.
    new_top = [(c['x_left'], c['z_deep'])]

    # Solid polygon inner boundary uses zmin + epsilon so that no vertex of
    # new_top lands exactly on the closing bottom edge (which is at zmin).
    # A vertex ON a non-adjacent edge makes the polygon self-intersecting and
    # causes triangulation to fail.  1 µm lift is geometrically negligible.
    zmin_solid = zmin + 1e-6

    for i, seg in enumerate(segs):
        name = seg['name']
        is_porous = name in active
        d = active.get(name, 0.0)

        if is_porous:
            ipts = _clipped_inner_pts(seg['P_start'], seg['P_end'], d, zmin_solid)
        else:
            ipts = []

        # First segment: explicitly add its start point
        if i == 0:
            new_top.append(tuple(seg['P_start']))
            if is_porous:
                # Connector: surface start → inner boundary start (all but last)
                for pt in ipts[:-1]:
                    new_top.append(pt)

        # Add end point: last inner pt (porous) or surface end (non-porous)
        elif is_porous:
            # Start point comes from the junction connector; keep the
            # intermediate points (corner where the boundary reaches zmin)
            for pt in ipts[1:-1]:
                new_top.append(pt)

        end_pt = ipts[-1] if is_porous else tuple(seg['P_end'])
        new_top.append(end_pt)

        # Junction connector to next segment
        if i + 1 < len(segs):
            next_seg = segs[i + 1]
            next_name = next_seg['name']
            is_porous_next = next_name in active
            d_next = active.get(next_name, 0.0)

            if is_porous and not is_porous_next:
                # porous → non-porous: rise back to surface junction
                new_top.append(tuple(seg['P_end']))
            elif not is_porous and is_porous_next:
                # non-porous → porous: descend to inner boundary start
                ipts_next = _clipped_inner_pts(next_seg['P_start'], next_seg['P_end'], d_next, zmin_solid)
                new_top.append(ipts_next[0])
            elif is_porous and is_porous_next:
                # porous → porous: add step only if inner levels differ
                ipts_next = _clipped_inner_pts(next_seg['P_start'], next_seg['P_end'], d_next, zmin_solid)
                if not np.allclose(end_pt, ipts_next[0], atol=1e-12):
                    new_top.append(ipts_next[0])

    # If last segment was porous, close back to original surface endpoint
    if segs and segs[-1]['name'] in active:
        new_top.append(tuple(segs[-1]['P_end']))

    # Append wall, collector, closing (same logic as build_clash_profile_polygon_xz)
    x_after = c['x_crest_end']
    wall_pts = []
    collector_pts = []

    if z_wall_top is not None and z_wall_top > c['z_crest'] + 1e-15:
        x_wall0 = c['x_crest_end']
        x_wall1 = x_wall0 + t_wall
        wall_pts = [
            (x_wall0, z_wall_top),
            (x_wall1, z_wall_top),
            (x_wall1, c['z_crest']),
        ]
        x_after = x_wall1

    if use_collector and L_back > 0.0:
        z_collector_floor = zmin + COLLECTOR_ZMIN_MARGIN
        if z_collector_floor >= c['z_crest'] - 1e-12:
            z_collector_floor = c['z_crest'] - 0.01
        if z_collector_floor <= zmin + 1e-12:
            z_collector_floor = zmin + 1e-3
        x_collector_end = x_after + L_back
        collector_pts = [
            (x_after,          z_collector_floor),
            (x_collector_end,  z_collector_floor),
            (x_collector_end,  c['z_crest']),
        ]
        x_after = x_collector_end

    if use_collector:
        x_end = x_after + 0.02
    else:
        x_end = x_after + L_back

    new_top += wall_pts + collector_pts
    new_top.append((x_end, c['z_crest']) if use_collector else (x_end, zmin))
    solid_raw = new_top + [(x_end, zmin), (c['x_left'], zmin)]
    solid_poly = ensure_ccw(
        np.array(_remove_spikes(remove_consecutive_duplicates(solid_raw)), float)
    )

    # Step E: merged polygon (if all active zones are contiguous)
    ordered_active = [s['name'] for s in segs if s['name'] in active]
    merged_poly = None

    if len(ordered_active) > 1:
        # Check contiguity
        is_contiguous = all(
            ordered_names.index(ordered_active[k + 1]) ==
            ordered_names.index(ordered_active[k]) + 1
            for k in range(len(ordered_active) - 1)
        )
    else:
        is_contiguous = len(ordered_active) == 1

    if is_contiguous and ordered_active:
        outer_pts = []
        inner_pts_list = []
        for zone in ordered_active:
            seg = seg_dict[zone]
            d = active[zone]
            if not outer_pts:
                outer_pts.append(seg['P_start'].tolist())
            outer_pts.append(seg['P_end'].tolist())
            ipts = _clipped_inner_pts(seg['P_start'], seg['P_end'], d, zmin)
            for pt in ipts:
                if not inner_pts_list or not np.allclose(inner_pts_list[-1], pt, atol=1e-12):
                    inner_pts_list.append(list(pt))

        merged_pts = outer_pts + inner_pts_list[::-1]
        merged_poly = _clip_poly_zmin(ensure_ccw(np.array(merged_pts, float)), zmin)
        if len(merged_poly) == 0:
            merged_poly = None

    return {
        'porous_polys': porous_polys,
        'solid_poly':   solid_poly,
        'merged_poly':  merged_poly,
        'segments':     segs,
        'active_zones': ordered_active,
    }


def save_porous_stls(
    porous_data: dict,
    Ly: float,
    out_dir,
    case_name: str,
    merged: bool = True,
) -> dict:
    """
    Save porous zone STL(s) and the modified solid STL.

    Parameters
    ----------
    porous_data : dict
        Output of build_porous_zone_data().
    Ly : float
        Spanwise extrusion thickness [m].
    out_dir : Path or str
        Directory for STL output (must exist).
    case_name : str
        Base name used for file naming.
    merged : bool
        True  → one merged porous STL  ({case_name}_porous.stl)
        False → one STL per zone       ({case_name}_porous_{zone}.stl)
        Falls back to per-zone if zones are non-contiguous.

    Returns
    -------
    dict with keys:
        'solid_stl'   : Path  — always saved
        'merged_stl'  : Path or None
        'porous_stls' : {zone_name: Path}  — populated when per-zone output
    """
    from pathlib import Path
    out_dir = Path(out_dir)
    result = {'solid_stl': None, 'merged_stl': None, 'porous_stls': {}}

    # solid
    solid_path = out_dir / f"{case_name}_solid.stl"
    extrude_to_stl(porous_data['solid_poly'], Ly, str(solid_path))
    result['solid_stl'] = solid_path

    # porous
    use_merged = merged and (porous_data['merged_poly'] is not None)

    if use_merged:
        merged_path = out_dir / f"{case_name}_porous.stl"
        extrude_to_stl(porous_data['merged_poly'], Ly, str(merged_path))
        result['merged_stl'] = merged_path
    else:
        if merged and porous_data['merged_poly'] is None:
            import warnings
            warnings.warn(
                "Porous zones are non-contiguous; saving per-zone STL files instead."
            )
        for zone, poly in porous_data['porous_polys'].items():
            path = out_dir / f"{case_name}_porous_{zone}.stl"
            extrude_to_stl(poly, Ly, str(path))
            result['porous_stls'][zone] = path

    return result


# Multi-layer porous structure

def build_layered_porous_data(
    params: dict,
    layers: list[dict],
    active_zones: dict | None = None,
    core_is_solid: bool = True,
    back_slope_wedge: bool = False,
) -> dict:
    """Build multi-layer concentric porous zone polygons.

    Parameters
    ----------
    params : dict
        CLASH parameter dict.
    layers : list[dict]
        Ordered outermost → innermost.  Each dict has keys:
        name, default_thickness, zone_thickness_overrides,
        physics_default, zone_physics_overrides.
    active_zones : dict | None
        {zone_name: True/False} filter.  If None, all zones are active.
    core_is_solid : bool
        If True, innermost region is solid; otherwise it is porous.
    back_slope_wedge : bool
        If True, the 'back_slope' layers fill only the wedge between the crest
        end and the back slope, down to z_crest - d (horizontal inner
        boundary), instead of a band along the whole back slope.

    Returns
    -------
    dict with keys:
        layer_polys   : {(layer_idx, zone_name): np.ndarray}
        solid_poly    : np.ndarray  (core / solid region)
        segments      : list[dict]
        active_zones  : list[str]
        layer_configs : list[dict]
        # backward-compat aliases
        porous_polys  : {zone_name: np.ndarray}  (layer 0 polys)
        merged_poly   : np.ndarray | None
    """
    import warnings

    zmin = float(params["zmin"])
    segs = _compute_surface_segments(params)
    seg_dict = {s["name"]: s for s in segs}
    ordered_names = [s["name"] for s in segs]

    # Resolve active zone list
    if active_zones is None:
        az_list = ordered_names[:]
    else:
        az_list = [z for z in ordered_names if active_zones.get(z, False)]

    if not az_list:
        raise ValueError("No active porous zones selected.")

    def _inner_pts(seg, d, z_floor):
        if back_slope_wedge and seg["name"] == "back_slope":
            return _wedge_inner_pts(seg["P_start"], seg["P_end"], d, z_floor)
        return _clipped_inner_pts(seg["P_start"], seg["P_end"], d, z_floor)

    # Build layer-zone polygons
    layer_polys: dict[tuple[int, str], np.ndarray] = {}
    cumulative_d: dict[str, float] = {z: 0.0 for z in az_list}

    for li, layer in enumerate(layers):
        d_default = float(layer["default_thickness"])
        overrides = layer.get("zone_thickness_overrides", {})

        for zone in az_list:
            d_this = float(overrides.get(zone, d_default))
            if d_this <= 0:
                continue

            seg = seg_dict[zone]
            d_outer = cumulative_d[zone]
            d_inner = d_outer + d_this

            if back_slope_wedge and zone == "back_slope":
                # Wedge: outer and inner boundaries are horizontal; the band
                # closes along the back slope itself, so the first layer
                # starts from the crest end only.
                if d_outer < 1e-14:
                    outer_pts = [tuple(seg["P_start"])]
                else:
                    outer_pts = _inner_pts(seg, d_outer, zmin)
                inner_pts = _inner_pts(seg, d_inner, zmin)
                band = list(outer_pts) + list(reversed(inner_pts))
            else:
                # Parallelogram between the surface offset by d_outer and by
                # d_inner, clipped at zmin below. Clipping the full (convex)
                # band avoids the back-tracking edges along zmin of a band
                # built from two separately clipped boundaries.
                dz_o = np.array([0.0, -d_outer])
                dz_i = np.array([0.0, -d_inner])
                band = [seg["P_start"] + dz_o, seg["P_end"] + dz_o,
                        seg["P_end"] + dz_i, seg["P_start"] + dz_i]

            band_arr = np.array(band, float)
            clipped = _clean_ring(_clip_poly_zmin(ensure_ccw(band_arr), zmin))
            if clipped is not None:
                layer_polys[(li, zone)] = clipped

            cumulative_d[zone] = d_inner

    # Build solid polygon (same junction-walking logic as build_porous_zone_data)
    zmin_solid = zmin + 1e-6
    R_c = params["R_c"]
    L_back = float(params["L_back"])
    use_collector = bool(params.get("use_collector", ENABLE_COLLECTOR))
    t_wall = float(params["t_wall"])
    c = _compute_clash_coords(params)

    z_wall_top = None
    if R_c is not None and not np.isnan(R_c):
        z_wall_top = float(params["SWL"]) + float(R_c)

    new_top = [(c["x_left"], c["z_deep"])]

    for i, seg in enumerate(segs):
        name = seg["name"]
        d_total = cumulative_d.get(name, 0.0)
        is_porous = name in az_list and d_total > 0

        if is_porous:
            ipts = _inner_pts(seg, d_total, zmin_solid)
        else:
            ipts = []

        if i == 0:
            new_top.append(tuple(seg["P_start"]))
            if is_porous:
                for pt in ipts[:-1]:
                    new_top.append(pt)

        elif is_porous:
            # Start point comes from the junction connector; keep the
            # intermediate points (corner where the boundary reaches zmin)
            for pt in ipts[1:-1]:
                new_top.append(pt)

        end_pt = ipts[-1] if is_porous else tuple(seg["P_end"])
        new_top.append(end_pt)

        if i + 1 < len(segs):
            next_seg = segs[i + 1]
            next_name = next_seg["name"]
            next_d = cumulative_d.get(next_name, 0.0)
            is_porous_next = next_name in az_list and next_d > 0

            if is_porous and not is_porous_next:
                new_top.append(tuple(seg["P_end"]))
            elif not is_porous and is_porous_next:
                ipts_next = _inner_pts(next_seg, next_d, zmin_solid)
                new_top.append(ipts_next[0])
            elif is_porous and is_porous_next:
                ipts_next = _inner_pts(next_seg, next_d, zmin_solid)
                if not np.allclose(end_pt, ipts_next[0], atol=1e-12):
                    new_top.append(ipts_next[0])

    if segs and segs[-1]["name"] in az_list and cumulative_d.get(segs[-1]["name"], 0) > 0:
        new_top.append(tuple(segs[-1]["P_end"]))

    # Append wall, collector, closing
    x_after = c["x_crest_end"]
    wall_pts = []
    collector_pts = []

    if z_wall_top is not None and z_wall_top > c["z_crest"] + 1e-15:
        x_wall0 = c["x_crest_end"]
        x_wall1 = x_wall0 + t_wall
        wall_pts = [
            (x_wall0, z_wall_top),
            (x_wall1, z_wall_top),
            (x_wall1, c["z_crest"]),
        ]
        x_after = x_wall1

    if use_collector and L_back > 0.0:
        z_collector_floor = zmin + COLLECTOR_ZMIN_MARGIN
        if z_collector_floor >= c["z_crest"] - 1e-12:
            z_collector_floor = c["z_crest"] - 0.01
        if z_collector_floor <= zmin + 1e-12:
            z_collector_floor = zmin + 1e-3
        x_collector_end = x_after + L_back
        collector_pts = [
            (x_after, z_collector_floor),
            (x_collector_end, z_collector_floor),
            (x_collector_end, c["z_crest"]),
        ]
        x_after = x_collector_end

    if use_collector:
        x_end = x_after + 0.02
    else:
        x_end = x_after + L_back

    new_top += wall_pts + collector_pts
    new_top.append((x_end, c["z_crest"]) if use_collector else (x_end, zmin))
    solid_raw = new_top + [(x_end, zmin), (c["x_left"], zmin)]
    solid_poly = ensure_ccw(
        np.array(_remove_spikes(remove_consecutive_duplicates(solid_raw)), float)
    )

    # Backward-compat aliases for layer 0
    porous_polys = {}
    for (li, zone), poly in layer_polys.items():
        if li == 0:
            porous_polys[zone] = poly

    return {
        "layer_polys": layer_polys,
        "solid_poly": solid_poly,
        "segments": segs,
        "active_zones": az_list,
        "layer_configs": [dict(l) for l in layers],
        "back_slope_wedge": bool(back_slope_wedge),
        # backward compat
        "porous_polys": porous_polys,
        "merged_poly": None,
    }


def save_layered_porous_stls(
    layered_data: dict,
    Ly: float,
    out_dir,
    case_name: str,
    core_is_solid: bool = True,
) -> dict:
    """Save STLs for each layer-zone combination and the solid/porous core.

    Returns
    -------
    dict with keys:
        solid_stl   : Path | None  (None when core is porous)
        core_stl    : Path | None  (set when core is porous)
        layer_stls  : {(layer_idx, zone_name): Path}
    """
    from pathlib import Path as _Path
    out_dir = _Path(out_dir)
    result = {"solid_stl": None, "core_stl": None, "layer_stls": {}}

    # Core polygon — solid wall or porous zone
    if core_is_solid:
        solid_path = out_dir / f"{case_name}_solid.stl"
        extrude_to_stl(layered_data["solid_poly"], Ly, str(solid_path))
        result["solid_stl"] = solid_path
    else:
        core_path = out_dir / f"{case_name}_core.stl"
        extrude_to_stl(layered_data["solid_poly"], Ly, str(core_path))
        result["core_stl"] = core_path

    # Layer-zone STLs
    layer_configs = layered_data.get("layer_configs", [])
    for (li, zone), poly in layered_data["layer_polys"].items():
        lname = layer_configs[li]["name"] if li < len(layer_configs) else f"layer{li}"
        stl_path = out_dir / f"{case_name}_{lname}_{zone}.stl"
        extrude_to_stl(poly, Ly, str(stl_path))
        result["layer_stls"][(li, zone)] = stl_path

    return result


# OBSTACLE GEOMETRY — separate STL approach
# Obstacles are drawn as standalone XZ polygons, extruded to
# their own STL files, then merged with the main structure STL.
# No polygon-merging into the profile → no triangulation issues.

ANCHOR_POINTS = [
    "crest_seaward",        # (x_u_end, z_crest)
    "crest_landward",       # (x_crest_end, z_crest)
    "crown_wall_seaward",   # (x_wall0, z_wall_top) — falls back to crest_landward
    "crown_wall_landward",  # (x_wall1, z_wall_top) — falls back to back_start
    "back_start",           # (x_back_start, z_crest)
]


def resolve_anchor_point(
    anchor_name: str, params: dict, meta: dict,
) -> tuple[float, float]:
    """Resolve a named anchor to (x, z) coordinates on the profile."""
    SWL = float(params["SWL"])
    z_crest = SWL + float(params["A_c"])

    if anchor_name == "crest_seaward":
        return (meta["x_u_end"], z_crest)
    if anchor_name == "crest_landward":
        return (meta["x_crest_end"], z_crest)
    if anchor_name == "crown_wall_seaward":
        if meta.get("x_wall0") is not None and meta.get("z_wall_top") is not None:
            return (meta["x_wall0"], meta["z_wall_top"])
        return (meta["x_crest_end"], z_crest)
    if anchor_name == "crown_wall_landward":
        if meta.get("x_wall1") is not None and meta.get("z_wall_top") is not None:
            return (meta["x_wall1"], meta["z_wall_top"])
        return (meta["x_back_start"], z_crest)
    if anchor_name == "back_start":
        return (meta["x_back_start"], z_crest)
    raise ValueError(f"Unknown anchor: {anchor_name}")


def extrude_obstacle_stl(
    vertices_xz: list[tuple[float, float]],
    Ly: float,
    out_name: str,
) -> None:
    """Extrude a 2D XZ obstacle polygon to a 3D STL (separate from the structure).

    Parameters
    ----------
    vertices_xz : list[(x, z)]
        Absolute XZ coordinates of the obstacle polygon (closed, CCW).
    Ly : float
        Extrusion thickness in Y [m].
    out_name : str
        Output STL file path.
    """
    poly = ensure_ccw(np.array(
        remove_consecutive_duplicates(vertices_xz), dtype=float,
    ))
    extrude_to_stl(poly, Ly, out_name)


def merge_stl_files(
    stl_paths: list,
    out_path: str,
) -> None:
    """Merge multiple STL files into a single combined STL.

    Simply concatenates triangle facets from all input meshes.

    Parameters
    ----------
    stl_paths : list[str | Path]
        Paths to input STL files.
    out_path : str
        Path for the merged output STL.
    """
    from pathlib import Path
    meshes = []
    for p in stl_paths:
        p = Path(p)
        if p.exists():
            meshes.append(mesh.Mesh.from_file(str(p)))
    if not meshes:
        return
    combined_data = np.concatenate([m.data for m in meshes])
    combined = mesh.Mesh(combined_data)
    combined.save(str(out_path))


def union_obstacles_with_poly(
    struct_poly: np.ndarray,
    obstacle_polys: list,
) -> "np.ndarray | None":
    """
    Compute the 2D XZ boolean union of the structure polygon with all obstacle polygons
    using Shapely. Returns the merged exterior boundary as an Nx2 array (CCW), or None if
    Shapely is unavailable or the union fails.

    This produces a single clean outer boundary with no interior seams — obstacles are
    truly fused into the structure so that segment extraction only shows exposed surfaces.
    """
    try:
        from shapely.geometry import Polygon as SPolygon
        from shapely.ops import unary_union
    except ImportError:
        return None

    try:
        shapes = [SPolygon(struct_poly)]
        for obs in obstacle_polys:
            if len(obs) >= 3:
                shapes.append(SPolygon(obs))

        result = unary_union(shapes)
        if result.is_empty:
            return None

        # Handle both Polygon and MultiPolygon (take the largest)
        from shapely.geometry import MultiPolygon
        if isinstance(result, MultiPolygon):
            result = max(result.geoms, key=lambda g: g.area)

        coords = np.array(result.exterior.coords, dtype=float)
        if len(coords) < 4:
            return None
        # Remove closing duplicate
        if np.allclose(coords[0], coords[-1]):
            coords = coords[:-1]
        return ensure_ccw(coords)
    except Exception:
        return None

