#!/usr/bin/env python3
"""
Figures of a single overtopping event at the overtopping section: field snapshots
around the crest edge, a detailed view of the section, and the matching
discharge, flow depth and velocity time series.
"""

from __future__ import annotations

import argparse
import gzip
import json
import re
from pathlib import Path

import matplotlib
import matplotlib.ticker
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection, PolyCollection
from matplotlib.colors import ListedColormap
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
from matplotlib.ticker import MaxNLocator

# User settings
N_SNAPSHOTS = 9        # snapshots of Figure C (three per row)
SPEED_MAX = None       # top of the speed colour bar [m/s]; None: from the event
DPI = 600              # PNG resolution; the PDF is vector (mesh of Figure D (a): DPI)

ZONE_NAME = "overtopping"
OUT_DIR = "event_figures"

W_FIG = 7.2            # figure width [in]: full text width of a two-column article page
ARROW_PT = 9.0         # length of the velocity arrows [points]
WATER_TOP = 0.55       # water fraction colours: part of the "Blues" map used (1: full map, down to dark navy;
                       # lower: lighter water)
MESH_GREY = 0.80       # grey of the mesh lines (0: black, 1: white)
RC = {"font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "xtick.labelsize": 7.5,
      "ytick.labelsize": 7.5, "legend.fontsize": 7, "axes.grid": False, "savefig.facecolor": "white"}
C_STRUCT, C_ZONE, C_WATER, C_U, C_EVENT = "#c8a27a", "#d62728", "#1f77b4", "#7b3294", "0.85"
C_ARROW = "#ff8c00"


def _water_cmap():
    return ListedColormap(matplotlib.colormaps["Blues"](np.linspace(0, WATER_TOP, 256)), name="water")


# OpenFOAM ascii files
def _read_text(path):
    path = Path(path)
    if not path.exists() and path.with_name(path.name + ".gz").exists():
        with gzip.open(path.with_name(path.name + ".gz"), "rt", errors="ignore") as f:
            txt = f.read()
    else:
        txt = path.read_text(errors="ignore")
    i = txt.find("FoamFile")
    if i >= 0:                                            # skip the header (it may hold a 'meta { }' block)
        k, depth = txt.index("{", i) + 1, 1
        while depth:
            depth += {"{": 1, "}": -1}.get(txt[k], 0)
            k += 1
        if re.search(r"format\s+binary", txt[i:k]):
            raise RuntimeError(f"{path}: binary format; the event figures need writeFormat ascii")
        txt = txt[k:]
    return txt


def _exists(path):
    path = Path(path)
    return path.exists() or path.with_name(path.name + ".gz").exists()


def _list_block(txt, start=0):
    """(n, body) of the first list 'N (' ... ')' after position start."""
    m = re.compile(r"(\d+)\s*\(").search(txt, start)
    n, a = int(m.group(1)), m.end()
    depth, k = 1, a
    while depth:
        c = txt[k]
        if c == "(":
            depth += 1
        elif c == ")":
            depth -= 1
        k += 1
    return n, txt[a:k - 1]


def read_points(path):
    n, body = _list_block(_read_text(path))
    return np.array(body.replace("(", " ").replace(")", " ").split(), float).reshape(n, 3)


def read_faces(path):
    n, body = _list_block(_read_text(path))
    return [list(map(int, f.split())) for f in re.findall(r"\d+\s*\(([\d\s]+)\)", body)][:n]


def read_labels(path):
    n, body = _list_block(_read_text(path))
    return np.array(body.split()[:n], np.int64)


def read_boundary(path):
    """{patch: (startFace, nFaces, type)}"""
    out = {}
    for name, blk in re.findall(r"(\w+)\s*\{([^}]*)\}", _read_text(path)):
        nf, sf = re.search(r"nFaces\s+(\d+)", blk), re.search(r"startFace\s+(\d+)", blk)
        tp = re.search(r"\btype\s+(\w+)", blk)
        if nf and sf:
            out[name] = (int(sf.group(1)), int(nf.group(1)), tp.group(1) if tp else "")
    return out


def read_facezone(path, name=ZONE_NAME):
    """Face labels of the faceZone 'name' (empty if the file or the zone is missing)."""
    if not _exists(path):
        return np.zeros(0, np.int64)
    txt = _read_text(path)
    m = re.search(rf"\b{re.escape(name)}\s*\{{", txt)
    if m is None:
        return np.zeros(0, np.int64)
    j = txt.index("faceLabels", m.end())
    n, body = _list_block(txt, j)
    return np.array(body.split()[:n], np.int64)


def read_internal_field(path, ncells=None):
    """internalField of a vol/surface scalar or vector field."""
    txt = _read_text(path)
    i = txt.index("internalField")
    first = txt[i:txt.index(";", i) if ";" in txt[i:i + 300] else i + 300]
    if "nonuniform" not in first:
        m = re.search(r"uniform\s+(\(([^)]*)\)|[-\d.eE+]+)", first)
        val = np.array(m.group(1).strip("()").split(), float)
        if ncells is None:
            raise RuntimeError(f"{path}: uniform field, size unknown")
        return np.tile(val, (ncells, 1)) if val.size == 3 else np.full(ncells, val[0])
    vec = "List<vector>" in txt[i:i + 200]
    n, body = _list_block(txt, i)
    a = np.array(body.replace("(", " ").replace(")", " ").split(), float)
    return a.reshape(n, 3) if vec else a[:n]


def _is_time(name):
    return re.fullmatch(r"[-+]?(\d+\.?\d*|\.\d+)([eE][-+]?\d+)?", name) is not None


def _processors(case):
    p = [d for d in Path(case).glob("processor*") if d.is_dir() and d.name[9:].isdigit()]
    return sorted(p, key=lambda d: int(d.name[9:]))


def field_times(case):
    """{time: folder name} of the full-field outputs (alpha.water and U written)."""
    case = Path(case)
    procs = _processors(case)
    base = procs[0] if procs else case
    out = {}
    for d in base.iterdir():
        if d.is_dir() and _is_time(d.name) and float(d.name) > 0:
            if _exists(d / "alpha.water") and _exists(d / "U"):
                out[float(d.name)] = d.name
    return dict(sorted(out.items()))


# Mesh, overtopping section and case data
class Mesh2D:
    """Mesh of a 2D COAST case as polygons in the x-z plane, and its full fields."""

    def __init__(self, case):
        case = Path(case)
        pm = case / "constant" / "polyMesh"
        self.case = case
        points = read_points(pm / "points")
        faces = read_faces(pm / "faces")
        owner = read_labels(pm / "owner")
        bnd = read_boundary(pm / "boundary")
        front = "front" if "front" in bnd else next((k for k, v in bnd.items() if v[2] == "empty"), None)
        if front is None:
            raise RuntimeError("No 'front' (empty) patch in constant/polyMesh/boundary: not a 2D case")
        s0, nf, _ = bnd[front]
        self.ncells = int(owner.max()) + 1
        poly = [None] * self.ncells
        for f in range(s0, s0 + nf):
            poly[owner[f]] = points[faces[f]][:, [0, 2]]
        missing = [k for k, p in enumerate(poly) if p is None]
        if missing:
            raise RuntimeError(f"{len(missing)} cells have no face on the patch '{front}'")
        self.poly = poly
        self.centre = np.array([p.mean(0) for p in poly])
        P = points[:, [0, 2]]
        self.xlim = (float(P[:, 0].min()), float(P[:, 0].max()))
        self.zlim = (float(P[:, 1].min()), float(P[:, 1].max()))
        self.width_y = float(points[:, 1].max() - points[:, 1].min())
        self.procs = _processors(case)
        self._addr = None

    def _addressing(self):
        if self._addr is None:
            self._addr = [read_labels(p / "constant/polyMesh/cellProcAddressing") for p in self.procs]
        return self._addr

    def field(self, tname, name):
        """Cell values of a field at the output time 'tname' (decomposed or reconstructed case)."""
        if not self.procs:
            return read_internal_field(self.case / tname / name, self.ncells)
        out = None
        for pdir, addr in zip(self.procs, self._addressing()):
            v = read_internal_field(pdir / tname / name, len(addr))
            if out is None:
                out = np.full((self.ncells,) + v.shape[1:], np.nan)
            out[addr] = v
        return out

    def in_window(self, xlim, zlim, margin=0.0):
        c = self.centre
        return np.where((c[:, 0] > xlim[0] - margin) & (c[:, 0] < xlim[1] + margin) &
                        (c[:, 1] > zlim[0] - margin) & (c[:, 1] < zlim[1] + margin))[0]


class OvertoppingSection:
    """Faces of the faceZone 'overtopping' (the section of overtopping.dat), read in the
    processor folders that hold them (or in constant/polyMesh when reconstructed)."""

    def __init__(self, case, name=ZONE_NAME):
        case = Path(case)
        self.parts = []                                   # (folder, face labels in that folder)
        segs, nx = [], []
        for folder in (_processors(case) or [case]):
            pm = folder / "constant" / "polyMesh"
            lab = read_facezone(pm / "faceZones", name)
            if lab.size == 0:
                continue
            pts, fcs = read_points(pm / "points"), read_faces(pm / "faces")
            for f in lab:
                P = pts[fcs[f]]
                a = 0.5 * sum(np.cross(P[k], P[(k + 1) % len(P)]) for k in range(len(P)))
                nx.append(a[0] / np.linalg.norm(a))
                y0 = P[:, 1].min()
                segs.append(P[P[:, 1] < y0 + 0.5 * (P[:, 1].max() - y0)][:, [0, 2]])
            self.parts.append((folder, lab))
        if not self.parts:
            raise RuntimeError(f"faceZone '{name}' not found in constant/polyMesh/faceZones")
        self.segments = segs
        self.nx = np.array(nx)

    def fluxes(self, tname, ly):
        """Water flux through each face, q_f = (alpha phi)_f n_x,f / Ly [m^3/s per m];
        None if alphaPhi is not written. NaN for a face on a processor boundary."""
        out = []
        for folder, lab in self.parts:
            d = folder / tname
            aname = next((n for n in ("alphaPhi0.water", "alphaPhi.water") if _exists(d / n)), None)
            if aname is None:
                return None
            aphi = read_internal_field(d / aname, int(lab.max()) + 1)
            out += [aphi[f] if f < len(aphi) else np.nan for f in lab]
        return np.array(out) * self.nx / ly


def read_section_line(case):
    """(x, z_bottom, z_top) of the first set of the 'line' function (x01), as in Step 4."""
    txt = (Path(case) / "system" / "controlDict").read_text(errors="ignore")
    m = re.search(r"^\s*line\s*\{", txt, flags=re.MULTILINE)
    if m is None:
        raise RuntimeError("No 'line' function in system/controlDict")
    pat = re.compile(r"(\w+)\s*\{\s*type\s+uniform;.*?start\s*\(([^)]*)\)\s*;\s*end\s*\(([^)]*)\)\s*;", re.S)
    sets = sorted(pat.findall(txt[m.end():]))
    if not sets:
        raise RuntimeError("No uniform line set in the 'line' function of system/controlDict")
    _, start, end = sets[0]
    x0, _, z0 = map(float, start.split())
    z1 = float(end.split()[2])
    return x0, min(z0, z1), max(z0, z1)


def read_case_info(case, mesh=None):
    """Section, still water level, width and cell size at the crest of the case."""
    case = Path(case)
    x_sec, z_bot, z_top = read_section_line(case)
    info = dict(x_sec=x_sec, z_bot=z_bot, z_top=z_top, swl=None, ly=None, cell=None, x_toe=None)
    try:
        run = json.loads((case / "run_info.json").read_text())
        g = run.get("domain_grid", {})
        info["swl"] = float(g["SWL"]) if "SWL" in g else None
        info["ly"] = float(g["Ly"]) if "Ly" in g else None
        try:
            info["cell"] = min(float(g["dx_target"]), float(g["dz_target"])) / 2 ** int(g["LEVEL_RUNUP"])
        except (KeyError, TypeError, ValueError):
            pass
        xt = run.get("structure_geometry", {}).get("x_toe")
        info["x_toe"] = float(xt) if xt is not None else None
    except (OSError, ValueError):
        pass
    if info["ly"] is None:
        info["ly"] = mesh.width_y if mesh is not None else 1.0
    return info


def read_overtopping_dat(case):
    """t and q+ = max(Q_x, 0) / Ly from every postProcessing/overtopping/<start>/overtopping.dat."""
    base = Path(case) / "postProcessing" / "overtopping"
    T, Q = [], []
    for d in sorted((p for p in base.iterdir() if p.is_dir() and _is_time(p.name)), key=lambda p: float(p.name)):
        f = d / "overtopping.dat"
        if f.exists():
            a = re.findall(r"^\s*([-+\d.eE]+)\s+\(\s*([-+\d.eE]+)", f.read_text(errors="ignore"), re.M)
            if a:
                a = np.array(a, float)
                T.append(a[:, 0])
                Q.append(a[:, 1])
    if not T:
        raise FileNotFoundError("postProcessing/overtopping/*/overtopping.dat not found")
    t, q = np.concatenate(T), np.concatenate(Q)
    t, i = np.unique(t[::-1], return_index=True)        # restart: keep the last value written
    return t, q[::-1][i]


def read_events(case):
    """Events of Step 1, with h_max and u_max of Step 4 when available."""
    case = Path(case)
    ev = pd.read_csv(case / "overtopping_events.csv")
    f = case / "flow_depth_velocity_events.csv"
    if f.exists():
        fd = pd.read_csv(f)
        keep = [c for c in ("event_id", "h_max", "u_max", "t_h_max") if c in fd.columns]
        ev = ev.merge(fd[keep], on="event_id", how="left")
    return ev.sort_values("event_id").reset_index(drop=True)


class CaseData:
    """Everything the two figures need, read once per case."""

    def __init__(self, case, mesh=None, section=None):
        self.case = Path(case)
        self.mesh = mesh if mesh is not None else Mesh2D(self.case)
        self.section = section if section is not None else OvertoppingSection(self.case)
        self.info = read_case_info(self.case, self.mesh)
        self.times = field_times(self.case)
        t, q = read_overtopping_dat(self.case)
        self.t_q, self.q_pos = t, np.maximum(q, 0.0) / self.info["ly"]
        ts = pd.read_csv(self.case / "flow_depth_velocity_timeseries.csv")
        self.t_l, self.h, self.u = ts["t"].to_numpy(), ts["h"].to_numpy(), ts["u"].to_numpy()
        self.events = read_events(self.case)


# Windows and snapshot times
def default_windows(info, mesh):
    """Windows (xlim, zlim) of the figures, scaled with the length Lz of line x01.
    (With Lz = 0.20 m they are the windows of the COAST article figures.)"""
    x, zb, zt = info["x_sec"], info["z_bot"], info["z_top"]
    lz = zt - zb
    x_toe = info.get("x_toe")
    xa0 = x_toe - 0.25 * (x - x_toe) if x_toe is not None and x_toe < x else mesh.xlim[0]
    return {
        "snapshots": ((x - lz, x + lz), (zb - 0.75 * lz, zb + 0.70 * lz)),
        "zoom": ((x - 0.575 * lz, x + 0.625 * lz), (zb - 0.20 * lz, zb + 1.20 * lz)),
        "overview": ((max(xa0, mesh.xlim[0]), mesh.xlim[1]), (mesh.zlim[0], min(zt + 0.70 * lz, mesh.zlim[1]))),
    }


def select_frames(times, t_start, t_end, n=N_SNAPSHOTS, t_first=None, step=None):
    """Full-field output times shown in Figure C.
    Default: n outputs, every k-th one, from the output just before t_start; k is the
    number of outputs of the event divided by n, rounded, and the last snapshot is at
    most one step after the first output after t_end (short events get fewer).
    t_first, step [s]: the outputs nearest t_first + i*step, i = 0..n-1."""
    tt = np.array(sorted(times))
    if tt.size == 0:
        return []
    if t_first is not None and step:
        want = t_first + step * np.arange(n)
        idx = np.unique([int(np.argmin(np.abs(tt - w))) for w in want])
        return [float(v) for v in tt[idx]]
    i0 = max(int(np.searchsorted(tt, t_start, side="right")) - 1, 0)
    i1 = min(int(np.searchsorted(tt, t_end, side="left")), tt.size - 1)
    k = max(1, int(round((i1 - i0 + 1) / n)))
    idx = np.arange(i0, min(i1 + k, tt.size - 1) + 1, k)[:n]
    return [float(v) for v in tt[idx]]


def snapshot_count(times_arr, t_start, t_end):
    """Number of full-field outputs between the event window bounds (inclusive of the
    output just before and just after)."""
    if times_arr.size == 0:
        return 0
    i0 = max(int(np.searchsorted(times_arr, t_start, side="right")) - 1, 0)
    i1 = min(int(np.searchsorted(times_arr, t_end, side="left")), times_arr.size - 1)
    return i1 - i0 + 1


def _event_maxima(d, e):
    """h_max, t_h_max, u_max, t_u_max of the event on line x01 (as in Step 4)."""
    m = (d.t_l >= e.t_start) & (d.t_l <= e.t_end)
    if not m.any() and d.t_l.size:
        dt = float(np.median(np.diff(d.t_l))) if d.t_l.size > 1 else 0.0
        m = np.abs(d.t_l - 0.5 * (e.t_start + e.t_end)) <= 0.5 * dt
    if not m.any():
        return np.nan, np.nan, np.nan, np.nan
    th, hh, uu = d.t_l[m], d.h[m], d.u[m]
    ih = int(np.argmax(hh))
    if np.isfinite(uu).any():
        iu = int(np.nanargmax(uu))
        return float(hh[ih]), float(th[ih]), float(uu[iu]), float(th[iu])
    return float(hh[ih]), float(th[ih]), np.nan, np.nan


def _power(vmax):
    """(factor, text) to write small values as multiples of 10^k (k <= -3) on an axis."""
    if not np.isfinite(vmax) or vmax <= 0 or vmax >= 0.01:
        return 1.0, ""
    k = int(np.floor(np.log10(vmax)))
    sup = str(k).replace("-", "⁻").translate(str.maketrans("0123456789", "⁰¹²³⁴⁵⁶⁷⁸⁹"))
    return 10.0 ** -k, f"10{sup} "


def _nice_ceil(v):
    if not np.isfinite(v) or v <= 0:
        return 1.0
    e = 10.0 ** np.floor(np.log10(v))
    return next(m * e for m in (1, 1.5, 2, 2.5, 3, 4, 5, 6, 8, 10) if m * e >= v * (1 - 1e-9))


# Drawing
def _cells(ax, mesh, values, sel, lw, edge=None, rasterized=False):
    edge = str(MESH_GREY) if edge is None else edge
    pc = PolyCollection([mesh.poly[k] for k in sel], array=np.nan_to_num(values[sel]), cmap=_water_cmap(),
                        clim=(0, 1), edgecolor=edge, linewidth=lw, rasterized=rasterized)
    ax.add_collection(pc)
    ax.set_facecolor(C_STRUCT)
    return pc


def _arrow_cells(mesh, sel, alpha, xlim, zlim, origin, bin_size):
    """One wet cell (alpha >= 0.5) per square display bin: the cell nearest the bin centre.
    Its own velocity is shown (no averaging); the bins only limit the overlap of arrows."""
    c = mesh.centre
    a = np.nan_to_num(alpha[sel])
    wet = sel[(a >= 0.5) & (c[sel, 0] >= xlim[0]) & (c[sel, 0] <= xlim[1]) &
              (c[sel, 1] >= zlim[0]) & (c[sel, 1] <= zlim[1])]
    if wet.size == 0:
        return wet
    bins = np.floor((c[wet] - origin) / bin_size).astype(int)
    centres = origin + (bins + 0.5) * bin_size
    order = np.argsort(np.sum((c[wet] - centres) ** 2, axis=1), kind="stable")
    _, first = np.unique(bins[order], axis=0, return_index=True)
    return wet[order[first]]


def _arrow_components(ux, uz):
    """Arrows of equal length (ARROW_PT) in the exact x-z direction of the velocity [in]."""
    speed = np.hypot(ux, uz)
    moving = np.isfinite(speed) & (speed > 0)
    f = np.divide(ARROW_PT / 72.0, speed, out=np.zeros_like(speed), where=moving)
    return np.where(moving, ux * f, 0.0), np.where(moving, uz * f, 0.0)


def _locators(ax):
    ax.xaxis.set_major_locator(MaxNLocator(nbins=4, steps=[1, 2, 2.5, 5, 10]))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6, steps=[1, 2, 2.5, 5, 10]))


def figure_snapshots(d, e, frames, fields, fluxes, window, speed_max=None, maxima=None):
    """Figure C. fields: {t: (alpha, U)}; fluxes: {t: q_f or None}. Returns (fig, notes)."""
    notes = []
    mesh, info = d.mesh, d.info
    xlim, zlim = window
    lz = info["z_top"] - info["z_bot"]
    sel = mesh.in_window(xlim, zlim, margin=0.1 * (xlim[1] - xlim[0]))
    bin_size = (xlim[1] - xlim[0]) / 16
    origin = np.array([info["x_sec"], info["z_bot"]])
    arrows = {tt: _arrow_cells(mesh, sel, fields[tt][0], xlim, zlim, origin, bin_size) for tt in frames}
    speeds = {tt: np.hypot(fields[tt][1][w, 0], fields[tt][1][w, 2]) for tt, w in arrows.items()}
    v_max = max((float(np.nanmax(v)) if np.isfinite(v).any() else 0.0) for v in speeds.values())
    if speed_max is None or speed_max <= 0:
        speed_max = _nice_ceil(v_max)
    elif v_max > speed_max:
        notes.append(f"Cell speeds up to {v_max:.2f} m/s exceed the colour bar (0-{speed_max:g} m/s): "
                     f"the fastest arrows have the top colour.")
    speed_norm = matplotlib.colors.Normalize(vmin=0, vmax=speed_max)
    speed_map = matplotlib.cm.ScalarMappable(norm=speed_norm, cmap="autumn_r")
    have_flux = all(fluxes[tt] is not None for tt in frames)
    if not have_flux:
        notes.append("alphaPhi0.water is not written at these times: the faces of the section are not "
                     "coloured by their water flux.")
    qf_lim = max((float(np.nanmax(fluxes[tt])) if have_flux else 0.0) for tt in frames)
    qf_lim = qf_lim if qf_lim > 0 else 1e-6

    # layout [in]
    L, R, gap_x = 0.64, 0.06, 0.06
    ncol = min(3, len(frames))
    nrow = int(np.ceil(len(frames) / ncol))
    pw = (W_FIG - L - R - (ncol - 1) * gap_x) / ncol
    ph = pw * (zlim[1] - zlim[0]) / (xlim[1] - xlim[0])
    top_leg, title_h, gap_row = 0.80, 0.17, 0.05
    snap_h = nrow * (title_h + ph) + (nrow - 1) * gap_row
    mid, ts_ph, ts_gap, bot = 0.72, 1.0, 0.10, 0.36
    H = top_leg + snap_h + mid + 3 * ts_ph + 2 * ts_gap + bot

    def box(x, y_top, w, h):                                 # inches from the top-left corner
        return [x / W_FIG, 1 - (y_top + h) / H, w / W_FIG, h / H]

    fr = np.array(frames)
    spacing = float(np.median(np.diff(fr))) if fr.size > 1 else 0.1 * max(e.t_end - e.t_start, 0.1)
    dec = int(max(1, min(4, -np.floor(np.log10(spacing) + 1e-9))))      # decimals of the snapshot times

    with matplotlib.rc_context(RC):
        fig = Figure(figsize=(W_FIG, H))
        pc = lc = None
        for n, tt in enumerate(frames):
            r, cc = divmod(n, ncol)
            ax = fig.add_axes(box(L + cc * (pw + gap_x), top_leg + r * (title_h + ph + gap_row) + title_h, pw, ph))
            a, U = fields[tt]
            pc = _cells(ax, mesh, a, sel, 0.22)
            w = arrows[tt]
            if w.size:
                du, dz = _arrow_components(U[w, 0], U[w, 2])
                mv = np.hypot(du, dz) > 0
                ax.quiver(mesh.centre[w[mv], 0], mesh.centre[w[mv], 1], du[mv], dz[mv], speeds[tt][mv],
                          angles="xy", scale_units="inches", units="inches", scale=1,
                          cmap=speed_map.cmap, norm=speed_norm, width=0.009, headwidth=3.8, headlength=4.5,
                          headaxislength=4, minlength=0, edgecolor="none", linewidth=0,
                          pivot="tail", zorder=4)
            if have_flux:
                ax.add_collection(LineCollection(d.section.segments, colors="k", lw=3.4, zorder=5))
                lc = LineCollection(d.section.segments, array=fluxes[tt], cmap="Greens", lw=2.2, zorder=5)
                lc.set_clim(0, qf_lim)
                ax.add_collection(lc)
            else:
                ax.add_collection(LineCollection(d.section.segments, colors=C_ZONE, lw=2.6, zorder=5))
            ax.set_xlim(*xlim)
            ax.set_ylim(*zlim)
            ax.set_aspect("equal")
            ax.set_title(f"({n + 1})  t = {tt:.{dec}f} s", pad=2)
            ax.tick_params(labelleft=cc == 0, labelbottom=n + ncol >= len(frames), length=2, labelsize=7)
            _locators(ax)
            if cc == 0:
                ax.set_ylabel("z [m]")
            if n + ncol >= len(frames):
                ax.set_xlabel("x [m]", labelpad=1)
        handles = [Patch(fc=C_STRUCT, label="structure"),
                   Line2D([], [], color="k" if have_flux else C_ZONE, lw=3.2, label="overtopping section"),
                   Line2D([], [], color=C_ARROW, lw=1.0, marker=">", ms=3,
                          label="velocity at selected wet cells (α ≥ 0.5)")]
        fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(L / W_FIG - 0.01, 1 - 0.03 / H), ncol=3,
                   frameon=False, fontsize=6.8, handlelength=1.8, columnspacing=1.2)
        f_fac, f_txt = _power(qf_lim)
        bars = [(pc, "Water fraction α [-]")] + ([(lc, f"Face flux [{f_txt}m³/s per m]")] if have_flux else []) + \
               [(speed_map, "Cell speed [m/s]")]
        bar_gap = 0.45
        bar_w = (W_FIG - L - R - 2 * bar_gap) / 3
        for k, (mp, lab) in enumerate(bars):
            cax = fig.add_axes(box(L + k * (bar_w + bar_gap), 0.48, bar_w, 0.09))
            cb = fig.colorbar(mp, cax=cax, orientation="horizontal")
            cax.set_title(lab, fontsize=6.8, pad=3)
            cax.tick_params(labelsize=6.3, length=2, pad=1)
            if mp is lc:
                cax.xaxis.set_major_locator(MaxNLocator(nbins=4, steps=[1, 2, 4, 5, 10]))
                cax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v * f_fac:g}"))

        # time series of the event on a common time axis
        y_ts = top_leg + snap_h + mid
        cell = info.get("cell")
        h_scale, h_unit = (100.0, "cm") if np.nanmax(np.r_[d.h, 0]) < 1.0 else (1.0, "m")
        th = [Patch(fc=C_EVENT, alpha=0.5, label="event window"),
              Patch(fc=C_ZONE, alpha=0.15, label="event volume"),
              Line2D([], [], color="k", marker="*", ms=6, ls="", label="maximum reported by COAST")]
        if cell:
            th.append(Line2D([], [], color="k", ls=":", lw=0.7, label=f"one cell ({cell * h_scale:.3g} {h_unit})"))
        fig.legend(handles=th, loc="lower left", bbox_to_anchor=(L / W_FIG - 0.01, 1 - (y_ts - 0.2) / H),
                   ncol=len(th), frameon=False, fontsize=6.8, handlelength=1.5, columnspacing=1.2,
                   handletextpad=0.5)
        t0 = min(fr[0], e.t_start) - 1.2 * spacing
        t1 = max(fr[-1], e.t_end) + 1.2 * spacing
        k = (d.t_q > t0) & (d.t_q < t1)
        kl = (d.t_l > t0) & (d.t_l < t1)
        hm, thm, um, tum = maxima if maxima is not None else _event_maxima(d, e)
        axs = [fig.add_axes(box(L, y_ts + j * (ts_ph + ts_gap), W_FIG - L - R, ts_ph)) for j in range(3)]
        q_top = d.q_pos[k].max() if k.any() and d.q_pos[k].max() > 0 else 1e-3
        q_fac, q_txt = _power(q_top)
        axs[0].plot(d.t_q[k], d.q_pos[k] * q_fac, color=C_ZONE, lw=1.1)
        ke = k & (d.t_q >= e.t_start) & (d.t_q <= e.t_end)
        axs[0].fill_between(d.t_q[ke], 0, d.q_pos[ke] * q_fac, color=C_ZONE, alpha=0.15, lw=0)
        axs[0].plot(fr, np.interp(fr, d.t_q, d.q_pos) * q_fac, "o", ms=3.5, mfc=C_ZONE, mec="k", mew=0.5)
        axs[0].set_ylim(0, q_top * q_fac * 1.15)
        ifr = [int(np.argmin(np.abs(d.t_l - tt))) for tt in fr] if d.t_l.size else []
        axs[1].plot(d.t_l[kl], d.h[kl] * h_scale, color=C_WATER, lw=1.1, marker="o", ms=1.6)
        axs[1].plot(fr, d.h[ifr] * h_scale, "o", ms=3.5, mfc=C_WATER, mec="k", mew=0.5)
        if cell:
            axs[1].axhline(cell * h_scale, color="k", ls=":", lw=0.7)
        if np.isfinite(hm):
            axs[1].plot(thm, hm * h_scale, "*", color="k", ms=7)
            axs[1].annotate(f"$h_{{max}}$ = {hm * h_scale:.3g} {h_unit}", (thm, hm * h_scale), xytext=(6, 1),
                            textcoords="offset points", fontsize=7, va="bottom", ha="left")
        top_h = max(hm if np.isfinite(hm) else 0.0, cell or 0.0) * h_scale
        axs[1].set_ylim(0, top_h * 1.4 if top_h > 0 else 1.0)
        axs[2].plot(d.t_l[kl], d.u[kl], color=C_U, lw=1.1, marker="o", ms=1.6)
        uf = d.u[ifr] if len(ifr) else np.zeros(0)
        ok = np.isfinite(uf)
        axs[2].plot(fr[ok], uf[ok], "o", ms=3.5, mfc=C_U, mec="k", mew=0.5)
        if np.isfinite(um):
            axs[2].plot(tum, um, "*", color="k", ms=7)
            axs[2].annotate(f"$u_{{max}}$ = {um:.2f} m/s", (tum, um), xytext=(6, 1), textcoords="offset points",
                            fontsize=7, va="bottom", ha="left")
        u_lo = min(0.0, np.nanmin(d.u[kl])) if np.isfinite(d.u[kl]).any() else 0.0
        axs[2].set_ylim(u_lo * 1.4, um * 1.4 if np.isfinite(um) and um > 0 else 1.0)
        if not np.isfinite(um):
            axs[2].text(0.5, 0.5, "u not resolved during this event (layer too thin)", transform=axs[2].transAxes,
                        ha="center", va="center", fontsize=7, color="0.3")
        for j, (ax, yl) in enumerate(zip(axs, [f"$q^+$\n[{q_txt}m³/s per m]", f"$h$\n[{h_unit}]", "$u$\n[m/s]"])):
            ax.set_ylabel(yl, linespacing=1.1)
            ax.text(0.012, 0.95, f"({chr(ord('a') + j)})", transform=ax.transAxes, ha="left", va="top",
                    fontsize=8, fontweight="bold", zorder=10)
            ax.axvspan(e.t_start, e.t_end, color=C_EVENT, alpha=0.5, lw=0, zorder=0)
            for n, tt in enumerate(frames):
                ax.axvline(tt, color="0.4", ls=":", lw=0.7)
                if j == 0:
                    ax.text(tt, 1.02, f"({n + 1})", transform=ax.get_xaxis_transform(), ha="center",
                            va="bottom", fontsize=7)
            ax.set_xlim(t0, t1)
            ax.tick_params(labelbottom=j == 2, length=2)
            ax.grid(True, alpha=0.25, lw=0.5)
            ax.yaxis.set_major_locator(MaxNLocator(nbins=4, steps=[1, 2, 2.5, 5, 10]))
            ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
            ax.yaxis.set_label_coords(-0.05, 0.5)
        axs[2].set_xlabel("time [s]")
    return fig, notes


def figure_section(d, t, alpha, zoom, overview):
    """Figure D: (a) the flow over the structure at t, with the mesh; (b) zoom on the
    crest edge with the mesh cells and the overtopping section."""
    mesh, info = d.mesh, d.info
    asp = [(w[1][1] - w[1][0]) / (w[0][1] - w[0][0]) for w in (overview, zoom)]
    L, R, G, top, bot = 0.5, 0.75, 0.55, 0.50, 0.38
    hp = (W_FIG - L - R - G) / (1 / asp[0] + 1 / asp[1])
    wa, wb = hp / asp[0], hp / asp[1]
    H = top + hp + bot
    with matplotlib.rc_context(RC):
        fig = Figure(figsize=(W_FIG, H))
        axa = fig.add_axes([L / W_FIG, bot / H, wa / W_FIG, hp / H])
        axb = fig.add_axes([(L + wa + G) / W_FIG, bot / H, wb / W_FIG, hp / H])
        m_a = 0.03 * (overview[0][1] - overview[0][0])
        pc = _cells(axa, mesh, alpha, mesh.in_window(*overview, margin=m_a), 0.08,
                    edge=str(0.75 * MESH_GREY), rasterized=True)          # thin lines: a little darker
        if info.get("swl") is not None:
            axa.axhline(info["swl"], color="k", ls=":", lw=0.8)
        axa.add_patch(Rectangle((zoom[0][0], zoom[1][0]), zoom[0][1] - zoom[0][0], zoom[1][1] - zoom[1][0],
                                fill=False, ec="k", lw=0.9))
        axa.annotate("(b)", (zoom[0][0], zoom[1][1]), xytext=(0, 2), textcoords="offset points",
                     fontsize=7.5, va="bottom")
        _cells(axb, mesh, alpha, mesh.in_window(*zoom, margin=0.1 * (zoom[0][1] - zoom[0][0])), 0.3)
        axb.add_collection(LineCollection(d.section.segments, colors=C_ZONE, lw=2.8))
        for ax, (xl, zl) in ((axa, overview), (axb, zoom)):
            ax.set_xlim(*xl)
            ax.set_ylim(*zl)
            ax.set_aspect("equal")
            ax.set_xlabel("x [m]")
            ax.set_ylabel("z [m]")
            ax.tick_params(length=2)
        axb.xaxis.set_major_locator(MaxNLocator(nbins=3, steps=[1, 2, 2.5, 5, 10]))
        axa.set_title(f"(a) Water fraction at t = {t:g} s", pad=3)
        axb.set_title("(b) Zoom on the crest edge", pad=3)
        handles = [Patch(fc=C_STRUCT, label="structure")]
        if info.get("swl") is not None:
            handles.append(Line2D([], [], color="k", ls=":", lw=0.8, label="still water level (SWL)"))
        handles.append(Line2D([], [], color=C_ZONE, lw=2.6, label="overtopping section"))
        fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(L / W_FIG - 0.01, 1 - 0.03 / H),
                   ncol=len(handles), frameon=False, fontsize=6.8, handlelength=1.8, columnspacing=1.0)
        cax = fig.add_axes([(W_FIG - R + 0.20) / W_FIG, bot / H, 0.10 / W_FIG, hp / H])
        cb = fig.colorbar(pc, cax=cax, orientation="vertical")
        cb.set_label("Water fraction α [-]", fontsize=6.8, labelpad=5)
        cax.tick_params(labelsize=6.3, length=2, pad=2)
    return fig


# Driver
def make_event_figures(case, event_id, n_snapshots=N_SNAPSHOTS, speed_max=SPEED_MAX, windows=None,
                       t_first=None, step=None, out_dir=None, data=None, dpi=DPI):
    """Figures C and D of one event. Returns a dict with the files written, the snapshot
    times, the event values and notes. 'data': a CaseData already read (the app caches it)."""
    d = data if data is not None else CaseData(case)
    sel = d.events[d.events.event_id == int(event_id)]
    if sel.empty:
        raise ValueError(f"Event {event_id} is not in overtopping_events.csv")
    e = sel.iloc[0]
    win = default_windows(d.info, d.mesh)
    win.update(windows or {})
    notes = []
    if not d.times:
        raise RuntimeError("No full-field output (alpha.water and U) in the time folders")
    frames = select_frames(list(d.times), e.t_start, e.t_end, n_snapshots, t_first, step)
    fields, fluxes = {}, {}
    for tt in frames:
        tn = d.times[tt]
        try:
            fields[tt] = (d.mesh.field(tn, "alpha.water"), d.mesh.field(tn, "U"))
            fluxes[tt] = d.section.fluxes(tn, d.info["ly"])
        except (OSError, ValueError, RuntimeError, IndexError) as err:
            notes.append(f"Output {tn} s skipped (cannot be read: {err}).")
    frames = [tt for tt in frames if tt in fields]
    if not frames:
        raise RuntimeError("None of the snapshot times could be read")
    dt_w = float(np.median(np.diff(list(d.times)))) if len(d.times) > 1 else np.nan
    inside = [tt for tt in frames if e.t_start <= tt <= e.t_end]
    if len(inside) < 3:
        notes.append(f"Only {len(inside)} full-field output(s) inside the event window "
                     f"({e.t_start:.2f}-{e.t_end:.2f} s, duration {e.t_end - e.t_start:.2f} s): the fields are "
                     f"written every {dt_w:.3g} s (controlDict writeInterval).")
    maxima = _event_maxima(d, e)

    # Find peak discharge and the nearest field output for Figure D.
    m = (d.t_q >= e.t_start) & (d.t_q <= e.t_end)
    t_peak = float(d.t_q[m][np.argmax(d.q_pos[m])]) if m.any() else 0.5 * (e.t_start + e.t_end)
    t_sec = min(d.times, key=lambda tt: abs(tt - t_peak))
    a_sec = fields[t_sec][0] if t_sec in fields else d.mesh.field(d.times[t_sec], "alpha.water")

    out = Path(out_dir) if out_dir is not None else d.case / OUT_DIR
    out.mkdir(parents=True, exist_ok=True)
    stem = f"event{int(e.event_id)}"
    fig_c, n_c = figure_snapshots(d, e, frames, fields, fluxes, win["snapshots"], speed_max, maxima)
    notes += n_c
    fig_d = figure_section(d, t_sec, a_sec, win["zoom"], win["overview"])
    files = {}
    for key, fig in (("snapshots", fig_c), ("section", fig_d)):
        for ext in ("pdf", "png"):
            p = out / f"{stem}_{key}.{ext}"
            fig.savefig(p, dpi=dpi)
            files[f"{key}_{ext}"] = p

    hm, thm, um, tum = maxima
    lines = [
        f"Event {int(e.event_id)} of {len(d.events)} (overtopping_events.csv, ranked by volume)",
        f"Event window: {e.t_start:.3f}-{e.t_end:.3f} s (duration {e.t_end - e.t_start:.3f} s)",
        f"Volume V = {e.volume:.4e} m3 per m; peak discharge {float(e.get('q_peak', np.nan)):.4g} m3/s per m "
        f"at {t_peak:.3f} s",
        f"h_max = {hm:.4g} m at {thm:.2f} s" + (f" ({hm / d.info['cell']:.1f} cells)" if d.info.get("cell") else ""),
        f"u_max = {um:.3g} m/s at {tum:.2f} s" if np.isfinite(um) else "u_max: not resolved",
        f"Figure C snapshots: {', '.join(f'{tt:g}' for tt in frames)} s",
        f"Figure D: full-field output at {t_sec:g} s (nearest the peak discharge)",
        "Windows (x, z) [m]: " + "; ".join(f"{k} x {v[0][0]:.4g}-{v[0][1]:.4g}, z {v[1][0]:.4g}-{v[1][1]:.4g}"
                                         for k, v in win.items()),
    ] + notes
    (out / f"{stem}_info.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    files["info"] = out / f"{stem}_info.txt"
    return dict(files=files, frames=frames, t_peak=t_peak, t_section=t_sec, notes=notes, summary=lines,
                windows=win)


def main():
    ap = argparse.ArgumentParser(description="Figures C (snapshots) and D (section) of one overtopping event.")
    ap.add_argument("event_id", type=int, help="event number in overtopping_events.csv (1 = largest volume)")
    ap.add_argument("--n", type=int, default=N_SNAPSHOTS, help="number of snapshots (default 9)")
    ap.add_argument("--speed-max", type=float, default=0.0, help="top of the speed colour bar [m/s]; 0: automatic")
    ap.add_argument("--first", type=float, default=None, help="time of the first snapshot [s] (with --step)")
    ap.add_argument("--step", type=float, default=None, help="time between snapshots [s] (with --first)")
    ap.add_argument("--out", default=OUT_DIR, help="output folder (default event_figures/)")
    a = ap.parse_args()
    res = make_event_figures(Path("."), a.event_id, a.n, a.speed_max or None, t_first=a.first, step=a.step,
                             out_dir=a.out)
    print("\n".join(res["summary"]))
    print("Saved: " + ", ".join(str(p) for p in res["files"].values()))


if __name__ == "__main__":
    main()
