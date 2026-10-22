"""
Writes the mesh and geometry dictionaries of an OpenFOAM case from the structure
STL and the target cell sizes.
"""

from __future__ import annotations

from pathlib import Path
import re
import shutil
import math
import numpy as np
from stl import mesh


def clone_case_from_template(
    template_dir: str | Path,
    new_case_dir: str | Path,
    *,
    overwrite: bool = False
) -> Path:
    """
    Copy a template OpenFOAM case folder and rename it to new_case_dir.

    - template_dir: e.g. "Main_CASE_FOAM"
    - new_case_dir: e.g. "P13_case_01"
    - overwrite=False: if target exists -> error
    - overwrite=True: delete target then copy
    """
    template_dir = Path(template_dir)
    new_case_dir = Path(new_case_dir)

    if not template_dir.exists():
        raise FileNotFoundError(f"Template case folder not found: {template_dir.resolve()}")

    if new_case_dir.exists():
        if not overwrite:
            raise FileExistsError(
                f"Target case folder already exists: {new_case_dir.resolve()}\n"
                f"Set overwrite=True to replace it."
            )
        shutil.rmtree(new_case_dir)

    shutil.copytree(template_dir, new_case_dir)
    return new_case_dir


def select_template_case(flow_model: str) -> Path:
    """
    Return the OpenFOAM template case folder based on flow model.

    Parameters
    ----------
    flow_model : str
        "laminar" or "RANS"

    Returns
    -------
    Path
        Template case directory
    """
    flow_model = flow_model.lower()

    templates = {
        "laminar": Path("Laminar_OF_Main"),
        "rans":    Path("Rans_OF_Main"),
    }

    if flow_model not in templates:
        raise ValueError(
            f"Invalid flow_model='{flow_model}'. "
            f"Expected one of {list(templates.keys())}."
        )

    tpl = templates[flow_model]
    if not tpl.exists():
        raise FileNotFoundError(f"Template case not found: {tpl.resolve()}")

    return tpl


def copy_stl_to_trisurface(case_dir: str | Path, stl_path: str | Path) -> Path:
    case_dir = Path(case_dir)
    tri_dir = case_dir / "constant" / "triSurface"
    tri_dir.mkdir(parents=True, exist_ok=True)

    stl_path = Path(stl_path)
    if not stl_path.exists():
        raise FileNotFoundError(f"STL not found: {stl_path.resolve()}")

    dst = tri_dir / stl_path.name
    shutil.copy2(stl_path, dst)
    return dst

def update_blockMeshDict_vertices_and_cells(
    case_dir: str | Path,
    dom: dict,
    dx_target: float = 0.04,
    dz_target: float = 0.04,
    Ny: int = 1,
    backup_suffix: str = ".bak"
):
    """
    Rewrites the vertices and cell counts of blockMeshDict:
    - Detect hex order from first 'hex(...) (Nx Ny Nz)'
    - Replace vertices block
    - Replace the first hex(...) cell counts to (Nx Ny Nz)
    """
    case_dir = Path(case_dir)
    bmd_path = case_dir / "system" / "blockMeshDict"
    if not bmd_path.exists():
        raise FileNotFoundError(f"blockMeshDict not found: {bmd_path.resolve()}")

    Dxmin, Dxmax = float(dom["Dxmin"]), float(dom["Dxmax"])
    Dymin, Dymax = float(dom["Dymin"]), float(dom["Dymax"])
    Dzmin, Dzmax = float(dom["Dzmin"]), float(dom["Dzmax"])

    Lx = float(Dxmax - Dxmin)
    Lz = float(Dzmax - Dzmin)

    Nx = int(math.ceil(Lx / dx_target))
    Nz = int(math.ceil(Lz / dz_target))

    bak_path = bmd_path.with_suffix(backup_suffix)
    if not bak_path.exists():
        shutil.copy2(bmd_path, bak_path)

    text = bmd_path.read_text()

    # Detect hex order
    mhex = re.search(r"hex\s*\(\s*([0-9\s]+)\s*\)\s*\(\s*([0-9]+\s+[0-9]+\s+[0-9]+)\s*\)", text)
    if not mhex:
        raise RuntimeError("Could not find a 'hex ( ... ) (Nx Ny Nz)' pattern in blockMeshDict.")

    hex_order = tuple(int(s) for s in mhex.group(1).split())

    # Update vertices to match existing hex order
    if hex_order == (0, 1, 5, 4, 3, 2, 6, 7):
        verts_map = {
            0: (Dxmin, Dymin, Dzmin),
            1: (Dxmax, Dymin, Dzmin),
            5: (Dxmax, Dymax, Dzmin),
            4: (Dxmin, Dymax, Dzmin),
            3: (Dxmin, Dymin, Dzmax),
            2: (Dxmax, Dymin, Dzmax),
            6: (Dxmax, Dymax, Dzmax),
            7: (Dxmin, Dymax, Dzmax),
        }
        verts = [verts_map[i] for i in range(8)]
    else:
        verts = [
            (Dxmin, Dymin, Dzmin),  # 0
            (Dxmax, Dymin, Dzmin),  # 1
            (Dxmax, Dymax, Dzmin),  # 2
            (Dxmin, Dymax, Dzmin),  # 3
            (Dxmin, Dymin, Dzmax),  # 4
            (Dxmax, Dymin, Dzmax),  # 5
            (Dxmax, Dymax, Dzmax),  # 6
            (Dxmin, Dymax, Dzmax),  # 7
        ]

    verts_str = "vertices\n(\n"
    for x, y, z in verts:
        verts_str += f"    ({x:.6f} {y:.6f} {z:.6f})\n"
    verts_str += ");\n"

    pattern_vertices = r"vertices\s*\(\s*.*?\s*\)\s*;"
    text, nvert = re.subn(pattern_vertices, verts_str, text, flags=re.DOTALL)
    if nvert != 1:
        raise RuntimeError(f"Could not uniquely replace vertices block (matches={nvert}).")

    def repl_blocks(match):
        hex_part = match.group(1)
        rest = match.group(3)
        return f"hex ({hex_part}) ({Nx} {Ny} {Nz}){rest}"

    pattern_block_line = r"hex\s*\(\s*([0-9\s]+)\s*\)\s*\(\s*([0-9]+\s+[0-9]+\s+[0-9]+)\s*\)(\s*[^;]*;)"
    text, nblk = re.subn(pattern_block_line, repl_blocks, text, count=1)
    if nblk != 1:
        raise RuntimeError(f"Could not update blocks cell counts (matches={nblk}).")

    bmd_path.write_text(text)

    return dict(Nx=Nx, Ny=Ny, Nz=Nz, dx_target=dx_target, dz_target=dz_target, backup=bak_path)

def update_surfaceFeatureExtractDict(case_dir: str | Path, stl_name: str, backup_suffix: str = ".bak"):
    """
    Updates the STL references in surfaceFeatureExtractDict:
    - Replace: file "something.stl"
    - Replace dictionary key like: someName.stl { ... }  -> {stl_name} {
    """
    case_dir = Path(case_dir)
    sfe_path = case_dir / "system" / "surfaceFeatureExtractDict"
    if not sfe_path.exists():
        raise FileNotFoundError(f"surfaceFeatureExtractDict not found: {sfe_path.resolve()}")

    bak_path = sfe_path.with_suffix(backup_suffix)
    if not bak_path.exists():
        shutil.copy2(sfe_path, bak_path)

    text = sfe_path.read_text()

    text, n_file = re.subn(
        r'file\s+"[^"]+\.stl"',
        f'file "{stl_name}"',
        text
    )

    text, n_key = re.subn(
        r'([A-Za-z0-9_\-]+)\.stl\s*\{',
        f'{stl_name} {{',
        text
    )

    sfe_path.write_text(text)
    return dict(replacements_file=n_file, replacements_key=n_key, backup=bak_path)

def write_snappyHexMeshDict_OF2206(
    case_dir: str | Path,
    dom: dict,
    stl_path: str | Path,
    *,
    Hs_plus: float,
    Hs_extra: float = 0.0,
    SWL: float = 0.0,
    LEVEL_INTERFACE: int = 2,
    LEVEL_RUNUP: int = 2,
    x_ht: float | None = None,
    runup_x_end: float | None = None,
    runup_top_above_stl: float = 0.10,
    backup_suffix: str = ".bak_headerfix",
    stl_zmax_ref: float | None = None,
):
    """
    Writes the snappyHexMeshDict of the case:
    - Compute STL bbox
    - interfaceBand: z in [SWL-Hs_plus, SWL+Hs_plus], x in [Dxmin,Dxmax], y in [stl_ymin, stl_ymax]
    - runupBand: x in [clamped x_ht, runup_x_end or stl_xmax], z in [min(stl_zmin,SWL-Hs_plus), max(stl_zmax, SWL+K_up*Hs_plus)]
    - Write a clean ASCII OpenFOAM 2206 header and controls
    stl_zmax_ref: top of the structure used for the runup band when stl_path is
    only the solid part of a porous structure (None: top of stl_path).
    """
    case_dir = Path(case_dir)
    shm_path = case_dir / "system" / "snappyHexMeshDict"

    stl_path = Path(stl_path)
    if not stl_path.exists():
        raise FileNotFoundError(f"STL not found: {stl_path.resolve()}")
    stl_name = stl_path.name
    stl_base = stl_path.stem

    Dxmin, Dxmax = float(dom["Dxmin"]), float(dom["Dxmax"])
    Dymin, Dymax = float(dom["Dymin"]), float(dom["Dymax"])
    Dzmin, Dzmax = float(dom["Dzmin"]), float(dom["Dzmax"])

    m = mesh.Mesh.from_file(str(stl_path))
    X = m.x.flatten()
    Y = m.y.flatten()
    Z = m.z.flatten()
    stl_xmin, stl_xmax = float(X.min()), float(X.max())
    stl_ymin, stl_ymax = float(Y.min()), float(Y.max())
    stl_zmin, stl_zmax = float(Z.min()), float(Z.max())

    if x_ht is None:
        raise RuntimeError("x_ht is required and must be numeric.")
        
    Hs_plus_eff = Hs_plus + Hs_extra
    zmin_interface = SWL - Hs_plus_eff
    zmax_interface = SWL + Hs_plus_eff

    runup_bottom_below_stl = 0.0
    zmin_runup = stl_zmin - float(runup_bottom_below_stl)
    z_top_ref = stl_zmax if stl_zmax_ref is None else float(stl_zmax_ref)
    zmax_runup = z_top_ref + float(runup_top_above_stl)

    x_ht_clamped = max(stl_xmin, min(float(x_ht), stl_xmax))
    if runup_x_end is None:
        x_runup_end = stl_xmax
    else:
        x_runup_end = float(runup_x_end)
        if x_runup_end < x_ht_clamped:
            raise ValueError(
                f"runup_x_end ({x_runup_end}) must be >= x_ht_clamped ({x_ht_clamped})"
            )
    x_keep = 0.5 * (Dxmin + Dxmax)
    y_keep = 0.5 * (Dymin + Dymax)
    z_keep = Dzmin + 0.2 * (Dzmax - Dzmin)

    if shm_path.exists():
        bak = shm_path.with_suffix(backup_suffix)
        shutil.copy2(shm_path, bak)

    snappy_text = f"""/*---------------------------------------------------------------------------*\\\\
| =========                 |                                                 |
| \\\\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |
|  \\\\    /   O peration     | Version:  2206                                  |
|   \\\\  /    A nd           | Website:  www.openfoam.com                      |
|    \\\\/     M anipulation  |                                                 |
\\\\*---------------------------------------------------------------------------*/
FoamFile
{{
    version     2.0;
    format      ascii;
    arch        "LSB;label=32;scalar=64";
    class       dictionary;
    location    "system";
    object      snappyHexMeshDict;
}}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

castellatedMesh true;
snap            true;
addLayers       false;

geometry
{{
    {stl_name}
    {{
        type triSurfaceMesh;
        name coast;
    }}

    interfaceBand
    {{
        type searchableBox;
        min ({Dxmin:.6f} {stl_ymin:.6f} {zmin_interface:.6f});
        max ({Dxmax:.6f} {stl_ymax:.6f} {zmax_interface:.6f});
    }}

    runupBand
    {{
        type searchableBox;
        min ({x_ht_clamped:.6f} {stl_ymin:.6f} {zmin_runup:.6f});
        max ({x_runup_end:.6f} {stl_ymax:.6f} {zmax_runup:.6f});
    }}
}};

castellatedMeshControls
{{
    maxLocalCells 100000;
    maxGlobalCells 20000000;
    minRefinementCells 0;
    nCellsBetweenLevels 2;

    features
    (
        {{
            file "{stl_base}.eMesh";
            level 0;
        }}
    );

    refinementSurfaces
    {{
        coast
        {{
            level (1 1);
        }}
    }}

    resolveFeatureAngle 30;

    refinementRegions
    {{
        interfaceBand
        {{
            mode inside;
            levels ((0 {int(LEVEL_INTERFACE)}));
        }}

        runupBand
        {{
            mode inside;
            levels ((0 {int(LEVEL_RUNUP)}));
        }}
    }}

    locationInMesh (-0.1 0.5 0);   
    // locationInMesh ({x_keep:.6f} {y_keep:.6f} {z_keep:.6f});
    allowFreeStandingZoneFaces true;
}}

snapControls
{{
    nSmoothPatch 3;
    tolerance 2.0;
    nSolveIter 300;
    nRelaxIter 5;

    nFeatureSnapIter 10;
    implicitFeatureSnap false;
    explicitFeatureSnap true;
    multiRegionFeatureSnap true;
}}

addLayersControls
{{
    relativeSizes true;
    layers {{ }}

    expansionRatio 1.0;
    finalLayerThickness 0.3;
    minThickness 0.25;
    nGrow 0;

    featureAngle 30;
    nRelaxIter 5;

    nSmoothSurfaceNormals 1;
    nSmoothNormals 3;
    nSmoothThickness 10;

    maxFaceThicknessRatio 0.5;
    maxThicknessToMedialRatio 0.3;
    minMedianAxisAngle 90;

    nBufferCellsNoExtrude 0;
    nLayerIter 50;
    nRelaxedIter 20;
}}

meshQualityControls
{{
    #include "meshQualityDict"

    relaxed
    {{
        maxNonOrtho 90;
    }}

    nSmoothScale 4;
    errorReduction 0.75;
}}

writeFlags
(
    scalarLevels
    layerSets
    layerFields
);

mergeTolerance 1e-6;

// ************************************************************************* //
"""
    shm_path.write_bytes(snappy_text.encode("ascii", errors="strict"))
    return dict(
        stl_bbox=dict(xmin=stl_xmin, xmax=stl_xmax, ymin=stl_ymin, ymax=stl_ymax, zmin=stl_zmin, zmax=stl_zmax),
        x_ht_clamped=x_ht_clamped,
        runupBand_x=(x_ht_clamped, x_runup_end),
        interfaceBand=(zmin_interface, zmax_interface),
        runupBand=(zmin_runup, zmax_runup),
        locationInMesh=(x_keep, y_keep, z_keep),
        written=shm_path
    )
    
    
    
    
    
from pathlib import Path
import re, shutil

def update_waveProperties_input(
    case_dir: str | Path,
    *,
    seaLevel: float,
    Hs: float,
    Tp: float,
    h_deep: float,
    dom: dict,
    L0: float,
    n_relax_inlet: float = 1.0,
    breakwater_mode: bool = False,
    n_relax_outlet: float = 1.5,
    seed: int = 1234,
    backup_suffix: str = ".bak"
):
    case_dir = Path(case_dir)
    wp_candidates = [
    case_dir / "constant" / "waveProperties.input",
    ]

    wp = next((p for p in wp_candidates if p.exists()), None)
    if wp is None:
        raise FileNotFoundError(
            f"waveProperties file not found. Tried: {[str(p) for p in wp_candidates]}"
        )


    bak = wp.with_suffix(wp.suffix + backup_suffix)
    if not bak.exists():
        shutil.copy2(wp, bak)

    text = wp.read_text()

    # helper
    def set_scalar(key, value, s):
        pat = rf'(^\s*{re.escape(key)}\s+)([^;]+)(;)'
        s, n = re.subn(pat, rf'\g<1>{value}\3', s, flags=re.MULTILINE)
        if n == 0:
            raise ValueError(f"Could not find '{key} ...;'")
        return s

    # Scalars
    text = set_scalar("seaLevel", f"{seaLevel:.6g}", text)
    text = set_scalar("Hs", f"{Hs:.6g}", text)
    text = set_scalar("Tp", f"{Tp:.6g}", text)
    text = set_scalar("depth", f"{float(h_deep):.6g}", text)

    # Seed of the random wave phases. waves2Foam (randomPhase.C) reads only the
    # keyword 'seedForRandomPhase'; without it the phases are seeded from the
    # clock and change at every run. Older templates used 'seedForRandom'.
    seed_line = f"seedForRandomPhase  {int(seed)};"
    text, n = re.subn(r"^(\s*)seedForRandom(?:Phase)?\s+[^;]+;", rf"\g<1>{seed_line}",
                      text, flags=re.MULTILINE)
    if n == 0:
        text, n = re.subn(r"^(\s*)(phaseMethod\s+[^;]+;)", rf"\g<1>\g<2>\n\g<1>{seed_line}",
                          text, count=1, flags=re.MULTILINE)
    if n == 0:
        raise ValueError("Could not set seedForRandomPhase: no 'phaseMethod' entry found.")

    # Geometry-based relaxation zones
    Dxmin, Dxmax = float(dom["Dxmin"]), float(dom["Dxmax"])
    Dzmin, Dzmax = float(dom["Dzmin"]), float(dom["Dzmax"])

    inlet_zone_len = float(n_relax_inlet) * float(L0)
    x0 = Dxmin
    x1 = min(Dxmin + inlet_zone_len, Dxmax)

    outlet_zone_len = 0.0

    if breakwater_mode:
        # Inlet + outlet relaxation zones
        text = re.sub(r'^\s*relaxationNames\s*\([^)]*\)\s*;',
                      'relaxationNames     (inlet outlet);',
                      text, flags=re.MULTILINE)
        outlet_zone_len = float(n_relax_outlet) * float(L0)
    else:
        # Inlet only
        text = re.sub(r'^\s*relaxationNames\s*\([^)]*\)\s*;',
                      'relaxationNames     (inlet);',
                      text, flags=re.MULTILINE)

    text = re.sub(r'^\s*initializationName\s+\w+\s*;',
                  'initializationName  inlet;',
                  text, flags=re.MULTILINE)

    # Remove any existing outletCoeffs block (rewritten below if needed)
    text = re.sub(r'\noutletCoeffs\s*\{.*?\}\s*\n',
                  '\n', text, flags=re.DOTALL)

    # Update inlet relaxation zone coordinates
    # startX
    text = re.sub(
        r'(inletCoeffs\s*\{.*?relaxationZone\s*\{.*?startX\s*\()\s*[^)]*\)',
        lambda m: f"{m.group(1)}{x0:.6f} 0.0 {Dzmin:.6f})",
        text,
        flags=re.DOTALL
    )

    # endX
    text = re.sub(
        r'(inletCoeffs\s*\{.*?relaxationZone\s*\{.*?endX\s*\()\s*[^)]*\)',
        lambda m: f"{m.group(1)}{x1:.6f} 0.0 {Dzmax:.6f})",
        text,
        flags=re.DOTALL
    )

    # Add outlet relaxation zone for breakwater mode
    if breakwater_mode:
        ox0 = Dxmax - outlet_zone_len   # start of absorption
        ox1 = Dxmax                      # right wall
        outlet_block = f"""
outletCoeffs
{{
    waveType    potentialCurrent;
    U           (0 0 0);
    Tsoft       2.0;

    relaxationZone
    {{
        relaxationScheme Spatial;
        relaxationShape  Rectangular;
        beachType        Empty;

        relaxType   OUTLET;
        startX      ({ox0:.6f} 0.0 {Dzmin:.6f});
        endX        ({ox1:.6f} 0.0 {Dzmax:.6f});
        orientation (1.0 0.0 0.0);
    }}
}}
"""
        text = text.rstrip() + "\n" + outlet_block + "\n"

    wp.write_text(text)
    return {
        "file": str(wp),
        "backup": str(bak),
        "seaLevel": seaLevel,
        "Hs": Hs,
        "Tp": Tp,
        "depth": float(h_deep),
        "L0": float(L0),
        "n_relax_inlet": float(n_relax_inlet),
        "inlet_zone_len": float(inlet_zone_len),
        "breakwater_mode": breakwater_mode,
        "outlet_zone_len": outlet_zone_len,
    }

from pathlib import Path
import shutil

def update_setFieldsDict_water_box(
    case_dir: str | Path,
    dom: dict,
    *,
    SWL: float,
    x_container_start: float,
    x_end_margin: float = 0.02,
    breakwater_mode: bool = False,
    backup_suffix: str = ".bak"
) -> dict:
    """
    Completely rewrites system/setFieldsDict (full valid OpenFOAM dict).

    Normal mode (seawall/dike):
    - Box 1: fill water (alpha=1) from Dxmin to just before container
    - Box 2: drain landward/container region (alpha=0)

    Breakwater mode:
    - Single box: fill water (alpha=1) from Dxmin to Dxmax up to SWL
      (water on both sides of the structure)
    """
    case_dir = Path(case_dir)
    sfd = case_dir / "system" / "setFieldsDict"
    if not sfd.exists():
        raise FileNotFoundError(f"setFieldsDict not found: {sfd.resolve()}")

    # backup current file (even if broken)
    bak = sfd.with_suffix(sfd.suffix + backup_suffix)
    if not bak.exists():
        shutil.copy2(sfd, bak)

    # bounds
    Dxmin = float(dom["Dxmin"])
    Dxmax = float(dom["Dxmax"])
    Dymin = float(dom["Dymin"])
    Dymax = float(dom["Dymax"])
    Dzmin = float(dom["Dzmin"])

    swl = float(SWL)

    if breakwater_mode:
        # Water everywhere up to SWL — both sides of breakwater
        regions_block = f"""    // Fill water on both sides (breakwater mode)
    boxToCell
    {{
        box ({Dxmin:.6f} {Dymin:.6f} {Dzmin:.6f}) ({Dxmax:.6f} {Dymax:.6f} {swl:.6f});
        fieldValues
        (
            volScalarFieldValue alpha.water 1
        );
    }}"""
        result_extra = {"breakwater_mode": True}
    else:
        # Seawall/dike: fill seaward, drain landward
        x_fill_end = float(x_container_start) - float(x_end_margin)
        if x_fill_end <= Dxmin:
            raise ValueError(
                f"x_fill_end <= Dxmin. Dxmin={Dxmin:.6f}, x_fill_end={x_fill_end:.6f}. "
                f"Check x_container_start and x_end_margin."
            )
        regions_block = f"""    // 1) Fill water (seaward)
    boxToCell
    {{
        box ({Dxmin:.6f} {Dymin:.6f} {Dzmin:.6f}) ({x_fill_end:.6f} {Dymax:.6f} {swl:.6f});
        fieldValues
        (
            volScalarFieldValue alpha.water 1
        );
    }}

    // 2) Drain container/landward part (keep it dry)
    boxToCell
    {{
        box ({float(x_container_start):.6f} {Dymin:.6f} {Dzmin:.6f}) ({Dxmax:.6f} {Dymax:.6f} {swl:.6f});
        fieldValues
        (
            volScalarFieldValue alpha.water 0
        );
    }}"""
        result_extra = {
            "fill_box_min": (Dxmin, Dymin, Dzmin),
            "fill_box_max": (x_fill_end, Dymax, swl),
            "drain_box_min": (float(x_container_start), Dymin, Dzmin),
            "drain_box_max": (Dxmax, Dymax, swl),
        }

    content = f"""/*--------------------------------*- C++ -*----------------------------------*\\
| =========                 |                                                 |
| \\\\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |
|  \\\\    /   O peration     | Version:  v2112                                 |
|   \\\\  /    A nd           | Website:  www.openfoam.com                      |
|    \\\\/     M anipulation  |                                                 |
\\*---------------------------------------------------------------------------*/
FoamFile
{{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      setFieldsDict;
}}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

defaultFieldValues
(
    volScalarFieldValue alpha.water 0
);

regions
(
{regions_block}
);

// ************************************************************************* //
"""

    # atomic write (prevents truncation if something crashes)
    tmp = sfd.with_suffix(".tmp")
    tmp.write_text(content)
    tmp.replace(sfd)

    return {"file": str(sfd), "backup": str(bak), **result_extra}


import re
from pathlib import Path
from typing import Sequence

def _fmt(x: float) -> str:
    """OpenFOAM-friendly float formatting."""
    # keep enough precision but avoid ugly long repr
    return f"{float(x):.6g}"

def _make_line_sets_block(
    x_positions: Sequence[float],
    *,
    y: float,
    zmin: float,
    zmax: float,
    nPoints: int = 300,
    writeInterval: float = 0.1,
) -> str:
    """
    Build the 'line { ... }' functionObject block as a string.
    Set names are x0, x1, ... and can be changed below.
    """
    sets_txt = []
    for i, x in enumerate(x_positions, start=1):
        name = f"x{i:02d}"   # x01, x02, ... (clean & consistent)
        sets_txt.append(
f"""            {name}
            {{
                type    uniform;
                axis    distance;
                start   ({_fmt(x)} {_fmt(y)} {_fmt(zmin)});
                end     ({_fmt(x)} {_fmt(y)} {_fmt(zmax)});
                nPoints {int(nPoints)};
            }}"""
        )

    sets_joined = "\n".join(sets_txt)

    return f"""
    line
    {{
        type                  sets;
        libs                  ("libsampling.so");
        enabled               true;
        writeControl          runTime;
        writeInterval         {_fmt(writeInterval)};

        interpolationScheme   cellPoint;
        setFormat             raw;
        fixedLocations        false;

        fields
        (
            U
            p_rgh
            rho
            alpha.water
        );

        sets
        {{
{sets_joined}
        }}
    
""".rstrip()


def update_controlDict_case(
    case_dir: str | Path,
    *,
    endTime: float,
    deltaT: float,
    writeInterval: float,
    maxCo: float,
    maxAlphaCo: float,
    # line probes controls
    x_positions: Sequence[float],
    y_line: float,
    zmin: float,
    zmax: float,
    nPoints: int = 300,
    line_write_interval: float | None = None,
    backup_suffix: str = ".bak",
):
    """
    Update system/controlDict:
      - endTime, deltaT, writeInterval, maxCo, maxAlphaCo
      - rebuild the 'line { ... }' sampling block with chosen x_positions
        (written every line_write_interval; default: writeInterval)
    """
    case_dir = Path(case_dir)
    cd_path = case_dir / "system" / "controlDict"
    if not cd_path.exists():
        raise FileNotFoundError(f"controlDict not found: {cd_path}")

    txt = cd_path.read_text(encoding="utf-8", errors="ignore")

    # backup
    cd_path.with_suffix(cd_path.suffix + backup_suffix).write_text(txt, encoding="utf-8")

    # simple key replacements at top-level
    def repl_key(key: str, value: float) -> None:
        nonlocal txt
        # matches: key <spaces> number ;
        pat = re.compile(rf"(^\s*{re.escape(key)}\s+)([^;]+)(;)", re.MULTILINE)
        if not pat.search(txt):
            raise ValueError(f"Could not find key '{key}' in controlDict.")
        txt = pat.sub(rf"\g<1>{_fmt(value)}\g<3>", txt, count=1)

    repl_key("endTime", endTime)
    repl_key("deltaT", deltaT)
    repl_key("writeInterval", writeInterval)
    repl_key("maxCo", maxCo)
    repl_key("maxAlphaCo", maxAlphaCo)

    # Use the same writeInterval for the freeSurface function object
    txt = re.sub(
        r"(\bfreeSurface\b\s*\{.*?\bwriteInterval\s+)([^;]+)(;)",
        rf"\g<1>{_fmt(writeInterval)}\g<3>",
        txt,
        flags=re.DOTALL,
        count=1
    )

    # rebuild the line block completely
    new_line_block = _make_line_sets_block(
        x_positions,
        y=y_line,
        zmin=zmin,
        zmax=zmax,
        nPoints=nPoints,
        writeInterval=writeInterval if line_write_interval is None else line_write_interval,
    )

    # Replace existing "line { ... }" block inside functions{...}
    # This pattern finds: line { ... } (non-greedy)
    line_pat = re.compile(r"(^\s*line\s*\{.*?^\s*\})", re.DOTALL | re.MULTILINE)
    if not line_pat.search(txt):
        raise ValueError("Could not find existing 'line { ... }' block in controlDict.")
    txt = line_pat.sub(new_line_block, txt, count=1)

    cd_path.write_text(txt, encoding="utf-8")
    return cd_path

from pathlib import Path
import numpy as np
from stl import mesh as stlmesh

def extract_xz_profile_segments_from_stl(stl_path: str | Path, y_plane: float = 0.0, atol: float = 1e-6, x_max_keep=None):
    """
    Returns list of segments [(p0,p1), (p1,p2), ...] where each p=(x,z),
    extracted from the boundary polyline of the STL on the plane y=y_plane.
    Requires numpy-stl (stl.mesh) only.
    """
    stl_path = Path(stl_path)

    m = stlmesh.Mesh.from_file(str(stl_path))

    # vertices per triangle: shape (ntri, 3, 3) for (x,y,z)
    V = m.vectors

    # select triangles lying on y=y_plane (all 3 vertices have y close to y_plane)
    tri_on_plane = np.all(np.isclose(V[:, :, 1], y_plane, atol=atol), axis=1)
    Vp = V[tri_on_plane]
    if Vp.shape[0] == 0:
        raise ValueError(f"No triangles found on y={y_plane}. Try y_plane=Ly (e.g. 1.0) or increase atol.")

    # Build edges from triangles, but in XZ only (y constant)
    # Keep full 3D points for indexing, but compare edges by their XZ endpoints.
    # Collect edges as pairs of points (pA,pB) in XZ
    def tri_edges_xz(tri):
        p0 = (float(tri[0,0]), float(tri[0,2]))
        p1 = (float(tri[1,0]), float(tri[1,2]))
        p2 = (float(tri[2,0]), float(tri[2,2]))
        return [(p0,p1), (p1,p2), (p2,p0)]

    edges = []
    for tri in Vp:
        edges.extend(tri_edges_xz(tri))

    # Normalize edges (unordered) so we can count duplicates
    # Use rounding to stabilize comparisons
    def norm_edge(e, nd=10):
        a, b = e
        a = (round(a[0], nd), round(a[1], nd))
        b = (round(b[0], nd), round(b[1], nd))
        return (a,b) if a <= b else (b,a)

    nedges = [norm_edge(e) for e in edges]
    uniq, counts = np.unique(nedges, axis=0, return_counts=True)

    # Boundary edges appear once
    bedges = uniq[counts == 1]
    if len(bedges) < 3:
        raise RuntimeError("Could not find a boundary loop (STL may be non-manifold on that plane).")

    # Build adjacency of boundary vertices in XZ
    adj = {}
    for a, b in bedges:
        a = (float(a[0]), float(a[1]))
        b = (float(b[0]), float(b[1]))
        adj.setdefault(a, []).append(b)
        adj.setdefault(b, []).append(a)

    # Traverse ALL connected components (one per disconnected boundary loop:
    # the main structure loop + one loop per merged obstacle).
    segments = []
    visited = set()

    while True:
        unvisited = [v for v in adj.keys() if v not in visited]
        if not unvisited:
            break
        start = min(unvisited, key=lambda p: (p[0], p[1]))

        loop = [start]
        prev = None
        cur = start
        visited.add(start)

        for _ in range(len(adj) + 10):
            nbrs = adj[cur]
            if not nbrs:
                break
            if prev is None:
                nxt = nbrs[0]
            else:
                nxt = nbrs[0] if nbrs[0] != prev else (nbrs[1] if len(nbrs) > 1 else None)
            if nxt is None or nxt == start or nxt in visited:
                break
            loop.append(nxt)
            visited.add(nxt)
            prev, cur = cur, nxt

        for i in range(len(loop) - 1):
            if loop[i] != loop[i + 1]:
                segments.append((loop[i], loop[i + 1]))
            
            
    if x_max_keep is not None:
        x_max_keep = float(x_max_keep)
        segments = [
            (p0, p1) for (p0, p1) in segments
            if max(p0[0], p1[0]) <= x_max_keep + 1e-9
        ]
        
    return segments



def segment_to_box(p0, p1, *, Ly: float, tol: float, x_max_keep=None):
    x0, z0 = p0
    x1, z1 = p1
    xmin, xmax = min(x0, x1) - tol, max(x0, x1) + tol

    if x_max_keep is not None:
        xmax = min(xmax, float(x_max_keep) + tol)

    zmin, zmax = min(z0, z1) - tol, max(z0, z1) + tol
    return (xmin, 0.0, zmin), (xmax, float(Ly), zmax)




import matplotlib.pyplot as plt
import matplotlib.cm as cm

def plot_stl_xz_with_segments_colored(stl_bbox, segments, x_max_keep=None, title="STL XZ profile with segments"):
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm

    if x_max_keep is not None:
        x_max_keep = float(x_max_keep)
        segments = [(p0,p1) for (p0,p1) in segments if max(p0[0], p1[0]) <= x_max_keep + 1e-9]

    nseg = len(segments)
    fig, ax = plt.subplots(figsize=(18, 6))
    cmap = cm.get_cmap("tab20", max(nseg, 1))

    for i, (p0, p1) in enumerate(segments, start=1):
        x0, z0 = p0
        x1, z1 = p1
        ax.plot([x0, x1], [z0, z1], linewidth=4, color=cmap(i-1), label=f"x{i:02d}")

    xmin = stl_bbox["xmin"] - 0.5
    xmax = (x_max_keep + 0.5) if x_max_keep is not None else (stl_bbox["xmax"] + 0.5)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(stl_bbox["zmin"] - 0.5, stl_bbox["zmax"] + 0.5)

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("z [m]")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), title="Segments")
    plt.tight_layout()
    plt.show()
    return fig, ax


from pathlib import Path

# Write topoSetDict
from pathlib import Path

def write_topoSetDict_force_and_overtopping(
    case_dir: str | Path,
    *,
    # overtopping faceSet (ALWAYS written)
    # Single vertical section at x = overtop_x, from z = overtop_zmin to
    # z = overtop_zmax, spanning the flume width Ly.
    overtop_x: float,
    overtop_zmin: float,
    overtop_zmax: float,
    Ly: float,
    overtop_half_width: float = 0.02,
    overtopping_faceset_name: str = "overtopping",
    # force faceSets (one per selected segment)
    include_force: bool = False,
    source_patch: str = "coast",
    force_entries: list[dict] | None = None,
    # each entry is a dict:
    #   set_name : str   — faceSet name, e.g. "force_seg_05"
    #   box_min  : tuple — (xmin, ymin, zmin)
    #   box_max  : tuple — (xmax, ymax, zmax)
    # legacy single-entry params (still accepted for backward compat)
    force_set_name: str = "x05",
    force_box_min: tuple[float, float, float] | None = None,
    force_box_max: tuple[float, float, float] | None = None,
    # porous cellZones (one entry per porous STL)
    porous_stl_entries: list | None = None,
    # each entry is a dict:
    #   stl_path      : str   — path relative to case dir, e.g.
    #                          "constant/triSurface/case_porous_upper_slope.stl"
    #   cellset_name  : str   — e.g. "porousCells_upper_slope"
    #   cellzone_name : str   — e.g. "porosity_upper_slope"  (matches porosityZones key)
    #   outside_point : tuple — a point known to be outside the STL, e.g. (-2, 0.5, 0.6)
    of_version: str = "v2406",
) -> Path:
    case_dir = Path(case_dir)
    out = case_dir / "system" / "topoSetDict"

    # overtopping actions (always)
    # The waves2Foam 'overtopping' function object sums the water flux times
    # the unit normal over EVERY face of each faceZone whose name starts with
    # "overtopping". The zone must therefore be a single surface: here, the
    # faces shared by the cells just seaward (side A) and just landward
    # (side B) of x = overtop_x. Helper set names must not start with
    # "overtopping" and are removed at the end, because setsToZones turns
    # every remaining set into a zone.
    xa = overtop_x - overtop_half_width
    xb = overtop_x + overtop_half_width
    actions = f"""
    // 1) Cells seaward (A) and landward (B) of the overtopping section
    {{
        name    ovSideA;
        type    cellSet;
        action  new;
        source  boxToCell;
        sourceInfo
        {{
            box ({_fmt(xa)} 0 {_fmt(overtop_zmin)})
                ({_fmt(overtop_x)} {_fmt(Ly)} {_fmt(overtop_zmax)});
        }}
    }}

    {{
        name    ovSideB;
        type    cellSet;
        action  new;
        source  boxToCell;
        sourceInfo
        {{
            box ({_fmt(overtop_x)} 0 {_fmt(overtop_zmin)})
                ({_fmt(xb)} {_fmt(Ly)} {_fmt(overtop_zmax)});
        }}
    }}

    // 2) Faces of side A ...
    {{
        name    {overtopping_faceset_name};
        type    faceSet;
        action  new;
        source  cellToFace;
        sourceInfo
        {{
            set ovSideA;
            option all;
        }}
    }}

    {{
        name    ovSideBFaces;
        type    faceSet;
        action  new;
        source  cellToFace;
        sourceInfo
        {{
            set ovSideB;
            option all;
        }}
    }}

    // 3) ... that are also faces of side B: one surface between A and B
    {{
        name    {overtopping_faceset_name};
        type    faceSet;
        action  subset;
        source  faceToFace;
        sourceInfo
        {{
            set ovSideBFaces;
        }}
    }}

    // 4) Remove boundary faces
    {{
        name    {overtopping_faceset_name};
        type    faceSet;
        action  delete;
        source  boundaryToFace;
    }}

    // 5) Remove the helper sets
    {{
        name    ovSideA;
        type    cellSet;
        action  remove;
    }}

    {{
        name    ovSideB;
        type    cellSet;
        action  remove;
    }}

    {{
        name    ovSideBFaces;
        type    faceSet;
        action  remove;
    }}
""".rstrip()

    # porous cellZone actions (optional, one pair per STL)
    if porous_stl_entries:
        for entry in porous_stl_entries:
            stl_path      = entry["stl_path"]
            cellset_name  = entry["cellset_name"]
            cellzone_name = entry["cellzone_name"]
            op = entry["outside_point"]
            actions += f"""

    // Porous zone: {cellzone_name}
    {{
        name    {cellset_name};
        type    cellSet;
        action  new;
        source  surfaceToCell;
        sourceInfo
        {{
            file        "{stl_path}";
            outsidePoints (({op[0]} {op[1]} {op[2]}));
            includeInside  true;
            includeOutside false;
            includeCut     true;
            nearDistance -1;
            curvature    -100;
        }}
    }}

    {{
        name    {cellzone_name};
        type    cellZoneSet;
        action  new;
        source  setToCellZone;
        sourceInfo
        {{
            set {cellset_name};
        }}
    }}""".rstrip()

    # force faceSet actions (optional, one pair per segment)
    if include_force:
        # Build unified list from force_entries or legacy single-entry params
        _entries = force_entries or []
        if not _entries and force_box_min is not None and force_box_max is not None:
            _entries = [{"set_name": force_set_name,
                         "box_min": force_box_min, "box_max": force_box_max}]
        if not _entries:
            raise ValueError("include_force=True requires force_entries or force_box_min/max")

        for fe in _entries:
            sn = fe["set_name"]
            bmin = fe["box_min"]
            bmax = fe["box_max"]
            actions += f"""

    // Force extraction faceSet: {sn}
    {{
        name    {sn};
        type    faceSet;
        action  new;
        source  patchToFace;
        sourceInfo
        {{
            patch {source_patch};
        }}
    }}

    {{
        name    {sn};
        type    faceSet;
        action  subset;
        source  boxToFace;
        sourceInfo
        {{
            box ({bmin[0]} {bmin[1]} {bmin[2]})
                ({bmax[0]} {bmax[1]} {bmax[2]});
        }}
    }}
""".rstrip()

    txt = f"""/*--------------------------------*- C++ -*----------------------------------*\\
| =========                 |                                                 |
| \\\\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |
|  \\\\    /   O peration     | Version:  {of_version}                                 |
|   \\\\  /    A nd           | Website:  www.openfoam.com                      |
|    \\\\/     M anipulation  |                                                 |
\\*---------------------------------------------------------------------------*/
FoamFile
{{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      topoSetDict;
}}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

actions
(
{actions}
);

// ************************************************************************* //
"""
    out.write_text(txt)
    return out


# Patch fvSchemes for porousWaveFoam
def add_porous_div_scheme(
    case_dir: str | Path,
    backup_suffix: str = ".bak",
) -> Path:
    """
    Insert the porous-media div scheme into system/fvSchemes.

    Adds after the ``div(rhoPhi,U)`` line:
        div((rhoPhi|interpolate(porosity)),U) Gauss limitedLinearV 1;

    Safe to call multiple times (skips if already present).
    """
    case_dir = Path(case_dir)
    fv = case_dir / "system" / "fvSchemes"
    if not fv.exists():
        raise FileNotFoundError(f"fvSchemes not found: {fv}")

    text = fv.read_text()
    porous_line = "    div((rhoPhi|interpolate(porosity)),U) Gauss limitedLinearV 1;"

    if "interpolate(porosity)" in text:
        return fv  # already present

    anchor = "div(rhoPhi,U)"
    idx = text.find(anchor)
    if idx == -1:
        raise ValueError(
            f"Cannot find '{anchor}' in {fv}. "
            "Check that divSchemes contains div(rhoPhi,U)."
        )
    eol = text.find("\n", idx)
    if eol == -1:
        eol = len(text)

    if backup_suffix:
        fv.rename(str(fv) + backup_suffix)

    new_text = text[:eol + 1] + porous_line + "\n" + text[eol + 1:]
    fv.write_text(new_text)
    return fv


# Write porosityZones
def write_porosity_zones(
    case_dir: str | Path,
    zones_physics: dict,
    backup_suffix: str = ".bak",
) -> Path:
    """
    Write ``constant/porosityZones`` for one or more porous cellZones.

    Parameters
    ----------
    case_dir : Path
        OpenFOAM case directory.
    zones_physics : dict
        ``{cellzone_name: {porosity, d50, KC, alpha, beta,
                           gammaAddedMass, resistanceFormulation}}``
        ``cellzone_name`` must match the ``cellZoneSet`` name in topoSetDict.
    backup_suffix : str
        If the file already exists, rename it with this suffix before overwriting.

    Returns
    -------
    Path to the written file.
    """
    case_dir = Path(case_dir)
    out = case_dir / "constant" / "porosityZones"

    if out.exists() and backup_suffix:
        out.rename(str(out) + backup_suffix)

    n = len(zones_physics)
    entries = ""
    for zone_name, p in zones_physics.items():
        porosity      = float(p.get("porosity", 0.44))
        d50           = float(p.get("d50", 0.06))
        KC            = float(p.get("KC", 128))
        alpha         = float(p.get("alpha", 500))
        beta          = float(p.get("beta", 2.0))
        gamma_am      = float(p.get("gammaAddedMass", 0.34))
        resist_form   = str(p.get("resistanceFormulation", "vanGent1995"))

        entries += f"""
    {zone_name}
    {{
        coordinateSystem
        {{
            e1  (1 0 0);
            e2  (0 0 1);
        }}

        origin (0 0 0);

        coordinateRotation
        {{
            type axesRotation;
            e1 (1 0 0);
            e2 (0 0 1);
        }}

        resistanceFormulation {resist_form};

        porosity {porosity};

        KC KC [0 0 0 0 0 0 0] {KC};

        gammaAddedMass {gamma_am};

        d50 d50 [0 1 0 0 0 0 0] {d50};

        alpha alpha [0 0 0 0 0 0 0] {alpha};

        beta beta [0 0 0 0 0 0 0] {beta};
    }}
"""

    txt = f"""/*--------------------------------*- C++ -*----------------------------------*\\
| =========                 |                                                 |
| \\\\      /  F ield         | OpenFOAM Extend Project: Open Source CFD        |
|  \\\\    /   O peration     |                                                 |
|   \\\\  /    A nd           |                                                 |
|    \\\\/     M anipulation  |                                                 |
\\*---------------------------------------------------------------------------*/
FoamFile
{{
    version     2.0;
    format      ascii;
    class       dictionary;
    location    "constant";
    object      porosityZones;
}}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

{n}
({entries})

// ************************************************************************* //
"""
    out.write_text(txt)
    return out


def build_layered_zones_physics(
    layers: list[dict],
    active_zones: list[str],
    core_physics: dict | None = None,
) -> dict:
    """Build {cellzone_name: physics_dict} for write_porosity_zones from layer configs.

    Naming convention: ``porosity_{layername}_{zonename}``.
    If *core_physics* is provided, adds a ``porosity_core`` entry.
    """
    zones_physics: dict[str, dict] = {}
    for layer in layers:
        lname = layer["name"]
        phys_default = dict(layer["physics_default"])
        zone_overrides = layer.get("zone_physics_overrides", {})
        for zone in active_zones:
            cellzone = f"porosity_{lname}_{zone}"
            phys = dict(phys_default)
            if zone in zone_overrides:
                phys.update(zone_overrides[zone])
            zones_physics[cellzone] = phys
    if core_physics is not None:
        zones_physics["porosity_core"] = dict(core_physics)
    return zones_physics


# Write createPatchDict
def write_createPatchDict_for_segment(
    case_dir: str | Path,
    *,
    patch_name: str,          # e.g. "x04"
    faceSet_name: str,        # e.g. "x04"
    patch_type: str = "wall",
    of_version: str = "v2406"
) -> Path:
    """
    Create system/createPatchDict (tutorial style) that makes a boundary patch
    from faceSet.
    """
    case_dir = Path(case_dir)
    out = case_dir / "system" / "createPatchDict"

    txt = f"""/*--------------------------------*- C++ -*----------------------------------*\\
| =========                 |                                                 |
| \\\\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |
|  \\\\    /   O peration     | Version:  {of_version}                                 |
|   \\\\  /    A nd           | Website:  www.openfoam.com                      |
|    \\\\/     M anipulation  |                                                 |
\\*---------------------------------------------------------------------------*/
FoamFile
{{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      createPatchDict;
}}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

pointSync false;

patches
(
    {{
        name {patch_name};

        patchInfo
        {{
            type {patch_type};
        }}

        constructFrom set;
        set {faceSet_name};
    }}
);

// ************************************************************************* //
"""
    out.write_text(txt)
    return out


def write_createPatchDict_for_segments(
    case_dir: str | Path,
    *,
    patches: list[dict],
    patch_type: str = "wall",
    of_version: str = "v2406",
) -> Path:
    """Create system/createPatchDict with multiple boundary patches from faceSets.

    Parameters
    ----------
    patches : list of dict
        Each dict has ``patch_name`` and ``faceSet_name``.
    """
    case_dir = Path(case_dir)
    out = case_dir / "system" / "createPatchDict"

    patch_blocks = ""
    for p in patches:
        patch_blocks += f"""
    {{
        name {p["patch_name"]};

        patchInfo
        {{
            type {patch_type};
        }}

        constructFrom set;
        set {p["faceSet_name"]};
    }}
"""

    txt = f"""/*--------------------------------*- C++ -*----------------------------------*\\
| =========                 |                                                 |
| \\\\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |
|  \\\\    /   O peration     | Version:  {of_version}                                 |
|   \\\\  /    A nd           | Website:  www.openfoam.com                      |
|    \\\\/     M anipulation  |                                                 |
\\*---------------------------------------------------------------------------*/
FoamFile
{{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      createPatchDict;
}}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

pointSync false;

patches
({patch_blocks});

// ************************************************************************* //
"""
    out.write_text(txt)
    return out


# Run topoSet and createPatch
def run_topo_and_createPatch(case_dir: str | Path):
    """
    Run topoSet and createPatch -overwrite inside the case directory.
    """
    import subprocess
    case_dir = Path(case_dir)

    subprocess.run(["topoSet"], cwd=str(case_dir), check=True)
    subprocess.run(["createPatch", "-overwrite"], cwd=str(case_dir), check=True)



from pathlib import Path
import re

def add_patch_by_copying_bc(case_dir, new_patch, src_patch="coast",
                            fields=("alpha.water","U","p_rgh","k","omega","nut","cellLevel", "pointLevel")):
    case_dir = Path(case_dir)
    zero_dir = case_dir / "0"
    if not zero_dir.is_dir():
        raise FileNotFoundError(f"Missing 0/ directory: {zero_dir}")

    for field in fields:
        fpath = zero_dir / field
        if not fpath.exists():
            print(f"⚠️ {field}: file not found, skip")
            continue

        txt = fpath.read_text()

        # Skip if already present
        if re.search(rf"\b{re.escape(new_patch)}\s*\{{", txt):
            print(f"✅ {field}: {new_patch} already exists")
            continue

        # Find src patch block (coast { ... })
        # This grabs from 'coast {' up to the matching '}' on its own indentation level in typical FO files
        m = re.search(rf"(^\s*{re.escape(src_patch)}\s*\{{.*?^\s*\}}\s*)",
                      txt, flags=re.DOTALL | re.MULTILINE)
        if not m:
            raise RuntimeError(f"{field}: could not find patch '{src_patch}' in boundaryField")

        src_block = m.group(1)
        new_block = src_block.replace(src_patch, new_patch, 1)

        # Insert right after the src block
        insert_pos = m.end(1)
        txt2 = txt[:insert_pos] + "\n" + new_block + txt[insert_pos:]

        fpath.write_text(txt2)
        print(f"✅ {field}: added {new_patch} (copied from {src_patch})")



from pathlib import Path
import re



def write_decomposeParDict(
    case_dir,
    n_procs,
    method="scotch",
    of_version="v2112"
):
    """
    Rewrite system/decomposeParDict.

    Supported methods:
      - scotch
      - simple  -> forced x-only decomposition: (n_procs 1 1)
    """
    case_dir = Path(case_dir)
    out = case_dir / "system" / "decomposeParDict"

    method = str(method).strip().lower()
    n_procs = int(n_procs)

    if method not in ("scotch", "simple"):
        raise ValueError(f"Unsupported decomposition method: {method}")

    extra_block = ""
    if method == "simple":
        extra_block = f"""
simpleCoeffs
{{
    n               ({n_procs} 1 1);
    delta           0.001;
}}

distributed     no;

roots           ();
"""

    txt = f"""/*--------------------------------*- C++ -*----------------------------------*\\
| =========                 |                                                 |
| \\\\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |
|  \\\\    /   O peration     | Version:  {of_version}                                 |
|   \\\\  /    A nd           | Website:  www.openfoam.com                      |
|    \\\\/     M anipulation  |                                                 |
\\*---------------------------------------------------------------------------*/
FoamFile
{{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      decomposeParDict;
}}

numberOfSubdomains {n_procs};

method {method};

// Keep both cells of every overtopping-zone face on the same processor:
// the waves2Foam overtopping function object fails on processor-boundary faces.
constraints
{{
    overtopping
    {{
        type    preserveFaceZones;
        zones   (overtopping);
        enabled true;
    }}
}}
{extra_block}
// ************************************************************************* //
"""
    out.write_text(txt)
    return out

from pathlib import Path
import re

def update_forces_patch(control_dict_path, new_patch_name):
    """
    Update the 'patches (...)' entry inside a forces function block
    in system/controlDict to use new_patch_name.
    Works on any 'patches (something);' in that forces block.
    """
    control_dict_path = Path(control_dict_path)
    text = control_dict_path.read_text()

    # Replace the first occurrence of 'patches (...) ;'
    new_text, n = re.subn(
        r"patches\s*\([^)]*\)\s*;",
        f"patches ({new_patch_name});",
        text,
        count=1
    )

    if n == 0:
        print("⚠️ update_forces_patch: no 'patches (...)' entry found to replace.")
    else:
        print(f"✅ update_forces_patch: updated patches to ({new_patch_name})")

    control_dict_path.write_text(new_text)


def update_forces_patches_multi(control_dict_path, patch_names: list[str]):
    """Replace the template forceImpact block with one force function per patch.

    Each patch gets its own ``forceImpact_<name>`` function object so that
    OpenFOAM writes separate ``postProcessing/forceImpact_<name>/`` directories.
    """
    control_dict_path = Path(control_dict_path)
    text = control_dict_path.read_text()

    # 1) Locate functions { ... } block
    m = re.search(r'\bfunctions\b\s*\{', text)
    if not m:
        print("update_forces_patches_multi: no 'functions { }' block found.")
        return

    start = m.end()
    depth = 1
    i = start
    while i < len(text) and depth > 0:
        if text[i] == '{':
            depth += 1
        elif text[i] == '}':
            depth -= 1
        i += 1
    functions_body = text[start:i-1]

    # 2) Remove existing forces blocks
    pattern = re.compile(
        r"""
        \n\s*[A-Za-z0-9_]+\s*\{
        [^{}]*?type\s+forces\s*;[^{}]*?
        \}
        """,
        re.VERBOSE | re.DOTALL
    )
    functions_body = pattern.sub("", functions_body)

    # 3) Append one block per patch
    for pname in patch_names:
        func_name = f"forceImpact_{pname}"
        functions_body += f"""

    {func_name}
    {{
        type            forces;
        libs            ("libforces.so");
        patches         ({pname});
        rhoInf          1000;
        log             true;
        CofR            (0 0 0);
        writeControl    runTime;
        writeInterval   0.1;
    }}
"""

    new_text = text[:start] + functions_body + text[i-1:]
    control_dict_path.write_text(new_text)
    print(f"update_forces_patches_multi: wrote {len(patch_names)} force functions: "
          + ", ".join(patch_names))


from pathlib import Path
import re

def remove_forces_function(control_dict_path):
    """
    Safely remove any function block with 'type forces;'
    INSIDE the 'functions { ... }' block of controlDict,
    without touching the FoamFile header or the rest.
    """
    control_dict_path = Path(control_dict_path)
    text = control_dict_path.read_text()

    # 1) Locate the 'functions { ... }' block
    m = re.search(r'\bfunctions\b\s*\{', text)
    if not m:
        print("⚠️ remove_forces_function: no 'functions {{ ... }}' block found.")
        return

    start = m.end()  # position just after 'functions{'
    # find the matching closing brace for the functions block
    depth = 1
    i = start
    while i < len(text) and depth > 0:
        c = text[i]
        if c == '{':
            depth += 1
        elif c == '}':
            depth -= 1
        i += 1

    # substring that contains only the inside of functions{...}
    functions_body = text[start:i-1]

    # 2) Inside this body, remove blocks that contain 'type forces;'
    pattern = re.compile(
        r"""
        (\n\s*                 # newline + leading spaces
         [A-Za-z0-9_]+         # function name, e.g. forceImpact
         \s*\{                 # opening brace
         [^{}]*?               # content, not crossing nested braces (simple case)
         type\s+forces\s*;     # 'type forces;'
         [^{}]*?               # rest of the block
         \}                    # closing brace
        )
        """,
        re.VERBOSE | re.DOTALL
    )

    new_functions_body, n = pattern.subn(
        "\n    // forces function removed because new_patch=False\n",
        functions_body
    )

    if n == 0:
        print("⚠️ remove_forces_function: no forces function block found.")
    else:
        print(f"✅ remove_forces_function: removed {n} forces function block(s).")

    # 3) Reconstruct the full text
    new_text = text[:start] + new_functions_body + text[i-1:]
    control_dict_path.write_text(new_text)



# Breakwater mode: outlet BC patch
def update_outlet_bc_for_breakwater(case_dir: str | Path) -> None:
    """Patch outlet U BC from fixedValue to inletOutlet for breakwater mode.

    The Laminar template uses fixedValue (0 0 0) which blocks flow at the
    outlet.  For breakwater mode with an outlet relaxation zone, the outlet
    needs to allow flow through.  RANS template already has inletOutlet, so
    this is a no-op there.
    """
    case_dir = Path(case_dir)
    u_file = case_dir / "0.orig" / "U"
    if not u_file.exists():
        return

    text = u_file.read_text()
    # Match the outlet block with fixedValue and replace
    old_outlet = re.search(
        r'(outlet\s*\{[^}]*type\s+)fixedValue\s*;'
        r'([^}]*value\s+uniform\s*\(\s*0\s+0\s+0\s*\)\s*;[^}]*\})',
        text, flags=re.DOTALL,
    )
    if old_outlet:
        new_outlet = (
            "outlet\n"
            "    {\n"
            "        type            inletOutlet;\n"
            "        inletValue      uniform (0 0 0);\n"
            "        value           uniform (0 0 0);\n"
            "    }"
        )
        text = text[:old_outlet.start()] + new_outlet + text[old_outlet.end():]
        u_file.write_text(text)


# Roughness utilities (RANS)
def update_nut_roughness_0orig(
    case_dir: str | Path,
    *,
    Ks: float,
    Cs: float,
    patch: str = "coast",
    file_relpath: str = "0.orig/nut",
    backup_suffix: str = ".bak"
) -> dict:
    """
    Update Ks and Cs for a given patch in the file 0.orig/nut (typically for RANS).

    This is designed for nut boundary conditions like:
        type    nutkRoughWallFunction;
        Ks      uniform <...>;
        Cs      uniform <...>;

    Parameters
    ----------
    case_dir : str | Path
        OpenFOAM case directory.
    Ks, Cs : float
        Roughness height [m] and roughness constant [-].
    patch : str
        Patch name to update (default: "coast").
    file_relpath : str
        Relative path from case_dir to nut file (default: "0.orig/nut").
    backup_suffix : str
        Backup suffix (default: ".bak").

    Returns
    -------
    dict with keys: file, backup, patch, Ks, Cs
    """
    case_dir = Path(case_dir)
    nut_path = case_dir / file_relpath
    if not nut_path.exists():
        raise FileNotFoundError(f"nut file not found: {nut_path.resolve()}")

    # backup once
    bak = nut_path.with_suffix(nut_path.suffix + backup_suffix)
    if not bak.exists():
        shutil.copy2(nut_path, bak)

    txt = nut_path.read_text(encoding="utf-8", errors="ignore")

    # Find patch block:   patchName { ... }
    pat_block = re.compile(
        rf"(^\s*{re.escape(patch)}\s*\{{.*?^\s*\}})",
        flags=re.DOTALL | re.MULTILINE
    )
    m = pat_block.search(txt)
    if not m:
        raise RuntimeError(
            f"Patch '{patch}' not found in {nut_path}. " 
            "Check the patch name in boundaryField."
        )

    block = m.group(1)

    # If it is not a rough-wall function, still update it but report this in the return value
    is_rough = bool(re.search(r"\bnutkRoughWallFunction\b", block))

    def _set_uniform_line(b: str, key: str, value: float) -> str:
        # Replace existing 'key uniform ...;' line if present
        line_pat = re.compile(rf"(^\s*{re.escape(key)}\s+uniform\s+)([^;]+)(;)",
                              flags=re.MULTILINE)
        if line_pat.search(b):
            return line_pat.sub(rf"\g<1>{value:.6g}\3", b, count=1)

        # Otherwise, insert after 'type ...;' line if present, else after opening brace
        type_pat = re.compile(r"(^\s*type\s+[^;]+;\s*$)", flags=re.MULTILINE)
        mt = type_pat.search(b)
        insert_line = f"        {key:<14}uniform {value:.6g};\n"
        if mt:
            pos = mt.end(1)
            return b[:pos] + "\n" + insert_line + b[pos:]
        # fallback: after first '{'
        brace_pos = b.find("{")
        if brace_pos != -1:
            brace_pos += 1
            return b[:brace_pos] + "\n" + insert_line + b[brace_pos:]
        return b + "\n" + insert_line

    block2 = _set_uniform_line(block, "Ks", float(Ks))
    block2 = _set_uniform_line(block2, "Cs", float(Cs))

    txt2 = txt[:m.start(1)] + block2 + txt[m.end(1):]
    nut_path.write_text(txt2, encoding="utf-8")

    return {
        "file": str(nut_path),
        "backup": str(bak),
        "patch": patch,
        "Ks": float(Ks),
        "Cs": float(Cs),
        "is_rough_wall_function": is_rough,
    }
