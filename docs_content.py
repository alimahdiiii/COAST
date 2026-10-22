"""
Documentation text displayed inside the application interface.
"""

HEADER = """\
# CLASH/EurOtop → OpenFOAM Workflow Simulation

**Affiliation:** CNRS UMR 6614 CORIA — Université de Rouen Normandie
**Domain:** Numerical simulation for Wave–Structure Interaction

---

This app builds a complete OpenFOAM case for wave–structure interaction starting from
CLASH-style parameters. From a small set of inputs (geometry, wave conditions, resolution,
runtime settings), it:

- Generates the 3D STL of the structure and flume.
- Sizes the numerical wave flume and configures the mesh and refinement zones.
- Sets wave generation, initial conditions, sampling lines, and optional impact patch
  (with topoSet and createPatch).
- Writes a parallel OpenFOAM case, creates a job script, runs it in tmux, and produces
  an HTML report.
"""

HOW_TO_USE = """\
## How to use this app

1. **Set the main parameters at the top**
   - Choose the flow model: `FLOW_MODEL = "laminar"` or `"rans"` (sidebar).
   - Fill in the **CLASH / geometry parameters** in the **Geometry** tab
     (slopes, depths, berm, crest, container, etc.).
   - Set the **wave conditions** at deep water and at the toe (`Hm0`, `Tp`, `Tm`,
     `Tm-1,0`, `SWL`) in the **Wave & Domain** tab.
   - Adjust **mesh / refinement parameters** (`dx_target`, `dz_target`,
     `LEVEL_INTERFACE`, `LEVEL_RUNUP`, etc.) in the **Mesh** tab.
   - Choose **simulation control parameters** (`END_TIME`, `DELTA_T`,
     `WRITE_INTERVAL`, CFL limits) in the **Simulation & Job** tab.
   - Optionally enable a **porous armour layer** in the **Porous Zones** tab.

2. **Generate the STL geometry**
   - Go to the **Geometry** tab and click **► Generate STL**.
   - The app will build and save the STL geometry and display a cross-section schematic.

3. **Size the domain**
   - Go to **Wave & Domain** and click **► Size Domain**.
   - The domain extent is computed from the STL bounding box and the peak period Tp.

4. **Build and run the case**
   - Go to the **Run** tab and click **► Build Case**.
   - Then click **► Launch Simulation** to start the waveFoam solver in a tmux session.
   - Use **► Generate Report** to produce the HTML case report.

5. **Monitor progress**
   - Use the **Monitor** tab to track running simulations and open their HTML reports.

> **Tip:** For a new scenario, edit only the changed parameter(s) and click the
> relevant action buttons in order (STL → Domain → Build → Launch).
"""

TABLE1 = """\
### Table 1 – Main input parameters

| # | Variable name | Unit | Description | Notes / link to EurOtop name |
|---|---------------|------|-------------|-------------------------------|
| 1 | `m` | [-] | Cotangent of foreshore slope in front of the structure | If no foreshore: set to 1000. |
| 2 | `dx_foreshore_ramp` | [m] | Foreshore ramp length used in flume configuration | CAD-only parameter. Used only when `m = 1000`. |
| 3 | `dx_toe_transition` | [m] | Toe height-transition ramp length (between `h` and `h_t`) | CAD-only parameter. Used only when `m = 1000` and `h_t != h`. |
| 4 | `dx_berm_connector` | [m] | Connector ramp length before toe berm `B_t` | CAD-only parameter. Used only when `m != 1000` and `h_t != h`. |
| 5 | `h_deep` | [m] | Water depth at offshore "deep water" section | Used with `Hm0_deep`, `Tp_deep` to size waves and domain. |
| 6 | `h` | [m] | Water depth at the structure toe | |
| 7 | `h_t` | [m] | Toe depth at berm start (structure toe level) | If no toe: set equal to h. |
| 8 | `B_t` | [m] | Toe berm width | If no toe: set to 0. |
| 9 | `h_b` | [m] | Water depth on the berm (for a sloping berm: at mid-berm) | Positive below SWL, negative above (`z_berm = SWL - h_b`). |
| 10 | `B` | [m] | Berm width (horizontal) | If no berm: set to 0. |
| 11 | `tan_alpha_B` | [-] | Berm slope, EurOtop database `tanαB` | 0 = horizontal berm; > 0 = berm rising landward. |
| 12 | `cot_alpha_d` | [-] | Cotangent of lower slope angle (between berm and SWL) | If no berm: it is slope below crest. |
| 13 | `cot_alpha_u` | [-] | Cotangent of upper slope angle (between berm and crest) | If no berm: equal to `cot_alpha_d`. |
| 14 | `A_c` | [m] | Crest freeboard: crest level – SWL | If no wall: also crest height. |
| 15 | `G_c` | [m] | Crest width on top of slope | |
| 16 | `R_c` | [m] | Wall freeboard (top of crown wall – SWL) | If no crown wall: set equal to `A_c` or `NaN`. |
| 17 | `G_c` | [m] | Promenade / crest width in front of crown wall | 0 if no horizontal promenade. |
| 18 | `Ly` | [m] | Flume thickness in spanwise (y) direction | Used for 3D STL extrusion; `Ny` is usually 1 (2D slice). |
| 19 | `zmin` | [m] | Bottom z-level of the STL and domain | Must be below all structural points (`zmin < min(z levels)`). |
| 20 | `SWL` | [m] | Still Water Level in the global z-system | Reference for all vertical levels. |
| 21 | `Hm0_deep` | [m] | Significant wave height in deep water | Matches input `H_m0`. |
| 22 | `Tp_deep` | [s] | Peak period in deep water | Used to compute deep-water wavelength `L0`. |
| 23 | `Tm_deep` | [s] | Mean period in deep water | Can be estimated or read from spectrum. |
| 24 | `Hm0_toe` | [m] | Significant height at the toe | Optional; if missing, toe conditions are derived from deep. |
| 25 | `Tp_toe` | [s] | Peak period at the toe | |
| 26 | `Tm_toe` | [s] | Mean period at the toe | |
| 27 | `Tm10_toe` | [s] | Spectral mean period Tm−1,0 at the toe | Same as `T_m-1,0,t`. |
| 28 | `n_pre` | [-] | Number of deep-water wavelengths upstream of the structure | Domain inlet length: `Dxmin = xmin_stl − n_pre*L0` |
| 29 | `n_post` | [-] | Number of wavelengths downstream | Domain outlet length: `Dxmax = xmax_stl + n_post*L0` |
| 30 | `top_clearance` | [m] | Extra air height above highest STL point | Sets `Dzmax = zmax_stl + top_clearance` |
| 31 | `dx_target` | [m] | Target base cell size in x for `blockMesh` | Used to compute `Nx` and refine interface/run-up bands. |
| 32 | `dz_target` | [m] | Target base cell size in z | Used to compute `Nz`. |
| 33 | `Ny` | [-] | Number of cells in y (spanwise direction) | For 2D: `Ny = 1`. |
| 34 | `LEVEL_INTERFACE` | [-] | Refinement level around free surface | Level in `interfaceBand` in `snappyHexMeshDict`. |
| 35 | `LEVEL_RUNUP` | [-] | Refinement level in run-up/impact region | Level in `runupBand`. |
| 36 | `RUNUP_TOP_ABOVE_STL` | [m] | Extra height of run-up refinement above STL | Used in `runupBand` z-extent. |
| 37 | `RUNUP_X_START_OFFSET` | [m] | Offset in x where run-up band starts from the toe | Controls `x_ht` used for refinement and sampling. |
| 38 | `END_TIME` | [s] | Final simulation time of `waveFoam` | Used in `controlDict`. |
| 39 | `DELTA_T` | [s] | Time step | Used in `controlDict`. |
| 40 | `WRITE_INTERVAL` | [s] | Output write interval | Used in `controlDict` and sampling. |
| 41 | `MAX_CO` | [-] | Maximum Courant number | CFL limiter. |
| 42 | `MAX_ALPHA_CO` | [-] | Maximum interface Courant number | Stability near free surface. |
| 43 | `FLOW_MODEL` | — | Choice of turbulence model: `"laminar"` or `"rans"` | Used to select template case folder. |
| 44 | `N_PROCS` | [-] | Number of MPI processes | Used for `decomposePar` and `mpirun`. |
| 45 | `new_patch` | bool | Whether to create an additional impact patch | If `True`, `topoSet` + `createPatch` are run. |
| 46 | `PATCH_NAME` | — | Name of the new impact patch | Used in `forces` function in `controlDict`. |
| 47 | `SEG_ID` | — | ID of STL segment used to define the patch | Used in patch-creation utilities. |
| | | | **Porous zone parameters** | |
| 48 | `USE_POROUS` | bool | Enable porous armour-layer generation | If `True`, porous STL(s) are carved from the structure surface and written to `STL_Geos/`. |
| 49 | `POROUS_ZONES` | dict | Zone names → normal thickness [m] | Keys: `foreshore`, `toe_slope`, `toe_berm`, `lower_slope`, `berm`, `upper_slope`, `crest`. |
| 50 | `POROUS_MERGED` | bool | Merge all porous zones into one STL | `True` = `{case_name}_porous.stl`; `False` = one file per zone. |
| 51 | `POROUS_PHYSICS_DEFAULT` | dict | Default resistance parameters applied to all zones | Keys: `resistanceFormulation`, `porosity`, `d50` [m], `KC` [-], `alpha` [-], `beta` [-], `gammaAddedMass` [-]. |
| 52 | `POROUS_PHYSICS_OVERRIDES` | dict | Per-zone overrides of resistance parameters | Dict of zone name → partial physics dict. Overrides only the specified keys for that zone. |
"""

TABLE2 = """\
### Table 2 – Derived x– and z–positions of key points in the CLASH profile

| Symbol / variable | Meaning | Formula in the code | Notes |
|-------------------|---------|---------------------|-------|
| `z_deep` | Bed level at deep-water section | `z_deep = SWL - h_deep` | Offshore reference depth. |
| `z_toe_water` | Bed level at toe (water side) | `z_toe_water = SWL - h` | Same vertical datum as EurOtop. |
| `z_toe_berm` | Bed level at toe berm start | `z_toe_berm = SWL - h_t` | If no toe: usually `h_t = h`. |
| `z_berm` | Berm level (mid-berm) | `z_berm = SWL - h_b` | Can be above or below SWL depending on `h_b`. |
| `z_berm_start`, `z_berm_end` | Berm level at its seaward / landward edge | `z_berm -/+ 0.5 * B * tan_alpha_B` | Equal to `z_berm` for a horizontal berm. |
| `z_crest` | Crest level | `z_crest = SWL + A_c` | Top of slope / promenade. |
| `z_wall_top` | Top of crown wall | `z_wall_top = SWL + R_c` (if defined) | Only if a crown wall is present. |
| `dz_fs` | Vertical difference between deep and toe | `dz_fs = z_toe_water - z_deep` | Used to compute horizontal foreshore distance. |
| `dx_fs` | Horizontal foreshore length | If flume (`m ~ 1000`): `dx_fs = dx_foreshore_ramp` (or 0); else `dx_fs = dz_fs * m` | `dx_foreshore_ramp` is used only in flume mode. |
| `x_toe` | x-position of structure toe | **User input** | All other x-positions are measured from here. |
| `x_deep` | x at deep-water bed point | `x_deep = x_toe - dx_fs` | Upstream limit of foreshore. |
| `x_left` | Leftmost point of STL polygon | `x_left = x_deep - x_margin` | Adds extra margin on the offshore side. |
| `x_toe0` | Start of toe (water side) | `x_toe0 = x_toe` | Reference point for toe-related transitions. |
| `x_toe_berm_start` | Start of toe berm | Flume: `x_toe0 + dx_toe_transition`; non-flume: `x_toe0 + dx_berm_connector` | Short connector ramp when `h_t != h`. |
| `x_toe1` | End of toe berm | `x_toe1 = x_toe_berm_start + B_t` | Horizontal width of toe berm. |
| `dx_d` | Horizontal length of lower slope | `dx_d = (z_berm_start - z_toe_berm) * cot_alpha_d` | Zero if `cot_alpha_d = 0`. |
| `x_d_end` | End of lower slope (start of berm) | `x_d_end = x_toe1 + dx_d` | Point where berm begins. |
| `x_berm_end` | End of berm | `x_berm_end = x_d_end + B` | |
| `dx_u` | Horizontal length of upper slope | `dx_u = (z_crest - z_berm_end) * cot_alpha_u` | |
| `x_u_end` | End of upper slope (start of crest) | `x_u_end = x_berm_end + dx_u` | |
| `x_crest_end_raw` | Raw end of crest before "recovery" | `x_crest_end_raw = x_u_end + G_c` | |
| `x_crest_end` | Final crest end | `x_crest_end = max(x_crest_end_raw, x_berm_end + 0.20)` | Ensures a minimum horizontal recovery length. |
| `x_wall0` | Start of crown wall | `x_wall0 = x_crest_end` (if wall exists) | Wall is placed immediately after the crest. |
| `x_wall1` | End of crown wall | `x_wall1 = x_wall0 + t_wall` | Crown-wall thickness. |
| `x_container_start` | Start of overtopping box | `x_container_start = x_after` (after wall) | Where the overtopping collector begins. |
| `x_container_end` | End of overtopping box | `x_container_end = x_after + L_back` | Horizontal length of collector `L_back`. |
| `x_end` | Rightmost x of STL | `x_end = x_after + 0.02` | Small tail extension after the box. |
"""

PARAM_CLASH_FIG = "python_tools/figures_DIR/Param_clash.png"

EUROTOP_REF = (
    "Reference: EurOtop (2016). Manual on wave overtopping of sea defences and "
    "related structures. Second Edition. Available online: "
    "https://www.overtopping-manual.com"
)
