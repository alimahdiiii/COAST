"""
Streamlit entry point of the COAST application: builds the case generation
interface on top of the python_tools modules.
"""

from __future__ import annotations

import io
import contextlib
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

# python_tools imports
from python_tools.clash_cad import (
    build_and_save_stl,
    build_layered_porous_data, save_layered_porous_stls, solid_top_at_x,
    extrude_obstacle_stl, merge_stl_files, union_obstacles_with_poly,
    extrude_to_stl,
    ANCHOR_POINTS, resolve_anchor_point,
)
from python_tools.clash_cad import build_clash_profile_polygon_xz
from python_tools.clash_plot import (
    plot_clash_schematic, plot_layered_porous_on_ax, plot_obstacles_on_ax,
)
from python_tools.canvas_tools import (
    extract_xz_outline_from_stl,
)
from python_tools.domain_tools import (
    compute_domain_from_stl_and_Tp,
    load_stl_bbox,
    deepwater_L0,
    plot_domain_and_stl,
    plot_domain_stl_with_refinement,
    plot_domain_stl_with_lines,
)
from python_tools.foam_case_tools import (
    select_template_case,
    clone_case_from_template,
    copy_stl_to_trisurface,
    update_blockMeshDict_vertices_and_cells,
    update_surfaceFeatureExtractDict,
    write_snappyHexMeshDict_OF2206,
    update_waveProperties_input,
    update_setFieldsDict_water_box,
    update_controlDict_case,
    write_topoSetDict_force_and_overtopping,
    write_createPatchDict_for_segment,
    write_createPatchDict_for_segments,
    write_decomposeParDict,
    write_porosity_zones,
    add_porous_div_scheme,
    build_layered_zones_physics,
    update_nut_roughness_0orig,
    update_outlet_bc_for_breakwater,
    extract_xz_profile_segments_from_stl,
    segment_to_box,
    plot_stl_xz_with_segments_colored,
)
from python_tools.tmux_run_case import launch_tmux_case, write_job_script
from python_tools.case_report import write_run_info, write_html_report_pro, register_figure
from theme import inject_theme, get_mpl_colors, get_fig_container_style, sidebar_logo, sidebar_footer

try:
    from python_tools.clash_plot import plot_layered_porous_on_ax as _check_porous_plot  # noqa: F401
    _HAS_POROUS_PLOT = True
except ImportError:
    _HAS_POROUS_PLOT = False

try:
    from python_tools.Mesh_plot import plot_refinement_bands
    _HAS_REFINE_PLOT = True
except ImportError:
    _HAS_REFINE_PLOT = False

# Constants
STL_DIR = Path("STL_Geos")
REPO_ROOT = Path(__file__).parent.resolve()
VALID_POROUS_ZONES = [
    "foreshore", "toe_slope", "toe_berm",
    "lower_slope", "berm", "upper_slope", "crest",
    "back_slope",
]

def _default_porous_layer(name: str = "armor") -> dict:
    """Return a default porous layer configuration dict."""
    return {
        "name": name,
        "default_thickness": 0.08,
        "zone_thickness_overrides": {},
        "physics_default": {
            "resistanceFormulation": "vanGent1995",
            "porosity": 0.6,
            "d50": 0.05,
            "KC": 128.0,
            "alpha": 500.0,
            "beta": 1.0,
            "gammaAddedMass": 0.34,
        },
        "zone_physics_overrides": {},
    }


# Session state initialisation
def _init_state():
    if "initialized" in st.session_state:
        return
    s = st.session_state

    # Identity
    s.case_name   = ""
    s.flow_model  = "rans"

    # Geometry
    s.SWL         = 0.0
    s.Ly          = 1.0
    s.BREAKWATER_MODE = False
    s.USE_COLLECTOR = False
    s.x_toe       = 0.0
    s.x_margin    = 0.0
    s.L_back      = 0.1
    s.t_wall      = 0.05
    s.ZMIN_THICKNESS = 0.001
    s.m           = 1000.0
    s.h_deep      = 0.748
    s.h           = 0.502
    s.h_t         = 0.286
    s.B_t         = 0.308
    s.h_b         = -0.16
    s.B           = 0.523
    s.tan_alpha_B = 0.0
    s.cot_alpha_d = 1.25
    s.cot_alpha_u = 3.0
    s.A_c         = 0.237
    s.G_c         = 0.154
    s.R_c         = 0.237
    s.dx_foreshore_ramp  = 0.15
    s.dx_toe_transition  = 0.15
    s.dx_berm_connector  = 0.20

    # Waves
    s.Hm0_deep   = 0.192
    s.Tp_deep    = 2.79
    s.Tm_deep    = 1.92
    s.Hm0_toe    = 0.192
    s.Tp_toe     = 2.79
    s.Tm_toe     = 1.92
    s.Tm10_toe   = 2.54
    s.WAVE_SEED  = 1234
    s.n_pre      = 2.0
    s.n_post     = 0.03
    s.top_clearance = 0.45
    s.domain_mode     = "Auto (n × L₀)"
    s.inlet_length_m  = 20.0
    s.outlet_length_m = 1.0

    # Mesh
    s.dx_target  = 0.05
    s.dz_target  = 0.05
    s.Ny         = 1
    s.LEVEL_INTERFACE = 2
    s.LEVEL_RUNUP     = 2
    s.RUNUP_TOP_ABOVE_STL = 0.2
    s.RUNUP_X_START_OFFSET = 1.0
    s.RUNUP_X_END  = 3.5
    s.Vertical_distance_plus_for_refinement_interface = 0.1
    s.Y_LINE       = 0.5
    s.NPOINTS      = 200
    # Sampling interval for overtopping depth and velocity
    s.LINE_WRITE_INTERVAL = 0.02

    # Simulation
    s.END_TIME       = 200.0
    s.DELTA_T        = 0.01
    s.WRITE_INTERVAL = 0.10
    s.MAX_CO         = 0.8
    s.MAX_ALPHA_CO   = 0.8
    s.N_PROCS        = 8
    s.DECOMP_METHOD  = "scotch"

    # Patch
    s.new_patch    = False
    s.SEG_IDS      = []          # list of selected segment IDs (1-indexed)
    s.SOURCE_PATCH = "coast"

    # Porous
    s.USE_POROUS   = True
    s.POROUS_MERGED = True
    s.porous_zone_active  = {z: (z in ("upper_slope", "lower_slope")) for z in VALID_POROUS_ZONES}
    s.porous_zone_d       = {z: 0.08 for z in VALID_POROUS_ZONES}
    s.porous_back_slope_wedge = False
    s.porosity            = 0.6
    s.d50                 = 0.05
    s.KC                  = 128.0
    s.alpha_porous        = 500.0
    s.beta_porous         = 1.0
    s.gammaAddedMass      = 0.34
    s.resistanceFormulation = "vanGent1995"
    # Multi-layer porous
    s.porous_layers = [_default_porous_layer("armor")]
    s.core_is_solid = True
    s.core_physics  = None
    # Obstacles
    s.obstacles = []
    s.obstacle_abs_verts = []
    s.canvas_xlim = None       # kept for backward compatibility
    s.canvas_ylim = None
    s.imported_stl_outline = None
    s.obstacle_input_mode = "Draw"

    # RANS roughness
    s.Ks = 0.01
    s.Cs = 0.5

    # State
    s.step_done   = {"stl": False, "domain": False, "case": False, "run": False}
    s.params      = {}
    s.porous_data = None
    s.porous_paths = None
    s.domain      = None
    s.stl_mesh    = None
    s.stl_bbox    = None
    s.L0          = None
    s.case_dir    = None
    s.stl_path    = None
    s.meta        = None
    s.snappy_info = None

    # Keep run outputs across page rerenders.
    s.build_log    = []          # list of log lines from last Build Case
    s.build_result = None        # ("success"|"error", message)
    s.launch_result = None       # ("success"|"error", message)

    s.theme_mode = "day"         # "night" or "day"

    # Scan cases when Refresh is clicked.
    s.monitor_jobs = {}
    s.monitor_cache = {}

    s.initialized = True


# Helpers

def _build_params() -> dict:
    s = st.session_state
    z_deep = float(s.SWL) - float(s.h_deep)
    zmin   = z_deep - float(s.ZMIN_THICKNESS)
    return dict(
        SWL          = float(s.SWL),
        m            = float(s.m),
        h_deep       = float(s.h_deep),
        h            = float(s.h),
        h_t          = float(s.h_t),
        B_t          = float(s.B_t),
        h_b          = float(s.h_b),
        B            = float(s.B),
        tan_alpha_B  = float(s.tan_alpha_B),
        cot_alpha_d  = float(s.cot_alpha_d),
        cot_alpha_u  = float(s.cot_alpha_u),
        A_c          = float(s.A_c),
        G_c          = float(s.G_c),
        R_c          = float(s.R_c) if not math.isnan(float(s.R_c)) else float("nan"),
        x_toe        = float(s.x_toe),
        x_margin     = float(s.x_margin),
        L_back       = float(s.L_back),
        use_collector = bool(s.USE_COLLECTOR),
        t_wall       = float(s.t_wall),
        zmin         = float(zmin),
        dx_foreshore_ramp   = float(s.dx_foreshore_ramp),
        dx_toe_transition   = float(s.dx_toe_transition),
        dx_berm_connector   = float(s.dx_berm_connector),
    )


def _capture(fn, *args, **kwargs):
    """Call fn(*args, **kwargs) and return (result, captured_stdout)."""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        result = fn(*args, **kwargs)
    return result, buf.getvalue()



def _show_fig(fig, fig_name: str | None = None):
    """
    Display a matplotlib figure as a zoomable, pannable PNG embedded in HTML.
    - Mouse-wheel to zoom, drag to pan, double-click to reset.
    - Coordinate tooltip: hovering inside the axes shows x/z data values.
    - More axis ticks applied before saving.
    If fig_name is given, also saves a PNG to figures/{case_name}/.
    """
    import base64, json

    # more ticks on both axes
    for ax in fig.axes:
        ax.xaxis.set_major_locator(plt.MaxNLocator(nbins=15))
        ax.yaxis.set_major_locator(plt.MaxNLocator(nbins=10))

    # Finalize the layout before reading axes positions.
    try:
        fig.tight_layout()
    except Exception:
        pass

    # Get axes bounds in image pixels.
    # Avoid tight cropping so axes positions match the PNG coordinates.
    dpi = fig.get_dpi() or 150
    fig_w_px = fig.get_figwidth()  * dpi
    fig_h_px = fig.get_figheight() * dpi

    axes_meta = []
    for ax in fig.axes:
        pos  = ax.get_position()   # Bbox in [0,1] figure fractions
        xlim = list(ax.get_xlim())
        ylim = list(ax.get_ylim())
        axes_meta.append({
            "left":   pos.x0 * fig_w_px,
            "right":  pos.x1 * fig_w_px,
            "top":    (1.0 - pos.y1) * fig_h_px,   # PNG y-axis is top-down
            "bottom": (1.0 - pos.y0) * fig_h_px,
            "xlim":   xlim,
            "ylim":   ylim,
            "xlabel": ax.get_xlabel() or "x",
            "ylabel": ax.get_ylabel() or "y",
        })

    # apply theme colors to the figure
    _mc = get_mpl_colors()
    fig.patch.set_facecolor(_mc['fig_bg'])
    for ax in fig.axes:
        ax.set_facecolor(_mc['ax_bg'])
        ax.tick_params(colors=_mc['ax_text'])
        ax.xaxis.label.set_color(_mc['ax_text'])
        ax.yaxis.label.set_color(_mc['ax_text'])
        ax.title.set_color(_mc['ax_title'])
        for spine in ax.spines.values():
            spine.set_color(_mc['ax_spine'])

    # Render PNG bytes without cropping.
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi)
    buf.seek(0)
    img_bytes = buf.read()
    b64 = base64.b64encode(img_bytes).decode()

    if fig_name:
        # Keep image bytes in session state until the case directory exists.
        # _run_report() saves them to case_dir/figures/.
        if "pending_figures" not in st.session_state:
            st.session_state.pending_figures = {}
        st.session_state.pending_figures[fig_name] = img_bytes

    plt.close(fig)

    uid  = (fig_name or "fig").replace(".", "_")
    meta = json.dumps(axes_meta)

    html = f"""
<style>
#tt-{uid}{{
  position:absolute;pointer-events:none;display:none;
  background:rgba(0,0,0,0.72);color:#fff;font:12px/1.4 monospace;
  padding:4px 8px;border-radius:4px;white-space:nowrap;z-index:99;
}}
</style>
<div id="c-{uid}" style="overflow:hidden;width:100%;height:500px;
     cursor:crosshair;{get_fig_container_style()};position:relative;">
  <img id="i-{uid}" src="data:image/png;base64,{b64}"
       style="transform-origin:0 0;user-select:none;max-width:none;" draggable="false"/>
  <div id="tt-{uid}"></div>
</div>
<script>
(function(){{
  var C=document.getElementById('c-{uid}'),
      I=document.getElementById('i-{uid}'),
      TT=document.getElementById('tt-{uid}');
  var sc=1, ox=0, oy=0, drag=false, sx, sy;
  var AXES={meta};

  function apply(){{ I.style.transform='translate('+ox+'px,'+oy+'px) scale('+sc+')'; }}
  function fitInit(){{
    var r=C.clientWidth/I.naturalWidth; sc=r; ox=0; oy=0; apply();
  }}
  if(I.complete){{ fitInit(); }} else {{ I.onload=fitInit; }}

  C.addEventListener('wheel',function(e){{
    e.preventDefault();
    var rect=C.getBoundingClientRect(), mx=e.clientX-rect.left, my=e.clientY-rect.top;
    var d=e.deltaY<0?1.12:0.89;
    ox=mx-(mx-ox)*d; oy=my-(my-oy)*d; sc*=d; apply();
  }},{{passive:false}});
  C.addEventListener('mousedown',function(e){{
    drag=true; sx=e.clientX-ox; sy=e.clientY-oy; C.style.cursor='grabbing';
  }});
  document.addEventListener('mouseup',function(){{ drag=false; C.style.cursor='crosshair'; }});
  document.addEventListener('mousemove',function(e){{
    if(drag){{ ox=e.clientX-sx; oy=e.clientY-sy; apply(); return; }}
    // coordinate tooltip
    var rect=C.getBoundingClientRect();
    var cx=e.clientX-rect.left, cy=e.clientY-rect.top;
    // cursor → image pixel coords
    var px=(cx-ox)/sc, py=(cy-oy)/sc;
    var hit=false;
    for(var i=0;i<AXES.length;i++){{
      var a=AXES[i];
      if(px>=a.left && px<=a.right && py>=a.top && py<=a.bottom){{
        var dx=a.xlim[0]+(px-a.left)/(a.right-a.left)*(a.xlim[1]-a.xlim[0]);
        var dy=a.ylim[1]-(py-a.top)/(a.bottom-a.top)*(a.ylim[1]-a.ylim[0]);
        TT.textContent=a.xlabel+': '+dx.toFixed(4)+' | '+a.ylabel+': '+dy.toFixed(4);
        // keep tooltip inside container
        var tx=cx+14, ty=cy-28;
        if(tx+180>C.clientWidth) tx=cx-190;
        if(ty<4) ty=cy+14;
        TT.style.left=tx+'px'; TT.style.top=ty+'px'; TT.style.display='block';
        hit=true; break;
      }}
    }}
    if(!hit) TT.style.display='none';
  }});
  C.addEventListener('mouseleave',function(){{ TT.style.display='none'; }});
  C.addEventListener('dblclick',function(){{ fitInit(); }});
}})();
</script>"""
    st.components.v1.html(html, height=520)


def _status_icon(done: bool) -> str:
    return "✅" if done else "❌"



# Sidebar
def _sidebar():
    with st.sidebar:
        # COAST logo above the case name
        sidebar_logo()

        st.session_state.case_name  = st.text_input("Case name", st.session_state.case_name)
        st.session_state.flow_model = st.selectbox(
            "Flow model", ["rans", "laminar"],
            index=0 if st.session_state.flow_model == "rans" else 1,
        )

        st.divider()
        st.subheader("Status")
        d = st.session_state.step_done
        st.write(f"{_status_icon(d['stl'])}  STL generated")
        st.write(f"{_status_icon(d['domain'])}  Domain sized")
        st.write(f"{_status_icon(d['case'])}  Case built")
        st.write(f"{_status_icon(d['run'])}  Simulation launched")
        # Institution logos
        sidebar_footer()

        # Theme toggle
        is_night = st.session_state.theme_mode == "night"
        icon = "☀️" if is_night else "🌙"
        left, _ = st.columns([1, 3])
        with left:
            if st.button(icon, key="theme_toggle", help="Toggle day / night theme"):
                st.session_state.theme_mode = "day" if is_night else "night"
                st.rerun()



# Tab 1: Geometry
def _tab_geometry():
    st.header("Geometry")

    with st.expander("Core depths", expanded=True):
        c1, c2, c3 = st.columns(3)
        st.session_state.SWL    = c1.number_input("SWL [m]",  value=float(st.session_state.SWL),  format="%.4f")
        st.session_state.m      = c2.number_input("m (foreshore slope)",  value=float(st.session_state.m),  format="%.1f")
        st.session_state.h_deep = c3.number_input("h_deep [m]", value=float(st.session_state.h_deep), format="%.4f")
        c4, c5, c6 = st.columns(3)
        st.session_state.h   = c4.number_input("h [m]",   value=float(st.session_state.h),   format="%.4f")
        st.session_state.h_t = c5.number_input("h_t [m]", value=float(st.session_state.h_t), format="%.4f")
        st.session_state.B_t = c6.number_input("B_t [m]", value=float(st.session_state.B_t), format="%.4f")
        c7, c8, c9 = st.columns(3)
        st.session_state.h_b = c7.number_input("h_b [m]", value=float(st.session_state.h_b), format="%.4f",
                                               help="Water depth on the berm (negative above SWL). For a sloping berm: depth at mid-berm.")
        st.session_state.B   = c8.number_input("B [m]",   value=float(st.session_state.B),   format="%.4f")
        st.session_state.tan_alpha_B = c9.number_input("tan_alpha_B [-]", value=float(st.session_state.tan_alpha_B), format="%.4f",
                                                       help="Berm slope (EurOtop database tanαB). 0 = horizontal berm; > 0 = berm rising landward.")

    with st.expander("Slopes"):
        c1, c2 = st.columns(2)
        st.session_state.cot_alpha_d = c1.number_input("cot_alpha_d", value=float(st.session_state.cot_alpha_d), format="%.3f")
        st.session_state.cot_alpha_u = c2.number_input("cot_alpha_u", value=float(st.session_state.cot_alpha_u), format="%.3f")

    with st.expander("Crest"):
        c1, c2, c3, c4 = st.columns(4)
        st.session_state.A_c    = c1.number_input("A_c [m]",   value=float(st.session_state.A_c),   format="%.4f")
        st.session_state.G_c    = c2.number_input("G_c [m]",   value=float(st.session_state.G_c),   format="%.4f")
        st.session_state.R_c    = c3.number_input("R_c [m]",   value=float(st.session_state.R_c),   format="%.4f")
        st.session_state.t_wall = c4.number_input("t_wall [m]", value=float(st.session_state.t_wall), format="%.4f")

    with st.expander("Structure type & Placement"):
        st.session_state.BREAKWATER_MODE = st.checkbox(
            "Breakwater mode (water on both sides)",
            value=bool(st.session_state.BREAKWATER_MODE),
            help="Enable for rubble mound breakwaters: water on both sides, "
                 "outlet absorption zone, no collector.",
        )
        if st.session_state.BREAKWATER_MODE:
            st.session_state.USE_COLLECTOR = False
            st.info("Breakwater mode: collector disabled, outlet relaxation zone will be added.")

        c1, c2, c3 = st.columns(3)
        st.session_state.x_toe   = c1.number_input("x_toe [m]",  value=float(st.session_state.x_toe),   format="%.4f")
        st.session_state.x_margin = c2.number_input("x_margin [m]", value=float(st.session_state.x_margin), format="%.4f")
        st.session_state.Ly      = c3.number_input("Ly [m]",     value=float(st.session_state.Ly),     format="%.3f")
        c4, c5, c6 = st.columns(3)
        st.session_state.ZMIN_THICKNESS = c4.number_input("ZMIN_THICKNESS [m]", value=float(st.session_state.ZMIN_THICKNESS), format="%.4f")
        st.session_state.L_back         = c5.number_input("L_back [m]",          value=float(st.session_state.L_back),         format="%.4f")
        if not st.session_state.BREAKWATER_MODE:
            st.session_state.USE_COLLECTOR  = c6.checkbox("USE_COLLECTOR",            value=bool(st.session_state.USE_COLLECTOR))

    with st.expander("CAD ramps"):
        c1, c2, c3 = st.columns(3)
        st.session_state.dx_foreshore_ramp = c1.number_input("dx_foreshore_ramp [m]", value=float(st.session_state.dx_foreshore_ramp), format="%.3f")
        st.session_state.dx_toe_transition = c2.number_input("dx_toe_transition [m]", value=float(st.session_state.dx_toe_transition), format="%.3f")
        st.session_state.dx_berm_connector = c3.number_input("dx_berm_connector [m]", value=float(st.session_state.dx_berm_connector), format="%.3f")

    st.divider()
    st.subheader("Porous Structure")

    s = st.session_state
    s.USE_POROUS = st.checkbox("Enable porous zones", value=bool(s.USE_POROUS))

    if s.USE_POROUS:
        # Active zones
        st.caption("Active zones")
        cols = st.columns(4)
        for i, zone in enumerate(VALID_POROUS_ZONES):
            col = cols[i % 4]
            s.porous_zone_active[zone] = col.checkbox(
                zone, value=bool(s.porous_zone_active.get(zone, False)),
                key=f"pz_active_{zone}",
            )
        if s.porous_zone_active.get("back_slope", False):
            s.porous_back_slope_wedge = st.checkbox(
                "back_slope: crest wedge only",
                value=bool(s.get("porous_back_slope_wedge", False)),
                key="pz_back_slope_wedge",
                help="Fill only the wedge between the crest end and the back slope, "
                     "down to z_crest - d (the level of the crest layer), instead of "
                     "a layer along the whole back slope.",
            )

        # Number of layers
        n_layers = st.number_input(
            "Number of layers", min_value=1, max_value=5,
            value=len(s.porous_layers), key="n_porous_layers",
        )
        _layer_names = ["armor", "filter", "sublayer", "core_porous", "extra"]
        while len(s.porous_layers) < n_layers:
            idx = len(s.porous_layers)
            s.porous_layers.append(_default_porous_layer(_layer_names[min(idx, 4)]))
        while len(s.porous_layers) > n_layers:
            s.porous_layers.pop()

        # Per-layer configuration
        for li, layer in enumerate(s.porous_layers):
            with st.expander(f"Layer {li+1}: {layer['name']}", expanded=(li == 0)):
                layer["name"] = st.text_input(
                    "Layer name", value=layer["name"], key=f"layer_name_{li}",
                )
                layer["default_thickness"] = st.number_input(
                    "Default thickness d [m]",
                    value=float(layer["default_thickness"]),
                    format="%.4f", key=f"layer_d_{li}",
                )

                # Zone thickness overrides
                with st.expander("Zone thickness overrides", expanded=False):
                    for zone in VALID_POROUS_ZONES:
                        if s.porous_zone_active.get(zone, False):
                            cur = float(layer["zone_thickness_overrides"].get(
                                zone, layer["default_thickness"]
                            ))
                            val = st.number_input(
                                f"d ({zone}) [m]", value=cur,
                                format="%.4f", key=f"layer_{li}_d_{zone}",
                            )
                            if abs(val - layer["default_thickness"]) > 1e-8:
                                layer["zone_thickness_overrides"][zone] = val
                            elif zone in layer["zone_thickness_overrides"]:
                                del layer["zone_thickness_overrides"][zone]

                # Resistance parameters
                with st.expander("Resistance parameters", expanded=False):
                    phys = layer["physics_default"]
                    c1, c2, c3 = st.columns(3)
                    phys["porosity"] = c1.number_input(
                        "porosity", value=float(phys["porosity"]),
                        format="%.4f", key=f"phys_por_{li}",
                    )
                    phys["d50"] = c2.number_input(
                        "d50 [m]", value=float(phys["d50"]),
                        format="%.4f", key=f"phys_d50_{li}",
                    )
                    phys["KC"] = c3.number_input(
                        "KC", value=float(phys["KC"]),
                        format="%.1f", key=f"phys_KC_{li}",
                    )
                    c4, c5, c6 = st.columns(3)
                    phys["alpha"] = c4.number_input(
                        "alpha", value=float(phys["alpha"]),
                        format="%.1f", key=f"phys_alpha_{li}",
                    )
                    phys["beta"] = c5.number_input(
                        "beta", value=float(phys["beta"]),
                        format="%.3f", key=f"phys_beta_{li}",
                    )
                    phys["gammaAddedMass"] = c6.number_input(
                        "gammaAddedMass", value=float(phys["gammaAddedMass"]),
                        format="%.4f", key=f"phys_gamma_{li}",
                    )

                # Zone physics overrides
                with st.expander("Zone physics overrides", expanded=False):
                    for zone in VALID_POROUS_ZONES:
                        if s.porous_zone_active.get(zone, False):
                            use_ov = st.checkbox(
                                f"Override {zone}",
                                value=bool(zone in layer.get("zone_physics_overrides", {})),
                                key=f"phys_ov_{li}_{zone}",
                            )
                            if use_ov:
                                ov = layer.setdefault("zone_physics_overrides", {}).setdefault(zone, {})
                                oc1, oc2 = st.columns(2)
                                ov["porosity"] = oc1.number_input(
                                    f"porosity ({zone})",
                                    value=float(ov.get("porosity", phys["porosity"])),
                                    format="%.4f", key=f"ov_por_{li}_{zone}",
                                )
                                ov["d50"] = oc2.number_input(
                                    f"d50 ({zone}) [m]",
                                    value=float(ov.get("d50", phys["d50"])),
                                    format="%.4f", key=f"ov_d50_{li}_{zone}",
                                )
                            elif zone in layer.get("zone_physics_overrides", {}):
                                del layer["zone_physics_overrides"][zone]

        # Core settings
        st.divider()
        s.core_is_solid = st.checkbox("Solid core", value=bool(s.core_is_solid))
        if not s.core_is_solid:
            st.caption("Core is porous — set its resistance parameters below.")
            if s.core_physics is None:
                s.core_physics = {
                    "resistanceFormulation": "vanGent1995",
                    "porosity": 0.4, "d50": 0.03, "KC": 128.0,
                    "alpha": 500.0, "beta": 1.0, "gammaAddedMass": 0.34,
                }
            cp = s.core_physics
            cc1, cc2, cc3, cc4 = st.columns(4)
            cp["porosity"] = cc1.number_input("Core porosity", value=float(cp["porosity"]), format="%.3f", key="core_porosity")
            cp["d50"] = cc2.number_input("Core d50 [m]", value=float(cp["d50"]), format="%.4f", key="core_d50")
            cp["alpha"] = cc3.number_input("Core alpha", value=float(cp["alpha"]), format="%.1f", key="core_alpha")
            cp["beta"] = cc4.number_input("Core beta", value=float(cp["beta"]), format="%.2f", key="core_beta")

        # Copy layer 0 settings to the legacy keys.
        if s.porous_layers:
            p0 = s.porous_layers[0]["physics_default"]
            s.porosity = p0["porosity"]
            s.d50 = p0["d50"]
            s.KC = p0["KC"]
            s.alpha_porous = p0["alpha"]
            s.beta_porous = p0["beta"]
            s.gammaAddedMass = p0["gammaAddedMass"]

    # Obstacles (separate STL)
    st.divider()
    _obs_hdr_l, _obs_hdr_r = st.columns([0.9, 0.1])
    _obs_hdr_l.subheader("Obstacles")
    if _obs_hdr_r.button("\u27f3", help="Refresh obstacle preview"):
        st.rerun()

    # Hide the obstacle editor until the user opens it.
    _show_obs_editor = st.checkbox(
        "\u2795 Add a new obstacle",
        value=False,
        key="show_obstacle_editor",
        help="Show the interactive view and inputs for adding/importing obstacles.",
    )

    if _show_obs_editor:  # obstacle editor visibility
        # Interactive plot and obstacle coordinates
        # Enter obstacle coordinates in meters.

        try:
            import plotly.graph_objects as go
            params_preview = _build_params()
            poly_preview, meta_preview = build_clash_profile_polygon_xz(
                params_preview, return_meta=True,
            )

            # Plot the structure and obstacles.
            _px = np.append(poly_preview[:, 0], poly_preview[0, 0])
            _pz = np.append(poly_preview[:, 1], poly_preview[0, 1])

            fig_obs = go.Figure()

            # Water fill below SWL
            _SWL = float(params_preview["SWL"])
            _xmin, _xmax = float(poly_preview[:, 0].min()), float(poly_preview[:, 0].max())
            _zmin = float(poly_preview[:, 1].min())
            fig_obs.add_shape(
                type="rect", x0=_xmin - 0.5, x1=_xmax + 0.5,
                y0=_zmin - 0.3, y1=_SWL,
                fillcolor="rgba(187,222,251,0.3)", line_width=0, layer="below",
            )
            # SWL line
            fig_obs.add_hline(
                y=_SWL, line_dash="dash", line_color="#1565c0",
                line_width=1.5, opacity=0.7,
                annotation_text=f"SWL = {_SWL:.2f} m",
                annotation_position="top left",
                annotation_font_size=10,
            )

            # Structure polygon
            fig_obs.add_trace(go.Scatter(
                x=_px.tolist(), y=_pz.tolist(), mode="lines",
                fill="toself", fillcolor="rgba(196,154,108,0.85)",
                line=dict(color="#4a3728", width=2),
                name="Structure", hoverinfo="skip",
            ))

            # Porous overlays
            if s.USE_POROUS and s.porous_data is not None:
                _pd = s.porous_data
                # Collect porous polygons as (label, polygon) pairs.
                _porous_items = []
                _lp = _pd.get("layer_polys")
                _pp = _pd.get("porous_polys")
                if _lp and isinstance(_lp, dict):
                    for _lname, _zones in _lp.items():
                        if isinstance(_zones, dict):
                            for _zname, _zpoly in _zones.items():
                                _porous_items.append((f"{_lname}/{_zname}", _zpoly))
                        else:
                            _porous_items.append((_lname, _zones))
                elif _pp and isinstance(_pp, dict):
                    for _zname, _zpoly in _pp.items():
                        _porous_items.append((_zname, _zpoly))
                for _plabel, _zpoly in _porous_items:
                    try:
                        _zp = np.array(_zpoly, float)
                        if _zp.ndim == 2 and _zp.shape[0] >= 3:
                            fig_obs.add_trace(go.Scatter(
                                x=np.append(_zp[:, 0], _zp[0, 0]).tolist(),
                                y=np.append(_zp[:, 1], _zp[0, 1]).tolist(),
                                mode="lines", fill="toself",
                                fillcolor="rgba(100,200,100,0.25)",
                                line=dict(color="green", width=1, dash="dot"),
                                name=f"porous {_plabel}",
                                hoverinfo="skip", showlegend=False,
                            ))
                    except Exception:
                        pass

            # Existing obstacles
            for _oi, _obs in enumerate(s.obstacles):
                _verts = _obs.get("vertices_rel", [])
                if len(_verts) < 3:
                    continue
                _ox = [v[0] for v in _verts] + [_verts[0][0]]
                _oz = [v[1] for v in _verts] + [_verts[0][1]]
                fig_obs.add_trace(go.Scatter(
                    x=_ox, y=_oz, mode="lines", fill="toself",
                    fillcolor="rgba(30,80,200,0.25)",
                    line=dict(color="rgba(30,80,200,0.8)", width=2),
                    name=_obs.get("name", f"obstacle_{_oi}"),
                    hoverinfo="skip",
                ))

            # Equal axis scale with zoom and pan
            _margin_x = max(0.3, (_xmax - _xmin) * 0.08)
            _zmax_poly = float(poly_preview[:, 1].max())
            _margin_z = max(0.15, (_zmax_poly - _zmin) * 0.25)
            fig_obs.update_layout(
                xaxis=dict(
                    title="x [m]",
                    range=[_xmin - _margin_x, _xmax + _margin_x],
                    scaleanchor="y", scaleratio=1,
                ),
                yaxis=dict(
                    title="z [m]",
                    range=[_zmin - _margin_z * 0.5, _zmax_poly + _margin_z],
                ),
                height=500, margin=dict(l=50, r=20, t=30, b=50),
                dragmode="pan",
                showlegend=False,
                hovermode="closest",
            )

            st.plotly_chart(fig_obs, width="stretch",
                           config=dict(displayModeBar=True, scrollZoom=True),
                           key="obstacle_view_chart")

            st.caption(
                "Zoom: scroll wheel. Pan: drag. "
                "Hover to read (x, z) coordinates. "
                "Use the inputs below to add obstacles at exact positions."
            )

            # Obstacle inputs
            _add_mode = st.radio(
                "Add obstacle as", ["Rectangle", "Polygon vertices"],
                horizontal=True, key="obs_draw_mode",
            )

            if _add_mode == "Rectangle":
                st.caption(
                    "Read the coordinates from the figure above (hover to see x, z), "
                    "then enter the rectangle bounds below."
                )
                _rc1, _rc2 = st.columns(2)
                _rx0 = _rc1.number_input("Left x [m]", value=float(_xmin + 0.5),
                                         format="%.3f", key="obs_rect_x0")
                _rz0 = _rc1.number_input("Bottom z [m]", value=float(_SWL - 0.1),
                                         format="%.3f", key="obs_rect_z0")
                _rx1 = _rc2.number_input("Right x [m]", value=float(_xmin + 0.7),
                                         format="%.3f", key="obs_rect_x1")
                _rz1 = _rc2.number_input("Top z [m]", value=float(_SWL + 0.1),
                                         format="%.3f", key="obs_rect_z1")
                if st.button("Add rectangle obstacle", key="obs_add_rect"):
                    if abs(_rx1 - _rx0) > 1e-6 and abs(_rz1 - _rz0) > 1e-6:
                        _x0, _x1 = min(_rx0, _rx1), max(_rx0, _rx1)
                        _z0, _z1 = min(_rz0, _rz1), max(_rz0, _rz1)
                        s.obstacles.append({
                            "name": f"rect_obstacle_{len(s.obstacles)}",
                            "anchor": None,
                            "vertices_rel": [
                                (_x0, _z0), (_x1, _z0), (_x1, _z1), (_x0, _z1),
                            ],
                        })
                        st.success(
                            f"Added rectangle obstacle: "
                            f"({_x0:.3f}, {_z0:.3f}) → ({_x1:.3f}, {_z1:.3f}) m"
                        )
                        st.rerun()
                    else:
                        st.warning("Rectangle width and height must be > 0.")

            else:  # Polygon vertices
                st.caption(
                    "Enter polygon vertices as `x, z` pairs (one per line). "
                    "Read coordinates from the figure above."
                )
                _default_verts = (
                    f"{_xmin + 0.5:.3f}, {_SWL - 0.05:.3f}\n"
                    f"{_xmin + 0.7:.3f}, {_SWL - 0.05:.3f}\n"
                    f"{_xmin + 0.6:.3f}, {_SWL + 0.10:.3f}"
                )
                _vtxt = st.text_area(
                    "Vertices (x, z per line)",
                    value=_default_verts, height=120,
                    key="obs_poly_verts_text",
                )
                if st.button("Add polygon obstacle", key="obs_add_poly"):
                    try:
                        _pverts = []
                        for _ln in _vtxt.strip().splitlines():
                            _ln = _ln.strip()
                            if not _ln or _ln.startswith("#"):
                                continue
                            _parts = _ln.split(",")
                            if len(_parts) == 2:
                                _pverts.append((
                                    float(_parts[0].strip()),
                                    float(_parts[1].strip()),
                                ))
                        if len(_pverts) < 3:
                            st.error("Need at least 3 vertices.")
                        else:
                            s.obstacles.append({
                                "name": f"poly_obstacle_{len(s.obstacles)}",
                                "anchor": None,
                                "vertices_rel": _pverts,
                            })
                            st.success(f"Added polygon obstacle with {len(_pverts)} vertices.")
                            st.rerun()
                    except ValueError as e:
                        st.error(f"Parse error: {e}")

        except Exception as e:
            st.error(f"Interactive view failed: {e}")
            import traceback
            st.code(traceback.format_exc())

    # Import STL obstacle
    if _show_obs_editor:
        with st.expander("Import STL obstacle"):
            uploaded = st.file_uploader("Upload .stl file", type=["stl"], key="obs_stl_upload")
            y_plane = st.number_input("Y-plane for cross-section", value=0.0,
                                      format="%.3f", key="obs_y_plane")
            if st.button("Import obstacle from STL") and uploaded is not None:
                import tempfile
                with tempfile.NamedTemporaryFile(suffix=".stl", delete=False) as tmp:
                    tmp.write(uploaded.read())
                    tmp_path = tmp.name
                try:
                    outline = extract_xz_outline_from_stl(tmp_path, y_plane=y_plane)
                    stl_name = uploaded.name.replace(".stl", "").replace(" ", "_")
                    s.obstacles.append({
                        "name": f"stl_{stl_name}",
                        "anchor": None,
                        "vertices_rel": list(outline),
                    })
                    st.success(
                        f"Imported '{stl_name}' with {len(outline)} vertices. "
                        "Use the dimension controls below to position and resize it."
                    )
                    st.rerun()
                except Exception as e:
                    st.error(f"STL extraction failed: {e}")

    # Show current obstacles with editable dimensions
    if s.obstacles:
        st.markdown("**Current obstacles:**")
        for oi, obs in enumerate(s.obstacles):
            verts = obs.get("vertices_rel", [])
            if not verts:
                continue
            xs = [v[0] for v in verts]
            zs = [v[1] for v in verts]
            cur_w = max(xs) - min(xs)
            cur_h = max(zs) - min(zs)
            cx = (max(xs) + min(xs)) / 2
            cz = (max(zs) + min(zs)) / 2

            with st.expander(
                f"{oi+1}. {obs.get('name', 'obstacle')} — "
                f"width {cur_w:.3f} m × height {cur_h:.3f} m",
            ):
                st.caption(
                    f"Bounding box: **{cur_w:.4f} m** wide × **{cur_h:.4f} m** tall  "
                    f"| centroid ({cx:.3f}, {cz:.3f}) m  "
                    "— these are the real-world dimensions used in the generated STL."
                )
                c1, c2, c3 = st.columns(3)
                new_w = c1.number_input("Width [m]", value=cur_w, min_value=0.001,
                                        format="%.4f", key=f"obs_w_{oi}")
                new_h = c2.number_input("Height [m]", value=cur_h, min_value=0.001,
                                        format="%.4f", key=f"obs_h_{oi}")
                c4, c5, c6 = st.columns(3)
                new_cx = c4.number_input("Center X [m]", value=cx,
                                         format="%.4f", key=f"obs_cx_{oi}")
                new_cz = c5.number_input("Center Z [m]", value=cz,
                                         format="%.4f", key=f"obs_cz_{oi}")

                col_apply, col_rm = st.columns(2)
                if col_apply.button("Apply dimensions", key=f"obs_apply_{oi}"):
                    # Scale and translate vertices to new dimensions/position
                    sx = new_w / max(cur_w, 1e-12)
                    sz = new_h / max(cur_h, 1e-12)
                    new_verts = [
                        (new_cx + (x - cx) * sx, new_cz + (z - cz) * sz)
                        for x, z in verts
                    ]
                    s.obstacles[oi]["vertices_rel"] = new_verts
                    st.rerun()
                if col_rm.button("Remove", key=f"obs_rm_{oi}"):
                    s.obstacles.pop(oi)
                    st.rerun()

    st.divider()
    if st.button("\u25ba Generate STL", type="primary"):
        _run_generate_stl()


def _run_generate_stl():
    s = st.session_state
    try:
        params = _build_params()
        s.params = params

        STL_DIR.mkdir(exist_ok=True)
        out_name = STL_DIR / f"{s.case_name}.stl"

        with st.spinner("Building STL…"):
            poly, meta = build_and_save_stl(
                params, float(s.Ly), str(out_name),
                return_meta=True,
            )

        s.meta = meta

        # Resolve obstacle vertices and merge them with the structure.
        s.obstacle_abs_verts = []
        _merged_poly = None          # merged polygon for the schematic
        if s.obstacles:
            for oi, obs in enumerate(s.obstacles):
                verts_rel = obs.get("vertices_rel", [])
                anchor_name = obs.get("anchor")
                if anchor_name:
                    _ax, _az = resolve_anchor_point(anchor_name, params, meta)
                    abs_verts = [(_ax + dx, _az + dz) for dx, dz in verts_rel]
                else:
                    abs_verts = list(verts_rel)
                s.obstacle_abs_verts.append(abs_verts)

            # Merge polygons with Shapely.
            _merged_poly = union_obstacles_with_poly(poly, s.obstacle_abs_verts)
            if _merged_poly is not None:
                extrude_to_stl(_merged_poly, float(s.Ly), str(out_name))
                st.success(
                    f"STL saved — {len(s.obstacle_abs_verts)} obstacle(s) union-merged "
                    f"into structure: `{out_name}`"
                )
            else:
                # Combine separate STLs if the polygon merge fails.
                obstacle_stl_paths = []
                for oi, abs_verts in enumerate(s.obstacle_abs_verts):
                    obs_name = s.obstacles[oi].get("name", f"obstacle_{oi}")
                    obs_stl = STL_DIR / f"{s.case_name}_{obs_name}.stl"
                    extrude_obstacle_stl(abs_verts, float(s.Ly), str(obs_stl))
                    obstacle_stl_paths.append(obs_stl)
                merge_stl_files([out_name] + obstacle_stl_paths, str(out_name))
                st.success(
                    f"STL saved with {len(obstacle_stl_paths)} obstacle(s) merged "
                    f"(concatenated — install shapely for true union): `{out_name}`"
                )
        else:
            st.success(f"STL saved: `{out_name}`")

        s.stl_path = out_name

        # Porous zone data (multi-layer)
        if s.USE_POROUS:
            active_zones_filter = {
                z: bool(s.porous_zone_active.get(z, False))
                for z in VALID_POROUS_ZONES
            }
            any_active = any(active_zones_filter.values())
            if any_active:
                with st.spinner("Building porous zone data…"):
                    buf = io.StringIO()
                    with contextlib.redirect_stdout(buf):
                        porous_data = build_layered_porous_data(
                            params,
                            layers=s.porous_layers,
                            active_zones=active_zones_filter,
                            core_is_solid=s.core_is_solid,
                            back_slope_wedge=bool(s.get("porous_back_slope_wedge", False)),
                        )
                    porous_paths = save_layered_porous_stls(
                        layered_data=porous_data,
                        Ly=float(s.Ly),
                        out_dir=STL_DIR,
                        case_name=s.case_name,
                        core_is_solid=s.core_is_solid,
                    )
                s.porous_data  = porous_data
                s.porous_paths = porous_paths
                n_layers = len(s.porous_layers)
                n_stls = len(porous_paths["layer_stls"])
                st.success(
                    f"Porous STLs saved: {n_layers} layer(s), "
                    f"{n_stls} zone STL(s), zones: {porous_data['active_zones']}"
                )

        # Plot the merged polygon when available.
        _plot_poly = _merged_poly if _merged_poly is not None else poly
        fig, ax = plot_clash_schematic(params, _plot_poly)
        if s.USE_POROUS and s.porous_data is not None and _HAS_POROUS_PLOT:
            plot_layered_porous_on_ax(ax, s.porous_data)
        if s.obstacle_abs_verts and _merged_poly is None:
            # Otherwise, draw obstacles separately and expand the plot limits.
            plot_obstacles_on_ax(ax, s.obstacle_abs_verts, s.obstacles)
            _obs_x = [x for verts in s.obstacle_abs_verts for (x, _z) in verts]
            _obs_z = [z for verts in s.obstacle_abs_verts for (_x, z) in verts]
            if _obs_x:
                _cx0, _cx1 = ax.get_xlim()
                _cz0, _cz1 = ax.get_ylim()
                _marg = 0.1
                ax.set_xlim(min(_cx0, min(_obs_x) - _marg), max(_cx1, max(_obs_x) + _marg))
                ax.set_ylim(min(_cz0, min(_obs_z) - _marg), max(_cz1, max(_obs_z) + _marg))

        s.step_done["stl"]    = True
        s.step_done["domain"] = False
        s.step_done["case"]   = False
        s.step_done["run"]    = False
        _show_fig(fig, "01_clash_schematic")

    except Exception as exc:
        st.error(f"STL generation failed: {exc}")
        import traceback
        st.code(traceback.format_exc())


# Tab 2: Wave & Domain
def _tab_wave():
    st.header("Wave & Domain")

    with st.expander("Deep-water conditions", expanded=True):
        c1, c2, c3 = st.columns(3)
        st.session_state.Hm0_deep = c1.number_input("Hm0_deep [m]", value=float(st.session_state.Hm0_deep), format="%.4f")
        st.session_state.Tp_deep  = c2.number_input("Tp_deep [s]",  value=float(st.session_state.Tp_deep),  format="%.4f")
        st.session_state.Tm_deep  = c3.number_input("Tm_deep [s]",  value=float(st.session_state.Tm_deep),  format="%.4f")
        st.session_state.WAVE_SEED = int(c1.number_input(
            "Random phase seed", value=int(st.session_state.WAVE_SEED), step=1, min_value=0,
            help="Integer that fixes the random phases of the 100 wave components. The same "
                 "seed with the same spectrum gives exactly the same wave train; another seed "
                 "gives another wave train with the same spectrum (same Hm0 and Tp).",
        ))

    with st.expander("Toe conditions"):
        c1, c2, c3, c4 = st.columns(4)
        st.session_state.Hm0_toe  = c1.number_input("Hm0_toe [m]",  value=float(st.session_state.Hm0_toe),  format="%.4f")
        st.session_state.Tp_toe   = c2.number_input("Tp_toe [s]",   value=float(st.session_state.Tp_toe),   format="%.4f")
        st.session_state.Tm_toe   = c3.number_input("Tm_toe [s]",   value=float(st.session_state.Tm_toe),   format="%.4f")
        st.session_state.Tm10_toe = c4.number_input("Tm10_toe [s]", value=float(st.session_state.Tm10_toe), format="%.4f")

    with st.expander("Domain sizing"):
        if st.session_state.BREAKWATER_MODE and st.session_state.n_post < 1.5:
            st.session_state.n_post = 2.0
            st.info("Breakwater mode: n_post set to 2.0 for outlet relaxation zone.")
        st.session_state.domain_mode = st.radio(
            "Sizing mode", ["Auto (n × L₀)", "Manual (m)"],
            index=["Auto (n × L₀)", "Manual (m)"].index(st.session_state.domain_mode),
            horizontal=True, key="domain_mode_radio",
        )
        c1, c2, c3 = st.columns(3)
        if st.session_state.domain_mode == "Auto (n × L₀)":
            st.session_state.n_pre  = c1.number_input(
                "n_pre",  value=float(st.session_state.n_pre),  format="%.2f",
                help="Inlet length = n_pre × L₀",
            )
            st.session_state.n_post = c2.number_input(
                "n_post", value=float(st.session_state.n_post), format="%.4f",
                help="Outlet length = n_post × L₀ (≥1.5 for breakwater mode)",
            )
        else:
            st.session_state.inlet_length_m = c1.number_input(
                "Inlet length [m]",  value=float(st.session_state.inlet_length_m),  format="%.2f",
                help="Distance from structure toe to inlet boundary",
            )
            st.session_state.outlet_length_m = c2.number_input(
                "Outlet length [m]", value=float(st.session_state.outlet_length_m), format="%.2f",
                help="Distance from structure crest to outlet boundary",
            )
        st.session_state.top_clearance = c3.number_input(
            "top_clearance [m]", value=float(st.session_state.top_clearance), format="%.3f",
        )

    st.divider()
    if not st.session_state.step_done["stl"]:
        st.warning("Generate STL first (Geometry tab).")
    else:
        if st.button("► Size Domain", type="primary"):
            _run_size_domain()


def _run_size_domain():
    s = st.session_state
    try:
        out_name = s.stl_path
        if out_name is None or not Path(out_name).exists():
            st.error("STL file not found. Run 'Generate STL' first.")
            return

        with st.spinner("Computing domain…"):
            if s.domain_mode == "Manual (m)":
                stl_mesh, bb = load_stl_bbox(str(out_name))
                L0 = deepwater_L0(float(s.Tp_deep))
                dom = dict(
                    Dxmin=bb["xmin"] - float(s.inlet_length_m),
                    Dxmax=bb["xmax"] + float(s.outlet_length_m),
                    Dymin=bb["ymin"], Dymax=bb["ymax"],
                    Dzmin=bb["zmin"],
                    Dzmax=bb["zmax"] + float(s.top_clearance),
                )
                stl_bbox = bb
            else:
                dom, stl_mesh, stl_bbox, L0 = compute_domain_from_stl_and_Tp(
                    stl_file=str(out_name),
                    Tp=float(s.Tp_deep),
                    g=9.81,
                    n_pre=float(s.n_pre),
                    n_post=float(s.n_post),
                    top_clearance=float(s.top_clearance),
                )

        s.domain    = dom
        s.stl_mesh  = stl_mesh
        s.stl_bbox  = stl_bbox
        s.L0        = L0
        s.step_done["domain"] = True
        s.step_done["case"]   = False
        s.step_done["run"]    = False

        st.success(
            f"L0 = {L0:.3f} m | "
            f"x: [{dom['Dxmin']:.3f}, {dom['Dxmax']:.3f}] | "
            f"z: [{dom['Dzmin']:.3f}, {dom['Dzmax']:.3f}]"
        )

        fig, _ = plot_domain_and_stl(dom, stl_mesh, show=False)
        _show_fig(fig, "02_domain")

    except Exception as exc:
        st.error(f"Domain sizing failed: {exc}")
        import traceback
        st.code(traceback.format_exc())


# Tab 3: Mesh
def _tab_mesh():
    st.header("Mesh")

    with st.expander("Background mesh", expanded=True):
        c1, c2, c3 = st.columns(3)
        st.session_state.dx_target = c1.number_input("dx_target [m]", value=float(st.session_state.dx_target), format="%.4f")
        st.session_state.dz_target = c2.number_input("dz_target [m]", value=float(st.session_state.dz_target), format="%.4f")
        st.session_state.Ny        = c3.number_input("Ny (spanwise cells)", value=int(st.session_state.Ny), step=1, min_value=1)

    with st.expander("Refinement levels"):
        c1, c2 = st.columns(2)
        st.session_state.LEVEL_INTERFACE = c1.number_input("LEVEL_INTERFACE", value=int(st.session_state.LEVEL_INTERFACE), step=1, min_value=0)
        st.session_state.LEVEL_RUNUP     = c2.number_input("LEVEL_RUNUP",     value=int(st.session_state.LEVEL_RUNUP),     step=1, min_value=0)
        st.session_state.Vertical_distance_plus_for_refinement_interface = st.number_input(
            "Vertical_distance_plus_for_refinement_interface [m]",
            value=float(st.session_state.Vertical_distance_plus_for_refinement_interface),
            format="%.4f",
        )

    with st.expander("Runup band"):
        c1, c2, c3 = st.columns(3)
        st.session_state.RUNUP_X_START_OFFSET = c1.number_input("RUNUP_X_START_OFFSET [m]", value=float(st.session_state.RUNUP_X_START_OFFSET), format="%.4f")
        st.session_state.RUNUP_X_END          = c2.number_input("RUNUP_X_END [m]",          value=float(st.session_state.RUNUP_X_END),          format="%.4f")
        st.session_state.RUNUP_TOP_ABOVE_STL  = c3.number_input("RUNUP_TOP_ABOVE_STL [m]",  value=float(st.session_state.RUNUP_TOP_ABOVE_STL),  format="%.4f")

    st.divider()
    if not st.session_state.step_done["domain"]:
        st.warning("Size domain first (Wave & Domain tab).")
    else:
        if st.button("► Preview Refinement", type="primary"):
            _run_preview_refinement()


def _run_preview_refinement():
    s = st.session_state
    try:
        x_ht = float(s.stl_bbox["xmin"]) + float(s.RUNUP_X_START_OFFSET)

        # Domain + STL + refinement bands
        fig, _ = plot_domain_stl_with_refinement(
            s.domain, s.stl_mesh, s.stl_bbox,
            SWL=float(s.SWL),
            Hs_plus=float(s.Hm0_toe),
            Hs_extra=float(s.Vertical_distance_plus_for_refinement_interface),
            x_ht=x_ht,
            runup_top_above_stl=float(s.RUNUP_TOP_ABOVE_STL),
            runup_bottom_below_stl=0.0,
            runup_x_end=float(s.RUNUP_X_END),
        )
        _show_fig(fig, "03_refinement")

        # Lines in Domain (probe line at x_container_start or x_back_start)
        if s.meta is not None:
            use_collector = bool(s.params.get("use_collector", False))
            x_line = s.meta.get("x_container_start") if use_collector else s.meta.get("x_back_start")
            if x_line is not None:
                fig2, _ = plot_domain_stl_with_lines(
                    s.domain, s.stl_mesh,
                    x_lines=[float(x_line)],
                    title="Lines in Domain (overtopping / sampling plane)",
                )
                _show_fig(fig2, "04_lines_in_domain")

    except Exception as exc:
        st.error(f"Refinement preview failed: {exc}")
        import traceback
        st.code(traceback.format_exc())


# Impact-patch helpers

def _remove_obstacle_bottom_segments(segments, obstacle_abs_verts, atol=2e-3):
    """
    Post-process extracted STL boundary segments after multi-loop extraction:
    - Remove segments that correspond to the bottom face of an obstacle
      (the face touching the structure surface — not a real wall to measure force on).
    - Split structure surface segments that are partially covered by the obstacle footprint,
      keeping only the uncovered portions.

    Works for obstacles sitting on a flat horizontal surface.  For each obstacle the
    "bottom" is defined as all edges whose both endpoints are at z == min(obstacle z).

    Parameters
    ----------
    segments : list[(p0,p1)]  — raw segments from extract_xz_profile_segments_from_stl
    obstacle_abs_verts : list[list[(x,z)]]  — absolute 2-D vertices per obstacle
    """
    if not obstacle_abs_verts:
        return segments
    import numpy as np

    result = list(segments)
    for obs_verts in obstacle_abs_verts:
        if len(obs_verts) < 3:
            continue
        obs_pts = np.array(obs_verts, dtype=float)
        z_bot   = float(np.min(obs_pts[:, 1]))
        x_obs_min = float(np.min(obs_pts[:, 0]))
        x_obs_max = float(np.max(obs_pts[:, 0]))

        new_result = []
        for (p0, p1) in result:
            x0, z0 = p0
            x1, z1 = p1
            # Check near-horizontal segments at the bottom of the obstacle.
            if abs(z0 - z_bot) < atol and abs(z1 - z_bot) < atol:
                seg_xmin = min(x0, x1)
                seg_xmax = max(x0, x1)
                overlap_min = max(seg_xmin, x_obs_min)
                overlap_max = min(seg_xmax, x_obs_max)
                if overlap_max > overlap_min + atol:
                    # Clip the segment at the obstacle footprint.
                    if seg_xmin < x_obs_min - atol:
                        new_result.append(((seg_xmin, z_bot), (x_obs_min, z_bot)))
                    if seg_xmax > x_obs_max + atol:
                        new_result.append(((x_obs_max, z_bot), (seg_xmax, z_bot)))
                    continue  # drop the covered portion (incl. obstacle bottom edge)
            new_result.append((p0, p1))
        result = new_result
    return result


def _get_porous_segment_ids(segments, porous_data):
    """Return a set of 1-indexed segment IDs whose midpoint falls inside a porous zone polygon."""
    if not porous_data or not segments:
        return set()
    try:
        from matplotlib.path import Path as MplPath
        import numpy as np

        all_polys = []
        # layer_polys is nested: {layer_name: {zone_name: ndarray}}
        for layer_val in porous_data.get("layer_polys", {}).values():
            if isinstance(layer_val, dict):
                for zone_poly in layer_val.values():
                    all_polys.append(np.array(zone_poly, dtype=float))
            else:
                all_polys.append(np.array(layer_val, dtype=float))
        # porous_polys is flat: {zone_name: ndarray}
        for poly in porous_data.get("porous_polys", {}).values():
            all_polys.append(np.array(poly, dtype=float))

        if not all_polys:
            return set()

        porous_ids = set()
        for i, (p0, p1) in enumerate(segments, start=1):
            xm = (p0[0] + p1[0]) / 2.0
            zm = (p0[1] + p1[1]) / 2.0
            for poly in all_polys:
                if len(poly) >= 3 and MplPath(poly).contains_point((xm, zm)):
                    porous_ids.add(i)
                    break
        return porous_ids
    except Exception:
        return set()


def _plot_segments_plotly(stl_bbox, segments, x_max_keep=None, porous_seg_ids=None,
                           obstacle_abs_verts=None, obstacles=None, title=""):
    """Interactive Plotly figure for STL segment selection (zoom/pan built-in)."""
    import plotly.graph_objects as go

    if x_max_keep is not None:
        x_max_keep = float(x_max_keep)
        segments = [(p0, p1) for (p0, p1) in segments if max(p0[0], p1[0]) <= x_max_keep + 1e-9]

    porous_seg_ids = set(porous_seg_ids or [])

    # Colour palette (tab20-like)
    _palette = [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
        "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
        "#aec7e8", "#ffbb78", "#98df8a", "#ff9896", "#c5b0d5",
        "#c49c94", "#f7b6d2", "#c7c7c7", "#dbdb8d", "#9edae5",
    ]

    fig = go.Figure()

    for i, (p0, p1) in enumerate(segments, start=1):
        x0, z0 = p0
        x1, z1 = p1
        is_porous = i in porous_seg_ids
        color = "rgba(180,180,180,0.6)" if is_porous else _palette[(i - 1) % len(_palette)]
        dash  = "dash" if is_porous else "solid"
        label = f"Seg {i}" + (" [porous]" if is_porous else "")
        # midpoint for label
        xm, zm = (x0 + x1) / 2, (z0 + z1) / 2
        fig.add_trace(go.Scatter(
            x=[x0, x1], y=[z0, z1],
            mode="lines",
            line=dict(width=5, color=color, dash=dash),
            name=label,
            hovertemplate=f"<b>{label}</b><br>({x0:.3f},{z0:.3f}) → ({x1:.3f},{z1:.3f})<extra></extra>",
        ))
        # number label at midpoint
        fig.add_trace(go.Scatter(
            x=[xm], y=[zm],
            mode="text",
            text=[str(i)],
            textfont=dict(size=11, color=color),
            showlegend=False,
            hoverinfo="skip",
        ))

    # Dashed obstacle outlines for reference
    if obstacle_abs_verts:
        for j, verts in enumerate(obstacle_abs_verts):
            if len(verts) < 3:
                continue
            xs = [v[0] for v in verts] + [verts[0][0]]
            zs = [v[1] for v in verts] + [verts[0][1]]
            obs_name = (obstacles[j].get("name", f"obstacle_{j}") if obstacles and j < len(obstacles) else f"obstacle_{j}")
            fig.add_trace(go.Scatter(
                x=xs, y=zs,
                mode="lines",
                line=dict(color="rgba(120,120,120,0.5)", width=1, dash="dot"),
                name=f"{obs_name} (outline)",
                hovertemplate=f"<b>{obs_name}</b><extra></extra>",
            ))

    xmin = stl_bbox["xmin"] - 0.5
    xmax = (x_max_keep + 0.5) if x_max_keep is not None else (stl_bbox["xmax"] + 0.5)
    zmin = stl_bbox["zmin"] - 0.5
    zmax = stl_bbox["zmax"] + 0.5
    if obstacle_abs_verts:
        _oz = [z for verts in obstacle_abs_verts for (_x, z) in verts]
        _ox = [x for verts in obstacle_abs_verts for (x, _z) in verts]
        if _oz:
            zmin = min(zmin, min(_oz) - 0.5)
            zmax = max(zmax, max(_oz) + 0.5)
        if _ox:
            xmax = max(xmax, max(_ox) + 0.5)

    fig.update_layout(
        title=title,
        xaxis_title="x [m]",
        yaxis_title="z [m]",
        xaxis=dict(range=[xmin, xmax]),
        yaxis=dict(range=[zmin, zmax], scaleanchor="x", scaleratio=1),
        height=480,
        legend=dict(orientation="v", x=1.02, y=0.5, font=dict(size=11)),
        margin=dict(l=60, r=180, t=50, b=50),
        hovermode="closest",
    )
    return fig


# Tab 4: Simulation & Job
def _tab_sim():
    st.header("Simulation & Job")

    with st.expander("Simulation control", expanded=True):
        c1, c2, c3 = st.columns(3)
        st.session_state.END_TIME       = c1.number_input("END_TIME [s]",       value=float(st.session_state.END_TIME),       format="%.2f")
        st.session_state.DELTA_T        = c2.number_input("DELTA_T [s]",        value=float(st.session_state.DELTA_T),        format="%.5f")
        st.session_state.WRITE_INTERVAL = c3.number_input("WRITE_INTERVAL [s]", value=float(st.session_state.WRITE_INTERVAL), format="%.4f")
        c4, c5 = st.columns(2)
        st.session_state.MAX_CO      = c4.number_input("MAX_CO",      value=float(st.session_state.MAX_CO),      format="%.3f")
        st.session_state.MAX_ALPHA_CO = c5.number_input("MAX_ALPHA_CO", value=float(st.session_state.MAX_ALPHA_CO), format="%.3f")

    with st.expander("Overtopping flow sampling (flow depth, velocity)"):
        st.caption(
            "Vertical sampling line x01 at the overtopping section: flow depth and velocity "
            "of each overtopping event are computed from it (post-processing, Step 4)."
        )
        st.session_state.LINE_WRITE_INTERVAL = st.number_input(
            "Line write interval [s]", value=float(st.session_state.LINE_WRITE_INTERVAL),
            min_value=0.001, format="%.4f",
            help="Output interval of the sampling line. Short overtopping events need a fine "
                 "interval (0.01-0.02 s); disk use grows as 1/interval.",
        )

    if st.session_state.flow_model.lower() == "rans":
        with st.expander("RANS roughness (0.orig/nut)"):
            c1, c2 = st.columns(2)
            st.session_state.Ks = c1.number_input("Ks", value=float(st.session_state.Ks), format="%.5f")
            st.session_state.Cs = c2.number_input("Cs", value=float(st.session_state.Cs), format="%.4f")

    with st.expander("HPC / Decomposition"):
        c1, c2 = st.columns(2)
        st.session_state.N_PROCS      = c1.number_input("N_PROCS", value=int(st.session_state.N_PROCS), step=1, min_value=1)
        st.session_state.DECOMP_METHOD = c2.selectbox(
            "DECOMP_METHOD",
            ["scotch", "simple"],
            index=0 if st.session_state.DECOMP_METHOD == "scotch" else 1,
        )

    with st.expander("Impact patch (optional)"):
        st.session_state.new_patch = st.checkbox(
            "Create new force/impact patch", value=bool(st.session_state.new_patch)
        )
        if st.session_state.new_patch:
            if not st.session_state.step_done["stl"]:
                st.info("Generate STL first to see the segment layout.")
            else:
                # Number STL segments for patch selection.
                try:
                    s = st.session_state
                    x_choose = (
                        s.meta.get("x_container_start", s.stl_bbox["xmax"])
                        if s.meta else s.stl_bbox["xmax"]
                    )
                    segs = extract_xz_profile_segments_from_stl(
                        str(s.stl_path), y_plane=0.0, x_max_keep=x_choose
                    )
                    # Remove obstacle bottom faces and split structure surface at obstacle footprint
                    obs_verts = getattr(s, "obstacle_abs_verts", None)
                    if obs_verts:
                        segs = _remove_obstacle_bottom_segments(segs, obs_verts)

                    # detect porous segments
                    porous_seg_ids = _get_porous_segment_ids(segs, s.porous_data if s.USE_POROUS else None)

                    # Interactive patch-selection plot
                    _fig_ip = _plot_segments_plotly(
                        s.stl_bbox, segs,
                        x_max_keep=x_choose,
                        porous_seg_ids=porous_seg_ids,
                        obstacle_abs_verts=getattr(s, "obstacle_abs_verts", None),
                        obstacles=getattr(s, "obstacles", None),
                        title="STL profile — select segments for force measurement",
                    )
                    st.plotly_chart(_fig_ip, width="stretch")
                    st.caption(
                        f"{len(segs)} segments available (1–{len(segs)}).  "
                        "Use toolbar or scroll to zoom; drag to pan."
                    )

                    if porous_seg_ids:
                        st.info(
                            f"Segments {sorted(porous_seg_ids)} overlap with porous zones "
                            "(shown dashed/grey). Selecting them for force measurement is "
                            "not recommended — the resistance model already accounts for forces there."
                        )

                    # Exclude porous segments from the selection.
                    seg_options = [sid for sid in range(1, len(segs) + 1) if sid not in porous_seg_ids]
                    st.session_state.SEG_IDS = st.multiselect(
                        "Select segment(s) for force measurement",
                        options=seg_options,
                        default=[sid for sid in st.session_state.SEG_IDS if sid in seg_options],
                        format_func=lambda x: f"Segment {x}",
                    )
                except Exception as exc_seg:
                    st.warning(f"Could not extract segments: {exc_seg}")

            st.session_state.SOURCE_PATCH = st.text_input("SOURCE_PATCH", value=st.session_state.SOURCE_PATCH)


# Job monitoring helpers
def _resolve_tmux_bin() -> str:
    """Same resolution order as tmux_run_case.py."""
    import shutil, os
    tmux_env = os.environ.get("TMUX_BIN", "").strip()
    if tmux_env:
        return tmux_env
    tmux_repo = REPO_ROOT / "tools" / "tmux" / "bin" / "tmux"
    if tmux_repo.exists():
        return str(tmux_repo)
    t = shutil.which("tmux")
    if t:
        return t
    return "tmux"


def _is_terminal(status: str) -> bool:
    """Return True if the case status is permanently done (no need to re-fetch)."""
    return status == "KILLED" or status.startswith("FINISHED_")


def _scan_proc_for_cases(case_dirs: list) -> dict:
    """
    Single /proc scan across all given case directories.
    Returns {case_dir_str: (mem_kb, n_procs)} — avoids per-case O(N_procs) scans.
    """
    import os
    result = {str(cd): [0, 0] for cd in case_dirs}
    case_strs = list(result.keys())
    if not case_strs:
        return {}
    try:
        uid = os.getuid()
        for pid_s in os.listdir("/proc"):
            if not pid_s.isdigit():
                continue
            pid_path = Path("/proc") / pid_s
            try:
                if pid_path.stat().st_uid != uid:
                    continue
                cwd = os.readlink(pid_path / "cwd")
                matched = next(
                    (cs for cs in case_strs if cwd == cs or cwd.startswith(cs + "/")),
                    None,
                )
                if matched is None:
                    continue
                comm = (pid_path / "comm").read_text().strip()
                stat_parts = (pid_path / "stat").read_text().split()
                rss_pages = int(stat_parts[23])
                result[matched][0] += rss_pages * 4
                if comm in ("waveFoam", "porousWaveFoam", "snappyHexMesh", "mpirun"):
                    result[matched][1] += 1
            except Exception:
                continue
    except Exception:
        pass
    return {k: tuple(v) for k, v in result.items()}


def _get_job_info(case_dir: Path, proc_data: tuple = None) -> dict:
    """
    Collect monitoring info for one case directory (mirrors bin/joblist logic).
    Returns a dict with keys: status, alive, runtime_str, phase, cpu_pct, ram_mb, n_procs,
    current_time, end_time, solver_label.
    """
    import subprocess, time, os, re

    info = dict(
        status="UNKNOWN", alive=False, runtime_str="--:--",
        phase="?", cpu_pct="--", ram_mb="--",
        n_procs=0, current_time=None, end_time=None, solver_label=None,
    )

    # Status
    status_file = case_dir / "TMUX_STATUS.txt"
    if status_file.exists():
        info["status"] = status_file.read_text().strip()

    # Socket path used to calculate runtime
    sock_file = case_dir / "TMUX_SOCKET.txt"
    socket_path = sock_file.read_text().strip() if sock_file.exists() else ""

    # Skip the tmux check for finished or killed cases.
    if _is_terminal(info["status"]):
        info["alive"] = False
    else:
        tmux = _resolve_tmux_bin()
        if socket_path and Path(socket_path).exists():
            try:
                r = subprocess.run([tmux, "-S", socket_path, "ls"], capture_output=True, timeout=3)
                info["alive"] = (r.returncode == 0)
            except Exception:
                info["alive"] = False
        else:
            info["alive"] = False

        if not info["alive"] and info["status"] == "RUNNING":
            info["status"] = "DEAD?"

    # Measure runtime from the socket modification time; freeze it when stopped.
    now_ts = time.time()
    start_ts = None
    if socket_path:
        try:
            start_ts = Path(socket_path).stat().st_mtime if Path(socket_path).exists() else None
        except Exception:
            pass
    if start_ts is None:
        # fallback: log file mtime
        for lf in ["log.waveFoam", "log.porousWaveFoam", "log.snappyHexMesh"]:
            p = case_dir / lf
            if p.exists():
                try:
                    start_ts = p.stat().st_mtime
                except Exception:
                    pass
                break

    if start_ts is not None:
        if info["status"] == "RUNNING":
            end_ts = now_ts
        else:
            stop_ts = status_file.stat().st_mtime if status_file.exists() else now_ts
            end_ts = stop_ts
        elapsed = max(0, int(end_ts - start_ts))
        h, rem = divmod(elapsed, 3600)
        m, s = divmod(rem, 60)
        info["runtime_str"] = f"{h:02d}:{m:02d}:{s:02d}"

    # Solver log: current time + end time
    solver_log = None
    if (case_dir / "log.porousWaveFoam").exists():
        solver_log = case_dir / "log.porousWaveFoam"
        info["solver_label"] = "porousWaveFoam"
    elif (case_dir / "log.waveFoam").exists():
        solver_log = case_dir / "log.waveFoam"
        info["solver_label"] = "waveFoam"

    cd = case_dir / "system" / "controlDict"
    if cd.exists():
        for line in cd.read_text().splitlines():
            m_et = re.match(r"^\s*endTime\s+([\d.eE+\-]+)\s*;", line)
            if m_et:
                try:
                    info["end_time"] = float(m_et.group(1))
                except ValueError:
                    pass
                break

    if solver_log is not None:
        try:
            # Read last 5 KB to find latest "Time ="
            with open(solver_log, "rb") as f:
                f.seek(0, 2)
                size = f.tell()
                f.seek(max(0, size - 5120))
                tail = f.read().decode("utf-8", errors="ignore")
            for line in reversed(tail.splitlines()):
                m_t = re.match(r"\s*Time\s*=\s*([\d.eE+\-]+)", line)   # not "ExecutionTime = ..."
                if m_t:
                    info["current_time"] = float(m_t.group(1))
                    break
        except Exception:
            pass

    if info["solver_label"] and info["current_time"] is not None:
        cur = info["current_time"]
        end = info["end_time"]
        lbl = info["solver_label"]
        if end:
            pct = min(100.0, 100.0 * cur / end)
            info["phase"] = f"{lbl}: t={cur:.2f}/{end:.0f} ({pct:.0f}%)"
        else:
            info["phase"] = f"{lbl}: t={cur:.2f}"
    elif (case_dir / "log.snappyHexMesh").exists() or (case_dir / "log.snappy").exists():
        info["phase"] = "snappyHexMesh"

    # Use the existing process scan for CPU and memory usage.
    if proc_data is not None:
        mem_kb, n_procs = proc_data
    else:
        # Scan /proc if no process data was supplied.
        mem_kb, n_procs = 0, 0
        try:
            uid = os.getuid()
            for pid_s in os.listdir("/proc"):
                if not pid_s.isdigit():
                    continue
                pid_path = Path("/proc") / pid_s
                try:
                    if pid_path.stat().st_uid != uid:
                        continue
                    cwd = os.readlink(pid_path / "cwd")
                    if not (cwd == str(case_dir) or cwd.startswith(str(case_dir) + "/")):
                        continue
                    comm = (pid_path / "comm").read_text().strip()
                    stat_parts = (pid_path / "stat").read_text().split()
                    rss_pages = int(stat_parts[23])
                    mem_kb += rss_pages * 4
                    if comm in ("waveFoam", "porousWaveFoam", "snappyHexMesh", "mpirun"):
                        n_procs += 1
                except Exception:
                    continue
        except Exception:
            pass

    if mem_kb > 0:
        info["ram_mb"] = f"{mem_kb / 1024:.0f}"
    if n_procs > 0:
        info["n_procs"] = n_procs

    return info


_FILE_SERVER_PORT: int | None = None

def _ensure_file_server(port: int = 8502) -> int:
    """
    Start a simple read-only HTTP server in a background thread serving REPO_ROOT.
    If the port is already bound (previous Streamlit run still holds it), just
    reuse that port — the server is still serving our files.
    """
    global _FILE_SERVER_PORT
    if _FILE_SERVER_PORT is not None:
        return _FILE_SERVER_PORT

    import threading, http.server

    class _Handler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=str(REPO_ROOT), **kwargs)
        def log_message(self, *args):
            pass

    try:
        server = http.server.HTTPServer(("0.0.0.0", port), _Handler)
        server.allow_reuse_address = True
        t = threading.Thread(target=server.serve_forever, daemon=True)
        t.start()
    except OSError:
        # A server is already using this port.
        pass

    _FILE_SERVER_PORT = port
    return port


def _kill_case(case_dir: Path):
    """Kill the tmux session for case_dir and mark TMUX_STATUS.txt = KILLED."""
    import subprocess
    sock_file = case_dir / "TMUX_SOCKET.txt"
    socket_path = sock_file.read_text().strip() if sock_file.exists() else ""
    tmux = _resolve_tmux_bin()
    if socket_path and Path(socket_path).exists():
        try:
            subprocess.run([tmux, "-S", socket_path, "kill-server"], timeout=5)
        except Exception:
            pass
    (case_dir / "TMUX_STATUS.txt").write_text("KILLED\n")


# Tab 5: Run
def _tab_run():
    st.header("Run")

    # Build Case
    st.subheader("1. Build Case")
    ready_build = st.session_state.step_done["stl"] and st.session_state.step_done["domain"]
    if not ready_build:
        st.warning("Complete Geometry (STL) and Wave & Domain steps first.")
    if st.button("► Build Case", type="primary", disabled=not ready_build):
        _run_build_case()

    # Keep build output visible after rerenders.
    if st.session_state.build_log:
        with st.expander("Build log", expanded=True):
            st.code("\n".join(st.session_state.build_log))
    if st.session_state.build_result:
        kind, msg = st.session_state.build_result
        if kind == "success":
            st.success(msg)
        else:
            st.error(msg)

    st.divider()

    # Launch Simulation
    st.subheader("2. Launch Simulation")
    if not st.session_state.step_done["case"]:
        st.warning("Build the case first.")
    if st.button("► Launch Simulation", type="primary", disabled=not st.session_state.step_done["case"]):
        _run_launch()

    # Keep launch output visible after rerenders.
    if st.session_state.launch_result:
        kind, msg = st.session_state.launch_result
        if kind == "success":
            st.success(msg)
        else:
            st.error(msg)



# Tab 6: Monitor
def _tab_monitor():
    import pandas as pd

    st.header("Job Monitor")

    # Controls row
    col_ref, col_root, _ = st.columns([1, 3, 2])
    with col_ref:
        do_refresh = st.button("↻ Refresh", key="refresh_monitor")
    with col_root:
        scan_root = st.text_input(
            "Scan root", value=str(REPO_ROOT),
            help="Directory to search recursively for TMUX_SOCKET.txt files",
            label_visibility="collapsed",
        )

    # Scan case directories on Refresh.
    # Cache finished and killed cases.
    # Fetch active cases in parallel.
    if do_refresh or "monitor_jobs" not in st.session_state:
        # Search direct subdirectories first, then search recursively if empty.
        sock_files = sorted(Path(scan_root).glob("*/TMUX_SOCKET.txt"))
        if not sock_files:
            sock_files = sorted(Path(scan_root).glob("**/TMUX_SOCKET.txt"))
        all_case_dirs = [sf.parent.resolve() for sf in sock_files]

        # Clear the cache on Refresh or when scan_root changes.
        if do_refresh or st.session_state.get("monitor_scan_root") != scan_root:
            st.session_state.monitor_cache = {}

        cache = st.session_state.get("monitor_cache", {})

        # Use cached results for stopped cases and fetch the others.
        cached_jobs, active_dirs = {}, []
        for cd in all_case_dirs:
            if str(cd) in cache:
                cached_jobs[cd] = cache[str(cd)]
            else:
                active_dirs.append(cd)

        # Scan /proc once for all active cases.
        proc_map = _scan_proc_for_cases(active_dirs)

        # Fetch case details in parallel.
        from concurrent.futures import ThreadPoolExecutor, as_completed

        def _fetch(cd):
            return cd, _get_job_info(cd, proc_data=proc_map.get(str(cd), (0, 0)))

        fresh_jobs = {}
        with ThreadPoolExecutor(max_workers=min(8, max(1, len(active_dirs)))) as exe:
            for cd, ji in exe.map(lambda cd: _fetch(cd), active_dirs):
                fresh_jobs[cd] = ji

        # Cache cases that have finished or been killed.
        new_cache = dict(cache)
        for cd, ji in fresh_jobs.items():
            if _is_terminal(ji["status"]):
                new_cache[str(cd)] = ji
        st.session_state.monitor_cache    = new_cache
        st.session_state.monitor_scan_root = scan_root

        # Keep the original case order.
        all_jobs = {}
        for cd in all_case_dirs:
            all_jobs[cd] = cached_jobs.get(cd) or fresh_jobs.get(cd, {})
        st.session_state.monitor_jobs = all_jobs

    jobs = st.session_state.get("monitor_jobs", {})

    if not jobs:
        st.info(f"No cases found under `{scan_root}`. Click ↻ Refresh to scan.")
        return

    STATUS_COLOR = {
        "RUNNING":    "🟢",
        "FINISHED_0": "🔵",
        "KILLED":     "🟠",
        "DEAD?":      "🔴",
        "UNKNOWN":    "⚪",
    }

    rows = []
    for cd in jobs.keys():
        ji = jobs[cd]
        rows.append({
            "Icon":     STATUS_COLOR.get(ji["status"], "⚪"),
            "Case":     cd.name,
            "Status":   ji["status"],
            "Alive":    "Yes" if ji["alive"] else "No",
            "Runtime":  ji["runtime_str"],
            "Procs":    ji["n_procs"],
            "RAM (MB)": ji["ram_mb"],
            "Phase":    ji["phase"],
        })

    df = pd.DataFrame(rows)
    st.dataframe(df, width="stretch", hide_index=True)

    # Case selector + detail + kill
    st.divider()
    NONE_LABEL = "— none —"
    case_names = [NONE_LABEL] + [cd.name for cd in jobs.keys()]

    # Select the current case if it is listed.
    default_idx = 0
    if st.session_state.case_dir:
        cur_name = Path(st.session_state.case_dir).name
        if cur_name in case_names:
            default_idx = case_names.index(cur_name)

    selected_name = st.selectbox(
        "Select case for details / kill",
        case_names,
        index=default_idx,
    )

    if selected_name == NONE_LABEL:
        st.info("Select a case above to see details and the solver log.")
        return

    selected_dir = next(cd for cd in jobs.keys() if cd.name == selected_name)
    job          = jobs[selected_dir]

    # Status + metrics
    status = job["status"]
    badge  = STATUS_COLOR.get(status, "⚪")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Status",   f"{badge} {status}")
    c2.metric("Runtime",  job["runtime_str"])
    c3.metric("Procs",    str(job["n_procs"]))
    c4.metric("RAM (MB)", str(job["ram_mb"]))

    # Progress bar
    if job["current_time"] is not None and job["end_time"]:
        pct = min(1.0, job["current_time"] / job["end_time"])
        st.progress(pct, text=job["phase"])
    else:
        st.caption(f"Phase: {job['phase']}")

    # Open the report in a new browser tab.
    # Serve report files through the background HTTP server on port 8502.
    report_path = selected_dir / "run_report.html"
    if report_path.exists():
        # Compute URL relative to REPO_ROOT served by _file_server()
        try:
            rel = report_path.relative_to(REPO_ROOT)
        except ValueError:
            rel = report_path
        port = _ensure_file_server()
        host = st.context.headers.get("host", "localhost").split(":")[0]
        url  = f"http://{host}:{port}/{rel.as_posix()}"
        from theme import _get_palette as _gp
        _tp = _gp()
        st.markdown(
            f'<a href="{url}" target="_blank">'
            f'<button style="background:linear-gradient(135deg,{_tp["accent"]},{_tp["accent_hover"]});'
            f'color:{_tp["bg_primary"]};padding:8px 18px;'
            f'border:none;border-radius:6px;cursor:pointer;font-size:14px;font-weight:600;'
            f'font-family:IBM Plex Sans,sans-serif;transition:all 0.2s;">'
            f'📄 Open Report in New Tab</button></a>',
            unsafe_allow_html=True,
        )
    else:
        st.caption("No run_report.html found for this case.")

    # Open post-processing in a new tab.
    from theme import _get_palette as _gp
    _tp = _gp()
    _pp_url = f"/postprocessing?case_dir={selected_dir}"
    st.markdown(
        f'<a href="{_pp_url}" target="_blank">'
        f'<button style="background:linear-gradient(135deg,{_tp["accent2"]},{_tp["accent"]});'
        f'color:{_tp["bg_primary"]};padding:8px 18px;border:none;border-radius:6px;'
        f'cursor:pointer;font-size:14px;font-weight:600;'
        f'font-family:IBM Plex Sans,sans-serif;margin-top:6px;">'
        f'📊 Open Post-processing</button></a>',
        unsafe_allow_html=True,
    )

    st.divider()

    # Kill button
    can_kill = job["alive"] or job["status"] == "RUNNING"
    if st.button(
        f"⛔ Kill  {selected_name}",
        type="secondary",
        disabled=not can_kill,
        key="kill_selected",
    ):
        _kill_case(selected_dir)
        st.warning(f"`{selected_name}` has been killed. Refresh to update the table.")

    # Show the last 30 solver log lines.
    st.divider()
    solver_log = None
    for lf in ["log.porousWaveFoam", "log.waveFoam"]:
        lp = selected_dir / lf
        if lp.exists():
            solver_log = lp
            break

    if solver_log:
        st.caption(f"`tail -n 30 {solver_log.name}`")
        try:
            with open(solver_log, "rb") as f:
                f.seek(0, 2)
                size = f.tell()
                f.seek(max(0, size - 30 * 120))   # ~120 chars/line safety margin
                tail_txt = f.read().decode("utf-8", errors="ignore")
            lines = tail_txt.splitlines()[-30:]
            st.code("\n".join(lines), language="text")
        except Exception as exc:
            st.warning(f"Could not read log: {exc}")
    else:
        st.info("No solver log found yet for this case.")


def _run_build_case():
    import json as _json
    s = st.session_state
    s.build_log = []
    s.build_result = None

    def log(msg):
        s.build_log.append(msg)

    try:
        params     = s.params
        dom        = s.domain
        stl_bbox   = s.stl_bbox
        L0         = s.L0
        out_name   = s.stl_path
        porous_data  = s.porous_data
        porous_paths = s.porous_paths
        meta         = s.meta

        # Clone template
        log("Cloning template case…")
        template_dir = select_template_case(s.flow_model)
        case_dir = clone_case_from_template(template_dir, s.case_name, overwrite=True)
        s.case_dir = case_dir
        log(f"  -> {case_dir.resolve()}")

        # RANS roughness
        if s.flow_model.lower() == "rans":
            log("Updating nut roughness (0.orig/nut)…")
            update_nut_roughness_0orig(case_dir=case_dir, Ks=float(s.Ks), Cs=float(s.Cs), patch="coast")

        # Porous zone OpenFOAM config (multi-layer)
        if s.USE_POROUS and porous_data is not None:
            log("Writing porosityZones…")
            active_zones = porous_data["active_zones"]
            core_phys = s.core_physics if (not s.core_is_solid and s.core_physics) else None
            zones_physics = build_layered_zones_physics(
                s.porous_layers, active_zones,
                core_physics=core_phys,
            )
            write_porosity_zones(case_dir, zones_physics)
            add_porous_div_scheme(case_dir)
            log(f"  porosityZones written for: {list(zones_physics.keys())}")

        # Select main STL
        if s.USE_POROUS and porous_paths is not None:
            main_stl = porous_paths["solid_stl"] or porous_paths.get("core_stl") or out_name
        else:
            main_stl = out_name

        # Copy STLs to triSurface
        log("Copying STL(s) to triSurface…")
        copy_stl_to_trisurface(case_dir, main_stl)
        if s.USE_POROUS and porous_paths is not None:
            for stl_path in porous_paths["layer_stls"].values():
                if stl_path is not None:
                    copy_stl_to_trisurface(case_dir, stl_path)
            # Copy core STL if porous core
            if porous_paths.get("core_stl") is not None:
                copy_stl_to_trisurface(case_dir, porous_paths["core_stl"])

        # blockMeshDict
        log("Updating blockMeshDict…")
        bm_info = update_blockMeshDict_vertices_and_cells(
            case_dir=case_dir, dom=dom,
            dx_target=float(s.dx_target),
            dz_target=float(s.dz_target),
            Ny=int(s.Ny),
        )
        log(f"  cells: {bm_info['Nx']} x {bm_info['Ny']} x {bm_info['Nz']}")

        # surfaceFeatureExtractDict
        log("Updating surfaceFeatureExtractDict…")
        update_surfaceFeatureExtractDict(case_dir, stl_name=Path(main_stl).name)

        # snappyHexMeshDict
        log("Writing snappyHexMeshDict…")
        x_ht = float(stl_bbox["xmin"]) + float(s.RUNUP_X_START_OFFSET)
        trisurface_stl = Path(case_dir) / "constant" / "triSurface" / Path(main_stl).name
        sn = write_snappyHexMeshDict_OF2206(
            case_dir=case_dir,
            dom=dom,
            stl_path=trisurface_stl,
            Hs_plus=float(s.Hm0_toe),
            Hs_extra=float(s.Vertical_distance_plus_for_refinement_interface),
            SWL=float(s.SWL),
            LEVEL_INTERFACE=int(s.LEVEL_INTERFACE),
            LEVEL_RUNUP=int(s.LEVEL_RUNUP),
            x_ht=x_ht,
            runup_x_end=float(s.RUNUP_X_END),
            runup_top_above_stl=float(s.RUNUP_TOP_ABOVE_STL),
            stl_zmax_ref=float(stl_bbox["zmax"]),
        )
        s.snappy_info = sn

        # waveProperties
        log("Updating waveProperties…")
        bw_mode = bool(s.BREAKWATER_MODE)
        update_waveProperties_input(
            case_dir,
            seaLevel=float(s.SWL),
            Hs=float(s.Hm0_deep),
            Tp=float(s.Tp_deep),
            h_deep=float(s.h_deep),
            dom=dom,
            L0=float(L0),
            n_relax_inlet=1.0,
            breakwater_mode=bw_mode,
            n_relax_outlet=1.5,
            seed=int(s.WAVE_SEED),
        )

        # Outlet BCs for breakwater mode
        if bw_mode:
            log("Patching outlet BCs for breakwater mode…")
            update_outlet_bc_for_breakwater(case_dir)

        # setFieldsDict
        log("Updating setFieldsDict…")
        use_collector = bool(params.get("use_collector", False))
        if bw_mode:
            # Breakwater: water on both sides, use back_start as reference
            x_fill_end = meta.get("x_back_start", meta.get("x_end"))
        elif use_collector:
            x_fill_end = meta.get("x_container_start")
            if x_fill_end is None:
                raise RuntimeError("use_collector=True but x_container_start is None in meta.")
        else:
            x_fill_end = meta.get("x_back_start")
            if x_fill_end is None:
                raise RuntimeError("use_collector=False but x_back_start is missing in meta.")

        update_setFieldsDict_water_box(
            case_dir, dom=dom,
            SWL=float(s.SWL),
            x_container_start=x_fill_end,
            x_end_margin=0.02,
            breakwater_mode=bw_mode,
        )

        # controlDict
        log("Updating controlDict…")
        # Use the full structure height, including porous layers.
        # The solid STL can be lower in porous cases.
        stl_zmax   = float(stl_bbox["zmax"])
        zmax_runup = sn["runupBand"][1]
        x_line     = x_fill_end
        update_controlDict_case(
            case_dir=case_dir,
            endTime=float(s.END_TIME),
            deltaT=float(s.DELTA_T),
            writeInterval=float(s.WRITE_INTERVAL),
            maxCo=float(s.MAX_CO),
            maxAlphaCo=float(s.MAX_ALPHA_CO),
            x_positions=[x_line],
            y_line=float(s.Y_LINE),
            zmin=float(stl_zmax),
            zmax=float(zmax_runup),
            nPoints=int(s.NPOINTS),
            line_write_interval=float(s.LINE_WRITE_INTERVAL),
        )

        # topoSetDict + createPatchDict (impact patch)
        log("Writing topoSetDict…")
        # Nudge inward: rounding Dxmin can place the point just outside the mesh
        outside_pt = (
            round(dom["Dxmin"] + 0.01 * (dom["Dxmax"] - dom["Dxmin"]), 4),
            round(0.5 * float(s.Ly), 4),
            round(float(s.SWL), 4),
        )

        porous_topo_entries = []
        if s.USE_POROUS and porous_data is not None and porous_paths is not None:
            layer_configs = porous_data.get("layer_configs", s.porous_layers)
            for (li, zone), stl_pz in porous_paths["layer_stls"].items():
                if stl_pz is None:
                    continue
                lname = layer_configs[li]["name"] if li < len(layer_configs) else f"layer{li}"
                cellzone = f"porosity_{lname}_{zone}"
                porous_topo_entries.append(dict(
                    stl_path      = f"constant/triSurface/{stl_pz.name}",
                    cellset_name  = f"porousCells_{lname}_{zone}",
                    cellzone_name = cellzone,
                    outside_point = outside_pt,
                ))
            # Porous core entry
            if porous_paths.get("core_stl") is not None:
                porous_topo_entries.append(dict(
                    stl_path      = f"constant/triSurface/{porous_paths['core_stl'].name}",
                    cellset_name  = "porousCells_core",
                    cellzone_name = "porosity_core",
                    outside_point = outside_pt,
                ))

        # resolve segment info for force patches (multi-segment)
        x_choose = meta.get("x_container_start", stl_bbox["xmax"]) if meta else stl_bbox["xmax"]
        force_entries = []
        patch_list = []       # list of {"patch_name", "faceSet_name"} for createPatchDict
        patch_names = []      # flat list of patch names for job script

        seg_ids = list(s.SEG_IDS) if s.new_patch else []
        if seg_ids:
            segs = extract_xz_profile_segments_from_stl(str(out_name), y_plane=0.0, x_max_keep=x_choose)
            for seg_id in seg_ids:
                if not (1 <= seg_id <= len(segs)):
                    raise ValueError(f"SEG_ID={seg_id} out of range [1,{len(segs)}]")
                p0, p1 = segs[seg_id - 1]
                bmin, bmax = segment_to_box(p0, p1, Ly=float(s.Ly), tol=0.02, x_max_keep=x_choose)
                set_name = f"force_seg_{seg_id:02d}"
                patch_name = f"force_seg_{seg_id:02d}"
                force_entries.append({"set_name": set_name, "box_min": bmin, "box_max": bmax})
                patch_list.append({"patch_name": patch_name, "faceSet_name": set_name})
                patch_names.append(patch_name)

        # Start the overtopping section at the impermeable surface at x_fill_end.
        # This includes flow through the porous layers.
        # The x01 depth and velocity line starts at the full structure height.
        overtop_zmin = float(stl_zmax)
        if s.USE_POROUS and porous_data is not None:
            z_imp = solid_top_at_x(porous_data["solid_poly"], float(x_fill_end))
            if z_imp is not None and z_imp < overtop_zmin:
                overtop_zmin = float(z_imp)
                log(f"  overtopping section from z = {overtop_zmin:.4f} m "
                    f"(top of the impermeable part; structure top {stl_zmax:.4f} m)")

        write_topoSetDict_force_and_overtopping(
            case_dir,
            overtop_x=float(x_fill_end),
            overtop_zmin=overtop_zmin,
            overtop_zmax=float(zmax_runup),
            Ly=float(s.Ly),
            include_force=bool(force_entries),
            source_patch=s.SOURCE_PATCH,
            force_entries=force_entries if force_entries else None,
            porous_stl_entries=porous_topo_entries if porous_topo_entries else None,
        )

        if patch_list:
            write_createPatchDict_for_segments(
                case_dir=case_dir,
                patches=patch_list,
                patch_type="wall",
            )
            log(f"  createPatchDict written for {len(patch_list)} patch(es): {patch_names}")
            # Save segment figure for PDF report
            try:
                sel_label = ", ".join(str(sid) for sid in seg_ids)
                _fig_ip, _ = plot_stl_xz_with_segments_colored(
                    stl_bbox, segs, x_max_keep=x_choose,
                    title=f"Impact patches — segments [{sel_label}]",
                )
                _fig_dir = Path(case_dir) / "figures"
                _fig_dir.mkdir(parents=True, exist_ok=True)
                _fig_ip.savefig(
                    str(_fig_dir / "impact_patch_segments.png"),
                    dpi=150, bbox_inches="tight",
                )
                plt.close(_fig_ip)
            except Exception:
                pass

        # decomposeParDict
        log("Writing decomposeParDict…")
        write_decomposeParDict(case_dir, n_procs=int(s.N_PROCS), method=s.DECOMP_METHOD, of_version="v2206")

        # Generate job_Run.sh
        log("Writing job_Run.sh…")
        write_job_script(
            case_dir=case_dir,
            notebook_dir=REPO_ROOT,
            new_patch=bool(s.new_patch),
            patch_name="",
            n_procs=int(s.N_PROCS),
            use_porous=bool(s.USE_POROUS),
            patch_names=patch_names if patch_names else None,
        )
        log(f"  job_Run.sh written ({s.N_PROCS} procs, solver={'porousWaveFoam' if s.USE_POROUS else 'waveFoam'})")

        s.step_done["case"] = True
        s.step_done["run"]  = False
        log("Case built successfully.")

        s.build_result = ("success", f"Case built: `{case_dir.resolve()}`")

    except Exception as exc:
        import traceback
        s.build_result = ("error", f"Case build failed: {exc}\n\n{traceback.format_exc()}")


def _run_launch():
    s = st.session_state
    try:
        case_dir     = Path(s.case_dir).resolve()
        notebook_dir = REPO_ROOT

        with st.spinner("Launching tmux session…"):
            launch_tmux_case(
                case_dir=case_dir,
                notebook_dir=notebook_dir,
                new_patch=bool(s.new_patch),
                patch_name="",
                n_procs=int(s.N_PROCS),
                use_porous=bool(s.USE_POROUS),
            )

        session_name = (case_dir / "TMUX_SESSION.txt").read_text().strip()
        socket_path  = (case_dir / "TMUX_SOCKET.txt").read_text().strip()

        s.step_done["run"] = True
        s.launch_result = ("success", f"Launched tmux session `{session_name}`\nSocket: `{socket_path}`")

        # Auto-generate report after successful launch
        try:
            _run_report()
        except Exception:
            pass  # report failure should not mask a successful launch

    except Exception as exc:
        import traceback
        s.launch_result = ("error", f"Launch failed: {exc}\n\n{traceback.format_exc()}")


def _write_run_info_from_state(case_dir: Path):
    """
    Writes run_info.json directly from st.session_state, bypassing the
    inspect.currentframe() mechanism used by case_report.write_run_info()
    (which only works when called from a notebook where variables are locals).
    """
    import json, socket as _socket, datetime
    s = st.session_state
    p = s.params or {}
    m = s.meta  or {}

    porous_zones_config = {
        z: float(s.porous_zone_d.get(z, 0.08))
        for z in VALID_POROUS_ZONES
        if s.porous_zone_active.get(z, False)
    } if s.USE_POROUS else {}

    info = {
        "meta": {
            "case_name": s.case_name,
            "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
            "machine": _socket.gethostname(),
        },
        "patch_logic": {
            "new_patch":    bool(s.new_patch),
            "SEG_IDS":      list(s.SEG_IDS),
            "PATCH_NAMES":  [f"force_seg_{sid:02d}" for sid in s.SEG_IDS],
            "FLOW_MODEL":   s.flow_model,
            "case_name":    s.case_name,
            "USE_COLLECTOR": bool(s.USE_COLLECTOR),
            "BREAKWATER_MODE": bool(s.BREAKWATER_MODE),
        },
        "structure_geometry": {
            "m":           float(s.m),
            "h_deep":      float(s.h_deep),
            "h":           float(s.h),
            "h_t":         float(s.h_t),
            "B_t":         float(s.B_t),
            "h_b":         float(s.h_b),
            "B":           float(s.B),
            "tan_alpha_B": float(s.tan_alpha_B),
            "cot_alpha_d": float(s.cot_alpha_d),
            "cot_alpha_u": float(s.cot_alpha_u),
            "A_c":         float(s.A_c),
            "G_c":         float(s.G_c),
            "R_c":         float(s.R_c),
            "ZMIN_THICKNESS": float(s.ZMIN_THICKNESS),
            "x_toe":       float(s.x_toe),
            "x_margin":    float(s.x_margin),
            "L_back":      float(s.L_back),
            "t_wall":      float(s.t_wall),
            "x_container_start": m.get("x_container_start"),
            "x_container_end":   m.get("x_container_end"),
            "dx_foreshore_ramp":  float(s.dx_foreshore_ramp),
            "dx_toe_transition":  float(s.dx_toe_transition),
            "dx_berm_connector":  float(s.dx_berm_connector),
        },
        "wave": {
            "Hm0_deep":   float(s.Hm0_deep),
            "Tp_deep":    float(s.Tp_deep),
            "Tm_deep":    float(s.Tm_deep),
            "Hm0_toe":    float(s.Hm0_toe),
            "Tp_toe":     float(s.Tp_toe),
            "Tm_toe":     float(s.Tm_toe),
            "Tm10_toe":   float(s.Tm10_toe),
            "WAVE_SEED":  int(s.WAVE_SEED),
            "Tp":         float(s.Tp_deep),
            "g":          9.81,
            "n_pre":      float(s.n_pre),
            "n_post":     float(s.n_post),
            "top_clearance": float(s.top_clearance),
        },
        "domain_grid": {
            "Ly":          float(s.Ly),
            "SWL":         float(s.SWL),
            "dx_target":   float(s.dx_target),
            "dz_target":   float(s.dz_target),
            "Ny":          int(s.Ny),
            "LEVEL_INTERFACE": int(s.LEVEL_INTERFACE),
            "LEVEL_RUNUP":     int(s.LEVEL_RUNUP),
            "Vertical_distance_plus_for_refinement_interface":
                float(s.Vertical_distance_plus_for_refinement_interface),
            "RUNUP_TOP_ABOVE_STL":   float(s.RUNUP_TOP_ABOVE_STL),
            "RUNUP_X_START_OFFSET":  float(s.RUNUP_X_START_OFFSET),
            "Y_LINE":  float(s.Y_LINE),
            "NPOINTS": int(s.NPOINTS),
            "LINE_WRITE_INTERVAL": float(s.LINE_WRITE_INTERVAL),
        },
        "control": {
            "END_TIME":       float(s.END_TIME),
            "DELTA_T":        float(s.DELTA_T),
            "WRITE_INTERVAL": float(s.WRITE_INTERVAL),
            "MAX_CO":         float(s.MAX_CO),
            "MAX_ALPHA_CO":   float(s.MAX_ALPHA_CO),
        },
        "hpc": {
            "N_PROCS":      int(s.N_PROCS),
            "DECOMP_METHOD": s.DECOMP_METHOD,
        },
        "porous": {
            "USE_POROUS":   bool(s.USE_POROUS),
            "POROUS_ZONES":  porous_zones_config,
            "BACK_SLOPE_WEDGE": bool(s.get("porous_back_slope_wedge", False))
                                if "back_slope" in porous_zones_config else False,
            "layers": [
                {
                    "name": layer["name"],
                    "default_thickness": float(layer["default_thickness"]),
                    "zone_thickness_overrides": {
                        z: float(v) for z, v in layer.get("zone_thickness_overrides", {}).items()
                    },
                    "physics_default": {
                        k: (float(v) if isinstance(v, (int, float)) else v)
                        for k, v in layer["physics_default"].items()
                    },
                    "zone_physics_overrides": {
                        z: {k: (float(vv) if isinstance(vv, (int, float)) else vv) for k, vv in ov.items()}
                        for z, ov in layer.get("zone_physics_overrides", {}).items()
                    },
                }
                for layer in s.porous_layers
            ] if s.USE_POROUS else [],
            "core_is_solid": bool(s.core_is_solid),
            "core_physics": {
                k: (float(v) if isinstance(v, (int, float)) else v)
                for k, v in s.core_physics.items()
            } if (not s.core_is_solid and s.core_physics) else None,
        },
        "obstacles": [
            {
                "name": o.get("name", f"obstacle_{i}"),
                "anchor": o.get("anchor"),
                "vertices_rel": o.get("vertices_rel", []),
            }
            for i, o in enumerate(s.obstacles)
        ] if s.obstacles else [],
    }

    out = case_dir / "run_info.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        json.dump(info, f, indent=2)


def _run_report():
    s = st.session_state
    try:
        case_dir = Path(s.case_dir).resolve()

        with st.spinner("Writing report…"):
            _write_run_info_from_state(case_dir)
            # Save figures held in session state.
            fig_dir = case_dir / "figures"
            pending = st.session_state.get("pending_figures", {})
            if pending:
                fig_dir.mkdir(parents=True, exist_ok=True)
                for fig_name, img_bytes in pending.items():
                    (fig_dir / f"{fig_name}.png").write_bytes(img_bytes)
            # Register all PNGs in case_dir/figures/, including earlier results.
            if fig_dir.is_dir():
                for png in sorted(fig_dir.glob("*.png")):
                    register_figure(case_dir, f"figures/{png.name}",
                                    png.stem.replace("_", " ").title())
            write_html_report_pro(case_dir)

        htmls = sorted(case_dir.glob("*.html"))
        if htmls:
            st.success(f"Report: `{htmls[0]}`")
        else:
            st.success("Report generated (see case directory).")

    except Exception as exc:
        st.error(f"Report generation failed: {exc}")
        import traceback
        st.code(traceback.format_exc())


# Main
def main():
    st.set_page_config(
        page_title="COAST — Coastal Overtopping Analysis and Simulation Toolkit",
        page_icon="🌊",
        layout="wide",
        menu_items={
            "Get Help": None,
            "Report a bug": None,
            "About": (
                "### COAST — Coastal Overtopping Analysis and Simulation Toolkit\n\n"
                "CNRS UMR 6614 CORIA, Université de Rouen Normandie\n\n"
                "Use the ☀️ / 🌙 button at the bottom of the sidebar to switch themes."
            ),
        },
    )
    _init_state()
    # Add defaults for keys missing from an existing session.
    st.session_state.setdefault("LINE_WRITE_INTERVAL", 0.02)
    st.session_state.setdefault("WAVE_SEED", 1234)

    inject_theme()

    # Read the theme from the URL parameter.
    _qp = st.query_params.to_dict()
    if "theme" in _qp and _qp["theme"] in ("day", "night"):
        st.session_state.theme_mode = _qp["theme"]
        st.query_params.clear()
        st.rerun()

    _sidebar()

    def _render_tabs():
        tabs = st.tabs([
            "Geometry",
            "Wave & Domain",
            "Mesh",
            "Simulation & Job",
            "Run",
            "Monitor",
        ])
        with tabs[0]:
            _tab_geometry()
        with tabs[1]:
            _tab_wave()
        with tabs[2]:
            _tab_mesh()
        with tabs[3]:
            _tab_sim()
        with tabs[4]:
            _tab_run()
        with tabs[5]:
            _tab_monitor()

    _render_tabs()


if __name__ == "__main__":
    main()
