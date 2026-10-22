# Report_wave_analysis.py

from pathlib import Path
import numpy as np
import pandas as pd
import html as html_lib


# Helpers

def safe_x_label(xg: float) -> str:
    """Return a string like 'm8p00' for x = -8.0, used in filenames."""
    return f"{xg:.2f}".replace(".", "p").replace("-", "m")

import re


def collect_free_surface_frames(fig_dir: Path):
    frames = []
    for p in fig_dir.glob("freeSurface_on_stl_t_*.png"):
        m = re.search(r"_t_([0-9]+p[0-9]+|[0-9]+)\.png$", p.name)
        if not m:
            continue
        t = float(m.group(1).replace("p", "."))
        frames.append((t, p.name))
    frames.sort(key=lambda x: x[0])
    return frames

def html_free_surface_player(fig_rel_dir: str, mp4_name: str, frames: list[tuple[float, str]]) -> str:
    """
    Build HTML block: (1) MP4 player + (2) time slider showing the PNG at selected timestep.
    fig_rel_dir is relative to the HTML file location.
    """
    if not frames:
        return "<p class='info-text'><b>No free-surface frames found.</b></p>"

    js = ",\n".join([f"{{t:{t:.6f}, file:'{fn}'}}" for t, fn in frames])

    t0 = frames[0][0]
    tN = frames[-1][0]

    return f"""
    <h2>Free-surface animation (STL + interface)</h2>

    <p class="info-text">
      This section provides: (i) a video animation and (ii) a time-step selector.
      Use the slider to display the exact frame at a chosen CFD output time.
    </p>

    <div class="card figure-card" style="margin-bottom: 1rem;">
      <div class="figure-media">
        <video controls style="max-width: 100%; border-radius: 0.9rem;">
          <source src="{fig_rel_dir}/{mp4_name}" type="video/mp4">
          Your browser does not support the video tag.
        </video>
      </div>
      <p class="info-text" style="margin-top: 0.65rem;">
        Video: <code>{mp4_name}</code>
      </p>
    </div>

    <div class="card figure-card">
      <div class="fs-toolbar">
        <input id="fsSlider" type="range" min="0" max="{len(frames)-1}" value="0" step="1">
        <div class="fs-meta">
          <div class="fs-time"><b>t = <span id="fsTime">{t0:.3f}</span> s</b></div>
          <div class="fs-range">range: {t0:.3f} → {tN:.3f} s</div>
        </div>
      </div>

      <div class="figure-media">
        <img id="fsImg" class="zoomable" src="{fig_rel_dir}/{frames[0][1]}"
             alt="Free-surface frame"/>
      </div>
    </div>

    <script>
    (function() {{
      const FS = [
      {js}
      ];

      const slider = document.getElementById('fsSlider');
      const img = document.getElementById('fsImg');
      const tspan = document.getElementById('fsTime');

      if (!slider || !img || !tspan) return;

      function updateFS() {{
        const k = parseInt(slider.value);
        const item = FS[k];
        img.src = "{fig_rel_dir}/" + item.file;
        tspan.textContent = item.t.toFixed(3);
      }}

      slider.addEventListener('input', updateFS);
    }})();
    </script>
    """


def compute_m_minus1(freq, spec):
    """
    Spectral moment m_-1 = ∫ S(f)/f df (discrete sum with constant Δf).
    """
    if len(freq) < 2:
        return np.nan
    df = freq[1] - freq[0]
    mask = freq > 0
    if not np.any(mask):
        return np.nan
    return np.sum(spec[mask] / freq[mask]) * df


def generate_wave_analysis_report(
    case_dir,
    X_GAUGES,
    Hs_target,
    Tp_target,
    gamma_input,
    wave_figures,
    wa,
    load_eta_for_gauge,
    trimmed_data=None,
    Fmax=None,
    tmax=None,
    q_cfd=None,
    t1_o=None,
    t2_o=None,
    Ly_o=None,
    output_dir_name="Wave_Analysis",
    report_name="wave_report.html",
):
    """
    Build an interactive HTML wave analysis report for a given case.

    Parameters
    ----------
    case_dir : str or Path
        Root of the OpenFOAM case.
    X_GAUGES : iterable of float
        Gauge x-positions.
    Hs_target, Tp_target, gamma_input : float
        Target JONSWAP parameters.
    wave_figures : list of dict
        List like [{'file': 'Wave_Analysis/figures/eta_x_m8p00.png', 'caption': '...'}, ...].
    wa : module or object
        Wave-analysis utilities, must provide:
        - trim_transient(time, eta)
        - find_waves(time, eta)
        - compute_wave_spectrum_welch(time, eta, nBlock)
        - wave_spectral_properties(freq, spec)
        - wave_physical_properties(Hw, Tw)
    load_eta_for_gauge : callable
        Function (xg) -> (time, eta) for that gauge.
    trimmed_data : dict or None
        Optional dict[xg] = {'time': ..., 'eta': ...}.
    Fmax, tmax : float or None
        Optional peak impact force and its time.
    output_dir_name : str
        Directory name under case_dir where report goes.
    report_name : str
        Name of the HTML report file.

    Returns
    -------
    Path
        Path to the generated HTML report.
    """

    case_dir = Path(case_dir)
    report_dir = case_dir / output_dir_name
    fig_root = report_dir / "figures"

    # Free-surface animation assets (frames + video)
    report_dir.mkdir(parents=True, exist_ok=True)
    fig_root.mkdir(parents=True, exist_ok=True)

    frames_fs = collect_free_surface_frames(fig_root)

    # video name expected in Wave_Analysis/figures/
    fs_mp4_name = "freeSurface_on_stl.mp4"

    # Build the HTML block (paths relative to Wave_Analysis/wave_report.html)
    free_surface_html = html_free_surface_player(
        fig_rel_dir="figures",
        mp4_name=fs_mp4_name,
        frames=frames_fs
    )

    report_dir.mkdir(parents=True, exist_ok=True)
    fig_root.mkdir(parents=True, exist_ok=True)
    html_path = report_dir / report_name

    # Compute time-domain and spectral metrics for each gauge.

    metrics_per_gauge = []
    tables_html_per_gauge = {}

    for xg in X_GAUGES:
        print(f"[Wave report] Computing metrics for gauge x = {xg} m")

        # Prefer trimmed_data if already available
        if trimmed_data is not None and xg in trimmed_data:
            time = np.asarray(trimmed_data[xg]["time"])
            elevation = np.asarray(trimmed_data[xg]["eta"])
        else:
            time_raw, elev_raw = load_eta_for_gauge(xg)
            time, elevation, idx0, t0, info = wa.trim_transient(time_raw, elev_raw)
            time = np.asarray(time)
            elevation = np.asarray(elevation)

        # Time-domain analysis
        Hw, Tw, Hmin_list, Hmax_list, Tstart, Tend, tHmin, tHmax = wa.find_waves(time, elevation)

        (Hmean_td, Hglobal_td, Hmax_td, Hrms_td,
         Hstd_td, H1ov3_td, H1ov10_td, Tmean_td, Nwaves_td) = wa.wave_physical_properties(Hw, Tw)

        Hs_time = H1ov3_td
        Tz_time = Tmean_td
        Hmin_td = float(np.min(Hw))
        eta_mean = float(np.mean(elevation))
        eta_std  = float(np.std(elevation))

        # Spectral analysis (Welch)
        nBlock = max(16, len(time) // 6)
        freq, spec = wa.compute_wave_spectrum_welch(time, elevation, nBlock)

        (nth_moments, T01_spec, T02_spec,
         Hm0_spec, Hrms_spec, fp_spec, Tp_spec) = wa.wave_spectral_properties(freq, spec)

        m0_spec, m1_spec, m2_spec = nth_moments
        Hs_spec = Hm0_spec

        m_minus1 = compute_m_minus1(freq, spec)
        if m0_spec > 0 and not np.isnan(m_minus1):
            Tm10_spec = m_minus1 / m0_spec
        else:
            Tm10_spec = np.nan

        # Expected Hmax from Rayleigh extremes
        if m0_spec > 0.0 and Nwaves_td > 1:
            EHmax_spec = (
                2.0
                * (1.0 + 0.29 / np.log(Nwaves_td))
                * np.sqrt(2.0 * np.log(Nwaves_td))
                * np.sqrt(m0_spec)
            )
        else:
            EHmax_spec = np.nan

        # Detailed table for this gauge
        Hrms_target = Hs_target / np.sqrt(2.0)

        rows = [
            # Heights
            ("Hs (significant height)",            Hs_time,     Hs_spec,      Hs_target,           "m"),
            ("Hrms (RMS height)",                  Hrms_td,     Hrms_spec,    Hrms_target,         "m"),
            ("Hmean (mean height)",                Hmean_td,    np.nan,       np.nan,              "m"),
            ("Hstd (std of wave heights)",         Hstd_td,     np.nan,       np.nan,              "m"),
            ("H1/10 (mean highest 10%)",           H1ov10_td,   np.nan,       np.nan,              "m"),
            ("Hglobal = max(Hi) − min(Hi)",        Hglobal_td,  np.nan,       np.nan,              "m"),
            ("Hmax (observed maximum)",            Hmax_td,     np.nan,       np.nan,              "m"),
            ("Hmin (observed minimum)",            Hmin_td,     np.nan,       np.nan,              "m"),
            ("Expected Hmax (Rayleigh extreme)",   np.nan,      EHmax_spec,   np.nan,              "m"),

            # Elevation statistics
            ("η mean (surface elevation mean)",    eta_mean,    np.nan,       0.0,                 "m"),
            ("η std",                              eta_std,     np.sqrt(m0_spec) if m0_spec>0 else np.nan, np.nan, "m"),
            ("m0 (variance of η)",                 np.nan,      m0_spec,      (Hs_target**2)/16.0, "m²"),
            ("m-1",                                np.nan,      m_minus1,     np.nan,              "m²·s"),
            ("m1",                                 np.nan,      m1_spec,      np.nan,              "m²/Hz"),
            ("m2",                                 np.nan,      m2_spec,      np.nan,              "m²/Hz²"),

            # Periods
            ("Tz (mean zero-crossing period)",     Tz_time,     T02_spec,     np.nan,              "s"),
            ("Tm (mean period)",                   Tmean_td,    T01_spec,     np.nan,              "s"),
            ("Tm-10 (energy period m-1/m0)",       np.nan,      Tm10_spec,    np.nan,              "s"),
            ("Tp (peak period)",                   np.nan,      Tp_spec,      Tp_target,           "s"),
            ("fp (peak frequency)",                np.nan,      fp_spec,      1.0 / Tp_target,     "Hz"),

            # Count
            ("Number of waves N",                  Nwaves_td,   np.nan,       np.nan,              "-"),
        ]

        df_gauge = pd.DataFrame(
            rows,
            columns=["Quantity", "CFD (time-domain)", "CFD (spectrum)", "Target JONSWAP", "Unit"]
        )

        tables_html_per_gauge[xg] = df_gauge.to_html(
            index=False,
            float_format=lambda x: f"{x:8.4f}",
            classes="param-table"
        )

        # Compact metrics for global summary
        metrics_per_gauge.append(dict(
            xg        = float(xg),
            Hs_time   = float(Hs_time),
            Hs_spec   = float(Hs_spec),
            Hs_target = float(Hs_target),
            Hrms_td   = float(Hrms_td),
            Hrms_spec = float(Hrms_spec),
            Tz_time   = float(Tz_time),
            T02_spec  = float(T02_spec),
            Tp_spec   = float(Tp_spec),
            Tp_target = float(Tp_target),
            Tm10_spec = float(Tm10_spec) if not np.isnan(Tm10_spec) else np.nan,
            Nwaves    = int(Nwaves_td),
            Hmax_td   = float(Hmax_td),
            m0_spec   = float(m0_spec),
            m_minus1  = float(m_minus1) if not np.isnan(m_minus1) else np.nan,
        ))

    df_summary = pd.DataFrame(metrics_per_gauge)
    print("\n[Wave report] Global summary over gauges:")
    print(df_summary)

    # Map figures to gauges using wave_figures

    gauge_figs = {xg: [] for xg in X_GAUGES}
    force_figs = []

    for fig in wave_figures:
        fpath = fig["file"]      # e.g. "Wave_Analysis/figures/eta_x_m8p00.png"
        cap   = fig["caption"]
        lower = fpath.lower()

        matched = False
        for xg in X_GAUGES:
            safe = safe_x_label(xg)
            if f"x_{safe.lower()}" in lower:
                gauge_figs[xg].append(fig)
                matched = True
                break

        if ("force_time_" in lower) and not matched:
            force_figs.append(fig)

    # Static explanatory text (theory block)

    explanation_html = r"""
<section id="theory">
  <h2>Wave-parameter extraction and analysis</h2>

  <p>This report computes a complete set of wave statistics from the trimmed free-surface elevation signal 
  \(\eta(t)\) and compares: (i) <strong>time-domain</strong> (zero-crossing) estimates, 
  (ii) <strong>spectral-domain</strong> (Welch PSD) estimates, and 
  (iii) the <strong>target JONSWAP</strong> parameters used to prescribe the sea state.</p>

  <h3>Time-domain (zero-crossing) analysis</h3>

  <p>For each detected wave, the wave height is defined as the crest-to-trough difference:</p>
  <p>\[
  H_i = \eta_{\text{crest},i} - \eta_{\text{trough},i}.
  \]</p>
  <p>The corresponding wave period is the time between successive zero-crossings:</p>
  <p>\[
  T_i = t_{c,i+1} - t_{c,i}.
  \]</p>

  <p>From the sequences \(\{H_i\}\) and \(\{T_i\}\), standard time-domain parameters are computed:</p>

  <ul>
    <li>Significant wave height (time-domain)
    \[
    H_s^{(\text{time})} = H_{1/3},
    \]
    </li>
    <li>Root-mean-square wave height
    \[
    H_{rms}^{(\text{time})} = \sqrt{\frac{1}{N}\sum_{i=1}^{N} H_i^2},
    \]
    </li>
    <li>Observed extremes
    \[
    H_{\max}^{(\text{time})} = \max(H_i), \qquad H_{\min}^{(\text{time})} = \min(H_i),
    \]
    </li>
    <li>Mean zero-crossing period
    \[
    T_z^{(\text{time})} = \frac{1}{N}\sum_{i=1}^{N} T_i.
    \]
    </li>
  </ul>

  <p>Elevation statistics are also reported:</p>
  <p>\[
  \bar{\eta} = \frac{1}{M}\sum_{k=1}^{M}\eta(t_k), \qquad
  \sigma_\eta = \sqrt{\frac{1}{M}\sum_{k=1}^{M}\left(\eta(t_k)-\bar{\eta}\right)^2}.
  \]</p>

  <h3> Spectral-domain (Welch PSD) analysis</h3>

  <p>The wave energy spectrum (PSD) \(E(f) \equiv S_{\eta\eta}(f)\) is estimated from \(\eta(t)\) using Welch averaging.
  The spectral moments are defined as:</p>
  <p>\[
  m_n = \int_0^\infty f^n E(f)\,df.
  \]</p>

  <p>In particular,</p>
  <p>\[
  m_0 = \int_0^\infty E(f)\,df \approx \mathrm{Var}(\eta).
  \]</p>

  <p>Bulk spectral parameters are derived as:</p>

  <ul>
    <li>Significant wave height (spectral)
    \[
    H_{m0} = 4\sqrt{m_0},
    \]
    </li>
    <li>Root-mean-square wave height (narrow-band / Rayleigh estimate)
    \[
    H_{rms}^{(\text{spec})} = \frac{H_{m0}}{\sqrt{2}} = \sqrt{8m_0},
    \]
    </li>
    <li>Mean periods
    \[
    T_{01} = \frac{m_0}{m_1}, \qquad
    T_{02} = \sqrt{\frac{m_0}{m_2}},
    \]
    </li>
    <li>Peak frequency and peak period
    \[
    f_p = \arg\max_f E(f), \qquad T_p = \frac{1}{f_p}.
    \]
    </li>
  </ul>

  <h3> Target JONSWAP relations used for comparison</h3>

  <p>The target sea state is defined by \((H_s^{\text{target}}, T_p^{\text{target}}, \gamma)\).
  Using the standard relation \(H_{m0}=4\sqrt{m_0}\), the target spectral moment is:</p>
  <p>\[
  m_0^{\text{target}} = \left(\frac{H_s^{\text{target}}}{4}\right)^2 = 
  \frac{\left(H_s^{\text{target}}\right)^2}{16}.
  \]</p>

  <p>The corresponding RMS height (narrow-band estimate) is:</p>
  <p>\[
  H_{rms}^{\text{target}} = \frac{H_s^{\text{target}}}{\sqrt{2}}.
  \]</p>

  <p>The target peak frequency is:</p>
  <p>\[
  f_p^{\text{target}} = \frac{1}{T_p^{\text{target}}}.
  \]</p>
</section>
"""

    # Build the interactive HTML report

    summary_html = df_summary.to_html(
        index=False,
        float_format=lambda x: f"{x:0.4f}",
        classes="param-table"
    )

    # Optional force-analysis data
    force_peak_exist = (Fmax is not None) and (tmax is not None)
    force_figs_exist = isinstance(force_figs, list) and len(force_figs) > 0

    force_peak_text = ""
    if force_peak_exist:
        try:
            force_peak_text = (
                f"<p>Peak impact force: <strong>{Fmax:.2f} N</strong> "
                f"at t = {tmax:.3f} s.</p>"
            )
        except Exception:
            force_peak_text = ""

    force_figs_html = ""
    if force_figs_exist:
        for fig in force_figs:
            img_src = "../" + fig["file"]  # fig["file"] is relative to case root
            force_figs_html += f"""
        <div class="figure-block">
          <p class="figure-caption">{fig['caption']}</p>
          <img class="zoomable" src="{img_src}" alt="{fig['caption']}" />
        </div>
        """

    # Find the overtopping figure and build its HTML.
    overtopping_figs = []
    for fig in wave_figures:
        name_lower = fig["file"].lower()
        cap_lower = fig["caption"].lower()
        if "overtopping" in name_lower or "overtopping" in cap_lower:
            overtopping_figs.append(fig)

    if overtopping_figs:
        overtopping_fig_html = ""
        for fig in overtopping_figs:
            img_src = "../" + fig["file"]
            overtopping_fig_html += f"""
            <div class="figure-block">
              <p class="figure-caption">{fig['caption']}</p>
              <img class="zoomable" src="{img_src}" alt="{fig['caption']}" />
            </div>
            """
    else:
        overtopping_fig_html = "<p class='info-text'>No overtopping figures registered in this case.</p>"

    # Text with numeric q if available
    overtopping_meta_html = ""
    if q_cfd is not None:
        overtopping_meta_html += (
            f"<p class='info-text'>Mean overtopping discharge from CFD: "
            f"<strong>{q_cfd:.3e} m³/s/m</strong>.</p>"
        )
        if t1_o is not None and t2_o is not None:
            overtopping_meta_html += (
                f"<p class='info-text'>Averaging window: "
                f"t₁ ≈ {t1_o:.3f} s, t₂ ≈ {t2_o:.3f} s.</p>"
            )
        if Ly_o is not None:
            overtopping_meta_html += (
                f"<p class='info-text'>width used for per-unit-width normalization: "
                f"Lᵧ = {Ly_o:.3f} m.</p>"
            )

    # Explanatory text for the overtopping results
    overtopping_explanation_html = r"""
        <p>
          Overtopping is quantified from the instantaneous volumetric flux crossing a predefined measurement
          plane (faceZone) located at the landward edge of the structure crest. OpenFOAM writes this signal as
          a time series \(q_x(t)\) in <code>postProcessing/overtopping/.../overtopping.dat</code>.
        </p>

        <h3>Per-unit-width discharge and overtopping direction</h3>
        <p>
          The reported flux \(q_x(t)\) is converted to a discharge per unit width using the spanwise width \(L_y\):
        </p>
        <p>
        \[
          q(t) = \frac{q_x(t)}{L_y}.
        \]
        </p>
        <p>
          To count only flow in the overtopping direction, the analysis uses the positive part:
        </p>
        <p>
        \[
          q^+(t) = \max\bigl(q(t), 0\bigr).
        \]
        </p>

        <h3>Cumulative overtopped volume</h3>
        <p>
          The cumulative overtopped volume per unit width is obtained by time integration of \(q^+(t)\)
          (implemented with a cumulative trapezoidal rule):
        </p>
        <p>
        \[
          V(t) = \int_{0}^{t} q^+(\tau)\,d\tau.
        \]
        </p>

        <h3>Mean overtopping discharge</h3>
        <p>
          Over an averaging window \([t_1,t_2]\), the mean overtopping discharge is computed as the time-average
          of \(q^+(t)\):
        </p>
        <p>
        \[
          q_{\mathrm{mean}} = \frac{1}{t_2-t_1}\int_{t_1}^{t_2} q^+(t)\,dt.
        \]
        </p>

        <h3>Overtopping events and volume budget</h3>
        <p>
          Individual overtopping events are detected using hysteresis thresholds: an event starts when
          \(q^+(t)\ge q_{\mathrm{start}}\) and ends when \(q^+(t)\le q_{\mathrm{end}}\) (\(q_{\mathrm{end}}<q_{\mathrm{start}}\)).
          The event volume is the time integral of \(q^+(t)\) over the event duration.
        </p>
        <p>
          The total cumulative volume \(V_{\mathrm{tot}}\) is taken from the final value of \(V(t)\).
          The detected-event contribution \(V_{\mathrm{events}}\) is the sum of event volumes, and the remaining
          contribution is reported as background volume:
        </p>
        <p>
        \[
          V_{\mathrm{background}} = V_{\mathrm{tot}} - V_{\mathrm{events}}.
        \]
        </p>
    """

    # Gauge selector menu
    gauge_selector_options = ['<option value="all" selected>All gauges</option>']
    for xg in X_GAUGES:
        safe = safe_x_label(xg)
        gauge_selector_options.append(
            f'<option value="{safe}">x = {xg:.2f} m</option>'
        )
    gauge_selector_html = """
<label for="gauge-select">Select gauge:</label>
<select id="gauge-select">
  {options}
</select>
""".format(options="\n  ".join(gauge_selector_options))

    # Gauge sections
    gauge_sections = []
    for xg in X_GAUGES:
        safe = safe_x_label(xg)
        figs_html = ""

        # Order figures: elevation, trimming, waves, spectrum, distributions, others.
        figs = gauge_figs.get(xg, [])

        def sort_key(fig):
            name = fig["file"].lower()
            if "eta_x_" in name:
                return 0
            if "trim_x_" in name:
                return 1
            if "wave_detection" in name:
                return 2
            if "spectrum" in name:
                return 3
            if "pdf" in name or "cdf" in name or "poe" in name:
                return 4
            return 5

        for fig in sorted(figs, key=sort_key):
            img_src = "../" + fig["file"]
            figs_html += f"""
        <div class="figure-block">
          <p class="figure-caption">{fig['caption']}</p>
          <img class="zoomable"
               src="{img_src}"
               alt="{fig['caption']}" />
        </div>
        """

        section_html = f"""
    <section class="gauge-section" data-gauge="{safe}">
      <h3>Gauge at x = {xg:.2f} m</h3>
      <div class="card table-card">
        {tables_html_per_gauge[xg]}
      </div>
      <div class="gauge-figures">
        {figs_html}
      </div>
    </section>
    """
        gauge_sections.append(section_html)

    gauge_sections_html = "\n".join(gauge_sections)

    case_name_safe = html_lib.escape(str(case_dir.name))

    # HTML report layout
    html_text = f"""<!DOCTYPE html>
<html lang="en" data-theme="light">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width,initial-scale=1" />
  <title>Wave analysis report – {case_name_safe}</title>

  <!-- MathJax for equations (kept as-is) -->
  <script src="https://polyfill.io/v3/polyfill.min.js?features=es6"></script>
  <script id="MathJax-script" async
          src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js">
  </script>

  <style>
    /* ============================================================
       Premium scientific report dashboard (self-contained)
       - Light/Dark themes via CSS variables + data-theme
       - Responsive grid layout
       - Print/PDF optimized stylesheet (A4)
       ============================================================ */

    :root {{
      --font-sans: ui-sans-serif, system-ui, -apple-system, "Segoe UI", Roboto, Arial, "Noto Sans", "Liberation Sans", sans-serif;
      --font-mono: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace;
      --radius-lg: 18px;
      --radius-md: 14px;
      --radius-sm: 12px;

      --shadow-sm: 0 1px 2px rgba(16, 24, 40, 0.06), 0 1px 1px rgba(16, 24, 40, 0.04);
      --shadow-md: 0 12px 28px rgba(16, 24, 40, 0.10);
      --shadow-lg: 0 20px 50px rgba(16, 24, 40, 0.18);

      --ring: 0 0 0 4px rgba(56, 189, 248, 0.22);

      --space-1: 6px;
      --space-2: 10px;
      --space-3: 14px;
      --space-4: 18px;
      --space-5: 24px;
      --space-6: 32px;

      --sidebar-w: 296px;

      /* Light (default) */
      --bg: #f7f8fb;
      --bg2: #ffffff;
      --surface: rgba(255,255,255,0.72);
      --card: #ffffff;
      --text: #0b1220;
      --muted: rgba(11,18,32,0.62);
      --muted2: rgba(11,18,32,0.48);
      --border: rgba(15, 23, 42, 0.10);
      --border2: rgba(15, 23, 42, 0.16);

      --accent: #0284c7;
      --accent2: #6366f1;
      --accent-soft: rgba(2, 132, 199, 0.10);
      --good: #16a34a;

      --code-bg: rgba(2, 6, 23, 0.06);

      --sidebar-bg: #0b1220;
      --sidebar-text: rgba(255,255,255,0.86);
      --sidebar-muted: rgba(255,255,255,0.62);
      --sidebar-border: rgba(255,255,255,0.10);
      --sidebar-hover: rgba(255,255,255,0.08);
      --sidebar-active: rgba(56, 189, 248, 0.16);

      color-scheme: light;
    }}

    html[data-theme="dark"] {{
      --bg: #060815;
      --bg2: #0a1024;
      --surface: rgba(10,16,36,0.72);
      --card: rgba(8, 12, 28, 0.92);
      --text: rgba(255,255,255,0.92);
      --muted: rgba(255,255,255,0.70);
      --muted2: rgba(255,255,255,0.56);
      --border: rgba(148, 163, 184, 0.14);
      --border2: rgba(148, 163, 184, 0.20);

      --accent: #38bdf8;
      --accent2: #a78bfa;
      --accent-soft: rgba(56, 189, 248, 0.14);
      --good: #22c55e;

      --code-bg: rgba(148, 163, 184, 0.10);

      --sidebar-bg: linear-gradient(180deg, rgba(4,6,16,0.98), rgba(9,13,28,0.98));
      --sidebar-text: rgba(255,255,255,0.90);
      --sidebar-muted: rgba(255,255,255,0.62);
      --sidebar-border: rgba(148, 163, 184, 0.16);
      --sidebar-hover: rgba(148, 163, 184, 0.12);
      --sidebar-active: rgba(56, 189, 248, 0.18);

      color-scheme: dark;
    }}

    * {{ box-sizing: border-box; }}
    html, body {{ height: 100%; }}
    body {{
      margin: 0;
      font-family: var(--font-sans);
      color: var(--text);
      background:
        radial-gradient(1200px 700px at 20% -10%, rgba(99,102,241,0.18), transparent 60%),
        radial-gradient(1100px 680px at 90% 10%, rgba(2,132,199,0.18), transparent 55%),
        linear-gradient(180deg, var(--bg), var(--bg2));
    }}

    a {{ color: inherit; text-decoration: none; }}
    code, pre {{
      font-family: var(--font-mono);
      background: var(--code-bg);
      padding: 0.12rem 0.35rem;
      border-radius: 10px;
      border: 1px solid var(--border);
    }}

    /* Accessibility */
    .skip-link {{
      position: absolute;
      left: -999px;
      top: 12px;
      padding: 10px 14px;
      background: var(--card);
      border: 1px solid var(--border2);
      border-radius: 12px;
      box-shadow: var(--shadow-md);
      z-index: 99999;
    }}
    .skip-link:focus {{ left: 12px; outline: none; box-shadow: var(--shadow-md), var(--ring); }}

    :focus-visible {{
      outline: none;
      box-shadow: var(--ring);
      border-radius: 12px;
    }}

    /* Layout grid */
    .page {{
      display: grid;
      grid-template-columns: var(--sidebar-w) minmax(0, 1fr);
      min-height: 100vh;
    }}

    /* Sidebar */
    .sidebar {{
      position: sticky;
      top: 0;
      height: 100vh;
      overflow: auto;
      padding: var(--space-6) var(--space-5);
      background: var(--sidebar-bg);
      color: var(--sidebar-text);
      border-right: 1px solid var(--sidebar-border);
    }}

    .sidebar-top {{
      display: grid;
      gap: var(--space-2);
      margin-bottom: var(--space-5);
    }}

    .brand {{
      display: grid;
      gap: 6px;
      padding: 14px 14px;
      border: 1px solid var(--sidebar-border);
      border-radius: var(--radius-md);
      background: rgba(255,255,255,0.06);
      box-shadow: inset 0 1px 0 rgba(255,255,255,0.06);
    }}

    .brand .title {{
      font-weight: 650;
      letter-spacing: 0.02em;
      font-size: 1.05rem;
      margin: 0;
    }}
    .brand .case-name {{
      font-weight: 600;
      font-size: 0.88rem;
      color: rgba(255,255,255,0.88);
      word-break: break-word;
      margin: 0;
    }}
    .brand .meta {{
      font-size: 0.78rem;
      color: var(--sidebar-muted);
      margin: 0;
    }}

    .nav-label {{
      margin-top: var(--space-4);
      margin-bottom: var(--space-2);
      font-size: 0.72rem;
      letter-spacing: 0.14em;
      text-transform: uppercase;
      color: var(--sidebar-muted);
    }}

    .nav {{
      display: grid;
      gap: 8px;
    }}

    .nav-item {{
      border: 1px solid transparent;
      border-radius: 14px;
      padding: 12px 12px;
      text-align: left;
      font-size: 0.92rem;
      background: transparent;
      color: var(--sidebar-text);
      cursor: pointer;
      display: grid;
      grid-template-columns: 1fr auto;
      align-items: center;
      gap: 10px;
      transition: transform 120ms ease, background 120ms ease, border-color 120ms ease;
    }}

    .nav-item span.badge {{
      font-size: 0.72rem;
      padding: 2px 9px;
      border-radius: 999px;
      background: rgba(255,255,255,0.10);
      border: 1px solid rgba(255,255,255,0.12);
      color: rgba(255,255,255,0.84);
    }}

    .nav-item:hover {{
      background: var(--sidebar-hover);
      transform: translateY(-1px);
      border-color: rgba(255,255,255,0.10);
    }}

    .nav-item.active {{
      background: var(--sidebar-active);
      border-color: rgba(56, 189, 248, 0.28);
      box-shadow: 0 10px 24px rgba(0,0,0,0.25);
    }}

    .sidebar-section-title {{
      margin-top: var(--space-5);
      margin-bottom: var(--space-2);
      font-size: 0.72rem;
      letter-spacing: 0.14em;
      text-transform: uppercase;
      color: var(--sidebar-muted);
    }}

    .gauge-list {{
      list-style: none;
      padding: 0;
      margin: 0;
      font-size: 0.86rem;
      color: rgba(255,255,255,0.76);
      line-height: 1.45;
    }}

    /* Content */
    .content {{
      padding: var(--space-6);
      min-width: 0;
    }}

    /* Topbar (responsive + controls) */
    .topbar {{
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: var(--space-4);
      padding: 14px 16px;
      border: 1px solid var(--border);
      background: var(--surface);
      backdrop-filter: blur(10px);
      border-radius: var(--radius-lg);
      box-shadow: var(--shadow-sm);
      margin-bottom: var(--space-5);
      position: sticky;
      top: 12px;
      z-index: 50;
    }}

    .topbar-left {{
      display: flex;
      align-items: center;
      gap: 12px;
      min-width: 0;
    }}

    .case-badge {{
      display: inline-flex;
      align-items: center;
      gap: 8px;
      padding: 8px 12px;
      border-radius: 999px;
      border: 1px solid var(--border);
      background: rgba(255,255,255,0.55);
    }}
    html[data-theme="dark"] .case-badge {{
      background: rgba(8,12,28,0.62);
    }}

    .case-badge .label {{
      font-size: 0.78rem;
      color: var(--muted);
      letter-spacing: 0.08em;
      text-transform: uppercase;
    }}
    .case-badge .value {{
      font-weight: 650;
      font-size: 0.86rem;
      color: var(--text);
      word-break: break-word;
    }}

    .topbar-title {{
      display: grid;
      min-width: 0;
    }}
    .topbar-title h1 {{
      margin: 0;
      font-size: 1.25rem;
      letter-spacing: -0.01em;
      line-height: 1.15;
    }}
    .topbar-title .subtitle {{
      margin: 2px 0 0 0;
      font-size: 0.88rem;
      color: var(--muted);
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
    }}

    .topbar-actions {{
      display: flex;
      align-items: center;
      gap: 10px;
      flex-wrap: wrap;
      justify-content: flex-end;
    }}

    .btn {{
      display: inline-flex;
      align-items: center;
      justify-content: center;
      gap: 8px;
      padding: 9px 12px;
      border-radius: 14px;
      border: 1px solid var(--border2);
      background: var(--card);
      color: var(--text);
      cursor: pointer;
      box-shadow: var(--shadow-sm);
      transition: transform 120ms ease, box-shadow 120ms ease, border-color 120ms ease;
      font-size: 0.88rem;
      user-select: none;
    }}
    .btn:hover {{
      transform: translateY(-1px);
      box-shadow: var(--shadow-md);
      border-color: rgba(2,132,199,0.22);
    }}
    .btn.primary {{
      background: linear-gradient(135deg, rgba(2,132,199,0.14), rgba(99,102,241,0.12));
      border-color: rgba(2,132,199,0.22);
    }}

    .btn.icon {{
      width: 42px;
      padding: 9px 0;
    }}

    .btn .dot {{
      width: 9px; height: 9px;
      border-radius: 999px;
      background: var(--accent);
      box-shadow: 0 0 0 3px rgba(2,132,199,0.14);
    }}

    /* Sections */
    .content-section {{
      display: none;
      margin-bottom: var(--space-6);
      padding: var(--space-5);
      border-radius: var(--radius-lg);
      border: 1px solid var(--border);
      background: rgba(255,255,255,0.66);
      backdrop-filter: blur(10px);
      box-shadow: var(--shadow-md);
    }}
    html[data-theme="dark"] .content-section {{
      background: rgba(8,12,28,0.70);
    }}

    .content-section h2 {{
      margin: 0 0 var(--space-3) 0;
      font-size: 1.25rem;
      letter-spacing: -0.01em;
      line-height: 1.2;
      padding-bottom: var(--space-2);
      border-bottom: 1px solid var(--border);
    }}

    .content-section h3 {{
      margin: var(--space-5) 0 var(--space-2) 0;
      font-size: 1.05rem;
      letter-spacing: -0.01em;
    }}

    .info-text {{
      font-size: 0.95rem;
      color: var(--muted);
      line-height: 1.55;
      margin-top: 0;
    }}

    /* Cards + grids */
    .card {{
      background: var(--card);
      border-radius: var(--radius-lg);
      padding: var(--space-4) var(--space-4);
      border: 1px solid var(--border);
      box-shadow: var(--shadow-sm);
      overflow-x: auto;
    }}

    .table-card {{
      padding: 0;
    }}
    .table-card .param-table {{
      margin: 0;
      border-radius: var(--radius-lg);
      overflow: hidden;
    }}

    /* Toolbar in gauges */
    .toolbar-gauge {{
      display: flex;
      flex-wrap: wrap;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
      margin: 0 0 var(--space-4) 0;
      padding: 14px 14px;
      border-radius: 999px;
      border: 1px solid rgba(2,132,199,0.18);
      background: linear-gradient(135deg, rgba(2,132,199,0.10), rgba(99,102,241,0.08));
    }}

    .toolbar-gauge label {{
      font-weight: 650;
      color: var(--text);
      margin-right: 8px;
    }}

    .toolbar-gauge select {{
      border-radius: 999px;
      border: 1px solid var(--border2);
      padding: 9px 12px;
      background: var(--card);
      color: var(--text);
      font-size: 0.92rem;
      box-shadow: var(--shadow-sm);
    }}

    /* Tables (scientific, modern) */
    .param-table {{
      width: 100%;
      border-collapse: separate;
      border-spacing: 0;
      font-size: 0.92rem;
      color: var(--text);
    }}

    .param-table th,
    .param-table td {{
      padding: 10px 12px;
      border-bottom: 1px solid var(--border);
      vertical-align: top;
      white-space: nowrap;
    }}

    .param-table thead th {{
      position: sticky;
      top: 0;
      z-index: 2;
      background: linear-gradient(180deg, rgba(255,255,255,0.92), rgba(255,255,255,0.75));
      color: var(--text);
      font-weight: 700;
      border-bottom: 1px solid var(--border2);
    }}
    html[data-theme="dark"] .param-table thead th {{
      background: linear-gradient(180deg, rgba(8,12,28,0.96), rgba(8,12,28,0.78));
    }}

    .param-table tbody tr:hover td {{
      background: rgba(2,132,199,0.06);
    }}
    html[data-theme="dark"] .param-table tbody tr:hover td {{
      background: rgba(56,189,248,0.08);
    }}

    /* Optional sticky first column (helpful for wide tables) */
    .param-table th:first-child,
    .param-table td:first-child {{
      position: sticky;
      left: 0;
      z-index: 1;
      background: var(--card);
      border-right: 1px solid var(--border);
    }}
    .param-table thead th:first-child {{
      z-index: 3;
    }}

    /* Figures: elegant figure cards */
    .gauge-figures {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
      gap: var(--space-4);
      margin-top: var(--space-4);
    }}

    .figure-block {{
      background: var(--card);
      border-radius: var(--radius-lg);
      padding: 12px;
      border: 1px solid var(--border);
      box-shadow: var(--shadow-sm);
      overflow: hidden;
    }}

    .figure-caption {{
      font-size: 0.9rem;
      color: var(--muted);
      margin: 6px 4px 10px 4px;
      line-height: 1.4;
    }}

    .figure-media {{
      border-radius: 16px;
      overflow: hidden;
      border: 1px solid var(--border);
      background: rgba(0,0,0,0.02);
    }}

    img.zoomable {{
      width: 100%;
      height: auto;
      display: block;
      cursor: zoom-in;
      border-radius: 16px;
    }}

    /* Free-surface player styling (IDs preserved) */
    .fs-toolbar {{
      display: flex;
      gap: 14px;
      align-items: center;
      flex-wrap: wrap;
      margin-bottom: 12px;
    }}
    #fsSlider {{
      width: min(520px, 100%);
      accent-color: var(--accent);
    }}
    .fs-meta {{
      display: grid;
      gap: 2px;
    }}
    .fs-range {{
      font-size: 0.85rem;
      color: var(--muted2);
    }}

    /* Modal zoom viewer (IDs preserved) */
    .img-modal {{
      display: none;
      position: fixed;
      z-index: 9999;
      inset: 0;
      background: rgba(0,0,0,0.70);
      justify-content: center;
      align-items: center;
      padding: 18px;
    }}

    .img-modal.open {{
      display: flex;
    }}

    .modal-content {{
      position: relative;
      width: min(1100px, 96vw);
      height: min(86vh, 860px);
      background: rgba(255,255,255,0.86);
      border-radius: 20px;
      padding: 14px 14px 16px 14px;
      border: 1px solid rgba(255,255,255,0.40);
      box-shadow: var(--shadow-lg);
      display: grid;
      grid-template-rows: auto 1fr auto;
      gap: 10px;
      backdrop-filter: blur(12px);
    }}
    html[data-theme="dark"] .modal-content {{
      background: rgba(8,12,28,0.88);
      border-color: rgba(148,163,184,0.22);
    }}

    .modal-header {{
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 10px;
      padding: 6px 6px 0 6px;
    }}

    .modal-title {{
      font-size: 0.92rem;
      color: var(--muted);
      overflow: hidden;
      white-space: nowrap;
      text-overflow: ellipsis;
      max-width: 80%;
    }}

    .close-button {{
      border: 1px solid var(--border2);
      background: var(--card);
      color: var(--text);
      width: 40px;
      height: 40px;
      border-radius: 14px;
      cursor: pointer;
      box-shadow: var(--shadow-sm);
      font-size: 1.2rem;
      line-height: 1;
    }}

    .modal-img-wrapper {{
      overflow: auto;
      border-radius: 16px;
      border: 1px solid var(--border);
      background: rgba(0,0,0,0.03);
      display: grid;
      place-items: center;
      padding: 10px;
    }}

    .modal-img-wrapper img {{
      max-width: 100%;
      max-height: 100%;
      transform-origin: center center;
      border-radius: 14px;
    }}

    .zoom-controls {{
      font-size: 0.9rem;
      color: var(--muted);
      display: flex;
      align-items: center;
      gap: 10px;
      padding: 0 6px 6px 6px;
    }}

    .zoom-controls input[type=range] {{
      flex: 1;
      accent-color: var(--accent);
    }}

    /* Collapsible sidebar (mobile + optional desktop) */
    body.sidebar-collapsed .page {{
      grid-template-columns: 88px minmax(0, 1fr);
    }}
    body.sidebar-collapsed .sidebar {{
      padding: var(--space-6) 10px;
    }}
    body.sidebar-collapsed .brand .meta,
    body.sidebar-collapsed .brand .case-name,
    body.sidebar-collapsed .nav-label,
    body.sidebar-collapsed .sidebar-section-title,
    body.sidebar-collapsed .gauge-list {{
      display: none;
    }}
    body.sidebar-collapsed .nav-item {{
      grid-template-columns: 1fr;
      justify-items: center;
      padding: 12px 10px;
    }}
    body.sidebar-collapsed .nav-item span.badge {{
      display: none;
    }}

    /* Responsive */
    @media (max-width: 980px) {{
      .page {{
        grid-template-columns: 1fr;
      }}
      .sidebar {{
        position: relative;
        height: auto;
        border-right: none;
        border-bottom: 1px solid var(--sidebar-border);
      }}
      .content {{
        padding: var(--space-5);
      }}
      .topbar {{
        top: 8px;
      }}
    }}

    @media (max-width: 560px) {{
      .content-section {{
        padding: var(--space-4);
      }}
      .topbar {{
        flex-wrap: wrap;
        gap: 10px;
      }}
      .topbar-actions {{
        width: 100%;
        justify-content: flex-start;
      }}
      .case-badge {{
        width: 100%;
        justify-content: space-between;
      }}
    }}

    /* Print / Save PDF (A4) */
    @page {{
      size: A4;
      margin: 14mm 12mm;
    }}

    @media print {{
      html, body {{
        background: #ffffff !important;
      }}

      body {{
        color: #000 !important;
        font-family: "Times New Roman", Times, serif !important;
      }}

      .sidebar,
      .topbar,
      .img-modal {{
        display: none !important;
      }}

      .page {{
        display: block !important;
      }}

      .content {{
        padding: 0 !important;
      }}

      .content-section {{
        display: block !important;
        background: #fff !important;
        box-shadow: none !important;
        border: none !important;
        padding: 0 0 10mm 0 !important;
        break-inside: avoid;
      }}

      .content-section {{
        page-break-before: always;
      }}
      .content-section#intro {{
        page-break-before: auto;
      }}

      h1, h2, h3 {{
        color: #000 !important;
      }}

      .card, .figure-block {{
        box-shadow: none !important;
        border: 1px solid #bbb !important;
      }}

      .param-table thead th {{
        position: static !important;
        background: #f2f2f2 !important;
      }}

      /* Avoid cut-off */
      img, video {{
        max-width: 100% !important;
        height: auto !important;
      }}

      /* Optional figure auto-numbering (CSS counters only) */
      .content {{
        counter-reset: fig;
      }}
      .figure-block {{
        counter-increment: fig;
      }}
      .figure-block .figure-caption::before {{
        content: "Figure " counter(fig) " — ";
        font-weight: 700;
      }}

      /* Links: print as text color */
      a {{
        color: #000 !important;
        text-decoration: none !important;
      }}
    }}
  </style>
</head>

<body>
<a class="skip-link" href="#main">Skip to content</a>

<div class="page">
  <aside class="sidebar" aria-label="Report navigation">
    <div class="sidebar-top">
      <div class="brand" role="group" aria-label="Report identity">
        <p class="title">Wave report</p>
        <p class="case-name">{case_name_safe}</p>
        <p class="meta">Automatic OpenFOAM wave analysis</p>
      </div>
    </div>

    <div class="nav-label">Sections</div>
    <nav class="nav">
      <button class="nav-item active" data-target="intro" aria-current="page">
        Introduction <span class="badge">theory</span>
      </button>
      <button class="nav-item" data-target="summary">
        Global summary
      </button>
      <button class="nav-item" data-target="gauges">
        Gauge analysis
      </button>
      <button class="nav-item" data-target="force-analysis">
        Force analysis
      </button>
      <button class="nav-item" data-target="overtopping">
        Overtopping
      </button>
      <button class="nav-item" data-target="free-surface">
        Free-surface animation
      </button>
    </nav>

    <div class="sidebar-section-title">Gauges</div>
    <ul class="gauge-list">
      <li>Positions: {", ".join(f"x = {xg:.2f} m" for xg in X_GAUGES)}</li>
    </ul>
  </aside>

  <main id="main" class="content">
    <div class="topbar" role="region" aria-label="Report controls">
      <div class="topbar-left">
        <button id="sidebarToggle" class="btn icon" type="button" aria-label="Toggle sidebar" title="Toggle sidebar">
          ☰
        </button>

        <div class="topbar-title">
          <h1>Wave analysis</h1>
          <div class="subtitle">• Report dashboard • by Ali MAHDI </div>
        </div>
      </div>

      <div class="topbar-actions">
        <div class="case-badge" title="Case name">
          <span class="label">Case</span>
          <span class="value">{case_name_safe}</span>
        </div>

        <button id="themeToggle" class="btn primary" type="button" aria-label="Toggle light/dark theme" title="Toggle theme">
          <span class="dot" aria-hidden="true"></span>
          Theme
        </button>

        <button id="printBtn" class="btn" type="button" aria-label="Print / Save PDF" title="Print / Save PDF">
          Print / PDF
        </button>
      </div>
    </div>

    <!-- INTRO / THEORY -->
    <section id="intro" class="content-section">
      <h2>Introduction &amp; wave-parameter extraction</h2>
      <p class="info-text">
        This section documents how free-surface elevation signals \\(\\eta(t)\\) are processed to extract
        time-domain wave statistics, spectral parameters, and their relation to the prescribed JONSWAP
        sea state used in the numerical wave flume.
      </p>
      {explanation_html}
    </section>

    <!-- GLOBAL SUMMARY SECTION -->
    <section id="summary" class="content-section">
      <h2>Global summary over gauges</h2>
      <p class="info-text">
        Summary of key wave parameters at each surface-elevation gauge: comparison between time-domain
        and spectral estimates, and deviation from the target JONSWAP parameters.
      </p>
      <div class="card table-card">
        {summary_html}
      </div>
    </section>

    <!-- GAUGES SECTION -->
    <section id="gauges" class="content-section">
      <h2>Per-gauge analysis</h2>
      <div class="toolbar-gauge">
        <div style="display:flex; align-items:center; gap:10px; flex-wrap:wrap;">
          {gauge_selector_html}
        </div>
        <span class="info-text" style="margin:0;">
          Select a gauge to display only the corresponding tables and figures. Click any figure to open a zoomable view.
        </span>
      </div>
      {gauge_sections_html}
    </section>

    <!-- OVERTOPPING SECTION -->
    <section id="overtopping" class="content-section">
      <h2>Overtopping analysis</h2>
      <p class="info-text">
        This section presents overtopping post-processing from a measurement plane (faceZone) located at the
        landward edge of the crest. The instantaneous overtopping discharge is obtained from the water-phase
        volumetric flux through this plane and integrated in time to compute cumulative overtopped volumes
        and mean discharge per unit width.
      </p>
      {overtopping_meta_html}
      {overtopping_explanation_html}
      {overtopping_fig_html}
    </section>

    <!-- FREE-SURFACE SECTION -->
    <section id="free-surface" class="content-section">
      {free_surface_html}
    </section>

    <!-- FORCE ANALYSIS SECTION -->
    <section id="force-analysis" class="content-section">
      <h2>Impact force analysis</h2>
      <p class="info-text">
        This section summarizes time-history and peak values of impact force on the structure, when force
        sampling is enabled in the CFD case.
      </p>
      {force_peak_text}
      <div class="gauge-figures">
        {force_figs_html}
      </div>
      {"" if (force_figs_exist or force_peak_exist) else "<p class='info-text'>No force data available for this case.</p>"}
    </section>
  </main>
</div>

<!-- Modal for zooming images (IDs preserved) -->
<div id="img-modal" class="img-modal" role="dialog" aria-modal="true" aria-label="Zoomed figure viewer">
  <div class="modal-content">
    <div class="modal-header">
      <div id="modal-title" class="modal-title">Figure</div>
      <button id="close-modal" class="close-button" type="button" aria-label="Close">&times;</button>
    </div>
    <div class="modal-img-wrapper">
      <img id="modal-img" src="" alt="Zoomed figure" />
    </div>
    <div class="zoom-controls">
      <label for="zoom-slider">Zoom</label>
      <input id="zoom-slider" type="range" min="1" max="4" step="0.1" value="1" />
      <span id="zoom-value">100%</span>
    </div>
  </div>
</div>

<script>
document.addEventListener("DOMContentLoaded", function () {{

  // ============================================================
  // Theme toggle (data-theme + localStorage). Safe if JS disabled.
  // ============================================================
  const root = document.documentElement;
  const themeBtn = document.getElementById("themeToggle");
  const THEME_KEY = "wave_report_theme";

  function applyTheme(theme) {{
    root.setAttribute("data-theme", theme);
    try {{ localStorage.setItem(THEME_KEY, theme); }} catch (e) {{}}
  }}

  // Initialize theme
  (function initTheme() {{
    let saved = null;
    try {{ saved = localStorage.getItem(THEME_KEY); }} catch (e) {{}}
    if (saved === "light" || saved === "dark") {{
      applyTheme(saved);
    }} else {{
      // Respect OS preference by default
      const prefersDark = window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)").matches;
      applyTheme(prefersDark ? "dark" : "light");
    }}
  }})();

  if (themeBtn) {{
    themeBtn.addEventListener("click", function() {{
      const current = root.getAttribute("data-theme") || "light";
      applyTheme(current === "dark" ? "light" : "dark");
    }});
  }}

  // ============================================================
  // Sidebar collapse (UI only)
  // ============================================================
  const sidebarToggle = document.getElementById("sidebarToggle");
  const SIDEBAR_KEY = "wave_report_sidebar_collapsed";

  function setSidebarCollapsed(v) {{
    document.body.classList.toggle("sidebar-collapsed", !!v);
    try {{ localStorage.setItem(SIDEBAR_KEY, v ? "1" : "0"); }} catch (e) {{}}
  }}

  (function initSidebar() {{
    let saved = null;
    try {{ saved = localStorage.getItem(SIDEBAR_KEY); }} catch (e) {{}}
    if (saved === "1") setSidebarCollapsed(true);
  }})();

  if (sidebarToggle) {{
    sidebarToggle.addEventListener("click", function() {{
      setSidebarCollapsed(!document.body.classList.contains("sidebar-collapsed"));
    }});
  }}

  // ============================================================
  // Print button
  // ============================================================
  const printBtn = document.getElementById("printBtn");
  if (printBtn) {{
    printBtn.addEventListener("click", function() {{
      window.print();
    }});
  }}

  // ============================================================
  // LEFT NAV: switch sections (IDs preserved)
  // ============================================================
  const navItems = Array.from(document.querySelectorAll(".nav-item"));
  const sections = Array.from(document.querySelectorAll(".content-section"));

  function showSection(id) {{
    sections.forEach(sec => {{
      sec.style.display = (sec.id === id) ? "block" : "none";
    }});
  }}

  navItems.forEach(btn => {{
    btn.addEventListener("click", () => {{
      const target = btn.dataset.target;
      showSection(target);
      navItems.forEach(b => {{
        const isActive = (b === btn);
        b.classList.toggle("active", isActive);
        b.setAttribute("aria-current", isActive ? "page" : "false");
      }});
      // On small screens, keep content in focus
      try {{
        const main = document.getElementById("main");
        if (main) main.scrollIntoView({{behavior:"smooth", block:"start"}});
      }} catch (e) {{}}
    }});
  }});

  // Default: show Introduction
  showSection("intro");

  // ============================================================
  // Gauge filter inside the Gauge tab (logic preserved)
  // ============================================================
  const gaugeSelect = document.getElementById("gauge-select");
  const gaugeSections = Array.from(document.querySelectorAll(".gauge-section"));

  function updateGaugeVisibility(value) {{
    gaugeSections.forEach(sec => {{
      if (value === "all" || sec.dataset.gauge === value) {{
        sec.style.display = "block";
      }} else {{
        sec.style.display = "none";
      }}
    }});
  }}

  if (gaugeSelect) {{
    gaugeSelect.addEventListener("change", function () {{
      updateGaugeVisibility(this.value);
    }});
    updateGaugeVisibility("all");
  }}

  // ============================================================
  // Zoom modal logic (behavior preserved; styling improved)
  // ============================================================
  const modal = document.getElementById("img-modal");
  const modalImg = document.getElementById("modal-img");
  const closeBtn = document.getElementById("close-modal");
  const zoomSlider = document.getElementById("zoom-slider");
  const zoomValue = document.getElementById("zoom-value");
  const modalTitle = document.getElementById("modal-title");

  function openModalWith(src, alt) {{
    if (!modal || !modalImg) return;
    modal.classList.add("open");
    modalImg.src = src;
    modalImg.alt = alt || "Zoomed figure";
    if (modalTitle) modalTitle.textContent = alt || "Figure";
    if (zoomSlider) zoomSlider.value = 1;
    if (zoomValue) zoomValue.textContent = "100%";
    modalImg.style.transform = "scale(1)";
    // Lock scroll behind modal
    document.body.style.overflow = "hidden";
    // Focus close for accessibility
    try {{ if (closeBtn) closeBtn.focus(); }} catch (e) {{}}
  }}

  function closeModal() {{
    if (modal) modal.classList.remove("open");
    document.body.style.overflow = "";
  }}

  document.querySelectorAll("img.zoomable").forEach(img => {{
    img.addEventListener("click", () => {{
      openModalWith(img.src, img.alt);
    }});
  }});

  if (closeBtn) closeBtn.addEventListener("click", closeModal);

  if (modal) {{
    modal.addEventListener("click", (e) => {{
      if (e.target === modal) closeModal();
    }});
  }}

  // ESC closes
  document.addEventListener("keydown", (e) => {{
    if (e.key === "Escape" && modal && modal.classList.contains("open")) {{
      closeModal();
    }}
  }});

  if (zoomSlider && modalImg && zoomValue) {{
    zoomSlider.addEventListener("input", () => {{
      const scale = parseFloat(zoomSlider.value) || 1;
      modalImg.style.transform = "scale(" + scale + ")";
      zoomValue.textContent = Math.round(scale * 100) + "%";
    }});
  }}
}});
</script>

</body>
</html>
"""

    with html_path.open("w", encoding="utf-8") as f:
        f.write(html_text)

    print(f"[Wave report] HTML report written to: {html_path}")
    return html_path

