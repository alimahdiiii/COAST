#!/usr/bin/env python3
import numpy as np
from scipy.integrate import trapezoid
import matplotlib.pyplot as plt
from pathlib import Path

# User settings

# I/O
Ly = 1.0
fname = Path("postProcessing/overtopping/0/overtopping.dat")  # change if needed

# Averaging window
# WINDOW_MODE:
# "MANUAL": use t1_manual and t2_manual.
# "STATIONARY": use the running mean and standard deviation of q^+(t).
# "FIRST_OVERTOPPING": start when cumulative volume reaches the threshold.
WINDOW_MODE = "FIRST_OVERTOPPING"  # averaging window mode

# Manual window (used only if WINDOW_MODE="MANUAL")
t1_manual, t2_manual = 10.0, 150.0

# Parameters for STATIONARY mode
STATIONARY = dict(
    win_sec=60.0,             # running window length [s]
    min_stationary_sec=20.0,  # must stay stationary for at least this long [s]
    tol_mean_rel=0.15,        # relative tolerance on running mean (vs reference mean)
    tol_std_rel=1.0,          # relative tolerance on running std  (vs reference std)
    tail_ref_sec=60.0,        # reference interval at the end of the record [s]
    end_margin_sec=0.0,       # optionally cut the end by this margin [s]
)

# Parameters for FIRST_OVERTOPPING mode
FIRST = dict(
    Vthr_abs=1e-5,        # [m^3/m] threshold on cumulative volume to declare overtopping start
    t2_end_margin=0.0,    # [s] optionally cut the end by this margin
)

# Event detection with separate start and end thresholds
q_start = 1e-3        # [m^3/s/m] event starts when q_pos >= q_start
q_end   = 3e-4        # [m^3/s/m] event ends   when q_pos <= q_end (must be < q_start)

min_duration = 0.02   # [s] discard events shorter than this
min_volume   = 1e-6   # [m^3/m] discard events smaller than this


# Helpers

def read_overtopping_xcomponent(fname: Path):
    """
    Read overtopping.dat formatted like:
      Time: overtopping
      0.1   (-1.2e-07 0 0)
    Returns sorted arrays: t, qx (x-component)
    """
    t_list, qx_list = [], []
    with open(fname, "r") as f:
        for line in f:
            line = line.strip()
            if (not line) or line.startswith("Time"):
                continue

            parts = line.split()
            if len(parts) < 2:
                continue

            try:
                tt = float(parts[0])
            except ValueError:
                continue

            rest = " ".join(parts[1:]).replace("(", "").replace(")", "")
            nums = rest.split()
            if not nums:
                continue

            try:
                qx = float(nums[0])
            except ValueError:
                continue

            t_list.append(tt)
            qx_list.append(qx)

    if not t_list:
        raise RuntimeError(f"No data read from {fname}")

    t = np.array(t_list, dtype=float)
    qx = np.array(qx_list, dtype=float)
    order = np.argsort(t)
    return t[order], qx[order]


def detect_overtopping_events_hysteresis(t, q_pos, q_start, q_end, min_duration=0.0, min_volume=0.0):
    """
    Peak-triggered overtopping event detection using hysteresis.

    Event START: q_pos >= q_start
    Event END  : q_pos <= q_end  (q_end < q_start)

    Returns list of dicts:
      i0, i1, t_start, t_end, duration, volume, q_peak
    """
    t = np.asarray(t, dtype=float)
    q = np.asarray(q_pos, dtype=float)

    if t.size != q.size or t.size < 2:
        return []
    if q_end >= q_start:
        raise ValueError("q_end must be < q_start for hysteresis to work.")

    events = []
    in_event = False
    i0 = None

    for i in range(t.size):
        if not in_event:
            if q[i] >= q_start:
                in_event = True
                i0 = i
        else:
            if q[i] <= q_end:
                i1 = i
                t0 = float(t[i0])
                t1_ = float(t[i1])
                duration = t1_ - t0
                vol = float(trapezoid(q[i0:i1 + 1], t[i0:i1 + 1]))
                q_peak = float(np.max(q[i0:i1 + 1]))

                if (duration >= min_duration) and (vol >= min_volume):
                    events.append(dict(
                        i0=int(i0), i1=int(i1),
                        t_start=t0, t_end=t1_,
                        duration=float(duration),
                        volume=float(vol),
                        q_peak=float(q_peak),
                    ))

                in_event = False
                i0 = None

    # close if open at the end
    if in_event and i0 is not None:
        i1 = t.size - 1
        t0 = float(t[i0])
        t1_ = float(t[i1])
        duration = t1_ - t0
        vol = float(trapezoid(q[i0:i1 + 1], t[i0:i1 + 1]))
        q_peak = float(np.max(q[i0:i1 + 1]))

        if (duration >= min_duration) and (vol >= min_volume):
            events.append(dict(
                i0=int(i0), i1=int(i1),
                t_start=t0, t_end=t1_,
                duration=float(duration),
                volume=float(vol),
                q_peak=float(q_peak),
            ))

    return events


def running_stats(t, x, win_sec):
    """
    Running mean and std over a time window of length win_sec.
    Simple two-pointer sliding window (O(N)).
    Returns arrays: mu, sig, valid (same length).
    """
    t = np.asarray(t, float)
    x = np.asarray(x, float)
    n = t.size

    mu = np.full(n, np.nan)
    sig = np.full(n, np.nan)
    valid = np.zeros(n, dtype=bool)

    j0 = 0
    s1 = 0.0
    s2 = 0.0
    for j in range(n):
        s1 += x[j]
        s2 += x[j] * x[j]

        while t[j] - t[j0] > win_sec and j0 < j:
            s1 -= x[j0]
            s2 -= x[j0] * x[j0]
            j0 += 1

        m = j - j0 + 1
        if m >= 2:
            mean = s1 / m
            var = max(0.0, s2 / m - mean * mean)
            mu[j] = mean
            sig[j] = np.sqrt(var)
            valid[j] = True

    return mu, sig, valid


def find_window_stationary(t, q_pos, cfg):
    """
    AUTO: choose (t1, t2) by detecting a statistically stationary regime of q_pos
    using running mean and running std.

    Returns: t1, t2, debug dict (with 'used_fallback' flag)
    """
    win_sec = float(cfg["win_sec"])
    min_stationary_sec = float(cfg["min_stationary_sec"])
    tol_mean_rel = float(cfg["tol_mean_rel"])
    tol_std_rel = float(cfg["tol_std_rel"])
    tail_ref_sec = float(cfg["tail_ref_sec"])
    end_margin_sec = float(cfg["end_margin_sec"])

    mu, sig, valid = running_stats(t, q_pos, win_sec)

    t_end = t[-1]
    tail_mask = (t >= (t_end - tail_ref_sec)) & valid
    if tail_mask.sum() < 5:
        tail_mask = valid

    mu_ref = np.nanmedian(mu[tail_mask])
    sig_ref = np.nanmedian(sig[tail_mask])

    mu_scale = max(abs(mu_ref), 1e-12)
    sig_scale = max(abs(sig_ref), 1e-12)

    cond = valid.copy()
    cond &= (np.abs(mu - mu_ref) / mu_scale) <= tol_mean_rel
    cond &= (np.abs(sig - sig_ref) / sig_scale) <= tol_std_rel

    # Find first stable segment lasting >= min_stationary_sec
    i_start = None
    acc = 0.0
    last_i = None
    for i in range(t.size):
        if cond[i]:
            if last_i is None:
                acc = 0.0
            else:
                acc += (t[i] - t[last_i])
            last_i = i

            if acc >= min_stationary_sec:
                j = i
                while j > 0 and cond[j - 1]:
                    j -= 1
                i_start = j
                break
        else:
            last_i = None
            acc = 0.0

    if i_start is None:
        # fallback
        t1 = float(t[0])
        t2 = float(t_end - end_margin_sec)
        return t1, t2, dict(mu=mu, sig=sig, mu_ref=mu_ref, sig_ref=sig_ref, cond=cond, used_fallback=True)

    i_end = i_start
    while i_end + 1 < t.size and cond[i_end + 1]:
        i_end += 1

    t1 = float(t[i_start])
    t2_raw = float(t[i_end])
    t2 = min(t2_raw, float(t_end - end_margin_sec))
    if t2 <= t1:
        t2 = float(t_end - end_margin_sec)

    return t1, t2, dict(mu=mu, sig=sig, mu_ref=mu_ref, sig_ref=sig_ref, cond=cond, used_fallback=False)


def find_window_first_overtopping(t, V_pos, Vthr_abs=1e-5, t2_end_margin=0.0):
    """
    AUTO: choose (t1, t2) by skipping the initial no-overtopping part.

    Definition:
      t1 = first time when cumulative overtopped volume V_pos >= Vthr_abs
      t2 = t_end - t2_end_margin

    Returns: t1, t2, used_fallback
    """
    t = np.asarray(t, float)
    V = np.asarray(V_pos, float)

    idx = np.where(V >= Vthr_abs)[0]
    if idx.size == 0:
        return float(t[0]), float(t[-1] - t2_end_margin), True

    i0 = int(idx[0])
    t1 = float(t[i0])
    t2 = float(t[-1] - t2_end_margin)
    return t1, t2, False


# Main

if not fname.exists():
    raise FileNotFoundError(f"File not found: {fname}")

t, qx = read_overtopping_xcomponent(fname)

# Positive-only overtopping discharge per unit width
q_pos = np.maximum(qx, 0.0) / Ly  # [m^3/s/m]

# Cumulative overtopped volume
V_pos = np.zeros_like(t)
V_pos[1:] = np.cumsum(0.5 * (q_pos[1:] + q_pos[:-1]) * np.diff(t))

# Choose averaging window [t1, t2]
dbg = None
window_note = ""
used_fallback = False
t1, t2 = None, None

if WINDOW_MODE.upper() == "MANUAL":
    t1, t2 = float(t1_manual), float(t2_manual)
    window_note = "MANUAL: user-specified window."
elif WINDOW_MODE.upper() == "STATIONARY":
    t1, t2, dbg = find_window_stationary(t, q_pos, STATIONARY)
    window_note = "AUTO (STATIONARY): stationarity of running mean/std of q^+(t)."
    used_fallback = bool(dbg.get("used_fallback", False)) if dbg is not None else False
elif WINDOW_MODE.upper() == "FIRST_OVERTOPPING":
    t1, t2, used_fallback = find_window_first_overtopping(
        t, V_pos, Vthr_abs=FIRST["Vthr_abs"], t2_end_margin=FIRST["t2_end_margin"]
    )
    window_note = "AUTO (FIRST_OVERTOPPING): t1 = first significant overtopping (V_pos >= Vthr_abs)."
    if used_fallback:
        window_note += " Fallback used (threshold not reached)."
else:
    raise ValueError("WINDOW_MODE must be one of: MANUAL, STATIONARY, FIRST_OVERTOPPING")

if t2 <= t1:
    raise ValueError(f"Invalid averaging window: t1={t1} must be < t2={t2}")

mask = (t >= t1) & (t <= t2)
if mask.sum() < 2:
    raise RuntimeError(f"Not enough samples in averaging window [{t1}, {t2}]")

# Mean overtopping discharge over [t1,t2]
q_mean = trapezoid(q_pos[mask], t[mask]) / (t2 - t1)

# Detect events on full signal, then keep only those fully within [t1, t2]
events_all = detect_overtopping_events_hysteresis(
    t, q_pos,
    q_start=q_start,
    q_end=q_end,
    min_duration=min_duration,
    min_volume=min_volume
)
events = [e for e in events_all if e["t_start"] >= t1 and e["t_end"] <= t2]
n_events_outside = len(events_all) - len(events)

# Volume budget within the averaging window [t1, t2]
V_window_curve = float(trapezoid(q_pos[mask], t[mask]))
V_total_events = float(sum(e["volume"] for e in events)) if events else 0.0

in_event = np.zeros_like(t, dtype=bool)
for e in events:
    in_event[e["i0"]:e["i1"] + 1] = True

# Background: within window, outside detected events
q_bg_window = q_pos[mask] * (~in_event[mask])
V_background = float(trapezoid(q_bg_window, t[mask]))

V_closure = V_total_events + V_background
closure_err = V_window_curve - V_closure
closure_err_pct = 100.0 * closure_err / V_window_curve if V_window_curve != 0 else np.nan

# Sort and statistics
events_sorted = sorted(events, key=lambda e: e["volume"], reverse=True) if events else []
if events_sorted:
    vols = np.array([e["volume"] for e in events_sorted], dtype=float)
    peaks = np.array([e["q_peak"] for e in events_sorted], dtype=float)
else:
    vols = np.array([], dtype=float)
    peaks = np.array([], dtype=float)

# Save CSV
out_csv = Path("overtopping_events.csv")
if events_sorted:
    with out_csv.open("w") as f:
        f.write("event_id,t_start,t_end,duration,volume,q_peak\n")
        for k, e in enumerate(events_sorted, start=1):
            f.write(
                f"{k},"
                f"{e['t_start']:.12g},"
                f"{e['t_end']:.12g},"
                f"{e['duration']:.12g},"
                f"{e['volume']:.12g},"
                f"{e['q_peak']:.12g}\n"
            )

# Print results

print("=" * 60)
print("Overtopping post-processing summary")
print("=" * 60)

print("\nAveraging window selection:")
print(f"  Mode  : {WINDOW_MODE.upper()}")
print(f"  Rule  : {window_note}")
if WINDOW_MODE.upper() == "STATIONARY":
    print(f"  win_sec={STATIONARY['win_sec']} s, min_stationary_sec={STATIONARY['min_stationary_sec']} s, "
          f"tol_mean_rel={STATIONARY['tol_mean_rel']}, tol_std_rel={STATIONARY['tol_std_rel']}, "
          f"tail_ref_sec={STATIONARY['tail_ref_sec']} s, end_margin_sec={STATIONARY['end_margin_sec']} s")
    if used_fallback:
        print("  Note  : AUTO window fallback was used (stationary segment not found).")
elif WINDOW_MODE.upper() == "FIRST_OVERTOPPING":
    print(f"  Vthr_abs={FIRST['Vthr_abs']:.2e} m^3/m, t2_end_margin={FIRST['t2_end_margin']} s")
print(f"  Window: [t1, t2] = [{t1:.3f} s, {t2:.3f} s]")

print("\nMean overtopping discharge:")
print(f"  q_mean = {q_mean:.6e} m^3 s^-1 m^-1")
print(f"  (computed over the time window [{t1:.3f} s, {t2:.3f} s])")

print("\n" + "-" * 60)
print("Event definition (hysteresis-based, wave-like events)")
print("-" * 60)
print(f"  Event start threshold : q_start = {q_start:.2e} m^3 s^-1 m^-1")
print(f"  Event end threshold   : q_end   = {q_end:.2e} m^3 s^-1 m^-1")
print(f"  Minimum event duration: {min_duration:.2f} s")
print(f"  Minimum event volume  : {min_volume:.2e} m^3 m^-1")
print(f"\n  Total detected events (full signal)  : {len(events_all)}")
print(f"  Events outside averaging window      : {n_events_outside}  (discarded)")
print(f"  Events within averaging window [t1,t2]: {len(events)}")

print("\n" + "-" * 60)
print("Overtopped volume budget  (within averaging window [t1, t2])")
print("-" * 60)
print("  Total overtopped volume within window (from q^+(t)):")
print(f"    V_window = {V_window_curve:.6e} m^3 m^-1")
print("\n  Volume carried by detected wave-events (within window):")
print(f"    V_events = {V_total_events:.6e} m^3 m^-1  ({100.0 * V_total_events / V_window_curve:.1f} % of window total)")
print("\n  Background / tail volume (within window, outside detected events):")
print(f"    V_background = {V_background:.6e} m^3 m^-1  ({100.0 * V_background / V_window_curve:.1f} % of window total)")
print("\n  Volume closure check (within window):")
print(f"    V_events + V_background = {V_closure:.6e} m^3 m^-1")
print(f"    Closure error           = {closure_err:.2e} m^3 m^-1  ({closure_err_pct:.2f} %)")

if events_sorted:
    print("\n" + "-" * 60)
    print("Event statistics")
    print("-" * 60)
    print("  Event volume statistics [m^3 m^-1]:")
    print(f"    min  = {vols.min():.3e}")
    print(f"    mean = {vols.mean():.3e}")
    print(f"    max  = {vols.max():.3e}")

    print("\n  Event peak discharge statistics [m^3 s^-1 m^-1]:")
    print(f"    min  = {peaks.min():.3e}")
    print(f"    mean = {peaks.mean():.3e}")
    print(f"    max  = {peaks.max():.3e}")

    print("\n" + "-" * 60)
    print("Largest overtopping events (by volume)")
    print("-" * 60)
    for k, e in enumerate(events_sorted[:3], start=1):
        print(f"  Event #{k}:")
        print(f"    Volume         = {e['volume']:.6e} m^3 m^-1")
        print(f"    Time interval  = [{e['t_start']:.3f} s , {e['t_end']:.3f} s]")
        print(f"    Duration       = {e['duration']:.3f} s")
        print(f"    Peak discharge = {e['q_peak']:.6e} m^3 s^-1 m^-1")

    print("\n" + "-" * 60)
    print("Consistency check (event volume vs cumulative curve)")
    print("-" * 60)
    print("  For the three largest events:")
    for k, e in enumerate(events_sorted[:3], start=1):
        i0, i1 = e["i0"], e["i1"]
        dV_from_curve = V_pos[i1] - V_pos[i0]
        print(f"    Event #{k}: V_event = {e['volume']:.6e} ; ΔV_curve = {dV_from_curve:.6e}  (m^3 m^-1)")
    print("  → Agreement indicates consistent event detection and integration.")
else:
    print("\nNo events detected (try adjusting q_start/q_end/min_volume).")

print("\n" + "=" * 60)
print(f"Saved event table: {out_csv.resolve() if events_sorted else '(not written)'}")
print("=" * 60)

# Figures

# Figure 1: cumulative volume with averaging window
plt.figure(figsize=(10, 4))
plt.plot(t, V_pos, label="Cumulative overtopped volume ")
plt.axvspan(t1, t2, alpha=0.2, label=f"Averaging window")
plt.xlabel("Time [s]")
plt.ylabel("Cumulative volume [m$^3$ per m]")
plt.title("Cumulative overtopping volume vs time ")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("overtopping_fig01_cumulative_volume.png", dpi=200)
plt.show()
plt.close()

# Figure 2: q(t) with events + averaging window
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(t, q_pos, label=r"$q^+(t)$ ")
ax.axvspan(t1, t2, alpha=0.2, label=f"Averaging window ")
for e in events:
    ax.axvspan(e["t_start"], e["t_end"], alpha=0.15)
    i0, i1 = e["i0"], e["i1"]
    i_peak = i0 + int(np.argmax(q_pos[i0:i1 + 1]))
    ax.plot(t[i_peak], q_pos[i_peak], marker="o", markersize=4)
ax.set_xlabel("Time [s]")
ax.set_ylabel("Overtopping rate [m$^3$/s per m]")
ax.set_title(f"Overtopping rate vs time  — mean q = {q_mean:.3e} m$^3$/s per m")
ax.grid(True)
ax.legend()
fig.tight_layout()
fig.savefig("overtopping_fig02_rate_and_events.png", dpi=200)
plt.show()
plt.close(fig)

# Figure 3: positive discharge and points outside events
fig_bg, ax_bg = plt.subplots(figsize=(10, 3.5))
ax_bg.plot(t, q_pos, label=r"$q^+(t)$")
ax_bg.plot(
    t[~in_event], q_pos[~in_event],
    linestyle="None", marker=".", markersize=2,
    label="Outside detected events (contributes V_background)"
)
ax_bg.set_xlabel("Time [s]")
ax_bg.set_ylabel("Overtopping rate [m$^3$/s per m]")
ax_bg.set_title("Samples outside detected wave-events")
ax_bg.grid(True)
ax_bg.legend()
fig_bg.tight_layout()
fig_bg.savefig("overtopping_fig03_background_samples.png", dpi=200)
plt.show()
plt.close(fig_bg)

# Figure 4: STATIONARY mode diagnostic (optional)
if WINDOW_MODE.upper() == "STATIONARY" and dbg is not None:
    cond = dbg["cond"]
    fig3, ax3 = plt.subplots(figsize=(10, 4))
    ax3.plot(t, q_pos, label=r"$q^+(t)$")
    ax3.plot(t[cond], q_pos[cond], linestyle="None", marker=".", markersize=2,
             label="Stationary points (criterion satisfied)")
    ax3.axvspan(t1, t2, alpha=0.2, label="AUTO selected window")
    ax3.set_xlabel("Time [s]")
    ax3.set_ylabel("Overtopping rate [m$^3$/s per m]")
    ax3.set_title("STATIONARY-mode diagnostic ")
    ax3.grid(True)
    ax3.legend()
    fig3.tight_layout()
    fig3.savefig("overtopping_fig04_stationary_diagnostic.png", dpi=200)
    plt.show()
    plt.close(fig3)
