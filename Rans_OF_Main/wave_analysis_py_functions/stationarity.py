"""
Detects when a free-surface time series becomes approximately stationary, from
the stability of its variance in sliding windows, and trims the initial
transient.
"""

from __future__ import annotations

from typing import Any, Dict, Literal, Tuple
import numpy as np


def auto_detect_stationary_start(
    time,
    elevation,
    window_duration: float = 200.0,
    step_duration: float = 20.0,
    tail_fraction: float = 0.3,
    tolerance: float = 0.05,
    min_time: float = 0.0,
    consecutive_windows: int = 3,
) -> Tuple[int, float]:
    """
    Automatically estimate the time at which the signal becomes stationary.

    The method:
      1) computes a reference variance from the last `tail_fraction` of the record,
      2) slides a window from `min_time` forward,
      3) finds the first time where `consecutive_windows` windows have variance within
         `tolerance` of the reference variance.

    Parameters
    ----------
    time : array-like of float
        Time array [s]. Must be 1D and strictly increasing. Sampling is assumed ~uniform.
    elevation : array-like of float
        Surface elevation η(t) [m]. Same length as `time`.
    window_duration : float, default=200.0
        Duration [s] of the sliding window used to compute variance.
    step_duration : float, default=20.0
        Step between two windows [s].
    tail_fraction : float, default=0.3
        Fraction of the record at the end used to define the reference variance.
        Example: 0.3 -> last 30% of the data.
    tolerance : float, default=0.05
        Relative tolerance for variance:
            |var_window - var_ref| / var_ref <= tolerance
        Example: 0.05 means within 5%.
    min_time : float, default=0.0
        Do not start searching before this time [s].
    consecutive_windows : int, default=3
        Number of consecutive windows that must satisfy the tolerance to accept stationarity.

    Returns
    -------
    index_start : int
        Index in the arrays where stationary regime starts.
    time_start : float
        Corresponding time value [s] in the original time axis.

    Notes
    -----
    - If no stationary window is found, returns (0, time[0]) as a fallback.
    - Variance-based stationarity is a practical heuristic; it does not guarantee full stationarity
      of all statistics (e.g., skewness, kurtosis).
    """
    time = np.asarray(time, dtype=float)
    elevation = np.asarray(elevation, dtype=float)

    # basic checks
    if time.ndim != 1 or elevation.ndim != 1:
        raise ValueError("time and elevation must be 1D arrays.")
    if len(time) != len(elevation):
        raise ValueError("time and elevation must have the same length.")
    if len(time) < 3:
        raise ValueError("Need at least 3 samples.")
    if not np.all(np.isfinite(time)) or not np.all(np.isfinite(elevation)):
        raise ValueError("time/elevation contain NaN/Inf.")
    if np.any(np.diff(time) <= 0):
        raise ValueError("time must be strictly increasing.")

    # approximate dt
    dt = float(np.median(np.diff(time)))
    if dt <= 0:
        raise ValueError("Invalid dt computed from time.")
    n = len(time)

    if not (0.0 < tail_fraction < 1.0):
        raise ValueError("tail_fraction must be between 0 and 1 (exclusive).")
    if window_duration <= 0 or step_duration <= 0:
        raise ValueError("window_duration and step_duration must be > 0.")
    if consecutive_windows < 1:
        raise ValueError("consecutive_windows must be >= 1.")
    if tolerance < 0:
        raise ValueError("tolerance must be >= 0.")

    # Reference variance from tail
    tail_start_index = int((1.0 - tail_fraction) * n)
    tail_start_index = np.clip(tail_start_index, 0, n - 1)
    tail_eta = elevation[tail_start_index:]
    var_ref = float(np.var(tail_eta))

    # Sliding-window scan
    window_size = int(round(window_duration / dt))
    step_size = int(round(step_duration / dt))

    if window_size < 10:
        raise ValueError(
            "window_duration is too small compared to dt; increase it "
            f"(window_size={window_size})."
        )
    if step_size < 1:
        step_size = 1

    # start search index based on min_time
    start_index_min = int(np.searchsorted(time, min_time, side="left"))
    start_index_min = np.clip(start_index_min, 0, n - 1)

    good_windows_in_a_row = 0
    first_good_index = None

    for start in range(start_index_min, n - window_size, step_size):
        end = start + window_size
        segment = elevation[start:end]
        var_seg = float(np.var(segment))

        # relative difference from reference variance
        if var_ref == 0.0:
            rel_diff = np.inf if var_seg != 0.0 else 0.0
        else:
            rel_diff = abs(var_seg - var_ref) / var_ref

        if rel_diff <= tolerance:
            good_windows_in_a_row += 1
            if good_windows_in_a_row == 1:
                first_good_index = start
            if good_windows_in_a_row >= consecutive_windows:
                index_start = int(first_good_index)
                time_start = float(time[index_start])
                return index_start, time_start
        else:
            good_windows_in_a_row = 0
            first_good_index = None

    # Fallback
    return 0, float(time[0])


def trim_transient(
    time,
    elevation,
    method: Literal["auto", "manual"] = "auto",
    t_cut: float = 0.0,
    *,
    # auto-detection parameters (only used if method="auto")
    window_duration: float = 200.0,
    step_duration: float = 20.0,
    tail_fraction: float = 0.3,
    tolerance: float = 0.05,
    min_time: float = 0.0,
    consecutive_windows: int = 3,
    # behavior
    reset_time: bool = True,
) -> Tuple[np.ndarray, np.ndarray, int, float, Dict[str, Any]]:
    """
    Trim the initial transient part of a free-surface elevation signal.

    Single-call helper:
    - method="auto": automatically detect stationary start using `auto_detect_stationary_start`
    - method="manual": remove the first `t_cut` seconds

    Parameters
    ----------
    time : array-like of float
        Time vector [s], 1D and strictly increasing.
    elevation : array-like of float
        Free-surface elevation η(t) [m], same length as `time`.
    method : {"auto","manual"}, default="auto"
        Strategy for trimming.
    t_cut : float, default=0.0
        Manual cut time [s] (used only if method="manual").
    reset_time : bool, default=True
        If True, shift the trimmed time so it starts at 0 s.

    Auto-detection parameters
    -------------------------
    window_duration, step_duration, tail_fraction, tolerance, min_time, consecutive_windows
        Passed to `auto_detect_stationary_start` when method="auto".

    Returns
    -------
    time_out : numpy.ndarray
        Trimmed time [s]. If reset_time=True, starts at 0.
    elevation_out : numpy.ndarray
        Trimmed elevation [m].
    index_start : int
        Index where the kept signal starts (in original arrays).
    time_start : float
        Start time of the kept signal in the original time axis [s].
    info : dict
        Dictionary describing what was done (method and parameters).

    Examples
    --------
    # Auto:
    t2, eta2, idx0, t0, info = trim_transient(time, eta, method="auto")

    # Manual:
    t2, eta2, idx0, t0, info = trim_transient(time, eta, method="manual", t_cut=50.0)
    """
    time = np.asarray(time, dtype=float)
    elevation = np.asarray(elevation, dtype=float)

    # Check the input arrays.
    if time.ndim != 1 or elevation.ndim != 1:
        raise ValueError("time and elevation must be 1D arrays.")
    if len(time) != len(elevation):
        raise ValueError("time and elevation must have the same length.")
    if len(time) < 2:
        raise ValueError("Need at least 2 samples.")
    if np.any(np.diff(time) <= 0):
        raise ValueError("time must be strictly increasing.")

    method_l = method.lower().strip()
    if method_l == "manual":
        if t_cut < 0:
            raise ValueError("t_cut must be >= 0.")
        # find index where time >= t_cut
        index_start = int(np.searchsorted(time, t_cut, side="left"))
        if index_start >= len(time) - 1:
            raise ValueError("t_cut is too large: nothing left after trimming.")
        time_start = float(time[index_start])

    elif method_l == "auto":
        index_start, time_start = auto_detect_stationary_start(
            time,
            elevation,
            window_duration=window_duration,
            step_duration=step_duration,
            tail_fraction=tail_fraction,
            tolerance=tolerance,
            min_time=min_time,
            consecutive_windows=consecutive_windows,
        )
    else:
        raise ValueError('method must be "auto" or "manual".')

    time_out = time[index_start:].copy()
    elevation_out = elevation[index_start:].copy()

    if reset_time and len(time_out) > 0:
        time_out = time_out - time_out[0]

    info: Dict[str, Any] = {
        "method": method_l,
        "reset_time": reset_time,
        "t_cut": float(t_cut),
        "window_duration": float(window_duration),
        "step_duration": float(step_duration),
        "tail_fraction": float(tail_fraction),
        "tolerance": float(tolerance),
        "min_time": float(min_time),
        "consecutive_windows": int(consecutive_windows),
    }

    return time_out, elevation_out, int(index_start), float(time_start), info

