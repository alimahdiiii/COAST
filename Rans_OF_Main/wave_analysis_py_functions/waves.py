"""Wave-by-wave (time-domain) analysis using zero-crossing."""

from __future__ import annotations

from typing import Sequence, List, Tuple
import math
import statistics
import numpy as np

def find_waves(time, elevation, upcrossing=False):
    """
    Detect individual waves in a free-surface time series using zero-crossing analysis.

    Parameters
    ----------
    time : array-like
        1D array/list of time instants (must be same length as `elevation`).
    elevation : array-like
        1D array/list of free-surface elevation η(t).
    upcrossing : bool, optional
        - True  -> use zero-upcrossing definition (waves between successive upcrossings).
        - False -> use zero-downcrossing definition (waves between successive downcrossings).

    Returns
    -------
    wave_heights : list of float
        Total wave height for each detected wave (crest - trough), using *detrended* signal.
    wave_periods : list of float
        Wave periods (difference between successive crossing times).
    min_heights : list of float
        Trough elevations (absolute, i.e. mean level added back).
    max_heights : list of float
        Crest elevations (absolute).
    wave_start_times : list of float
        Times of the starting zero-crossing (up or down).
    wave_end_times : list of float
        Times of the ending zero-crossing (up or down).
    time_at_min_height : list of float
        Times when the trough occurs within each wave.
    time_at_max_height : list of float
        Times when the crest occurs within each wave.
    """

    # Convert inputs to NumPy arrays.
    time = np.asarray(time, dtype=float)
    elevation = np.asarray(elevation, dtype=float)

    if time.shape != elevation.shape:
        raise ValueError("time and elevation must have the same shape")

    if len(time) < 3:
        # Not enough points to form a wave
        return [], [], [], [], [], [], [], []

    # Mean water level and detrending
    average_elevation = float(statistics.mean(elevation))
    adjusted_elevation = elevation - average_elevation

    # Sign of the detrended elevation
    elevation_sign = np.sign(adjusted_elevation)

    # Storage for all results
    wave_heights = []
    max_heights = []
    min_heights = []
    wave_periods = []
    wave_start_times = []
    wave_end_times = []
    time_at_max_height = []
    time_at_min_height = []

    # Storage for crossings
    upcross_indices = []
    downcross_indices = []
    upcross_times = []
    downcross_times = []

    # Detect zero-crossings with linear interpolation
    for i in range(len(adjusted_elevation) - 1):
        s0 = elevation_sign[i]
        s1 = elevation_sign[i + 1]

        # Skip if both are exactly zero (flat at mean)
        if s0 == 0 and s1 == 0:
            continue

        # Detect sign change (crossing from + to - or - to +)
        if s0 * s1 < 0:
            # Linear interpolation between points i and i+1
            dt = time[i + 1] - time[i]
            if dt == 0:
                continue  # avoid division by zero if time is malformed

            alpha = (adjusted_elevation[i + 1] - adjusted_elevation[i]) / dt
            if alpha == 0:
                continue  # Skip intervals with zero slope.

            time_cross = time[i] - adjusted_elevation[i] / alpha

            # Classify as upcrossing or downcrossing
            if s0 > 0:
                # from + to - : downcrossing
                downcross_indices.append(i)
                downcross_times.append(time_cross)
            else:
                # from - to + : upcrossing
                upcross_indices.append(i)
                upcross_times.append(time_cross)

    # Choose which crossings define waves
    if upcrossing:
        crossing_times = upcross_times
        crossing_indices = upcross_indices
        # condition: crest before trough within the wave interval
        def order_ok(idx_min, idx_max):
            return idx_max < idx_min
    else:
        crossing_times = downcross_times
        crossing_indices = downcross_indices
        # condition: trough before crest within the wave interval
        def order_ok(idx_min, idx_max):
            return idx_min < idx_max

    n_cross = len(crossing_times) - 1
    if n_cross <= 0:
        # No full waves detected
        return [], [], [], [], [], [], [], []

    # Loop over each wave interval
    for k in range(n_cross):
        t_start = crossing_times[k]
        t_end = crossing_times[k + 1]
        start_idx = crossing_indices[k]
        end_idx = crossing_indices[k + 1]

        # Put the interval indices in order.
        if end_idx <= start_idx + 1:
            continue  # not enough points in this interval

        segment = adjusted_elevation[start_idx:end_idx]

        # Local trough / crest within this wave
        local_min = np.min(segment)
        local_max = np.max(segment)
        idx_min = int(np.argmin(segment))
        idx_max = int(np.argmax(segment))

        # Enforce physical order (depends on up/downcross convention)
        if not order_ok(idx_min, idx_max):
            continue

        # Compute wave characteristics
        H = local_max - local_min
        T = t_end - t_start

        # Convert local indices back to global indices
        g_idx_min = start_idx + idx_min
        g_idx_max = start_idx + idx_max

        # Store results (add back mean for min/max elevations)
        wave_heights.append(float(H))
        wave_periods.append(float(T))
        min_heights.append(float(average_elevation + local_min))
        max_heights.append(float(average_elevation + local_max))
        wave_start_times.append(float(t_start))
        wave_end_times.append(float(t_end))
        time_at_min_height.append(float(time[g_idx_min]))
        time_at_max_height.append(float(time[g_idx_max]))

    return (
        wave_heights,
        wave_periods,
        min_heights,
        max_heights,
        wave_start_times,
        wave_end_times,
        time_at_min_height,
        time_at_max_height,
    )

def wave_physical_properties(wave_heights, wave_periods):
    """
    Compute basic statistical / physical properties of a set of individual waves.

    Parameters
    ----------
    wave_heights : array-like
        1D list/array of individual wave heights H_i (from zero-crossing analysis).
    wave_periods : array-like
        1D list/array of individual wave periods T_i (same length as wave_heights).

    Returns
    -------
    mean_height : float
        Arithmetic mean of individual wave heights, <H>.
    global_height : float
        Range of heights: max(H_i) - min(H_i). (Not a standard name; more a spread indicator.)
    max_height : float
        Maximum individual wave height, H_max.
    rms_height : float
        Root-mean-square (RMS) wave height: sqrt( mean(H_i^2) ).
    deviation_height : float
        Standard deviation of individual heights around the mean, std(H_i).
    one_third_mean_height : float
        Mean of the highest one-third of waves, H_1/3 (also called significant wave height H_s).
    one_tenth_mean_height : float
        Mean of the highest one-tenth of waves, H_1/10 (often used as a more “extreme” wave height).
    mean_period : float
        Arithmetic mean of individual wave periods, <T>.
    num_waves : int
        Number of waves used in the statistics.
    """

    # Convert inputs to NumPy arrays.
    wave_heights = np.asarray(wave_heights, dtype=float)
    wave_periods = np.asarray(wave_periods, dtype=float)

    # Basic checks
    if wave_heights.size == 0 or wave_periods.size == 0:
        # Return zeros when no waves are detected.
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0

    if wave_heights.size != wave_periods.size:
        raise ValueError("wave_heights and wave_periods must have the same length")

    num_waves = wave_heights.size

    # Global range of heights (not a standard parameter, but kept for compatibility)
    global_height = float(np.max(wave_heights) - np.min(wave_heights))

    # Maximum individual wave height
    max_height = float(np.max(wave_heights))

    # Mean individual height
    mean_height = float(np.mean(wave_heights))

    # RMS height: sqrt( mean(H_i^2) )
    rms_height = float(math.sqrt(np.mean(wave_heights**2)))

    # Standard deviation of heights
    deviation_height = float(np.std(wave_heights))

    # Sort heights descending (largest waves first)
    sorted_heights = np.sort(wave_heights)[::-1]

    # Highest third: H_1/3 (significant wave height)
    # Include at least one wave when the record is nonempty.
    one_third_count = max(1, num_waves // 3)  # N/3 largest waves
    one_third_mean_height = float(np.mean(sorted_heights[:one_third_count]))

    # Highest tenth: H_1/10 (more extreme indicator)
    one_tenth_count = max(1, num_waves // 10)  # N/10 largest waves
    one_tenth_mean_height = float(np.mean(sorted_heights[:one_tenth_count]))

    # Mean period
    mean_period = float(np.mean(wave_periods))

    return (
        mean_height,
        global_height,
        max_height,
        rms_height,
        deviation_height,
        one_third_mean_height,
        one_tenth_mean_height,
        mean_period,
        num_waves,
    )
