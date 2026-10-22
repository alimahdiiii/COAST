"""
Building blocks for the analysis of free-surface elevation time series: reading,
transient trimming, zero-crossing wave detection, time-domain and spectral
statistics, and plotting.
"""
from .Report_wave_analysis import safe_x_label, compute_m_minus1,generate_wave_analysis_report
from .io import read_two_columns, read_targets_from_waveProperties, read_relaxation_and_depth
from .stationarity import auto_detect_stationary_start
from .waves import find_waves, wave_physical_properties
from .spectrum import (
    compute_wave_spectrum_welch,
    compute_nth_moment_of_spectrum,
    wave_spectral_properties,
    jonswap_spectrum,
)
from .stationarity import auto_detect_stationary_start, trim_transient
from .stats import calculate_pdf_cdf
from .utils import find_closest_index
from .reflection import (
    solve_dispersion,
    mansard_funke_separation,
    suggest_reflection_gauges,
)
from .plotting import (
    plot_wave_periods,
    plot_freq_wave,
    plot_pdf,
    plot_cdf,
    plot_poe,
)

__all__ = [
    'read_two_columns',
    'auto_detect_stationary_start',
    'trim_transient',
    'find_waves',
    'wave_physical_properties',
    'compute_wave_spectrum_welch',
    'compute_nth_moment_of_spectrum',
    'wave_spectral_properties',
    'jonswap_spectrum',
    'calculate_pdf_cdf',
    'find_closest_index',
    'plot_wave_periods',
    'plot_freq_wave',
    'plot_pdf',
    'plot_cdf',
    'plot_poe',
    'read_relaxation_and_depth',
    'solve_dispersion',
    'mansard_funke_separation',
    'suggest_reflection_gauges',
]
