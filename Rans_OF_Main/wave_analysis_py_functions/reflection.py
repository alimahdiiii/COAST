"""Wave reflection analysis — Mansard & Funke (1980) 3-gauge method."""

from __future__ import annotations

import numpy as np
from scipy.integrate import trapezoid


def solve_dispersion(
    omega: float,
    h: float,
    g: float = 9.81,
    tol: float = 1e-10,
    max_iter: int = 50,
) -> float:
    """Solve omega^2 = g*k*tanh(k*h) for wavenumber k (Newton-Raphson).

    Parameters
    ----------
    omega : float
        Angular frequency [rad/s].
    h : float
        Water depth [m].
    g : float
        Gravity [m/s^2].

    Returns
    -------
    k : float
        Wavenumber [rad/m].
    """
    if omega <= 0 or h <= 0:
        return 0.0
    # Deep-water initial guess
    k = omega ** 2 / g
    for _ in range(max_iter):
        tanh_kh = np.tanh(k * h)
        f_val = omega ** 2 - g * k * tanh_kh
        f_der = -g * (tanh_kh + k * h * (1.0 - tanh_kh ** 2))
        if abs(f_der) < 1e-30:
            break
        dk = f_val / f_der
        k -= dk
        if k < 0:
            k = omega ** 2 / g
        if abs(dk) < tol:
            break
    return k


def _solve_dispersion_array(
    freq: np.ndarray,
    h: float,
    g: float = 9.81,
) -> np.ndarray:
    """Solve dispersion relation for an array of frequencies [Hz]."""
    k = np.zeros_like(freq, dtype=float)
    for i, f in enumerate(freq):
        if f > 0:
            k[i] = solve_dispersion(2.0 * np.pi * f, h, g)
    return k


def mansard_funke_separation(
    eta1: np.ndarray,
    eta2: np.ndarray,
    eta3: np.ndarray,
    dt: float,
    x1: float,
    x2: float,
    x3: float,
    h: float,
    g: float = 9.81,
    f_min: float = 0.0,
    f_max: float | None = None,
) -> dict:
    """Separate incident/reflected waves using the Mansard & Funke (1980) method.

    Parameters
    ----------
    eta1, eta2, eta3 : np.ndarray
        Free-surface elevation at gauges 1, 2, 3 (same length).
    dt : float
        Sampling interval [s].
    x1, x2, x3 : float
        Gauge x-positions [m].
    h : float
        Water depth [m].
    f_min, f_max : float | None
        Frequency band [Hz]; components outside are zeroed.

    Returns
    -------
    dict with keys: freq, S_inc, S_ref, Hm0_inc, Hm0_ref, Kr,
                    eta_inc, eta_ref, A_inc, A_ref
    """
    eta1 = np.asarray(eta1, dtype=float)
    eta2 = np.asarray(eta2, dtype=float)
    eta3 = np.asarray(eta3, dtype=float)
    N = len(eta1)

    # FFT (real-valued signals → rfft)
    E1 = np.fft.rfft(eta1)
    E2 = np.fft.rfft(eta2)
    E3 = np.fft.rfft(eta3)
    freq = np.fft.rfftfreq(N, dt)
    Nf = len(freq)
    df = freq[1] - freq[0] if Nf > 1 else 1.0

    # Gauge spacings relative to gauge 1
    dx12 = x2 - x1
    dx13 = x3 - x1

    # Solve dispersion for all frequencies
    k_arr = _solve_dispersion_array(freq, h, g)

    # Output arrays
    A_inc = np.zeros(Nf, dtype=complex)
    A_ref = np.zeros(Nf, dtype=complex)

    for j in range(Nf):
        f = freq[j]
        if f <= 0:
            continue
        if f < f_min:
            continue
        if f_max is not None and f > f_max:
            continue

        k = k_arr[j]
        if k <= 0:
            continue

        # Transfer matrix H (3 x 2)
        # eta_j = A_I * exp(-i*k*(xj-x1)) + A_R * exp(+i*k*(xj-x1))
        # Gauge 1 is reference (xj - x1 = 0, dx12, dx13)
        H = np.array([
            [1.0, 1.0],
            [np.exp(-1j * k * dx12), np.exp(1j * k * dx12)],
            [np.exp(-1j * k * dx13), np.exp(1j * k * dx13)],
        ])

        b = np.array([E1[j], E2[j], E3[j]])

        # Least-squares: (H^H H)^{-1} H^H b
        HH = H.conj().T @ H
        det_HH = np.linalg.det(HH)

        # Skip near-singular frequencies (kΔ ≈ nπ)
        if abs(det_HH) < 1e-12:
            continue

        x_sol = np.linalg.solve(HH, H.conj().T @ b)
        A_inc[j] = x_sol[0]
        A_ref[j] = x_sol[1]

    # Incident and reflected spectra (one-sided PSD)
    # |A|^2 scaled to spectral density: S = 2*|A|^2 / (N^2 * df)
    scale = 2.0 / (N ** 2 * df) if df > 0 else 0.0
    S_inc = scale * np.abs(A_inc) ** 2
    S_ref = scale * np.abs(A_ref) ** 2
    # DC and Nyquist not doubled
    S_inc[0] *= 0.5
    S_ref[0] *= 0.5
    if N % 2 == 0:
        S_inc[-1] *= 0.5
        S_ref[-1] *= 0.5

    # Spectral moments → Hm0
    m0_inc = float(trapezoid(S_inc, freq))
    m0_ref = float(trapezoid(S_ref, freq))
    Hm0_inc = 4.0 * np.sqrt(max(m0_inc, 0.0))
    Hm0_ref = 4.0 * np.sqrt(max(m0_ref, 0.0))
    Kr = Hm0_ref / Hm0_inc if Hm0_inc > 0 else 0.0

    # Reconstruct time series at gauge 1 (reference position)
    E_inc_ts = np.zeros(Nf, dtype=complex)
    E_ref_ts = np.zeros(Nf, dtype=complex)
    E_inc_ts[:] = A_inc  # at x1 (reference), phase factor is 1
    E_ref_ts[:] = A_ref

    eta_inc = np.fft.irfft(E_inc_ts, n=N)
    eta_ref = np.fft.irfft(E_ref_ts, n=N)

    return {
        "freq": freq,
        "S_inc": S_inc,
        "S_ref": S_ref,
        "Hm0_inc": Hm0_inc,
        "Hm0_ref": Hm0_ref,
        "Kr": Kr,
        "eta_inc": eta_inc,
        "eta_ref": eta_ref,
        "A_inc": A_inc,
        "A_ref": A_ref,
    }


def suggest_reflection_gauges(
    relax_end_x: float,
    Tp: float,
    h: float,
    g: float = 9.81,
    margin: float = 0.05,
) -> tuple[float, float, float]:
    """Suggest 3 gauge positions for the Mansard & Funke method.

    Gauge 1: just after the inlet relaxation zone end.
    Gauge 2: Gauge 1 + L/4
    Gauge 3: Gauge 2 + L/10

    Returns (x1, x2, x3).
    """
    omega_p = 2.0 * np.pi / Tp
    k_p = solve_dispersion(omega_p, h, g)
    L = 2.0 * np.pi / k_p if k_p > 0 else 0.0

    x1 = relax_end_x + margin
    x2 = x1 + L / 4.0
    x3 = x2 + L / 10.0
    return x1, x2, x3
