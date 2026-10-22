"""Spectral (frequency-domain) analysis: Welch PSD, moments, and target spectra."""

from __future__ import annotations

from typing import Sequence, Tuple, List
import math
import numpy as np
from scipy.integrate import trapezoid
import scipy.signal as signal
import pandas as pd

def compute_wave_spectrum_welch(time, data, nBlocks):  
    """
    Compute the wave power spectral density (PSD)  with welch method
    """

    # Time step and sampling frequency.
    time_step = time[1] - time[0]
    sampling_frequency = 1 / time_step

    # Compute PSD
    frequencies, energy_spectrum = signal.welch(data, sampling_frequency, nperseg=nBlocks) 


    return (frequencies, energy_spectrum)

def compute_nth_moment_of_spectrum(frequencies, spectrum, moment_order):
    """
    Compute the nth moment of an energy spectrum. The nth moment is a statistical property
    of the spectrum that is used to describe its shape concerning the frequency axis.

    Parameters:
    frequencies (array-like): The frequency axis of the spectrum, assumed to be evenly spaced.
    spectrum (array-like): The energy spectrum values corresponding to each frequency.
    moment_order (int): The order of the moment to compute. A moment_order of 0 corresponds
                        to the total energy in the spectrum.

    Returns:
    nth_moment (float): The computed nth moment of the energy spectrum.
    """

    # Frequency spacing
    frequency_step = frequencies[1] - frequencies[0]

    # Start with a zero moment.
    nth_moment = 0.0

    # The zeroth moment is the area under the spectrum.
    if moment_order == 0:
        nth_moment = frequency_step * sum(spectrum)
    else:
        # Integrate higher-order moments.
        for i, frequency in enumerate(frequencies):
            nth_moment += (frequency ** moment_order) * spectrum[i] * frequency_step

    return nth_moment

def wave_spectral_properties(freq, spec):
    """
    Calculates the spectral properties of waves.

    This function computes various spectral properties of a wave given its frequency 
    and spectral data. These properties include the nth moments, T01 (average period),
    Hm0 (significant wave height), maximum energy frequency, and maximum energy period.

    Args:
    freq (list): Frequency data.
    spec (list): Spectral data.

    Returns:
    tuple: nth moments, T01, Hm0, max energy frequency, max energy period.
    """

    # Find the peak frequency.
    max_energy_freq = freq[np.argmax(spec)]

    # Convert peak frequency to period.
    # Return zero if the peak frequency is zero.
    max_energy_period = 1 / max_energy_freq if max_energy_freq != 0 else 0

    # Compute moments m0, m1, and m2.
    nth_moments = [compute_nth_moment_of_spectrum(freq, spec, i) for i in range(3)]

    # Mean spectral period T01 = m0 / m1
    T01 = nth_moments[0] / nth_moments[1] if nth_moments[1] != 0 else 0

    # Mean zero crossing period
    T02 = math.sqrt(nth_moments[0] / nth_moments[2]) if nth_moments[2] != 0 else 0

    # Significant wave height Hm0 = 4 * sqrt(m0)
    Hm0 = 4.000 * math.sqrt(nth_moments[0])

    # Root-mean-square wave height
    Hrms = math.sqrt(8*nth_moments[0])


    # Return spectral properties.
    return nth_moments, T01, T02, Hm0, Hrms, max_energy_freq, max_energy_period

def jonswap_spectrum(freq, Hs, Tp, gamma=3.3, g=9.81):
    """
    Compute the JONSWAP spectrum on a given frequency grid.

    Parameters
    ----------
    freq : array-like
        Frequency array [Hz].
    Hs : float
        Significant wave height [m].
    Tp : float
        Peak period [s].
    gamma : float, optional
        Peak enhancement factor (default 3.3).
    g : float, optional
        Gravity acceleration [m/s^2].

    Returns
    -------
    Sj : numpy array
        JONSWAP spectral density S(f) [m^2/Hz] evaluated at 'freq',
        normalized so that Hm0 = Hs (i.e. 4*sqrt(m0) = Hs).
    """

    freq = np.asarray(freq)
    Sj = np.zeros_like(freq)

    # Avoid division by zero
    positive = freq > 0
    f = freq[positive]

    # Peak frequency
    fp = 1.0 / Tp

    # Sigma parameter (width of peak)
    sigma = np.where(f <= fp, 0.07, 0.09)

    # Phillips constant (initial guess, will rescale later)
    alpha_PM = 0.0081

    # Pierson–Moskowitz base spectrum
    # S_PM(f) = alpha g^2 (2π)^-4 f^-5 exp[-1.25 (fp/f)^4]
    S_PM = (alpha_PM * g**2 * (2*np.pi)**-4 *
            f**-5 * np.exp(-1.25 * (fp / f)**4))

    # JONSWAP peak enhancement factor
    r = np.exp(- (f - fp)**2 / (2.0 * sigma**2 * fp**2))
    G = gamma**r

    Sj_pos = S_PM * G

    # Put back into full array
    Sj[positive] = Sj_pos

    # Normalise so that Hm0 = Hs
    # m0 = ∫ S(f) df  ≈ trapz(S, f)
    m0_num = trapezoid(Sj, freq)
    target_m0 = Hs**2 / 16.0          # because Hs = 4 * sqrt(m0)
    scale = target_m0 / m0_num
    Sj *= scale

    return Sj
