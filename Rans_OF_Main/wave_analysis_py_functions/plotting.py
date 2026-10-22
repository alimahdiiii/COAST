"""Plotting helpers (Matplotlib)."""

from __future__ import annotations

from typing import Sequence
import statistics
import numpy as np
import matplotlib.pyplot as plt

from .waves import find_waves
from .utils import find_closest_index

def plot_wave_periods(time, elevation, num_periods):
    """
    Plots the first 'num_periods' average periods of wave data.

    Args:
    time (list): Time data for the waves.
    elevation (list): Elevation data for the waves.
    num_periods (int): Number of average periods to plot.
    """

    # Calculate initial waves and average period
    Hwaves, Twaves, Hmin, Hmax, Tstart, Tend, tHmin, tHmax = find_waves(time, elevation)

    # Find the end of the first 'num_periods' average periods.
    ilp = find_closest_index(time, num_periods * np.average(Twaves))

    # Detect waves for the first 'num_periods' average periods
    Hwaves, Twaves, Hmin, Hmax, Tstart, Tend, tHmin, tHmax = find_waves(time[:ilp], elevation[:ilp])

    # Plotting
    plt.plot(time[:ilp], elevation[:ilp], '-', linewidth=2, color='indigo', zorder=20)
    plt.scatter(tHmax, Hmax, s=10, color='indigo', zorder=20)
    plt.scatter(tHmin, Hmin, s=10, color='hotpink', zorder=20)

    for i in range(len(Tstart)):
        # Plot period
        xline = [Tstart[i], Tend[i]]
        yline = [statistics.mean(elevation[:ilp]), statistics.mean(elevation[:ilp])]
        plt.plot(xline, yline, linewidth=5, alpha=0.5, zorder=30)

        # Plot crest
        w = Twaves[i] / 3
        xline = [tHmax[i] - w, tHmax[i] + w]
        yline = [Hmax[i], Hmax[i]]
        plt.plot(xline, yline, linewidth=1, color='indigo', zorder=20)

        # Plot trough
        xline = [tHmin[i] - w, tHmin[i] + w]
        yline = [Hmin[i], Hmin[i]]
        plt.plot(xline, yline, linewidth=1, color='hotpink', zorder=20)

    plt.xlabel('Time [s]', fontsize=14)
    plt.ylabel('Elevation [m]', fontsize=14)
    plt.title('Wave Detection')
    plt.grid(True)
    plt.tight_layout()

def plot_freq_wave(xf, yf, max_freq=-1, plot_line=True, plot_bar=False, title='Wave PSD'):
    """
    Plots the frequency wave data.

    Args:
    xf (list): X-axis data (frequency).
    yf (list): Y-axis data (spectral density).
    max_freq (float): Maximum frequency to plot. If -1, plots all data.
    plot_line (bool): If True, plots a line.
    plot_bar (bool): If True, plots bars.
    title (str): Title of the plot.
    """

    # Apply the frequency limit if provided.
    if max_freq > 0:
        xf, yf = zip(*[(x, y) for x, y in zip(xf, yf) if x <= max_freq])

    # Plot the data as a line or scatter plot
    if plot_line:
        plt.plot(xf, yf, '-', linewidth=2, color='indigo', zorder=20)
    else:
        plt.scatter(xf, yf, s=5, color='indigo', label='probes', zorder=20)

    # Draw vertical lines when bar is True.
    if plot_bar:
        plt.vlines(xf, 0, yf, color='indigo', linewidth=10)

    # Set plot labels and title
    plt.xlabel('f [Hz]', fontsize=14)
    plt.ylabel('Spectral Density [$m^2/Hz$]', fontsize=14)
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()

def plot_pdf(xpdf, pdf):
    """
    Plots the Probability Density Function (PDF) for wave heights.

    Args:
    xpdf (list): x-axis data.
    pdf (list): Probability density function data.
    output_dir (str): Directory for saving plots.
    output_name (str): Name for the output file.
    notes (str): Additional notes to add to the plot title.
    """
    dx = xpdf[1] - xpdf[0]
    plt.figure()
    plt.bar(xpdf, pdf, width=0.8 * dx, color='darkorange', edgecolor='indigo', alpha=0.5)
    plt.xlabel('H [m]', fontsize=14)
    plt.ylabel('Probability Density Function P($H$)', fontsize=14)
    plt.title(f'Wave Heights $H_i$ PDF')
    plt.tight_layout()

def plot_cdf(xpdf, cdf):
    """
    Plots the Cumulative Density Function (CDF) for wave heights.

    Args:
    xpdf (list): x-axis data.
    cdf (list): Cumulative density function data.
    output_dir (str): Directory for saving plots.
    output_name (str): Name for the output file.
    notes (str): Additional notes to add to the plot title.
    """
    dx = xpdf[1] - xpdf[0]
    plt.figure()
    plt.bar(xpdf, cdf, width=0.8 * dx, color='darkorange', edgecolor='indigo', alpha=0.5)
    plt.xlabel('H [m]', fontsize=14)
    plt.ylabel('Cumulative Function C($H$)', fontsize=14)
    plt.title(f'Cumulative Function $H_i$')
    plt.tight_layout()

def plot_poe(xpdf, poe, poe_Rayleigh, rayleigh_scale: bool = True):
    """
    Plots the Probability of Exceedence (POE) for wave heights.

    Args:
    xpdf (list): x-axis data.
    poe (list): Probability of exceedance data.
    poe_Rayleigh (list): Probability of exceedance according to Rayleigh theory.
    rayleigh_scale (bool): if True (default), the vertical axis is a Rayleigh
        probability axis: it is linear in sqrt(-ln Q) and labelled with Q, so the
        Rayleigh distribution Q = exp(-H^2/Hrms^2) is a straight line and any
        deviation of the data from it is visible directly. If False, linear Q axis.
    """
    xpdf = np.asarray(xpdf, float)
    poe = np.asarray(poe, float)
    poe_Rayleigh = np.asarray(poe_Rayleigh, float)
    plt.figure()
    if not rayleigh_scale:
        dx = xpdf[1] - xpdf[0]
        plt.bar(xpdf, poe, width=0.8 * dx, color='darkorange', edgecolor='indigo', alpha=0.5, label='wave data')
        plt.plot(xpdf, poe_Rayleigh, label='Rayleigh theory')
        plt.ylabel('Probability of Exceedence Q($H$)', fontsize=14)
    else:
        def ray(q):
            return np.sqrt(-np.log(q))
        ok = (poe > 0) & (poe < 1)
        okr = (poe_Rayleigh > 0) & (poe_Rayleigh < 1)
        plt.plot(xpdf[ok], ray(poe[ok]), 'o', color='darkorange', mec='indigo', ms=5, label='wave data')
        plt.plot(xpdf[okr], ray(poe_Rayleigh[okr]), '-', label='Rayleigh theory')
        ticks = np.array([0.9, 0.5, 0.2, 0.1, 0.05, 0.01, 0.001, 0.0001])
        plt.yticks(ray(ticks), [f'{100 * q:g}%' for q in ticks])
        plt.ylim(0, ray(1e-4))
        plt.grid(True, alpha=0.3)
        plt.ylabel('Probability of Exceedence Q($H$)\n(Rayleigh scale)', fontsize=12)
    plt.xlabel('H [m]', fontsize=14)
    plt.title(f'Probability of Exceedence of $H_i$')
    plt.legend(loc='upper left' if rayleigh_scale else 'upper right')
    plt.tight_layout()
