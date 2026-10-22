"""Statistical distributions (PDF/CDF/PoE)."""

from __future__ import annotations

from typing import Sequence, Tuple
import math
import numpy as np
import pandas as pd

def calculate_pdf_cdf(data, number_of_bins, analyze_fluctuations=False):
    """
    Compute the Probability Density Function (PDF) and Cumulative Distribution Function (CDF)
    of the given data. Optionally analyze only the fluctuations around the mean.

    Parameters:
    data (array-like): The data to be analyzed, such as wave height or elevation.
    number_of_bins (int): The number of bins to use for the PDF.
    analyze_fluctuations (bool): If True, analyze the fluctuations around the mean value of the data.

    Returns:
    bin_centers (numpy array): The positions of the bin centers.
    pdf (numpy array): The PDF of the data.
    cdf (numpy array): The CDF of the data.
    poe (numpy array): The Probability of Exceedance (PoE) of the data.
    poe_Rayleigh (numpy array): The Rayleigh distribution fit for the data.
    """

    # Subtract the mean if requested.
    if analyze_fluctuations:
        mean_value = np.mean(data)
        fluctuating_data = [x - mean_value for x in data]
    else:
        fluctuating_data = data

    # Mean square for the Rayleigh fit
    squared_data = [value ** 2 for value in fluctuating_data]
    mean_squared = np.mean(squared_data)

    # Estimate the PDF with a histogram.
    pdf_values, bin_edges = np.histogram(fluctuating_data, bins=number_of_bins, density=True)

    # Bin width and centers
    bin_width = bin_edges[1] - bin_edges[0]
    bin_centers = (bin_edges[1:] + bin_edges[:-1]) * 0.5

    # Integrate the PDF to get the CDF.
    cdf_values = np.cumsum(pdf_values) * bin_width

    # Integrate from the upper tail to get the exceedance probability.
    poe_values = np.cumsum(pdf_values[::-1])[::-1] * bin_width

    # Rayleigh exceedance probability for comparison
    poe_Rayleigh = [math.exp(-center**2 / mean_squared) for center in bin_centers]

    return bin_centers, pdf_values, cdf_values, poe_values, poe_Rayleigh
