"""Small utilities."""

from typing import Sequence
import numpy as np

def find_closest_index(input_list, target_value):
    """
    Find the index of the closest value in the list to the target value.
    """
    array = np.asarray(input_list)
    index = (np.abs(array - target_value)).argmin()
    return index
