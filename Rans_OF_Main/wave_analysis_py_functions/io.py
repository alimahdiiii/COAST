"""I/O helpers."""

import fileinput
from typing import List, Tuple

def read_two_columns(file_path):
    """
    Read a 2-column text file while eliminating lines that begin with '#'.

    Args:
    file_path (str): The path to the text file.

    Returns:
    tuple: Two lists containing the data from the first and second columns, respectively.
    """
    col_one, col_two = [], []

    with fileinput.input(file_path) as file:
        for line in file:
            # Split the line into fields.
            words = line.strip().split()

            # Skip empty lines and lines starting with '#'
            if words and words[0] != '#':
                col_one.append(float(words[0]))
                col_two.append(float(words[1]))

    return col_one, col_two
    
    
from pathlib import Path
import re

def read_targets_from_waveProperties(case_dir="."):
    """
    Read Hs, Tp and (optionally) gamma from constant/waveProperties.input
    inside an OpenFOAM case.

    Parameters
    ----------
    case_dir : str or Path
        Path to the case directory (where 'constant' folder lives).

    Returns
    -------
    Hs_target : float
    Tp_target : float
    gamma_input : float or None
        If no gamma-like key is found, returns None.
    """
    case_dir = Path(case_dir)
    wp_candidates = [
        case_dir / "constant" / "waveProperties.input",
        case_dir / "constant" / "waveProperties",
    ]

    wp = next((p for p in wp_candidates if p.exists()), None)
    if wp is None:
        raise FileNotFoundError(
            f"waveProperties file not found. Tried: {[str(p) for p in wp_candidates]}"
        )

    text = wp.read_text()

    def get_scalar(key):
        m = re.search(
            rf'^\s*{re.escape(key)}\s+([^;]+);',
            text,
            flags=re.MULTILINE
        )
        if not m:
            return None
        try:
            return float(m.group(1))
        except ValueError:
            return None

    Hs = get_scalar("Hs")
    Tp = get_scalar("Tp")

    if Hs is None or Tp is None:
        raise ValueError("Could not read Hs and/or Tp from waveProperties.input")

    # Try a few possible names for gamma
    gamma = None
    for key in ("gamma", "gammaInput", "peakEnhancementFactor", "enhancementFactor"):
        gamma = get_scalar(key)
        if gamma is not None:
            break

    return Hs, Tp, gamma


def read_relaxation_and_depth(case_dir="."):
    """Read inlet relaxation zone end position, water depth, and sea level
    from constant/waveProperties.input.

    Returns
    -------
    dict with keys 'relax_end_x', 'depth', 'seaLevel'.
    """
    case_dir = Path(case_dir)
    wp_candidates = [
        case_dir / "constant" / "waveProperties.input",
        case_dir / "constant" / "waveProperties",
    ]
    wp = next((p for p in wp_candidates if p.exists()), None)
    if wp is None:
        raise FileNotFoundError(
            f"waveProperties file not found. Tried: {[str(p) for p in wp_candidates]}"
        )

    text = wp.read_text()

    def get_scalar(key):
        m = re.search(
            rf'^\s*{re.escape(key)}\s+([^;]+);',
            text, flags=re.MULTILINE,
        )
        if not m:
            return None
        try:
            return float(m.group(1))
        except ValueError:
            return None

    depth = get_scalar("depth")
    if depth is None:
        raise ValueError("Could not read 'depth' from waveProperties")

    seaLevel = get_scalar("seaLevel") or 0.0

    # Parse endX from relaxationZone block: endX  (-3.149 0.0 0.7);
    m_endx = re.search(
        r'endX\s*\(\s*([^\s)]+)',
        text,
    )
    if m_endx is None:
        raise ValueError("Could not read relaxation zone 'endX' from waveProperties")
    relax_end_x = float(m_endx.group(1))

    return {
        "relax_end_x": relax_end_x,
        "depth": depth,
        "seaLevel": seaLevel,
    }

