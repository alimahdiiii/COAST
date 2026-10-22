from __future__ import annotations
from pathlib import Path
import re
import subprocess


def _time_from_name(p: Path) -> float:
    # filename: freeSurface_on_stl_t_56p4.png  -> 56.4
    m = re.search(r"_t_([0-9]+p[0-9]+|[0-9]+)\.png$", p.name)
    if not m:
        return float("inf")
    return float(m.group(1).replace("p", "."))


def make_mp4_from_frames(
    figures_dir: str | Path,
    *,
    pattern: str = "freeSurface_on_stl_t_*.png",
    out_mp4: str | Path = "freeSurface_on_stl.mp4",
    fps: int = 15,
) -> Path:
    """
    Uses ffmpeg to build an mp4 from PNG frames.
    """
    figures_dir = Path(figures_dir)
    out_mp4 = Path(out_mp4)

    frames = sorted(figures_dir.glob(pattern), key=_time_from_name)
    if not frames:
        raise FileNotFoundError(f"No frames found in {figures_dir} with pattern {pattern}")

    # Write an ffmpeg file list for the ordered frames.
    lst = figures_dir / "_frames.txt"
    with lst.open("w", encoding="utf-8") as f:
        for p in frames:
            # ffmpeg concat expects: file 'path'
            f.write(f"file '{p.name}'\n")

    # ffmpeg concat -> mp4
    cmd = [
        "ffmpeg", "-y",
        "-r", str(fps),
        "-f", "concat", "-safe", "0",
        "-i", str(lst),
        "-vf",  "crop=trunc(iw/2)*2:trunc(ih/2)*2,format=yuv420p",
        "-movflags", "+faststart",
        str(out_mp4),
    ]

    # run inside figures_dir so paths are relative
    subprocess.run(cmd, cwd=str(figures_dir), check=True)

    # cleanup list file
    try:
        lst.unlink()
    except Exception:
        pass

    return out_mp4

