# tmux_run_case.py
from pathlib import Path
import subprocess
import shlex
import os
import shutil

def _solver_name(use_porous: bool) -> str:
    return "porousWaveFoam" if use_porous else "waveFoam"


def _porous_copy_block(use_porous: bool) -> str:
    if not use_porous:
        return ""
    return (
        "\n# Copy porosityZones to each processor directory\n"
        "for p in processor*/; do\n"
        "    cp constant/porosityZones \"$p/constant/\"\n"
        "done\n"
    )


def _first_existing(*candidates):
    for c in candidates:
        if c and Path(c).is_file():
            return str(Path(c))
    return None


def _env_source_block(solver: str) -> str:
    """
    Shell lines that load OpenFOAM + waves2Foam inside job_Run.sh.

    Each bashrc is resolved in this order:
      OpenFOAM:   $COAST_OPENFOAM_BASHRC, then $WM_PROJECT_DIR/etc/bashrc
      waves2Foam: $COAST_WAVES2FOAM_BASHRC, then $WAVES_DIR/bin/bashrc,
                  then $WM_PROJECT_USER_DIR/applications/utilities/waves2Foam/bin/bashrc
    If nothing is found, the job relies on the environment inherited from tmux
    and stops early with a clear message if the solver is not on PATH.
    """
    env = os.environ
    of_rc = _first_existing(
        env.get("COAST_OPENFOAM_BASHRC"),
        env.get("WM_PROJECT_DIR") and f"{env['WM_PROJECT_DIR']}/etc/bashrc",
    )
    w2f_rc = _first_existing(
        env.get("COAST_WAVES2FOAM_BASHRC"),
        env.get("WAVES_DIR") and f"{env['WAVES_DIR']}/bin/bashrc",
        env.get("WM_PROJECT_USER_DIR")
        and f"{env['WM_PROJECT_USER_DIR']}/applications/utilities/waves2Foam/bin/bashrc",
    )

    # Must come before `set -euo pipefail`: OpenFOAM's bashrc is not `set -u` safe.
    lines = []
    for rc in (of_rc, w2f_rc):
        if rc:
            lines.append(f"source {shlex.quote(rc)}")
    lines.append("set -euo pipefail")
    lines.append(
        f'command -v {solver} >/dev/null 2>&1 || {{ echo "❌ {solver} not found on PATH. '
        f'Source OpenFOAM and waves2Foam in ~/.bashrc, or set COAST_OPENFOAM_BASHRC / '
        f'COAST_WAVES2FOAM_BASHRC before starting COAST." >&2; exit 1; }}'
    )
    return "\n".join(lines)


def write_job_script(case_dir, notebook_dir, new_patch, patch_name, n_procs,
                     use_porous=False, patch_names=None):
    """
    Generate job_Run.sh inside case_dir (without launching it).
    Call this during 'Build Case' so the script is ready before launch.

    Parameters
    ----------
    patch_names : list[str] | None
        When provided, overrides ``new_patch`` / ``patch_name`` (legacy single
        patch). Each name gets its own BC copy and force function object.
    """
    case_dir     = Path(case_dir).resolve()
    notebook_dir = Path(notebook_dir).resolve()
    REPO_ROOT    = Path(__file__).resolve().parents[1]

    job = case_dir / "job_Run.sh"

    # PYTHONPATH
    pythonpath = f"{notebook_dir}:{REPO_ROOT}:{REPO_ROOT}/python_tools"

    # Resolve multi-patch vs legacy single-patch
    if patch_names is None and new_patch and patch_name:
        patch_names = [patch_name]

    # Patch logic
    if patch_names:
        # Build Python lines for each patch
        bc_lines = []
        for pn in patch_names:
            bc_lines.append(f'add_patch_by_copying_bc(".", new_patch="{pn}", src_patch="coast")')
        bc_code = "\n".join(bc_lines)
        names_list = ", ".join(f'"{pn}"' for pn in patch_names)

        patch_block = f"""
python3 - <<'PY'
from foam_case_tools import add_patch_by_copying_bc, update_forces_patches_multi
{bc_code}
update_forces_patches_multi("./system/controlDict", [{names_list}])
PY

# ALWAYS build sets (includes overtopping, and force set too when new_patch=True)
rm -rf constant/polyMesh/sets constant/polyMesh/faceZones constant/polyMesh/cellZones
topoSet > log.topoSet 2>&1
setsToZones -noFlipMap > log.setsToZones 2>&1

# ONLY create the boundary patch when requested
createPatch -overwrite > log.createPatch 2>&1
"""
    else:
        patch_block = """
python3 - <<'PY'
from foam_case_tools import remove_forces_function
remove_forces_function("./system/controlDict")
print("new_patch=False: forces function removed from controlDict")
PY

# ALWAYS build sets (overtopping set still needed even if no force patch)
rm -rf constant/polyMesh/sets constant/polyMesh/faceZones constant/polyMesh/cellZones
topoSet > log.topoSet 2>&1

setsToZones -noFlipMap > log.setsToZones 2>&1

foamToVTK -faceZones overtopping
"""

    # Write job_Run.sh
    job_text = f"""#!/bin/bash
{_env_source_block(_solver_name(use_porous))}
cd "$(dirname "$0")"

export PYTHONPATH="{pythonpath}"
export N_PROCS="{n_procs}"

foamCleanTutorials
blockMesh
surfaceFeatureExtract
decomposePar -force

mpirun -n {n_procs} snappyHexMesh -parallel -overwrite > log.snappy 2>&1
reconstructParMesh -constant
extrudeMesh

rm -rf 0
cp -r 0.orig 0
{patch_block}
setWaveParameters
setWaveField
setFields
decomposePar -force
{_porous_copy_block(use_porous)}
mpirun -n {n_procs} {_solver_name(use_porous)} -parallel > log.{_solver_name(use_porous)} 2>&1

mpirun -n {n_procs} postProcess -func writeCellCentres -parallel -time 0 > log.cellCentres 2>&1
mpirun -n {n_procs} postProcess -func writeCellVolumes -parallel -time 0 > log.cellVolumes 2>&1

echo "✅ job_Run.sh finished OK"
"""
    job.write_text(job_text)
    job.chmod(job.stat().st_mode | 0o111)
    print(f"✅ job_Run.sh written in {case_dir}")
    return job


def launch_tmux_case(case_dir, notebook_dir, new_patch, patch_name, n_procs,
                     use_porous=False):
    """
    Launch job_Run.sh (which must already exist) in a tmux session.

    tmux resolution order:
      1) TMUX_BIN environment variable
      2) repo-bundled tmux: tools/tmux/bin/tmux
      3) system tmux (which tmux)
    """
    case_dir = Path(case_dir).resolve()
    REPO_ROOT = Path(__file__).resolve().parents[1]

    job = case_dir / "job_Run.sh"
    if not job.exists():
        raise RuntimeError(
            f"job_Run.sh not found in {case_dir}.\n"
            "Run 'Build Case' first to generate it."
        )

    # Resolve tmux binary
    tmux_env = os.environ.get("TMUX_BIN", "").strip()
    tmux_repo = REPO_ROOT / "tools" / "tmux" / "bin" / "tmux"

    if tmux_env:
        tmux_bin = tmux_env
    elif tmux_repo.exists():
        tmux_bin = str(tmux_repo)
    else:
        tmux_sys = shutil.which("tmux")
        if tmux_sys is None:
            raise RuntimeError(
                "tmux not found.\n"
                "Run: ./tools/install_tmux.sh\n"
                "or install tmux system-wide\n"
                "or set TMUX_BIN=/path/to/tmux"
            )
        tmux_bin = tmux_sys

    # Session / socket bookkeeping
    session = case_dir.name
    socket  = Path(f"/tmp/tmux_{case_dir.name}.sock")

    (case_dir / "TMUX_SESSION.txt").write_text(session + "\n")
    (case_dir / "TMUX_SOCKET.txt").write_text(str(socket) + "\n")

    # Kill previous session
    kill_cmd = f'{tmux_bin} -S {shlex.quote(str(socket))} kill-server'
    subprocess.run(
        ["bash", "-lc", kill_cmd],
        cwd=str(case_dir),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    # Start tmux session
    inner = (
        f"cd {shlex.quote(str(case_dir))} && "
        "echo RUNNING > TMUX_STATUS.txt && "
        f"./{job.name} > TMUX_OUT.log 2>&1; "
        "rc=$?; echo FINISHED_$rc > TMUX_STATUS.txt; "
        "exec bash"
    )

    start_cmd = (
        f'{tmux_bin} -S {shlex.quote(str(socket))} '
        f'new-session -d -s {shlex.quote(session)} {shlex.quote(inner)}'
    )

    res = subprocess.run(
        ["bash", "-lc", start_cmd],
        cwd=str(case_dir),
        capture_output=True,
        text=True,
    )

    if res.returncode != 0:
        print("❌ tmux start failed")
        print(res.stderr)
        raise SystemExit(1)

    print("✅ Started tmux session")
    print("SESSION:", session)
    print("SOCKET :", socket)
    print("List  :", f"{tmux_bin} -S {socket} ls")
    print("Attach:", f"{tmux_bin} -S {socket} attach -t {session}")

