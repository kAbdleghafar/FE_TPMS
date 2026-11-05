"""Run CalculiX (open-source, Abaqus-compatible input) and read its results.

CalculiX serves two roles: an independent open-source reference solver for
verification, and a stand-in to dry-run decks before they go to Abaqus.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import time
from typing import Optional

import numpy as np


def run_ccx(inp_path: str, ccx: Optional[str] = None, threads: int = 4, timeout: float = 3600) -> dict:
    """Run ``ccx`` on an input deck. Returns paths and wall time."""
    exe = ccx or shutil.which("ccx") or shutil.which("ccx_2.21") or shutil.which("ccx_2.22")
    if exe is None:
        raise FileNotFoundError("CalculiX 'ccx' not found (apt install calculix-ccx / conda install calculix)")
    wd = os.path.dirname(os.path.abspath(inp_path))
    job = os.path.splitext(os.path.basename(inp_path))[0]
    env = dict(os.environ, OMP_NUM_THREADS=str(threads), CCX_NPROC_EQUATION_SOLVER=str(threads))
    t0 = time.perf_counter()
    p = subprocess.run([exe, "-i", job], cwd=wd, env=env, capture_output=True, text=True, timeout=timeout)
    dt = time.perf_counter() - t0
    log = p.stdout + p.stderr
    with open(os.path.join(wd, job + ".log"), "w") as fh:
        fh.write(log)
    if p.returncode != 0 or "*ERROR" in log:
        raise RuntimeError(f"ccx failed (see {job}.log):\n" + "\n".join(log.splitlines()[-15:]))
    return dict(frd=os.path.join(wd, job + ".frd"), dat=os.path.join(wd, job + ".dat"),
                seconds=dt, log=log)


def read_frd(path: str) -> list:
    """Parse an ASCII .frd file.

    Returns a list of result blocks ``{"name", "step", "components", "nodes",
    "values"}`` in file order; nodes are 1-based ids, values (n, ncomp).
    """
    blocks = []
    cur = None
    step = 0
    with open(path) as fh:
        for line in fh:
            if line.startswith("    1PSTEP"):
                step = int(line[10:].split()[1]) if len(line[10:].split()) > 1 else step + 1
            elif line.startswith(" -4"):
                cur = dict(name=line[5:13].strip(), step=step, components=[], nodes=[], rows=[])
            elif cur is not None and line.startswith(" -5"):
                cur["components"].append(line[5:13].strip())
            elif cur is not None and line.startswith(" -1"):
                nid = int(line[3:13])
                body = line[13:].rstrip("\n")
                vals = [float(body[i:i + 12]) for i in range(0, len(body) - 11, 12)]
                cur["nodes"].append(nid)
                cur["rows"].append(vals)
            elif cur is not None and line.startswith(" -3"):
                comps = [c for c in cur["components"] if c != "ALL"]
                rows = [r[:len(comps)] for r in cur["rows"]]
                blocks.append(dict(name=cur["name"], step=cur["step"], components=comps,
                                   nodes=np.asarray(cur["nodes"]), values=np.asarray(rows, float)))
                cur = None
    return blocks


def frd_field(blocks: list, name: str, n_nodes: int, which: int = -1) -> np.ndarray:
    """Nodal array (n_nodes, ncomp) of result ``name`` (e.g. 'DISP', 'STRESS')
    from the ``which``-th occurrence (default: last step). Missing nodes = NaN."""
    sel = [b for b in blocks if b["name"] == name]
    if not sel:
        raise KeyError(f"{name} not in frd (have {sorted({b['name'] for b in blocks})})")
    b = sel[which]
    out = np.full((n_nodes, b["values"].shape[1]), np.nan)
    out[b["nodes"] - 1] = b["values"]
    if name == "STRESS":  # SXX SYY SZZ SXY SYZ SZX -> xx yy zz yz xz xy
        out = out[:, [0, 1, 2, 4, 5, 3]]
    return out


# CalculiX numbers the C3D8 Gauss points lexicographically (xi fastest), tpmsfem
# in element-node order -> swap points 3/4 and 7/8 (self-inverse permutation).
CCX_IP_ORDER = {8: [0, 1, 3, 2, 4, 5, 7, 6]}


def read_dat_stress(path: str, n_elem: int, nq: int, which: int = -1) -> np.ndarray:
    """Integration-point stresses from ``*EL PRINT`` in the .dat file,
    returned as (n_elem, nq, 6) in tpmsfem component order [xx,yy,zz,yz,xz,xy]
    and tpmsfem integration-point order (verified for C3D8; for C3D10 compare
    element averages unless the point order has been verified)."""
    sections = []
    cur = None
    with open(path) as fh:
        for line in fh:
            if line.lstrip().startswith("stresses (elem, integ.pnt."):
                cur = []
                sections.append(cur)
                continue
            if cur is None:
                continue
            parts = line.split()
            if len(parts) == 8:
                try:
                    cur.append([float(v) for v in parts])
                except ValueError:
                    cur = None
            elif parts and cur:
                cur = None
    if not sections:
        raise KeyError("no stress section in .dat (add *EL PRINT, ELSET=EALL / S)")
    a = np.asarray(sections[which])
    out = np.full((n_elem, nq, 6), np.nan)
    e = a[:, 0].astype(int) - 1
    q = a[:, 1].astype(int) - 1
    out[e, q] = a[:, [2, 3, 4, 7, 6, 5]]        # sxx syy szz sxy sxz syz -> xx yy zz yz xz xy
    if nq in CCX_IP_ORDER:
        out = out[:, CCX_IP_ORDER[nq]]
    return out
