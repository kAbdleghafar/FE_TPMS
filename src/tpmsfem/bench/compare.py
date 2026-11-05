"""Compare tpmsfem results with a reference solver (CalculiX, Abaqus, ...).

Reference data formats:

* CalculiX: ``.frd`` (nodal U) + ``.dat`` (integration-point S via *EL PRINT)
* Abaqus:   CSVs written by ``tools/abaqus_odb_export.py`` (run with
  ``abaqus python``): ``<prefix>_U.csv`` and ``<prefix>_S.csv``
* anything else: two CSVs in the same layout.

Command line::

    python -m tpmsfem.bench.compare ours.npz --abaqus job_last   [--json out.json]
    python -m tpmsfem.bench.compare ours.npz --ccx job.frd job.dat
"""
from __future__ import annotations

import argparse
import json

import numpy as np

from ..fem import constitutive as C
from .metrics import percentile_diff, rel_l2, rel_max


def save_ours(model, path: str):
    """Save the fields used for comparison (U nodal, S at integration points)."""
    np.savez_compressed(path, U=model.displacement(), S=model.stress(),
                        active=model.active_host, cell_type=model.mesh.cell_type,
                        RF=model.reactions())


def load_abaqus_csv(prefix: str, n_nodes: int, n_elem: int):
    """Read ``<prefix>_U.csv`` (node,u1,u2,u3) and ``<prefix>_S.csv``
    (elem,ip,s11,s22,s33,s12,s13,s23); returns U (n,3), S_elem_mean (ne,6)."""
    u = np.loadtxt(prefix + "_U.csv", delimiter=",", skiprows=1)
    U = np.full((n_nodes, 3), np.nan)
    U[u[:, 0].astype(int) - 1] = u[:, 1:4]
    s = np.loadtxt(prefix + "_S.csv", delimiter=",", skiprows=1)
    e = s[:, 0].astype(int) - 1
    vals = s[:, [2, 3, 4, 7, 6, 5]]               # 11 22 33 12 13 23 -> xx yy zz yz xz xy
    S = np.zeros((n_elem, 6))
    cnt = np.bincount(e, minlength=n_elem).astype(float)
    for k in range(6):
        S[:, k] = np.bincount(e, weights=vals[:, k], minlength=n_elem)
    with np.errstate(invalid="ignore"):
        S /= cnt[:, None]
    S[cnt == 0] = np.nan
    return U, S


def compare(U, S_elem, U_ref, S_elem_ref, active=None) -> dict:
    """Standard metric set (element-mean stresses are IP-order independent)."""
    if active is not None:
        S_elem, S_elem_ref = S_elem[active], S_elem_ref[active]
    vm, vm_ref = np.asarray(C.von_mises(S_elem)), np.asarray(C.von_mises(S_elem_ref))
    umag, umag_ref = np.linalg.norm(U, axis=1), np.linalg.norm(U_ref, axis=1)
    return dict(
        U_rel_l2=rel_l2(U, U_ref), U_rel_max=rel_max(U, U_ref),
        U_max=float(np.nanmax(umag)), U_max_ref=float(np.nanmax(umag_ref)),
        S_rel_l2=rel_l2(S_elem, S_elem_ref),
        vonMises_rel_l2=rel_l2(vm, vm_ref),
        vonMises_percentiles=percentile_diff(vm, vm_ref),
    )


def _main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ours", help=".npz written by tpmsfem.bench.compare.save_ours")
    ap.add_argument("--abaqus", help="CSV prefix from tools/abaqus_odb_export.py")
    ap.add_argument("--ccx", nargs=2, metavar=("FRD", "DAT"))
    ap.add_argument("--json")
    a = ap.parse_args(argv)
    d = np.load(a.ours, allow_pickle=True)
    U, Sq, act = d["U"], d["S"], d["active"]
    S = Sq.mean(axis=1)
    if a.abaqus:
        U_ref, S_ref = load_abaqus_csv(a.abaqus, len(U), len(Sq))
    elif a.ccx:
        from ..io.calculix import frd_field, read_dat_stress, read_frd

        U_ref = frd_field(read_frd(a.ccx[0]), "DISP", len(U))
        S_ref = read_dat_stress(a.ccx[1], len(Sq), Sq.shape[1]).mean(axis=1)
    else:
        ap.error("give --abaqus or --ccx")
    res = compare(U, S, U_ref, S_ref, active=act)
    print(json.dumps(res, indent=1))
    if a.json:
        json.dump(res, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    _main()
