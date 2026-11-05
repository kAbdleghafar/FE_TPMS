"""Command line entry point: ``tpmsfem info | compare | voxelize``."""
from __future__ import annotations

import argparse
import sys


def main(argv=None):
    ap = argparse.ArgumentParser(prog="tpmsfem")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("info", help="backend / device information")
    c = sub.add_parser("compare", help="compare results with Abaqus/CalculiX (see tpmsfem.bench.compare)")
    c.add_argument("rest", nargs=argparse.REMAINDER)
    v = sub.add_parser("voxelize", help="voxel mesh of a TPMS block -> .vtu/.inp")
    v.add_argument("out")
    v.add_argument("--surface", default="gyroid")
    v.add_argument("--kind", default="sheet")
    v.add_argument("--cell", type=float, default=3.0)
    v.add_argument("--density", type=float, default=0.3)
    v.add_argument("--size", type=float, default=6.0)
    v.add_argument("--h", type=float, default=0.1)
    a = ap.parse_args(argv)

    if a.cmd == "info":
        from . import __version__
        from .backend import device_info

        print(f"tpmsfem {__version__}: {device_info()}")
    elif a.cmd == "compare":
        from .bench.compare import _main

        _main(a.rest)
    elif a.cmd == "voxelize":
        from .geometry import TPMS, box
        from .mesh import voxelize

        lat = TPMS(a.surface, cell_size=a.cell, kind=a.kind, density=a.density)
        m = voxelize(box((0, 0, 0), (a.size,) * 3) & lat, (0, 0, 0), (a.size,) * 3, a.h)
        m.save(a.out)
        print(m.summary(), "->", a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
