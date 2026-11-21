"""Export a keyhole run (tools/lbm/keyhole_runs/<tag>.json, _sections.npz, _final.npz) to VTK
files for ParaView. No extra dependencies: VTK XML ImageData with raw appended data.

Writes, next to the run (or to ``--out``):
  <tag>_vtk/section_0000.vti ...   the centreline x-z section every saved step (cell data
                                    ``fill`` = metal fill level, ``T`` [degC]; field data
                                    ``time_us``, ``laser_on``, ``x_laser_um``, ``depth_um``)
  <tag>_section.pvd                 the time series of those sections (open this one)
  <tag>_final.vti                   the 3-D state at the end of the run (``fill``, ``flag``
                                    0 gas / 1 interface / 2 liquid / 3 wall, ``T_peak`` [degC])

Coordinates in micrometres: x along the track, y across it, z up with z = 0 at the original
plate surface (negative: into the plate).

In ParaView:
  * section: open <tag>_section.pvd, Apply, colour by ``fill`` (or ``T``), press Play; or
    Cell Data to Point Data -> Contour fill = 0.5 for the keyhole wall line.
  * 3-D end state: open <tag>_final.vti, Cell Data to Point Data -> Contour fill = 0.5 (the
    surface, end crater and any pores) or Threshold T_peak >= 1655 (melted volume, Ti-6Al-4V).

    python tools/lbm/keyhole_to_vtk.py tools/lbm/keyhole_runs/T3_dx3.33um.json
"""
import argparse
import json
import os
import sys

import numpy as np


def write_vti(path, cell_data, origin, spacing, field_data=None):
    """VTK XML ImageData with cell data, raw appended binary (UInt64 headers)."""
    shape = next(iter(cell_data.values())).shape
    nx, ny, nz = shape
    vtk_types = {np.dtype(np.float32): "Float32", np.dtype(np.float64): "Float64", np.dtype(np.int8): "Int8",
                 np.dtype(np.uint8): "UInt8", np.dtype(np.int32): "Int32"}
    blocks, offset, arrays_xml = [], 0, []

    def add(name, a, ncomp=1, tag="DataArray"):
        nonlocal offset
        a = np.ascontiguousarray(a)
        raw = a.tobytes()
        blocks.append(np.uint64(len(raw)).tobytes() + raw)
        arrays_xml.append(f'<{tag} type="{vtk_types[a.dtype]}" Name="{name}" NumberOfComponents="{ncomp}" '
                          f'NumberOfTuples="{a.size // ncomp}" format="appended" offset="{offset}"/>')
        offset += 8 + len(raw)

    field_xml = ""
    if field_data:
        start = len(arrays_xml)
        for k, v in field_data.items():
            add(k, np.atleast_1d(np.asarray(v, dtype=np.float64)))
        field_xml = "<FieldData>" + "".join(arrays_xml[start:]) + "</FieldData>"
        del arrays_xml[start:]
    cell_start = len(arrays_xml)
    for k, v in cell_data.items():
        # VTK image order: x fastest -> transpose (nx, ny, nz) to Fortran order
        add(k, np.asarray(v).transpose(2, 1, 0).ravel())
    cells_xml = "".join(arrays_xml[cell_start:])
    ext = f"0 {nx} 0 {ny} 0 {nz}"
    head = (f'<?xml version="1.0"?>\n<VTKFile type="ImageData" version="1.0" byte_order="LittleEndian" '
            f'header_type="UInt64">\n<ImageData WholeExtent="{ext}" Origin="{origin[0]} {origin[1]} {origin[2]}" '
            f'Spacing="{spacing} {spacing} {spacing}">\n{field_xml}\n<Piece Extent="{ext}">\n'
            f'<CellData Scalars="{next(iter(cell_data))}">{cells_xml}</CellData>\n</Piece>\n</ImageData>\n'
            f'<AppendedData encoding="raw">\n_')
    with open(path, "wb") as f:
        f.write(head.encode())
        for b in blocks:
            f.write(b)
        f.write(b"\n</AppendedData>\n</VTKFile>\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run", help="the run's JSON (tools/lbm/keyhole_runs/<tag>.json)")
    ap.add_argument("--out", default=None, help="output folder (default: next to the run)")
    ap.add_argument("--every", type=int, default=1, help="write every N-th saved section")
    a = ap.parse_args()
    base = a.run[:-5] if a.run.endswith(".json") else a.run
    tag = os.path.basename(base)
    out = a.out or os.path.dirname(os.path.abspath(a.run))
    info = json.load(open(base + ".json"))
    dx = info["dx"] * 1e6
    z0 = -info["z_surface"] * 1e6
    jc = int(round(info["y"] / info["dx"]))
    log = info.get("log", [])
    by_t = {round(r["t"] * 1e9): r for r in log}
    written = []
    if os.path.exists(base + "_sections.npz"):
        s = np.load(base + "_sections.npz")
        d = os.path.join(out, f"{tag}_vtk")
        os.makedirs(d, exist_ok=True)
        entries = []
        for n in range(0, len(s["t"]), a.every):
            t = float(s["t"][n])
            r = by_t.get(round(t * 1e9), {})
            fd = dict(time_us=t * 1e6, laser_on=float(bool(r.get("laser_on", False))),
                      x_laser_um=(r.get("x_l") or r.get("x_last") or np.nan) * 1e6 if r else np.nan,
                      depth_um=r.get("depth", np.nan) * 1e6 if r else np.nan)
            fn = f"section_{n:04d}.vti"
            write_vti(os.path.join(d, fn), dict(fill=s["fill"][n][:, None, :].astype(np.float32),
                                                T=s["T"][n][:, None, :].astype(np.float32)),
                      (0.0, jc * dx, z0), dx, fd)
            entries.append(f'<DataSet timestep="{t * 1e6:.4f}" part="0" file="{tag}_vtk/{fn}"/>')
        pvd = os.path.join(out, f"{tag}_section.pvd")
        with open(pvd, "w") as f:
            f.write('<?xml version="1.0"?>\n<VTKFile type="Collection" version="0.1">\n<Collection>\n'
                    + "\n".join(entries) + "\n</Collection>\n</VTKFile>\n")
        written.append(f"{pvd} ({len(entries)} sections, time in us)")
    if os.path.exists(base + "_final.npz"):
        z = np.load(base + "_final.npz")
        fin = os.path.join(out, f"{tag}_final.vti")
        write_vti(fin, dict(fill=z["fill"].astype(np.float32), flag=z["flag"].astype(np.int8),
                            T_peak=z["T_peak"].astype(np.float32)), (0.0, 0.0, z0), dx,
                  dict(time_us=log[-1]["t"] * 1e6 if log else np.nan))
        written.append(fin)
    for w in written:
        print("wrote", w)
    if not written:
        sys.exit(f"nothing to export for {base}")


if __name__ == "__main__":
    main()
