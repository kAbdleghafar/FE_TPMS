# Tutorial 23: The command line

**Goal.** Use the `tpmsfem` command for quick jobs without writing Python.

## Backend and device

```bash
tpmsfem info
```

prints the array backend (NumPy or CuPy), the device, and whether numba is available.

## A voxel mesh of a TPMS block

```bash
tpmsfem voxelize lattice.vtu --surface gyroid --kind sheet --cell 3 --density 0.3 --size 6 --h 0.1
tpmsfem voxelize lattice.inp --surface diamond --kind network --cell 2 --density 0.25 --size 4 --h 0.05
```

writes a hex8 mesh in any format meshio knows (from the extension: `.vtu`, `.inp`, `.msh`, ...).

## Compare with Abaqus

```bash
tpmsfem compare ours.npz --abaqus abq_prefix
```

compares displacement and stress fields with an Abaqus export, field by field (Tutorial 12).

## Help

```bash
tpmsfem --help
tpmsfem voxelize --help
```

## Long studies

The scripts in `benchmarks/literature/` and `tools/lbm/` are command-line programs too; each
documents its usage at the top, stores results in `benchmarks/literature/results/` or
`tools/lbm/keyhole_runs/`, and resumes after an interruption. The table in
[manual.md](../manual.md) §22 lists them.
