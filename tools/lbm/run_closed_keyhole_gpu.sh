#!/usr/bin/env bash
# Closed-keyhole production runs on a CUDA GPU (CuPy with the fused CUDA kernels for the flow,
# numba for the rays). Each run checkpoints every CKPT minutes (default 10). After an
# interruption, rerun the same command: finished cases are skipped and the interrupted one
# resumes from its last checkpoint. Results go to tools/lbm/keyhole_runs/; the closure figure is
# drawn at the end.
#
#   bash tools/lbm/run_closed_keyhole_gpu.sh                 # everything below, at DX (default 3.33 um)
#   DX=2.5e-6 bash tools/lbm/run_closed_keyhole_gpu.sh turn  # one group at 2.5 um
#   PYTHON=.venv/Scripts/python bash tools/lbm/run_closed_keyhole_gpu.sh   # a specific interpreter
#
# Groups:
#   off      Ti-6Al-4V T3 track, laser off abruptly at the end vs ramped down over 100 um
#   turn     Martin et al. (2019) turn points: M200 at constant power and with the
#            constant-keyhole-number power schedule first, then M100 and M300
#   overhang CoCrFeMnNi slab over powder at the gyroid settings: 90, 150, 250 um slabs
set -euo pipefail
cd "$(dirname "$0")/../.."
export TPMSFEM_BACKEND=cupy
DX=${DX:-3.33e-6}
CKPT=${CKPT:-10}
PY=${PYTHON:-python}
SETS=("$@")
[ ${#SETS[@]} -eq 0 ] && SETS=(off turn overhang)
for g in "${SETS[@]}"; do
  case "$g" in
    off)
      "$PY" tools/lbm/keyhole_ti64.py T3 --dx "$DX" --track 300e-6 --cool 200e-6 --checkpoint "$CKPT"
      "$PY" tools/lbm/keyhole_ti64.py T3 --dx "$DX" --track 300e-6 --cool 200e-6 --scenario ramp --ramp 100e-6 --checkpoint "$CKPT"
      ;;
    turn)
      "$PY" tools/lbm/keyhole_ti64.py M200 --dx "$DX" --scenario turn --cool 200e-6 --checkpoint "$CKPT"
      "$PY" tools/lbm/keyhole_ti64.py M200 --dx "$DX" --scenario turn-ke --cool 200e-6 --checkpoint "$CKPT"
      for c in M100 M300; do "$PY" tools/lbm/keyhole_ti64.py "$c" --dx "$DX" --scenario turn --cool 200e-6 --checkpoint "$CKPT"; done
      ;;
    overhang)
      for t in 90e-6 150e-6 250e-6; do "$PY" tools/lbm/keyhole_overhang.py --slab "$t" --dx "$DX" --checkpoint "$CKPT"; done
      ;;
    *) echo "unknown group $g"; exit 1 ;;
  esac
done
"$PY" tools/lbm/keyhole_figure.py closure tools/lbm/keyhole_runs/T3_dx*um.json tools/lbm/keyhole_runs/T3_ramp_dx*um.json \
  tools/lbm/keyhole_runs/M*_turn*_dx*um.json || true
