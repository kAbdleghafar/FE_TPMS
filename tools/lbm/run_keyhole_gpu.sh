#!/usr/bin/env bash
# Keyhole stage 2 production runs on a CUDA GPU (CuPy backend; numba for the ray tracer).
# Each case writes tools/lbm/keyhole_runs/<case>_dx2.5um.json (+ sections, final fields) and
# checkpoints every 20 min: rerun the same command to resume after an interruption.
#
#   bash tools/lbm/run_keyhole_gpu.sh            # all six cases at 2.5 um
#   bash tools/lbm/run_keyhole_gpu.sh T3 T5      # selected cases
#   DX=3.33e-6 bash tools/lbm/run_keyhole_gpu.sh # coarser and ~2.4x faster
#
# First, a short timing run (about a minute of GPU time) to see the speed:
#   TPMSFEM_BACKEND=cupy python tools/lbm/keyhole_ti64.py T3 --track 50e-6 --cool 0 --out /tmp/kh_timing
set -euo pipefail
cd "$(dirname "$0")/../.."
export TPMSFEM_BACKEND=cupy
DX=${DX:-2.5e-6}
CASES=("$@")
[ ${#CASES[@]} -eq 0 ] && CASES=(T1 T2 T3 T4 T5 T6)
for c in "${CASES[@]}"; do
  echo "=== $c (dx $DX) ==="
  python tools/lbm/keyhole_ti64.py "$c" --dx "$DX"
done
python tools/lbm/keyhole_figure.py tools/lbm/keyhole_runs
