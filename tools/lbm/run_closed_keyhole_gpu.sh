#!/usr/bin/env bash
# Closed-keyhole production runs on a CUDA GPU (CuPy for the flow, numba for the rays).
# Each run checkpoints every 20 min: rerun the same command to resume. Results go to
# tools/lbm/keyhole_runs/; the closure figure is drawn at the end.
#
#   bash tools/lbm/run_closed_keyhole_gpu.sh                 # everything below, at DX (default 3.33 um)
#   DX=2.5e-6 bash tools/lbm/run_closed_keyhole_gpu.sh turn  # one group at 2.5 um
#
# Groups:
#   off      Ti-6Al-4V T3 track, laser off abruptly at the end vs ramped down over 100 um
#   turn     Martin et al. (2019) turn points: M100, M200, M300 at constant power, M200 with the
#            constant-keyhole-number power schedule
#   overhang CoCrFeMnNi slab over powder at the gyroid settings: 90, 150, 250 um slabs
set -euo pipefail
cd "$(dirname "$0")/../.."
export TPMSFEM_BACKEND=cupy
DX=${DX:-3.33e-6}
SETS=("$@")
[ ${#SETS[@]} -eq 0 ] && SETS=(off turn overhang)
for g in "${SETS[@]}"; do
  case "$g" in
    off)
      python tools/lbm/keyhole_ti64.py T3 --dx "$DX" --track 300e-6 --cool 200e-6
      python tools/lbm/keyhole_ti64.py T3 --dx "$DX" --track 300e-6 --cool 200e-6 --scenario ramp --ramp 100e-6
      ;;
    turn)
      for c in M100 M200 M300; do python tools/lbm/keyhole_ti64.py "$c" --dx "$DX" --scenario turn --cool 200e-6; done
      python tools/lbm/keyhole_ti64.py M200 --dx "$DX" --scenario turn-ke --cool 200e-6
      ;;
    overhang)
      for t in 90e-6 150e-6 250e-6; do python tools/lbm/keyhole_overhang.py --slab "$t" --dx "$DX"; done
      ;;
    *) echo "unknown group $g"; exit 1 ;;
  esac
done
python tools/lbm/keyhole_figure.py closure tools/lbm/keyhole_runs/T3_dx*um.json tools/lbm/keyhole_runs/T3_ramp_dx*um.json \
  tools/lbm/keyhole_runs/M*_turn*_dx*um.json || true
