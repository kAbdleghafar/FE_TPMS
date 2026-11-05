#!/usr/bin/env bash
# Runs every literature benchmark (resumable: finished cases are cached in results/literature/*.json).
# The two Barsoum configurations run in parallel (the AMG solver is mostly single-threaded).
set -e
cd "$(dirname "$0")/../.."
python benchmarks/literature/van_viet_elrich_2025.py
OMP_NUM_THREADS=2 python benchmarks/literature/ahmed_barsoum_2022.py --config periodic > results/literature/periodic.log 2>&1 &
OMP_NUM_THREADS=2 python benchmarks/literature/ahmed_barsoum_2022.py --config finite > results/literature/finite.log 2>&1 &
wait
python benchmarks/literature/ahmed_barsoum_2022.py --report
python benchmarks/literature/ahmed_barsoum_2022_rs.py
