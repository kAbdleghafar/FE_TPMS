#!/usr/bin/env bash
# Runs every literature benchmark (resumable: finished cases are cached in results/literature/*.json).
# The two Barsoum configurations run in parallel (the AMG solver is mostly single-threaded).
set -e
cd "$(dirname "$0")/../.."
mkdir -p results/literature
python benchmarks/literature/van_viet_elrich_2025.py
OMP_NUM_THREADS=2 python benchmarks/literature/ahmed_barsoum_2022.py --config periodic > results/literature/periodic.log 2>&1 &
OMP_NUM_THREADS=2 python benchmarks/literature/ahmed_barsoum_2022.py --config finite --topologies P-S,IWP-S,IWP-L,G-S > results/literature/finite.log 2>&1 &
wait
python benchmarks/literature/ahmed_barsoum_2022.py --report
python benchmarks/literature/ahmed_barsoum_2022_sensitivity.py
for r in "0.3 24" "0.2 28" "0.1 40"; do set -- $r; python benchmarks/literature/ahmed_barsoum_2022_secant.py --rho $1 --n $2; done
python benchmarks/literature/ahmed_barsoum_2022_rs.py
python benchmarks/literature/ahmed_barsoum_2022_analysis.py
# other groups
python benchmarks/literature/nist_ambench2018.py --variant plate
python benchmarks/literature/nist_ambench2018.py --variant rigid
python benchmarks/literature/nist_ambench2018.py --report
python benchmarks/literature/nist_ambench2018_plots.py
python benchmarks/literature/naghavi_2022.py --config periodic
python benchmarks/literature/naghavi_2022.py --config specimen
python benchmarks/literature/naghavi_2022.py --report
