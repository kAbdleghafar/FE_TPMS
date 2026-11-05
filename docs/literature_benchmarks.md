# Benchmarks against published results (Barsoum; El-Rich)

This page compares tpmsfem with finite element results published by the Khalifa
University groups of **Imad Barsoum** and **Marwan El-Rich**. Every published
number below is transcribed from the paper, with its location; the transcriptions
are in `benchmarks/literature/data/*.json`. Every tpmsfem number is reproduced by
`benchmarks/literature/run_all.sh`.

Unlike the CalculiX comparison (identical decks, agreement to 1e-7), these are
**independent re-implementations from the paper text**. They use a different element
type (voxel hex8 vs. their C3D10 / Ansys tets), our own geometry generation, and our
reading of anything the paper leaves out. Each assumption is listed. Differences of a
few percent are expected. Larger ones are discussed.

Each tpmsfem value is solved on three grids and Richardson-extrapolated (generalised
for non-constant refinement ratio, Celik et al. 2008). The grid-convergence index
(GCI) is reported as its numerical uncertainty.

RESULTS_PLACEHOLDER
