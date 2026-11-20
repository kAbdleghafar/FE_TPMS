# Tutorial 7: Where could twins or HCP start, and how much HCP forms?

**Goal.** For a compressed FCC lattice (CoCrFeMnNi), map where the local stress can extend
stacking faults, the precursor of deformation twins and of the FCC → HCP transformation
(Level 0). Then, for a metastable alloy that does transform (Fe50Mn30Co10Cr10), estimate how
much HCP forms (Level 1a). About 1 minute.

**You will learn**
* the leading-partial criterion and its threshold;
* orientation sampling;
* the Olson–Cohen kinetics fitted to measured HCP fractions;
* what the full studies found.

## Background

A stacking fault in FCC extends when the shear stress on the leading Shockley partial,
τ\*, overcomes the fault energy, τ\* ≥ τ_c. For CoCrFeMnNi, τ_c = 360 MPa reproduces the
measured twinning onset (720 MPa uniaxial, Laplanche et al. 2016) in the best-oriented grains.
A lower τ_c stands for a lower fault energy (cryogenic temperature, Mn/Fe-rich variants).

## Step 1: Schmid factors and the onset stress of single crystals

```python
import numpy as np
from tpmsfem.post import fcc_faults as FF

R = FF.random_rotations(2000, seed=0)                      # random grain orientations
m_c = FF.uniaxial_factor(R, (0, 0, 1), -1)                 # leading-partial factor, compression
m_t = FF.uniaxial_factor(R, (0, 0, 1), +1)                 # tension
print(f"compression: max {m_c.max():.3f}, median {np.median(m_c):.3f};  tension: max {m_t.max():.3f}, median {np.median(m_t):.3f}")
g001 = np.eye(3)[None]                                     # a crystal with [001] along the load
print(f"[001] crystal: compression {FF.uniaxial_factor(g001, (0, 0, 1), -1)[0]:.3f}, "
      f"tension {FF.uniaxial_factor(g001, (0, 0, 1), +1)[0]:.3f}")
on = FF.onset_stress_uniaxial(R, 20.0)                     # uniaxial stress to extend a 20 mJ/m^2 fault
print(f"onset stress for 20 mJ/m^2, best grain: {np.nanmin(np.where(np.isfinite(on), on, np.nan)):.0f} MPa")
```

Output (from the test run):

```text
compression: max 0.500, median 0.360;  tension: max 0.500, median 0.437
[001] crystal: compression 0.471, tension 0.236
onset stress for 20 mJ/m^2, best grain: 272 MPa
```

The leading partial is pushed ahead in some orientations and held back in others: the
asymmetry between tension and compression in twinning comes from this.

## Step 2: a compressed lattice

```python
import tpmsfem as tf
from tpmsfem.fem.materials import Material
from tpmsfem.geometry import TPMS, lattice_cube
from tpmsfem.mesh import voxelize

cantor = Material("CoCrFeMnNi_LPBF", E=203e3, nu=0.25, yield_stress=757.0, hardening=1500.0)
mesh = voxelize(lattice_cube(TPMS("gyroid", cell_size=4.0, density=0.35), 4.0, plate=0.5), (0, 0, 0), (4, 4, 4), 0.25)
md = tf.SolidModel(mesh, cantor)
z = mesh.points[:, 2]
top = np.flatnonzero(z > 4 - 1e-9)
md.fix(np.flatnonzero(z < 1e-9))
md.fix(top, (0, 1), 0.0)
rot = FF.random_rotations(300)
for strain in (0.01, 0.02, 0.03):
    md.fix(top, (2,), -strain * 4.0)
    md.solve(n_inc=2)
    sig = md.stress().mean(axis=1)                         # element stresses [MPa]
    yielded = md.plastic_strain()[1].mean(axis=1) > 0      # faults nucleate where dislocations move
    tau = FF.leading_partial_stress(sig, rot)              # (elements, orientations)
    f360 = ((tau >= 360.0).mean(axis=1) * yielded).mean()
    f240 = ((tau >= 240.0).mean(axis=1) * yielded).mean()
    print(f"{100 * strain:.0f} % strain: faults can extend in {100 * f360:.1f} % (tau_c 360 MPa), "
          f"{100 * f240:.1f} % (240 MPa) of the lattice")
```

Output (from the test run):

```text
1 % strain: faults can extend in 3.9 % (tau_c 360 MPa), 22.2 % (240 MPa) of the lattice
2 % strain: faults can extend in 6.1 % (tau_c 360 MPa), 28.8 % (240 MPa) of the lattice
3 % strain: faults can extend in 7.5 % (tau_c 360 MPa), 31.7 % (240 MPa) of the lattice
```

## Step 3: the stricter free-partials bound

If both partials glide freely (Byun's picture), only the difference of their shear stresses
works against the fault. That needs about twice the measured twinning stress:

```python
g_lead = FF.fault_energy_capacity(sig, rot, criterion="leading")    # mJ/m^2 the stress can overcome
g_byun = FF.fault_energy_capacity(sig, rot, criterion="byun")
print(f"largest fault energy overcome: leading-partial {g_lead.max():.0f} mJ/m^2, free partials {g_byun.max():.0f} mJ/m^2 "
      f"(CoCrFeMnNi: ~20-30 mJ/m^2)")
```

Output (from the test run):

```text
largest fault energy overcome: leading-partial 74 mJ/m^2, free partials 21 mJ/m^2 (CoCrFeMnNi: ~20-30 mJ/m^2)
```

## Step 4: HCP embryos (Olson–Cohen)

An HCP embryo n planes thick costs γ_SF − 2σ(1 − 1/n) per unit area: thick embryos form more
easily when the FCC/HCP interface energy σ is large.

```python
for n in (1, 2, 4, 8):
    print(f"n = {n}: effective fault energy {FF.olson_cohen_threshold(25.0, 8.0, n):5.1f} mJ/m^2")
```

Output (from the test run):

```text
n = 1: effective fault energy  25.0 mJ/m^2
n = 2: effective fault energy  17.0 mJ/m^2
n = 4: effective fault energy  13.0 mJ/m^2
n = 8: effective fault energy  11.0 mJ/m^2
```

## Step 5: how much HCP (Level 1a)

`hcp_kinetics` turns the plastic strain into an HCP fraction with an Olson–Cohen law fitted to
LPBF Fe50Mn30Co10Cr10 (0.94 % HCP as built, 28.1 % at 20 % strain). With one measured point,
two shapes bound the small-strain part.

```python
from tpmsfem.post import hcp_kinetics as HK

peeq = md.plastic_strain()[1].mean(axis=1)                 # the lattice at 3 % strain
for shape in ("exponential", "sigmoidal"):
    law = HK.femncocr_lpbf(shape)
    f = law.fraction(peeq)
    print(f"{shape:11s}: HCP mean {100 * f.mean():.2f} %, 95th percentile {100 * np.percentile(f, 95):.2f} %, "
          f"f(20 % strain) = {100 * law.fraction(np.array([0.2 - 744.9 / 200e3]))[0]:.1f} %")
```

Output (from the test run):

```text
exponential: HCP mean 2.36 %, 95th percentile 8.69 %, f(20 % strain) = 28.1 %
sigmoidal  : HCP mean 1.53 %, 95th percentile 4.82 %, f(20 % strain) = 28.1 %
```

`StrainInducedHCP.fit(eps_p, f, f0, shape=...)` fits your own measurements.

## Results so far (full studies)

The full studies use the graded gyroid column of the CoCrFeMnNi study (4 × 4 × 8 mm, walls
0.2 → 0.6 mm, 0.1 mm voxels) and the grain-CA textures:

| compression (nominal) | 1 % | 2 % | 3 % |
|---|---|---|---|
| faults can extend, τ_c 360 MPa, stress-free start | 6.2 % | 10.4 % | 13.1 % |
| faults can extend, τ_c 240 MPa | 27.8 % | 43.6 % | 51.7 % |
| from the as-built residual stress, τ_c 360 MPa | 9.4 % | 13.6 % | 16.2 % |
| FeMnCoCr gyroid: HCP lattice mean (exp. / sigm.) | 1.3 / 1.0 % | 1.8 / 1.2 % | 2.4 / 1.4 % |

* Faults can extend as soon as the walls yield, because the as-built yield (757 MPa) exceeds
  the twinning stress (720 MPa); they sit in the bending hinges of the inclined walls.
* The as-built residual stresses bring the onset forward: 6.5× the fault-prone volume at 0.4 %
  strain.
* In the metastable alloy little HCP forms at test strains (2–3.5 % at 5 %), but 8–12 % in the
  hinges.

Details: [process_models.md](../process_models.md) §7.5–§7.6; scripts
`benchmarks/literature/cantor_gyroid_faults*.py`, `femncocr_gyroid_hcp.py`.

## Limits
* Each grain takes its element's stress (Sachs-type); grain-to-grain stresses need crystal
  plasticity.
* Level 0 says where faults can extend, not whether twins or HCP form: that is set by the
  FCC/HCP free-energy difference.

## Next
[Tutorial 8: implants and fatigue](08_implants_fatigue.md).
