"""Where can the local stress extend stacking faults in an FCC crystal (the precursor of
deformation twins and of the FCC -> HCP transformation)?

A perfect dislocation b = a/2<110> on a {111} plane glides as a leading and a trailing Shockley
partial (b_p = a/sqrt(6), a/6<112>) separated by a stacking fault of energy gamma. Extended faults
on every {111} plane make a twin, on every second plane HCP (epsilon) martensite; which one forms
is set by the chemistry (the FCC/HCP free-energy difference), not by the stress. Two limits for
the stress at which a fault extends:

* ``criterion="leading"`` (default): the leading partial runs away from a held trailing partial
  (a source pinned at an obstacle, as in twin nucleation models) once

      gamma* = tau_L b_p >= gamma,

  tau_L being the shear stress resolved on the leading partial's {111}<112> system (the twinning /
  transformation systems of crystal-plasticity models). Calibration: the twinning stress of
  CoCrFeMnNi, 720 MPa true stress at 77 K and 293 K, i.e. a resolved 235 MPa with a Taylor factor
  of 3.06 (Laplanche et al., Acta Mater. 118 (2016) 152), corresponds to gamma = 235 MPa x b_p =
  34.5 mJ/m^2, consistent with the measured stacking-fault energy, 30 +- 5 mJ/m^2 (Okamoto et
  al., Sci. Rep. 6 (2016) 35863).
* ``criterion="byun"``: both partials glide freely; the equilibrium separation
  d = 2 K / (2 gamma - (tau_L - tau_T) b_p) (Copley & Kear, Acta Metall. 16 (1968) 227; Byun, Acta
  Mater. 51 (2003) 3063) diverges once gamma* = (tau_L - tau_T) b_p / 2 >= gamma. With gamma =
  30 mJ/m^2 this puts the onset in the best-oriented grains at ~1.4 GPa in uniaxial tension, twice
  the measured twinning stress: a strict upper bound.

``gamma*`` is the largest fault energy the local stress can overcome. Comparing it with several
thresholds brackets twin and HCP nucleation: the single fault (gamma_SF), and a thick HCP embryo
of n planes, whose effective fault energy falls to gamma_SF - 2 sigma (1 - 1/n) (Olson & Cohen,
Metall. Trans. A 7 (1976) 1897; sigma: FCC/HCP interface energy).

Bookkeeping. For a {111} plane with normal n, the three Shockley partials that turn the stacking
ABC into a fault when the crystal on the +n side moves by them are, for n = [111],
a/6 [1 1 -2], a/6 [1 -2 1], a/6 [-2 1 1] (the "+" family; for the other normals they follow by
the sign change n = diag(s) [111]). These are the 12 twinning / transformation systems (with
sense). A full dislocation b = t_i - t_j glides with leading partial t_i and trailing partial
-t_j; for ``"byun"`` only dislocations that glide forward count (resolved shear tau_b > 0, within
``active`` of the grain's largest tau_b).

Checks (tests/test_fcc_faults.py): tension along [001] gives a small leading-partial Schmid
factor (0.24) and compression a large one (0.47), as observed for twinning in FCC single crystals
([001] twins in compression, not in tension); the free-partials criterion narrows the faults in
[001] tension and widens them in compression.

The local stress of a grain is taken as the stress of the finite element it sits in (a
Sachs-type assumption: grain-to-grain stress differences are not resolved).

Stresses: Voigt order [xx, yy, zz, yz, xz, xy] (as tpmsfem.fem), MPa. Orientations: rotation
matrices whose rows are the crystal <100> axes in the sample frame (crystal = R @ sample), as in
tpmsfem.process.grain_ca and phasefield3d. Lattice parameter: CoCrFeMnNi, 0.3597 nm.
"""
from __future__ import annotations

import numpy as np

A_CANTOR = 0.3597e-9          # m, CoCrFeMnNi lattice parameter (room temperature)

_N111 = np.array([[1, 1, 1], [-1, 1, 1], [1, -1, 1], [1, 1, -1]], dtype=float)
_PLUS111 = np.array([[1, 1, -2], [1, -2, 1], [-2, 1, 1]], dtype=float)


def shockley_systems():
    """The 24 dissociated slip systems (12 slip systems x 2 senses) in the crystal frame.

    Returns unit vectors (n, b, t_lead, t_trail), each (24, 3): plane normal, full Burgers
    vector direction, leading and trailing partial directions (b = |t| (t_lead + t_trail) * ...,
    i.e. b is parallel to t_lead + t_trail)."""
    n_out, b_out, l_out, t_out = [], [], [], []
    for n in _N111:
        D = np.diag(n)
        plus = (D @ _PLUS111.T).T                     # the "+" partials of this plane
        for i in range(3):
            for j in range(3):
                if i == j:
                    continue
                lead, trail = plus[i], -plus[j]
                b = lead + trail                      # a/6 <112> - a/6 <112> = a/2 <110>
                n_out.append(n / np.linalg.norm(n))
                b_out.append(b / np.linalg.norm(b))
                l_out.append(lead / np.linalg.norm(lead))
                t_out.append(trail / np.linalg.norm(trail))
    return np.array(n_out), np.array(b_out), np.array(l_out), np.array(t_out)


def twin_systems():
    """The 12 twinning / transformation systems {111}<112> with sense: unit plane normals and
    leading-partial directions (n, t), each (12, 3), crystal frame."""
    n_out, t_out = [], []
    for n in _N111:
        for t in (np.diag(n) @ _PLUS111.T).T:
            n_out.append(n / np.linalg.norm(n))
            t_out.append(t / np.linalg.norm(t))
    return np.array(n_out), np.array(t_out)


def _voigt_to_tensor(s):
    s = np.asarray(s, dtype=float)
    if s.shape[-2:] == (3, 3):
        return s
    t = np.empty(s.shape[:-1] + (3, 3))
    t[..., 0, 0], t[..., 1, 1], t[..., 2, 2] = s[..., 0], s[..., 1], s[..., 2]
    t[..., 1, 2] = t[..., 2, 1] = s[..., 3]
    t[..., 0, 2] = t[..., 2, 0] = s[..., 4]
    t[..., 0, 1] = t[..., 1, 0] = s[..., 5]
    return t


def _schmid_tensors(rot):
    """Symmetric Schmid tensors in the sample frame for every orientation: (M, 24, 3, 3) for the
    full dislocation, the leading and the trailing partial."""
    n, b, lead, trail = shockley_systems()
    R = np.asarray(rot, dtype=float).reshape(-1, 3, 3)
    ns = np.einsum("mji,sj->msi", R, n)            # sample-frame vectors: v_s = R^T v_c
    out = []
    for d in (b, lead, trail):
        ds = np.einsum("mji,sj->msi", R, d)
        P = 0.5 * (ds[..., :, None] * ns[..., None, :] + ns[..., :, None] * ds[..., None, :])
        out.append(P)
    return out


def _tau_lead(stress, rot, chunk: int):
    """Largest positive shear stress [MPa] on the 12 twinning / transformation systems, (N, M)."""
    S = _voigt_to_tensor(stress).reshape(-1, 3, 3)
    n, t = twin_systems()
    R = np.asarray(rot, dtype=float).reshape(-1, 3, 3)
    ns = np.einsum("mji,sj->msi", R, n)
    ts = np.einsum("mji,sj->msi", R, t)
    M = R.shape[0]
    P = (0.5 * (ts[..., :, None] * ns[..., None, :] + ns[..., :, None] * ts[..., None, :])).reshape(M * 12, 9)
    N = S.shape[0]
    out = np.empty((N, M))
    for k in range(0, N, chunk):
        tau = (S[k:k + chunk].reshape(-1, 9) @ P.T).reshape(-1, M, 12)
        out[k:k + chunk] = np.maximum(tau.max(axis=2), 0.0)
    return out


def _dtau(stress, rot, active: float, chunk: int, return_tau: bool):
    """Largest (tau_L - tau_T) [MPa] over the gliding systems, (N, M); optionally max tau_b."""
    S = _voigt_to_tensor(stress).reshape(-1, 3, 3)
    Pb, Pl, Pt = _schmid_tensors(rot)
    M = Pb.shape[0]
    Pb = Pb.reshape(M * 24, 9)
    Pd = (Pl - Pt).reshape(M * 24, 9)
    N = S.shape[0]
    out = np.empty((N, M))
    taub = np.empty((N, M)) if return_tau else None
    for k in range(0, N, chunk):
        s = S[k:k + chunk].reshape(-1, 9)
        tb = (s @ Pb.T).reshape(-1, M, 24)
        dt = (s @ Pd.T).reshape(-1, M, 24)
        tmax = tb.max(axis=2, keepdims=True)
        ok = (tb > 0.0) & (tb >= active * tmax)
        g = np.where(ok, dt, -np.inf).max(axis=2)
        out[k:k + chunk] = np.where(np.isfinite(g), g, 0.0)       # zero stress: nothing glides
        if return_tau:
            taub[k:k + chunk] = tmax[..., 0]
    return out, taub


def fault_energy_capacity(stress, rot, criterion: str = "leading", a: float = A_CANTOR, active: float = 0.9,
                          chunk: int = 4096):
    """gamma* [mJ/m^2] for every stress (N, 6 Voigt or N, 3, 3; MPa) and orientation (M, 3, 3),
    shape (N, M). ``criterion="leading"``: tau_L b_p, tau_L the largest positive shear stress on the
    12 {111}<112> systems. ``criterion="byun"``: the largest (tau_L - tau_T) b_p / 2 over the gliding
    slip systems (negative: every gliding dislocation is pushed together)."""
    bp = a / np.sqrt(6.0)
    if criterion == "leading":
        return _tau_lead(stress, rot, chunk) * 1e6 * bp * 1e3
    if criterion == "byun":
        dt, _ = _dtau(stress, rot, active, chunk, False)
        return dt * 1e6 * bp / 2.0 * 1e3
    raise ValueError(f"unknown criterion {criterion!r}")


def leading_partial_stress(stress, rot, chunk: int = 4096):
    """tau* [MPa]: the largest positive shear stress on the 12 {111}<112> twinning /
    transformation systems, for every stress (N, 6 Voigt or N, 3, 3; MPa) and orientation
    (M, 3, 3); shape (N, M). Faults extend where tau* >= tau_c (``fault_energy_capacity`` gives
    the same in fault-energy units, tau* b_p)."""
    return _tau_lead(stress, rot, chunk)


def uniaxial_factor(rot, axis=(0.0, 0.0, 1.0), sign: float = 1.0, criterion: str = "leading", active: float = 0.9):
    """Schmid factor of the criterion for a uniaxial stress along ``axis`` (sign +1 tension, -1
    compression), per orientation: the leading-partial Schmid factor (``"leading"``) or the
    leading-minus-trailing factor of the gliding systems (``"byun"``)."""
    l = np.asarray(axis, float)
    l = l / np.linalg.norm(l)
    s = (sign * np.outer(l, l))[None]
    if criterion == "leading":
        return _tau_lead(s, rot, 4096)[0]
    dt, _ = _dtau(s, rot, active, 4096, False)
    return dt[0]


def onset_stress_uniaxial(rot, gamma_mJ: float, criterion: str = "leading", a: float = A_CANTOR, **kw):
    """Uniaxial stress [MPa] at which gamma* reaches ``gamma_mJ`` (inf where it never does)."""
    f = uniaxial_factor(rot, criterion=criterion, **kw)
    bp = a / np.sqrt(6.0)
    k = 1.0 if criterion == "leading" else 2.0
    with np.errstate(divide="ignore"):
        return np.where(f > 0, k * gamma_mJ * 1e-3 / (f * bp) * 1e-6, np.inf)


def random_rotations(n: int, seed: int = 0) -> np.ndarray:
    """Uniformly random rotation matrices (n, 3, 3) (from random unit quaternions)."""
    rng = np.random.default_rng(seed)
    q = rng.normal(size=(n, 4))
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    w, x, y, z = q.T
    return np.stack([
        np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], -1),
        np.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], -1),
        np.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1)], 1)


def olson_cohen_threshold(gamma_sf: float, sigma_interface: float, n_planes: float = np.inf) -> float:
    """Effective fault energy of an HCP embryo n planes thick, gamma_SF - 2 sigma (1 - 1/n)
    [same unit as the inputs] (n = 1: a single stacking fault)."""
    return gamma_sf - 2.0 * sigma_interface * (1.0 - 1.0 / n_planes)
