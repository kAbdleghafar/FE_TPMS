"""Equiatomic CoCrFeMnNi (Cantor alloy): property set for the LPBF process models.

Thermal properties: Li, Huang, Sun & Lu, J. Mater. Sci. Technol. 77 (2021) 187-195,
doi:10.1016/j.jmst.2020.10.052, Tables 1-2:
- the simulation inputs of a laser-deposition study of CoCrFeMnNi;
- the paper does not say where they were measured, so treat them as literature model inputs,
  not verified measurements;
- k(T) and cp(T) are tabulated from 25 to 1000 C and are held constant above 1000 C here
  (an assumption, also used for the liquid).

Solidification. The alloy is reduced to an effective binary for the phase field
(``tpmsfem.process.phasefield.DiluteAlloy``):
- Co, Cr and Fe stay in the cell cores, while Mn and Ni go to the cell walls;
- the binary is the pseudo-binary CrFeCo-(MnNi), with k = 0.74 as quoted for that
  description; a laser-remelting study gives k_Mn = 0.81 and k_Ni = 0.92 (Clyne-Kurz fit);
- the solute is Mn + Ni, 40 at% of the alloy (~40 wt%);
- the freezing range is the least certain input. Li et al. use 1250-1340 C (90 K), while
  Piglione et al. (2018) call it "narrow". The study therefore runs dT0 = 40, 60 (baseline)
  and 90 K;
- D and Gamma are typical values for Fe/Ni-base liquids, the same as for IN625.

Validation target. Piglione, Dovgyy, Liu, Gourlay & Hooper, Mater. Lett. 224 (2018) 22-25,
doi:10.1016/j.matlet.2018.04.052 (open access):
- an 11-track single-layer LPBF build on 316L, Renishaw AM250 (pulsed);
- 200 W, 80 us exposure, 60 um point distance, 125 um hatch;
- cell spacing 0.52-0.64 um, mean 0.59 um;
- cells grow along <001> normal to the fusion line; build texture <001> along the build direction.
"""
import numpy as np

from tpmsfem.fem.materials import Material, ThermalProperties
from tpmsfem.process.phasefield import DiluteAlloy

LI2021 = dict(
    source="Li et al., J. Mater. Sci. Technol. 77 (2021) 187, Tables 1-2 (simulation inputs)",
    T_C=[25.0, 100.0, 200.0, 400.0, 600.0, 800.0, 1000.0],
    cp_J_kgK=[494.0, 522.0, 545.0, 570.0, 623.0, 628.0, 642.0],
    k_W_mK=[12.46, 14.19, 15.98, 18.96, 22.94, 24.71, 26.93],
    density_kg_m3=7980.0, latent_J_kg=2.20e5, solidus_C=1250.0, liquidus_C=1340.0, emissivity=0.85,
)

PIGLIONE2018 = dict(
    source="Piglione et al., Mater. Lett. 224 (2018) 22, doi:10.1016/j.matlet.2018.04.052",
    machine="Renishaw AM250 (pulsed, point-to-point)", power_W=200.0, exposure_us=80.0, point_distance_um=60.0,
    hatch_um=125.0, substrate="316L", cell_spacing_um=(0.52, 0.64), cell_spacing_mean_um=0.59,
)


def thermal_properties():
    d = LI2021
    T = np.r_[d["T_C"], 1250.0, 1340.0, 3500.0]
    k = np.r_[d["k_W_mK"], [d["k_W_mK"][-1]] * 3]
    cp = np.r_[d["cp_J_kgK"], [d["cp_J_kgK"][-1]] * 3]
    return ThermalProperties.from_si(T, k, cp, d["density_kg_m3"], d["latent_J_kg"], d["solidus_C"], d["liquidus_C"])


def material():
    # E, nu: room-temperature values of the order reported for CoCrFeMnNi; only the thermal
    # data are used by the process models
    return Material("CoCrFeMnNi", 200_000.0, 0.26, thermal=thermal_properties())


def pseudo_binary(freezing_range=60.0, k=0.74, eps4=0.02):
    """CrFeCo-(MnNi) pseudo-binary for the phase field; c_inf = 40 (wt% of Mn + Ni)."""
    return DiluteAlloy(c_inf=40.0, k=k, freezing_range=freezing_range, D=3.0e-9, gibbs_thomson=3.65e-7, eps4=eps4)


def effective_speed_m_s(p=PIGLIONE2018, delay_us=0.0):
    """Mean speed of a point-exposure scan: point distance / (exposure + delay)."""
    return p["point_distance_um"] * 1e-6 / ((p["exposure_us"] + delay_us) * 1e-6)


# ------------------------------------------------------------------ mechanical data
KIM2020 = dict(
    source="Kim, Yang & Lee, Sci. Rep. 10 (2020) 8045, Table 1: compressive yield strength of as-built SLM "
           "CoCrFeMnNi (Concept Laser Mlab, 90 W, 600 mm/s)",
    T_C=[25.0, 500.0, 600.0, 700.0, 800.0, 900.0], yield_MPa=[756.5, 594.4, 551.3, 545.8, 376.6, 223.0],
)
LAPLANCHE2015 = dict(
    source="Laplanche et al., J. Alloys Compd. 623 (2015) 348: E = 214 - 35/(exp(416/T) - 1) GPa, "
           "alpha = 23.7e-6 (1 - exp(-T/299)) 1/K (T in K), nu = 0.25",
)


def youngs_modulus_MPa(T_C):
    T = np.asarray(T_C, float) + 273.15
    return (214.0 - 35.0 / np.expm1(416.0 / T)) * 1e3


def cte_instantaneous(T_C):
    T = np.asarray(T_C, float) + 273.15
    return 23.7e-6 * (1.0 - np.exp(-T / 299.0))


def mechanical_material(hardening_MPa=1500.0):
    """Temperature-dependent elasto-plastic CoCrFeMnNi for the thermo-mechanical build model.

    - E(T) and the expansion: Laplanche et al. (2015), measured on wrought material to 1000 K
      and 1270 K; extrapolated above with their fits.
    - Yield strength: Kim et al. (2020), as-built SLM, compression, 25-900 C. Above 900 C it is
      continued to 80 MPa at 1100 C and 40 MPa at the solidus (an assumption; a lower floor
      makes the freshly deposited lump nearly strengthless and the Newton iteration fail).
    - Hardening: linear, ``hardening_MPa`` at room temperature and scaled with the yield
      strength at temperature (an assumption; no as-built flow curves were found).
    """
    from tpmsfem.fem.materials import FlowTable, TemperatureTable

    T = np.array([20.0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1250.0])
    sy = np.interp(T, np.r_[KIM2020["T_C"], 1100.0, 1250.0], np.r_[KIM2020["yield_MPa"], 80.0, 40.0])
    H = hardening_MPa * sy / sy[0]
    flow = FlowTable([0.0, 0.2], np.c_[sy, sy + 0.2 * H], T, 0.0, "Kim 2020 yield + linear hardening")
    alpha = TemperatureTable.mean_from_instantaneous(T, cte_instantaneous(T), 20.0)
    table = TemperatureTable(T=T, E=youngs_modulus_MPa(T), alpha=alpha, T_ref=20.0, T_cap=LI2021["solidus_C"])
    return Material("CoCrFeMnNi_LPBF", E=float(youngs_modulus_MPa(20.0)), nu=0.25, flow=flow, temperature=table,
                    thermal=thermal_properties(), density=LI2021["density_kg_m3"] * 1e-12)
