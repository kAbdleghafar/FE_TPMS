"""Ti-6Al-4V for the keyhole simulations: the property set of Gan et al., Nat. Commun. 12 (2021)
2379, Supplementary Table 1 (the same properties behind their keyhole-number scaling and their
multiphysics model; temperatures there in K):

* density 4420 kg/m^3 below the liquidus (1933 K), 3920 above;
* c_p = 0.1734 T + 452.72 J/(kg K) below the liquidus, 830 above;
* k = 0.0136 T + 1.3097 W/(m K) below the liquidus, 33.4 above;
* solidus 1877 K, liquidus 1933 K, latent heat of melting 2.86e5 J/kg;
* boiling point 3560 K, latent heat of vaporisation 9.255e6 J/kg, molar mass 48 g/mol;
* surface tension 1.65 N/m, d sigma / dT = -2.6e-4 N/(m K), viscosity 2.66e-3 Pa s;
* emissivity 0.3, flat-melt absorptivity 0.26 at 1070 nm.

The lattice works with one density, the liquid's (the flow is in the liquid). The solid's
larger volumetric heat capacity is kept by scaling its c_p with rho_solid / rho_liquid, so
the heat equation sees the measured rho c_p everywhere.

Validation data: the X-ray keyhole depths of Cunningham et al., Science 363 (2019) 849
(bare Ti-6Al-4V plates, argon, 1070 nm, 1/e^2 radius 47.5 or 70 um), as compiled in Gan et
al.'s Supplementary Data 1 (data/gan2021_keyhole_xray.json).
"""
import numpy as np

from tpmsfem.fem.materials import Material, ThermalProperties
from tpmsfem.process import Evaporation

K0 = 273.15
GAN2021 = dict(
    source="Gan et al., Nat. Commun. 12 (2021) 2379, Supplementary Table 1",
    rho_solid=4420.0, rho_liquid=3920.0, T_solidus_K=1877.0, T_liquidus_K=1933.0, latent_J_kg=2.86e5,
    T_boil_K=3560.0, L_v_J_kg=9.255e6, M_kg_mol=0.048, sigma_N_m=1.65, dsigma_dT=-2.6e-4, viscosity_Pa_s=2.66e-3,
    emissivity=0.3, eta_min=0.26,
)


def thermal_properties():
    d = GAN2021
    TK = np.r_[np.arange(300.0, 1933.0, 50.0), 1932.9, 1933.0, 6000.0]
    cp = np.where(TK < d["T_liquidus_K"], 0.1734 * TK + 452.72, 830.0)
    k = np.where(TK < d["T_liquidus_K"], 0.0136 * TK + 1.3097, 33.4)
    rho = np.where(TK < d["T_liquidus_K"], d["rho_solid"], d["rho_liquid"])
    cp_eff = cp * rho / d["rho_liquid"]                 # keep rho c_p with one lattice density
    return ThermalProperties.from_si(TK - K0, k, cp_eff, d["rho_liquid"], d["latent_J_kg"],
                                     d["T_solidus_K"] - K0, d["T_liquidus_K"] - K0)


def material():
    return Material("Ti-6Al-4V", 110_000.0, 0.34, thermal=thermal_properties(), density=GAN2021["rho_liquid"] * 1e-12)


def evaporation():
    d = GAN2021
    return Evaporation(T_boil=d["T_boil_K"] - K0, L_v=d["L_v_J_kg"], M=d["M_kg_mol"])
