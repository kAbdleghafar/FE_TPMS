"""Closed-form reference solutions used for verification (code vs. math)."""
from __future__ import annotations

import numpy as np


# --------------------------------------------------------------- Lame sphere
def lame_sphere(r, a, b, p, E, nu):
    """Thick-walled sphere, inner radius a, outer b, internal pressure p.
    Returns (sigma_r, sigma_theta, u_r)."""
    r = np.asarray(r, float)
    k = p * a**3 / (b**3 - a**3)
    sr = k * (1 - b**3 / r**3)
    st = k * (1 + b**3 / (2 * r**3))
    ur = k * r / E * ((1 - 2 * nu) + (1 + nu) * b**3 / (2 * r**3))
    return sr, st, ur


def hill_sphere_pressure(c, a, b, sy):
    """Internal pressure of an elastic-perfectly-plastic (Mises/Tresca) thick
    sphere whose plastic zone extends to radius c (Hill, 1950)."""
    return 2 * sy * (np.log(c / a) + (1 - c**3 / b**3) / 3)


def hill_sphere_outer_displacement(c, b, sy, E, nu):
    """Radial displacement of the outer surface for plastic radius c."""
    return sy * (1 - nu) * c**3 / (E * b**2)


# --------------------------------------------------------------- strips/bars
def bilayer_curvature(delta_eps, h):
    """Free curvature of a homogeneous strip of thickness h whose upper half
    carries an extra axial eigenstrain delta_eps (nu = 0, beam theory)."""
    return 1.5 * delta_eps / h


def clamped_isotropic_eigenstress(eps, E, nu):
    """Stress in a fully clamped body with isotropic eigenstrain eps."""
    return -E / (1 - 2 * nu) * eps


def uniaxial_linear_hardening(strain, E, sy, H):
    """Uniaxial stress for monotonic loading, J2 with linear isotropic hardening."""
    strain = np.asarray(strain, float)
    ey = sy / E
    return np.where(strain <= ey, E * strain, sy + E * H / (E + H) * (strain - ey))


def cantilever_tip_deflection(P, L, E, nu, b, h, timoshenko=True):
    """Tip deflection of a rectangular cantilever (Euler-Bernoulli + shear)."""
    I = b * h**3 / 12
    d = P * L**3 / (3 * E * I)
    if timoshenko:
        G = E / (2 * (1 + nu))
        d += P * L / (5 / 6 * G * b * h)
    return d


# --------------------------------------------------------------- meshes
def gmsh_sphere_shell_octant(a, b, size, order=2):
    """Tet mesh of the first octant of a thick sphere (for symmetric BCs)."""
    import os
    import tempfile

    import gmsh

    from ..mesh.mesh import read_mesh

    tmp = tempfile.mkdtemp()
    path = os.path.join(tmp, "oct.msh")
    gmsh.initialize(interruptible=False)
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        occ = gmsh.model.occ
        so = occ.addSphere(0, 0, 0, b)
        si = occ.addSphere(0, 0, 0, a)
        shell, _ = occ.cut([(3, so)], [(3, si)])
        box = occ.addBox(0, 0, 0, 2 * b, 2 * b, 2 * b)
        occ.intersect(shell, [(3, box)])
        occ.synchronize()
        gmsh.option.setNumber("Mesh.MeshSizeMax", size)
        gmsh.option.setNumber("Mesh.MeshSizeMin", size * 0.5)
        gmsh.model.mesh.generate(3)
        gmsh.model.mesh.setOrder(order)
        gmsh.write(path)
    finally:
        gmsh.finalize()
    return read_mesh(path, "tet10" if order == 2 else "tet4")
