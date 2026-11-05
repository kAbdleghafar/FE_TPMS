"""Hip-implant load cases.

Values follow the cited standards/literature as commonly used in FE studies;
check them against the current edition of the standard you test to.
"""
from __future__ import annotations

import numpy as np

from .geometry.fields import rotation_matrix


def iso7206_4_stem(head_center, F: float = 2300.0, alpha_deg: float = 10.0, beta_deg: float = 9.0,
                   embed_depth: float = 80.0):
    """ISO 7206-4-style stem fatigue set-up, expressed in the stem frame
    (stem axis +z, neck towards -x, anterior +y).

    In the test the stem is tilted by ``alpha`` in the frontal plane and
    ``beta`` in the sagittal plane and loaded vertically through the head
    centre; the stem is potted up to ``embed_depth`` below the head centre.
    Here the stem stays fixed and the load is rotated instead. Sign
    convention: positive ``alpha`` tilts the proximal stem medially (-x),
    positive ``beta`` posteriorly (-y) -- verify against the standard's figure.

    Returns ``(force_vector (3,), z_embed)``.
    """
    R = rotation_matrix((0, 1, 0), -alpha_deg) @ rotation_matrix((1, 0, 0), beta_deg)
    d = R.T @ np.array([0.0, 0.0, -1.0])
    z_embed = float(np.asarray(head_center)[2] - embed_depth)
    return F * d, z_embed


def gait_peak_hip_force(body_weight_N: float = 750.0, bw_ratio: float = 2.5,
                        frontal_deg: float = 13.0, sagittal_deg: float = 8.0):
    """Peak hip contact force during walking (order of 2.5 x body weight,
    cf. Bergmann et al., J Biomech 2001 / OrthoLoad), directed downward and
    tilted medially/posteriorly by the given angles, in the femoral frame
    (+z superior, -x medial, +y anterior). Returns the force on the head."""
    R = rotation_matrix((0, 1, 0), frontal_deg) @ rotation_matrix((1, 0, 0), -sagittal_deg)
    return body_weight_N * bw_ratio * (R @ np.array([0.0, 0.0, -1.0]))


def head_contact_faces(mesh, head_center, head_radius, force, half_angle_deg: float = 30.0,
                       tol: float = None):
    """Boundary faces of the head on a spherical cap centred on the line of
    action of ``force`` (through the head centre). Returns (faces, face_type)."""
    faces, ftype, _ = mesh.boundary_faces()
    c = np.asarray(head_center, float)
    fc = mesh.points[faces].mean(axis=1) - c
    r = np.linalg.norm(fc, axis=1)
    tol = tol if tol is not None else 0.15 * head_radius
    on_head = np.abs(r - head_radius) < tol
    d = -np.asarray(force, float) / np.linalg.norm(force)     # contact is on the side the force pushes from
    cosang = (fc @ d) / np.maximum(r, 1e-12)
    cap = on_head & (cosang > np.cos(np.deg2rad(half_angle_deg)))
    if not cap.any():
        raise ValueError("no head faces found in the contact cap")
    return faces[cap], ftype
