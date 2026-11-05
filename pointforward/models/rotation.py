"""Quaternion / rotation-matrix helpers.

Convention throughout: a quaternion is stored as (w, x, y, z) and the matrix it
maps to acts on COLUMN vectors, matching gsplat's expectation.

Note the scene-graph convention (see :mod:`.scene_graph`): points are row
vectors, so a box rotation ``R`` maps world -> box-local in column form and
``R.T`` maps box-local -> world.  When a dynamic Gaussian is pushed from
canonical space into the world at some timestamp, the linear part of that rigid
transform is ``A = R_box.T @ R_can``, and the Gaussian's own orientation matrix
composes on the left as ``A @ G``.
"""

from __future__ import annotations

import torch
from torch import Tensor


def quaternion_to_matrix(quats: Tensor) -> Tensor:
    """(..., 4) unit quaternions (w, x, y, z) -> (..., 3, 3) rotation matrices."""
    q = torch.nn.functional.normalize(quats, dim=-1)
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]

    ww, xx, yy, zz = w * w, x * x, y * y, z * z
    wx, wy, wz = w * x, w * y, w * z
    xy, xz, yz = x * y, x * z, y * z

    mat = torch.stack(
        (
            ww + xx - yy - zz, 2 * (xy - wz), 2 * (xz + wy),
            2 * (xy + wz), ww - xx + yy - zz, 2 * (yz - wx),
            2 * (xz - wy), 2 * (yz + wx), ww - xx - yy + zz,
        ),
        dim=-1,
    )
    return mat.reshape(*q.shape[:-1], 3, 3)


def matrix_to_quaternion(mat: Tensor) -> Tensor:
    """(..., 3, 3) rotation matrices -> (..., 4) quaternions (w, x, y, z).

    Uses the branch-free formulation of Shepperd's method: build the four
    candidate quaternions and pick, per element, the one whose leading term is
    largest.  This avoids the numerical blow-up of the naive ``w = sqrt(1+tr)/2``
    formula near 180-degree rotations.
    """
    m00, m01, m02 = mat[..., 0, 0], mat[..., 0, 1], mat[..., 0, 2]
    m10, m11, m12 = mat[..., 1, 0], mat[..., 1, 1], mat[..., 1, 2]
    m20, m21, m22 = mat[..., 2, 0], mat[..., 2, 1], mat[..., 2, 2]

    # Squared magnitudes of 2w, 2x, 2y, 2z (up to a common factor).
    q_abs = torch.stack(
        (
            1.0 + m00 + m11 + m22,
            1.0 + m00 - m11 - m22,
            1.0 - m00 + m11 - m22,
            1.0 - m00 - m11 + m22,
        ),
        dim=-1,
    ).clamp_min(0.0).sqrt()

    candidates = torch.stack(
        (
            torch.stack((q_abs[..., 0] ** 2, m21 - m12, m02 - m20, m10 - m01), dim=-1),
            torch.stack((m21 - m12, q_abs[..., 1] ** 2, m10 + m01, m02 + m20), dim=-1),
            torch.stack((m02 - m20, m10 + m01, q_abs[..., 2] ** 2, m12 + m21), dim=-1),
            torch.stack((m10 - m01, m02 + m20, m12 + m21, q_abs[..., 3] ** 2), dim=-1),
        ),
        dim=-2,
    )  # (..., 4, 4)

    # Normalise each candidate by 2 * its leading term, then take the best.
    denom = (2.0 * q_abs).unsqueeze(-1).clamp_min(1e-8)
    candidates = candidates / denom

    best = q_abs.argmax(dim=-1, keepdim=True)
    quat = torch.gather(candidates, -2, best.unsqueeze(-1).expand(*best.shape, 4)).squeeze(-2)
    return torch.nn.functional.normalize(quat, dim=-1)


def quaternion_multiply(a: Tensor, b: Tensor) -> Tensor:
    """Hamilton product a (x) b for (..., 4) quaternions (w, x, y, z).

    Corresponds to applying rotation ``b`` first, then ``a``.
    """
    aw, ax, ay, az = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    bw, bx, by, bz = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    return torch.stack(
        (
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ),
        dim=-1,
    )
