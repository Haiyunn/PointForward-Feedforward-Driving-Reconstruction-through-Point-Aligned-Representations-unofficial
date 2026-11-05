"""Point-aligned query construction (paper Sec. 3.1).

Builds the initial dense query set Q_0 = {q_{i,j}} by lifting every pixel of
every input view into world space using the (frozen) predicted depth and the
camera parameters.  Each query is the 12-dimensional vector

    q = [ xyz (3) | rgb (3) | plucker (6) ]

where xyz is the world-space coordinate of the pixel, rgb its colour, and
plucker the 6-D Plucker coordinate of the corresponding camera ray.
"""

from __future__ import annotations

import torch
from torch import Tensor

QUERY_DIM = 12
_XYZ = slice(0, 3)
_RGB = slice(3, 6)
_PLUCKER = slice(6, 12)


def pixel_grid(height: int, width: int, device: torch.device, dtype: torch.dtype) -> Tensor:
    """Pixel centre coordinates as an (H, W, 2) tensor of (u, v)."""
    v, u = torch.meshgrid(
        torch.arange(height, device=device, dtype=dtype),
        torch.arange(width, device=device, dtype=dtype),
        indexing="ij",
    )
    return torch.stack((u, v), dim=-1) + 0.5


def camera_rays(
    height: int,
    width: int,
    intrinsics: Tensor,
    cam_to_world: Tensor,
) -> tuple[Tensor, Tensor]:
    """Per-pixel ray origins and unit directions in world space.

    Args:
        height, width: image size.
        intrinsics:   (..., 3, 3) pinhole K.
        cam_to_world: (..., 4, 4) camera-to-world transform.

    Returns:
        origins    (..., H, W, 3) -- identical per view, broadcast for convenience.
        directions (..., H, W, 3) -- unit norm.
    """
    batch = intrinsics.shape[:-2]
    device, dtype = intrinsics.device, intrinsics.dtype

    uv = pixel_grid(height, width, device, dtype)  # (H, W, 2)
    ones = torch.ones_like(uv[..., :1])
    uv1 = torch.cat((uv, ones), dim=-1)  # (H, W, 3)

    k_inv = torch.linalg.inv(intrinsics)  # (..., 3, 3)
    # Camera-space direction for each pixel.
    dirs_cam = torch.einsum("...ij,hwj->...hwi", k_inv, uv1)

    rot = cam_to_world[..., :3, :3]  # (..., 3, 3)
    trans = cam_to_world[..., :3, 3]  # (..., 3)

    dirs_world = torch.einsum("...ij,...hwj->...hwi", rot, dirs_cam)
    dirs_world = dirs_world / dirs_world.norm(dim=-1, keepdim=True).clamp_min(1e-8)

    origins = trans.reshape(*batch, 1, 1, 3).expand(*batch, height, width, 3)
    return origins, dirs_world


def plucker(origins: Tensor, directions: Tensor) -> Tensor:
    """6-D Plucker coordinates (d, o x d) for rays given as origin + unit direction."""
    moment = torch.cross(origins, directions, dim=-1)
    return torch.cat((directions, moment), dim=-1)


def unproject(depth: Tensor, origins: Tensor, directions: Tensor) -> Tensor:
    """Lift per-pixel depth along the camera rays into world space.

    ``depth`` is interpreted as distance along the *ray direction*.  Because
    ``directions`` are normalised, a z-buffer depth must be converted by the
    caller; :func:`unproject_z` does that.

    Args:
        depth:      (..., H, W)
        origins:    (..., H, W, 3)
        directions: (..., H, W, 3)

    Returns:
        (..., H, W, 3) world-space points.
    """
    return origins + directions * depth.unsqueeze(-1)


def unproject_z(depth_z: Tensor, origins: Tensor, directions: Tensor, cam_forward: Tensor) -> Tensor:
    """Lift z-buffer depth (distance along the optical axis) into world space.

    Args:
        depth_z:     (..., H, W) depth measured along the camera forward axis.
        origins:     (..., H, W, 3)
        directions:  (..., H, W, 3) unit world-space ray directions.
        cam_forward: (..., 3) unit world-space camera forward (+Z) axis.

    Returns:
        (..., H, W, 3) world-space points.
    """
    # cos between ray and optical axis converts z-depth to ray distance.
    cos = torch.einsum("...hwi,...i->...hw", directions, cam_forward).clamp_min(1e-6)
    return unproject(depth_z / cos, origins, directions)


def build_dense_queries(
    images: Tensor,
    depth: Tensor,
    intrinsics: Tensor,
    cam_to_world: Tensor,
    z_depth: bool = True,
) -> tuple[Tensor, Tensor]:
    """Assemble the dense query set Q_0 for a single scene.

    Args:
        images:       (V, 3, H, W) in [0, 1].
        depth:        (V, H, W) predicted (or LiDAR-completed) depth.
        intrinsics:   (V, 3, 3)
        cam_to_world: (V, 4, 4)
        z_depth:      True if ``depth`` is along the optical axis, False if it is
                      distance along the ray.

    Returns:
        queries (V * H * W, 12) -- the 12-D query vectors.
        view_id (V * H * W,)    -- source view index of each query.
    """
    v, _, h, w = images.shape
    origins, directions = camera_rays(h, w, intrinsics, cam_to_world)

    if z_depth:
        forward = cam_to_world[:, :3, 2]
        forward = forward / forward.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        xyz = unproject_z(depth, origins, directions, forward)
    else:
        xyz = unproject(depth, origins, directions)

    rgb = images.permute(0, 2, 3, 1)  # (V, H, W, 3)
    plk = plucker(origins, directions)  # (V, H, W, 6)

    queries = torch.cat((xyz, rgb, plk), dim=-1).reshape(v * h * w, QUERY_DIM)
    view_id = (
        torch.arange(v, device=images.device)
        .view(v, 1)
        .expand(v, h * w)
        .reshape(-1)
    )
    return queries, view_id


def query_xyz(queries: Tensor) -> Tensor:
    """World (or canonical) 3D coordinate slice of a query tensor."""
    return queries[..., _XYZ]


def query_rgb(queries: Tensor) -> Tensor:
    return queries[..., _RGB]


def query_plucker(queries: Tensor) -> Tensor:
    return queries[..., _PLUCKER]


def set_query_xyz(queries: Tensor, xyz: Tensor) -> Tensor:
    """Return a copy of ``queries`` with the coordinate slice replaced."""
    out = queries.clone()
    out[..., _XYZ] = xyz
    return out
