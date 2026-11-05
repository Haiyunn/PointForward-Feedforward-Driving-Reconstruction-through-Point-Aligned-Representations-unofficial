"""Model components of PointForward."""

from .backbone import DEFAULT_BACKBONE, FeatureExtractor, sample_features
from .depth import ConstantDepth, FrozenInfiniDepth, dense_query_coord
from .fusion import SpatialTemporalFusion, box_dynamic_masks, project_points
from .fusion_variable import VariableViewSpatialTemporalFusion
from .gaussian_head import GaussianDecoder, temporal_opacity
from .gaussian_head_sh import GaussianSHDecoder
from .pointforward import Gaussians, PointForward, SceneInput, TargetViews
from .pointforward_rec import PointForwardRec, ReconstructionCache
from .pointforward_sh import PointForwardSH, PointForwardVariableViewSH
from .query import (
    QUERY_DIM,
    build_dense_queries,
    camera_rays,
    plucker,
    query_plucker,
    query_rgb,
    query_xyz,
    set_query_xyz,
    unproject,
    unproject_z,
)
from .renderer import FeatureToRGB, render_features
from .rotation import matrix_to_quaternion, quaternion_multiply, quaternion_to_matrix
from .scene_graph import (
    InstanceBoxes,
    assign_instances,
    canonicalise,
    decanonicalise,
    from_box_local,
    inside_box,
    sample_queries,
    to_box_local,
)

__all__ = [
    "DEFAULT_BACKBONE",
    "FeatureExtractor",
    "sample_features",
    "ConstantDepth",
    "FrozenInfiniDepth",
    "dense_query_coord",
    "SpatialTemporalFusion",
    "VariableViewSpatialTemporalFusion",
    "box_dynamic_masks",
    "project_points",
    "GaussianDecoder",
    "GaussianSHDecoder",
    "temporal_opacity",
    "Gaussians",
    "PointForward",
    "PointForwardSH",
    "PointForwardVariableViewSH",
    "PointForwardRec",
    "ReconstructionCache",
    "SceneInput",
    "TargetViews",
    "QUERY_DIM",
    "build_dense_queries",
    "camera_rays",
    "plucker",
    "query_plucker",
    "query_rgb",
    "query_xyz",
    "set_query_xyz",
    "unproject",
    "unproject_z",
    "FeatureToRGB",
    "render_features",
    "matrix_to_quaternion",
    "quaternion_multiply",
    "quaternion_to_matrix",
    "InstanceBoxes",
    "assign_instances",
    "canonicalise",
    "decanonicalise",
    "from_box_local",
    "inside_box",
    "sample_queries",
    "to_box_local",
]
