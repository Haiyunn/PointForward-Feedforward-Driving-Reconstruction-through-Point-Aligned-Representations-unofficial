"""Canonical model and entrypoint resolution for all PointForward variants."""

from __future__ import annotations

from torch import nn

from .models.pointforward import PointForward
from .models.pointforward_rec import PointForwardRec
from .models.pointforward_sh import PointForwardSH, PointForwardVariableViewSH


MODEL_VARIANTS = {
    "base": PointForward,
    "sh": PointForwardSH,
    "rec_cross_attention": PointForwardRec,
    "rec_temporal_attention": PointForwardRec,
    "crossdata_sh_variable": PointForwardVariableViewSH,
    "argoverse_sh_variable": PointForwardVariableViewSH,
}


def resolve_model_class(config: dict) -> type[nn.Module]:
    """Resolve ``model.variant`` and reject unknown or missing variants."""
    name = config.get("model", {}).get("variant", "base")
    if name not in MODEL_VARIANTS:
        choices = ", ".join(MODEL_VARIANTS)
        raise ValueError(f"unknown model.variant {name!r}; expected one of: {choices}")
    return MODEL_VARIANTS[name]


def build_model(config: dict, depth_model: nn.Module, *, inference: bool = False) -> nn.Module:
    """Build the exact configured model class without loading a checkpoint."""
    model_config = config["model"]
    model_class = resolve_model_class(config)
    num_views = config["data"].get("num_context", 4) * len(
        config["data"].get("cameras", (0, 1, 2))
    )
    kwargs = dict(
        depth_model=depth_model,
        backbone=model_config.get("backbone", "tf_efficientnetv2_s"),
        feat_dim=model_config.get("feat_dim", 128),
        hidden=model_config.get("hidden", 128),
        num_static=model_config.get("num_static", 100_000),
        num_dynamic=model_config.get("num_dynamic", 10_000),
        num_views=num_views,
        use_weighted_fusion=model_config.get("use_weighted_fusion", True),
        use_depth_cue=model_config.get("use_depth_cue", True),
        pretrained_backbone=model_config.get("pretrained_backbone", True),
        fusion_chunk=model_config.get("fusion_chunk", 65_536),
        decoder_chunk=model_config.get("decoder_chunk", 262_144),
        grad_checkpoint=False if inference else model_config.get("grad_checkpoint", True),
        depth_view_chunk=model_config.get("depth_view_chunk", 0),
        backbone_view_chunk=model_config.get("backbone_view_chunk", 0),
        stream_queries=model_config.get("stream_queries", False),
        freeze_backbone_bn=model_config.get("freeze_backbone_bn", False),
        refiner_view_chunk=model_config.get("refiner_view_chunk", 0),
    )
    if issubclass(model_class, PointForwardSH):
        kwargs["sh_degree"] = model_config.get("sh_degree", 3)
    else:
        kwargs["color_dim"] = model_config.get("color_dim", 128)
    if model_class is PointForwardRec:
        rec = config.get("rec", {})
        kwargs.update(
            scene_fusion=rec.get("scene_fusion", {}),
            gaussian_map=rec.get("gaussian_map", {}),
            temporal_attention=rec.get("temporal_attention"),
        )
    return model_class(**kwargs)


def resolve_entrypoint(config: dict, mode: str) -> str:
    """Return the configured train or inference module."""
    metadata = config.get("variant", {})
    key = f"{mode}_module"
    module = metadata.get(key)
    if not isinstance(module, str) or not module:
        raise ValueError(f"variant.{key} must be a nonempty module name")
    return module