import ast
from pathlib import Path
from unittest.mock import patch

import pytest
import torch
import yaml
from torch import nn

from pointforward.models import (
    PointForward,
    PointForwardRec,
    PointForwardSH,
    PointForwardVariableViewSH,
    SpatialTemporalFusion,
    VariableViewSpatialTemporalFusion,
)
from pointforward.variants import build_model, resolve_entrypoint, resolve_model_class
from scripts.train_rec import _warm_start, build_windows_dataset


CONFIG_DIR = Path(__file__).parents[1] / "configs" / "variants"


def test_offline_temporal_attention_matches_training():
    from pointforward.models.scene_fusion import SceneTemporalAttention
    attention = SceneTemporalAttention(8, 8, heads=2, layers=1, causal_inference=False)
    tokens = torch.randn(4, 8)
    training, _ = attention.train()(tokens)
    inference, _ = attention.eval()(tokens)
    torch.testing.assert_close(training, inference)


def test_offline_gaussians_do_not_pop_at_input_time():
    from pointforward.models.pointforward import Gaussians
    from pointforward.models.scene_fusion import SceneTemporalAttention
    model = PointForwardRec.__new__(PointForwardRec)
    nn.Module.__init__(model)
    model.temporal_attention = SceneTemporalAttention(8, 8, heads=2, layers=1, causal_inference=False)
    model.gaussian_voxel_size = 0.5
    model.eval()
    gaussians = Gaussians(
        torch.zeros(2, 3), torch.tensor([[1., 0., 0., 0.]]).repeat(2, 1),
        torch.ones(2, 3), torch.ones(2), torch.ones(2, 3),
        torch.ones(2), torch.tensor([0., 0.5]), torch.full((2,), -1),
    )
    before = model.pose_gaussians(gaussians, None, 0, 0.49999)["opacities"]
    after = model.pose_gaussians(gaussians, None, 0, 0.50001)["opacities"]
    assert before.min() > 0
    torch.testing.assert_close(before, after, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize(
    ("filename", "expected"),
    [
        ("01_base.yaml", PointForward),
        ("02_sh.yaml", PointForwardSH),
        ("03_1_rec_cross_attention.yaml", PointForwardRec),
        ("03_2_rec_temporal_attention.yaml", PointForwardRec),
        ("04_crossdata_arbitrary_views.yaml", PointForwardVariableViewSH),
        ("05_argoverse_sh.yaml", PointForwardVariableViewSH),
    ],
)
def test_canonical_config_resolves_model_and_entrypoints(filename, expected):
    config = yaml.safe_load((CONFIG_DIR / filename).read_text())
    assert resolve_model_class(config) is expected
    assert resolve_entrypoint(config, "train") == config["variant"]["train_module"]
    assert resolve_entrypoint(config, "infer") == config["variant"]["infer_module"]


def test_rec_temporal_attention_is_explicitly_separated():
    cross = yaml.safe_load((CONFIG_DIR / "03_1_rec_cross_attention.yaml").read_text())
    temporal = yaml.safe_load((CONFIG_DIR / "03_2_rec_temporal_attention.yaml").read_text())
    assert not cross["rec"].get("temporal_attention")
    assert temporal["rec"]["temporal_attention"]["layers"] > 0


def test_fixed_and_variable_view_heads_have_distinct_shapes():
    fixed = SpatialTemporalFusion(feat_dim=8, hidden=8, num_views=5, grad_checkpoint=False)
    variable = VariableViewSpatialTemporalFusion(feat_dim=8, hidden=8, grad_checkpoint=False)
    assert fixed.phi_v[0].in_features == 5
    assert variable.phi_v[0].in_features == 2


def test_variable_view_pooling_is_permutation_and_duplicate_count_invariant():
    torch.manual_seed(0)
    fusion = VariableViewSpatialTemporalFusion(feat_dim=8, hidden=8, grad_checkpoint=False)
    values = torch.tensor([[1.0, 4.0], [3.0, 2.0], [2.0, 8.0]])
    visible = torch.ones_like(values, dtype=torch.bool)
    expected = fusion._aggregate_sigma(values, visible)
    permutation = torch.tensor([2, 0, 1])
    assert torch.allclose(expected, fusion._aggregate_sigma(values[permutation], visible[permutation]))
    duplicated = values.repeat_interleave(2, dim=0)
    duplicated_visible = visible.repeat_interleave(2, dim=0)
    assert torch.allclose(expected, fusion._aggregate_sigma(duplicated, duplicated_visible))


def test_unknown_variant_is_rejected():
    with pytest.raises(ValueError, match="unknown model.variant"):
        resolve_model_class({"model": {"variant": "legacy_guess"}})


def test_legacy_config_without_variant_defaults_to_base():
    assert resolve_model_class({"model": {}}) is PointForward


@pytest.mark.parametrize(
    ("filename", "module"),
    [
        ("train_crossdata.py", "scripts.train_sh"),
        ("train_argoverse.py", "scripts.train_sh"),
        ("infer_crossdata.py", "scripts.infer_sh"),
        ("infer_argoverse.py", "scripts.infer_sh"),
    ],
)
def test_sh_wrappers_delegate_to_full_entrypoints(filename, module):
    path = Path(__file__).parents[1] / "scripts" / filename
    tree = ast.parse(path.read_text())
    imports = {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and any(alias.name == "main" for alias in node.names)
    }
    assert imports == {module}


def test_build_model_passes_inference_and_sh_options_to_resolved_class():
    captured = {}

    class CaptureSH(PointForwardSH):
        def __init__(self, **kwargs):
            nn.Module.__init__(self)
            captured.update(kwargs)

    config = {
        "data": {"num_context": 4, "cameras": [0, 1, 2, 3, 4]},
        "model": {
            "variant": "crossdata_sh_variable", "sh_degree": 2,
            "grad_checkpoint": True, "pretrained_backbone": False,
        },
    }
    with patch("pointforward.variants.resolve_model_class", return_value=CaptureSH):
        model = build_model(config, nn.Identity(), inference=True)
    assert isinstance(model, CaptureSH)
    assert captured["num_views"] == 20
    assert captured["sh_degree"] == 2
    assert captured["grad_checkpoint"] is False
    assert "color_dim" not in captured


def test_rec_window_builder_preserves_canonical_dataset_options():
    captured = {}

    class CaptureWindows:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    config = {
        "seed": 7,
        "rec": {"num_windows": 3},
        "data": {
            "layout": "drivestudio", "root": "/unused", "train_scenes": ["001"],
            "clip_stride": 20, "variable_views": True, "min_cameras": 3,
            "rigid_classes": ["CAR"], "apply_opencv2waymo": False,
        },
    }
    with patch("scripts.train_rec.WaymoDriveStudioWindows", CaptureWindows):
        build_windows_dataset(config, "train")
    assert captured["clip_stride"] == 20
    assert captured["variable_views"] is True
    assert captured["min_cameras"] == 3
    assert captured["rigid_classes"] == ("CAR",)
    assert captured["apply_opencv2waymo"] is False
    assert captured["num_windows"] == 3


def test_rec_warm_start_filters_shape_mismatches():
    model = nn.Linear(3, 2)
    state = {"weight": torch.zeros(4, 3), "bias": torch.ones(2), "extra": torch.ones(1)}
    missing, unexpected, incompatible = _warm_start(model, state)
    assert incompatible == ["weight"]
    assert "weight" in missing
    assert unexpected == ["extra"]
    torch.testing.assert_close(model.bias, torch.ones(2))