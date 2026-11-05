import torch

from pointforward.temporal_consistency import TemporalRenderConsistencyLoss


def sample():
    return dict(rgb=torch.ones(2, 3, 8, 8, requires_grad=True),
                truth=torch.ones(2, 3, 8, 8), depth=torch.full((2, 1, 8, 8), 2., requires_grad=True),
                alpha=torch.ones(2, 1, 8, 8), static=torch.ones(2, 1, 8, 8),
                intrinsic=torch.eye(3).repeat(2, 1, 1), pose=torch.eye(4).repeat(2, 1, 1))


def test_identity_and_cross_window():
    criterion = TemporalRenderConsistencyLoss()
    current = sample()
    losses, previous = criterion(current, 1)
    assert losses["consistency_loss"].item() == 0
    assert losses["consistency_coverage"].item() == 1
    assert not previous["rgb"].requires_grad
    losses, _ = criterion(current, 1, previous)
    losses["consistency_loss"].backward()
    assert torch.isfinite(current["depth"].grad).all()


def test_flicker_and_geometry_have_gradients():
    criterion = TemporalRenderConsistencyLoss()
    current = sample()
    with torch.no_grad():
        current["rgb"][1] += 0.1
        current["depth"][1] += 0.01
    losses, _ = criterion(current, 1)
    assert losses["consistency_rgb"] > 0
    assert losses["consistency_depth"] > 0
    losses["consistency_loss"].backward()
    assert current["rgb"].grad.abs().sum() > 0
    assert current["depth"].grad.abs().sum() > 0


def test_dynamic_exclusion_and_empty_visibility():
    criterion = TemporalRenderConsistencyLoss()
    current = sample()
    current["static"].zero_()
    losses, _ = criterion(current, 1)
    assert losses["consistency_coverage"] == 0
    assert losses["consistency_depth"] == 0
    assert torch.isfinite(losses["consistency_loss"])


def test_cross_window_different_resolutions():
    criterion = TemporalRenderConsistencyLoss()
    current = sample()
    _, previous = criterion(current, 1)
    for key in ("rgb", "truth", "depth", "alpha", "static"):
        current[key] = torch.nn.functional.interpolate(current[key], size=(4, 4), mode="nearest")
    current["intrinsic"][:, :2] *= 0.5
    losses, _ = criterion(current, 1, previous)
    assert losses["consistency_loss"].item() == 0