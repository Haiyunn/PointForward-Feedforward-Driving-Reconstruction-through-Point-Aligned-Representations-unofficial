"""CPU contracts for experimental, feed-forward Gaussian topology."""

import os
import unittest
from unittest.mock import patch

import torch
from torch import nn

from pointforward.models.gaussian_topology import (
    ACTIONS, KEEP, UPDATE, SPLIT, CLONE, MERGE, GaussianTopology, GaussianTopologyPolicy,
)
from pointforward.models.pointforward import Gaussians, TargetViews
from pointforward.variants import build_model
from scripts.train import initialise_from_checkpoint
from test_long_context_model import make_model, make_scene, DummyBackbone, DummyDepth


def make_gaussians():
    return Gaussians(
        means=torch.tensor([[0.01, 0.01, 2.01], [0.02, 0.01, 2.01], [1., 1., 3.]]),
        quats=torch.tensor([[1., 0., 0., 0.]]).repeat(3, 1),
        scales=torch.tensor([[0.05, 0.2, 0.05]]).repeat(3, 1),
        opacities=torch.tensor([0.3, 0.7, 0.5]), colors=torch.randn(3, 4),
        sigma=torch.ones(3), times=torch.zeros(3), instance_id=torch.full((3,), -1),
    )


def force_action(topology, action):
    with torch.no_grad():
        topology.policy.action_head.weight.zero_()
        topology.policy.action_head.bias.fill_(-20)
        topology.policy.action_head.bias[action] = 20
        topology.policy.pair_head[-1].weight.zero_()
        topology.policy.pair_head[-1].bias.fill_(20)


class GaussianTopologyTests(unittest.TestCase):
    def test_policy_shapes_keep_initialisation_and_action_gradients(self):
        torch.manual_seed(12)
        policy = GaussianTopologyPolicy(8)
        features = torch.randn(6, 8, requires_grad=True)
        prediction = policy(features, torch.full((6, 3), 0.05), torch.full((6,), 0.5))
        self.assertEqual(prediction["logits"].shape, (6, len(ACTIONS)))
        self.assertTrue((prediction["logits"].argmax(-1) == KEEP).all())
        torch.testing.assert_close(prediction["probabilities"].sum(-1), torch.ones(6))
        objective = (prediction["probabilities"] * torch.arange(5.)).sum()
        objective.backward()
        self.assertGreater(float(policy.action_head.weight.grad.abs().sum()), 0)

    def test_pair_score_is_symmetric(self):
        policy = GaussianTopologyPolicy(8)
        with torch.no_grad():
            policy.pair_head[-1].weight.normal_()
        left, right = torch.randn(4, 8), torch.randn(4, 8)
        distance = torch.rand(4)
        torch.testing.assert_close(
            policy.pair_logits(left, right, distance),
            policy.pair_logits(right, left, distance),
        )

    def test_initial_merge_candidate_exceeds_rasterizer_alpha_cutoff(self):
        topology = GaussianTopology(8, 4).train()
        result = topology(make_gaussians(), torch.ones(3, 8), torch.tensor([0, 1, 0]))
        merged = result.extras["topology_action"] == MERGE
        self.assertTrue(merged.any())
        self.assertTrue((result.opacities[merged] > 1 / 255).all())

    def test_keep_preserves_all_attributes_and_metadata(self):
        topology = GaussianTopology(8, 4).eval()
        source = make_gaussians()
        source.extras["tag"] = torch.arange(3)
        actual = topology(source, torch.ones(3, 8), torch.tensor([0, 1, 0]))
        for name in ("means", "quats", "scales", "opacities", "colors", "sigma", "times", "instance_id"):
            torch.testing.assert_close(getattr(actual, name), getattr(source, name))
        torch.testing.assert_close(actual.extras["tag"], source.extras["tag"])

    def test_split_replaces_parent_along_rotated_major_axis(self):
        topology = GaussianTopology(8, 4).eval()
        force_action(topology, SPLIT)
        source = make_gaussians()
        source.quats[:] = torch.tensor([2 ** -0.5, 0., 0., 2 ** -0.5])
        actual = topology(source, torch.ones(3, 8), torch.tensor([0, 1, 0]))
        self.assertEqual(actual.count, 6)
        torch.testing.assert_close(actual.means[:3] + actual.means[3:], 2 * source.means)
        torch.testing.assert_close((actual.means[:3] - source.means)[:, 0].abs(), torch.full((3,), 0.1))
        torch.testing.assert_close(actual.means[:3, 1:], source.means[:, 1:])
        torch.testing.assert_close(actual.scales[:3, 0], source.scales[:, 0])
        self.assertTrue((actual.scales[:3, 1] < source.scales[:, 1]).all())
        torch.testing.assert_close((1 - actual.opacities[:3]) ** 2, 1 - source.opacities)
        self.assertTrue((actual.extras["topology_action"] == SPLIT).all())

    def test_clone_keeps_parent_position_and_adds_distinct_child(self):
        topology = GaussianTopology(8, 4).eval()
        force_action(topology, CLONE)
        source = make_gaussians()
        actual = topology(source, torch.ones(3, 8), torch.tensor([0, 1, 0]))
        self.assertEqual(actual.count, 6)
        torch.testing.assert_close(actual.means[:3], source.means)
        self.assertTrue(((actual.means[3:] - source.means).norm(dim=-1) > 0).all())
        torch.testing.assert_close(actual.scales[:3], source.scales)

    def test_update_changes_parameters_without_changing_count(self):
        topology = GaussianTopology(8, 4).eval()
        force_action(topology, UPDATE)
        with torch.no_grad():
            topology.residual[-1].bias.fill_(0.5)
        source = make_gaussians()
        actual = topology(source, torch.ones(3, 8), torch.tensor([0, 1, 0]))
        self.assertEqual(actual.count, 3)
        for name in ("means", "quats", "scales", "opacities", "colors", "sigma"):
            self.assertFalse(torch.allclose(getattr(actual, name), getattr(source, name)), name)
        torch.testing.assert_close(actual.quats.norm(dim=-1), torch.ones(3))
        self.assertLessEqual(float((actual.means - source.means).abs().max().detach()), topology.max_update)

    def test_merge_reduces_count_and_unmatched_intent_falls_back_to_keep(self):
        topology = GaussianTopology(8, 4).eval()
        force_action(topology, MERGE)
        source = make_gaussians()
        actual = topology(source, torch.ones(3, 8), torch.tensor([0, 1, 0]))
        self.assertEqual(actual.count, 2)
        torch.testing.assert_close(actual.means[0], source.means[2])
        torch.testing.assert_close(actual.means[1], source.means[:2].mean(0))
        torch.testing.assert_close(actual.opacities[1], source.opacities[:2].mean())
        torch.testing.assert_close(actual.colors[1], source.colors[:2].mean(0))
        self.assertGreater(float(actual.scales[1, 0].detach()), float(source.scales[0, 0]))
        self.assertEqual(actual.extras["topology_partner"].tolist(), [-1, 1])

    def test_pair_safety_and_rejection_never_delete_points(self):
        features, views = torch.ones(3, 8), torch.tensor([0, 1, 0])
        topology = GaussianTopology(8, 4).eval()
        force_action(topology, MERGE)
        for reason in ("same_view", "instance", "time", "distance", "feature", "rotation", "lifespan"):
            source, tokens, source_views = make_gaussians(), features.clone(), views.clone()
            if reason == "same_view":
                source_views[1] = 0
            elif reason == "instance":
                source.instance_id[1] = 0
            elif reason == "time":
                source.times[1] = 0.5
            elif reason == "distance":
                source.means[1, 0] = 0.049
            elif reason == "feature":
                tokens[1] *= -1
            elif reason == "rotation":
                source.quats[1] = torch.tensor([0., 1., 0., 0.])
            else:
                source.sigma[1] = 3
            with self.subTest(reason=reason):
                self.assertEqual(topology.propose_pairs(source, tokens, source_views).shape, (0, 2))
                self.assertEqual(topology(source, tokens, source_views).count, 3)

    def test_soft_candidates_train_action_pair_quality_and_residual_heads(self):
        torch.manual_seed(32)
        topology = GaussianTopology(8, 4).train()
        source = make_gaussians()
        features = torch.ones(3, 8, requires_grad=True)
        with torch.no_grad():
            topology.policy.action_head.bias.zero_()
        actual = topology(source, features, torch.tensor([0, 1, 0]))
        self.assertEqual(actual.count, 19)
        objective = (actual.opacities * (actual.means.square().sum(-1) + actual.colors.square().sum(-1))).sum()
        objective = objective + 0.01 * actual.extras["topology_gate"].sum()
        objective.backward()
        for head in (topology.policy.action_head, topology.policy.quality_head,
                     topology.policy.pair_head[-1], topology.residual[-1],
                     topology.clone_offset, topology.split_offset):
            self.assertIsNotNone(head.weight.grad)
            self.assertTrue(torch.isfinite(head.weight.grad).all())
            self.assertGreater(float(head.weight.grad.abs().sum()), 0)
        self.assertTrue(torch.isfinite(features.grad).all())
        self.assertTrue((actual.opacities >= 0).all())
        self.assertTrue((actual.opacities <= 1).all())
        for value in actual.extras.values():
            self.assertEqual(value.shape[0], actual.count)

    def test_empty_input_and_explicit_budget(self):
        topology = GaussianTopology(8, 4, max_input_points=2)
        source = make_gaussians()
        with self.assertRaisesRegex(ValueError, "budget"):
            topology(source, torch.ones(3, 8), torch.tensor([0, 1, 0]))
        for name in ("means", "quats", "scales", "opacities", "colors", "sigma", "times", "instance_id"):
            setattr(source, name, getattr(source, name)[:0])
        actual = topology(source, torch.ones(0, 8), torch.empty(0, dtype=torch.long))
        self.assertEqual(actual.count, 0)

    def test_model_default_has_no_new_parameters_and_disabled_outputs_match(self):
        torch.manual_seed(13)
        default = make_model().eval()
        torch.manual_seed(13)
        disabled = make_model(topology={"enabled": False}).eval()
        self.assertIsNone(default.topology)
        self.assertEqual(default.state_dict().keys(), disabled.state_dict().keys())
        disabled.load_state_dict(default.state_dict(), strict=True)
        self.assertFalse(any(name.startswith("topology.") for name in default.state_dict()))
        features = torch.randn(3, 8)
        sampled = {"base_xyz": torch.randn(3, 3), "times": torch.zeros(3),
                   "instance_id": torch.full((3,), -1), "view_id": torch.zeros(3, dtype=torch.long)}
        left = default.decode_gaussians(features, torch.ones(3), sampled)
        right = disabled.decode_gaussians(features, torch.ones(3), sampled)
        torch.testing.assert_close(left.means, right.means, rtol=0, atol=0)
        torch.testing.assert_close(left.colors, right.colors, rtol=0, atol=0)

    def test_model_forward_integration_and_budget_loss_backward(self):
        model = make_model(num_views=2, topology={"enabled": True}).train()
        scene = make_scene(2)
        targets = TargetViews(scene.intrinsics, scene.cam_to_world, scene.times, scene.frame_index)

        def render_stub(gaussians, *args, **kwargs):
            rgb = (gaussians.colors[:, :3] * gaussians.opacities[:, None]).mean(0)
            return {"rgb": rgb[None, :, None, None].expand(2, 3, 2, 4)}

        with patch.object(model, "render", side_effect=render_stub):
            result = model(scene, targets, return_gaussians=True)
        self.assertEqual(result["gaussians"].count, 36)
        self.assertIn("topology_cost", result)
        (result["rgb"].square().mean() + 0.001 * result["topology_cost"]).backward()
        self.assertGreater(float(model.topology.policy.action_head.weight.grad.abs().sum()), 0)
        self.assertGreater(float(model.decoder.head_color.weight.grad.abs().sum()), 0)
        model.eval()
        force_action(model.topology, SPLIT)
        with patch.object(model, "render", side_effect=render_stub):
            result = model(scene, targets, return_gaussians=True)
        self.assertEqual(result["gaussians"].count, 12)
        torch.testing.assert_close(result["topology_cost"], torch.tensor(2.))

    def test_variant_builder_enables_base_and_rejects_other_variants(self):
        config = {"data": {"num_context": 2, "cameras": [0]}, "model": {
            "variant": "base", "feat_dim": 4, "hidden": 8, "color_dim": 8,
            "topology": {"enabled": True},
        }}
        with patch("pointforward.models.pointforward.FeatureExtractor", DummyBackbone), patch(
            "pointforward.models.pointforward.FeatureToRGB", return_value=nn.Identity()
        ):
            model = build_model(config, DummyDepth())
        self.assertIsInstance(model.topology, GaussianTopology)
        for variant in ("sh", "rec_cross_attention", "rec_temporal_attention", "crossdata_sh_variable"):
            config["model"]["variant"] = variant
            with self.assertRaisesRegex(ValueError, "base only"):
                build_model(config, DummyDepth())

    def test_warm_start_loads_base_but_not_optimizer_and_rejects_incompatible_weights(self):
        base = make_model()
        model = make_model(topology={"enabled": True})
        checkpoint = {"model": base.state_dict(), "optimizer": {"unused": True}, "step": 20000}
        with patch("scripts.train.torch.load", return_value=checkpoint):
            missing = initialise_from_checkpoint(model, "unused.pt")
        self.assertTrue(missing)
        self.assertTrue(all(name.startswith("topology.") for name in missing))
        torch.testing.assert_close(model.decoder.head_color.weight, base.decoder.head_color.weight)
        for issue in ("missing", "shape", "unexpected"):
            state = dict(base.state_dict())
            if issue == "missing":
                del state["decoder.head_color.weight"]
            elif issue == "shape":
                state["decoder.head_color.weight"] = torch.zeros(1)
            else:
                state["unrelated.parameter"] = torch.zeros(1)
            with self.subTest(issue=issue), patch("scripts.train.torch.load", return_value={"model": state}):
                with self.assertRaisesRegex(ValueError, "incompatible initial checkpoint"):
                    initialise_from_checkpoint(model, "unused.pt")

    def test_experiment_config_budget_and_model_dimensions(self):
        from pathlib import Path
        import yaml

        config = yaml.safe_load((Path(__file__).parents[1] / "configs/variants/01_base_topology.yaml").read_text())
        options = config["model"]
        views = config["data"]["num_context"] * len(config["data"]["cameras"])
        maximum = views * options["num_static"] + config["data"]["max_instances"] * options["num_dynamic"]
        self.assertLessEqual(maximum, options["topology"]["max_input_points"])
        self.assertTrue(options["topology"]["enabled"])
        self.assertEqual(options["variant"], "base")

    @unittest.skipUnless(os.environ.get("POINTFORWARD_TEST_CUDA") == "1", "opt-in gsplat CUDA smoke")
    def test_real_gsplat_image_loss_reaches_action_merge_and_update_heads(self):
        from dataclasses import replace
        from pointforward.models.gaussian_topology import ATTRIBUTES
        from pointforward.models.renderer import render_features

        torch.manual_seed(61)
        source = make_gaussians()
        source = replace(source, **{name: getattr(source, name).cuda() for name in ATTRIBUTES})
        topology = GaussianTopology(8, 4).cuda().train()
        features = torch.ones(3, 8, device="cuda", requires_grad=True)
        gaussians = topology(source, features, torch.tensor([0, 1, 0], device="cuda"))
        intrinsic = torch.tensor([[[20., 0., 16.], [0., 20., 16.], [0., 0., 1.]]], device="cuda")
        rendered, alpha, _ = render_features(
            gaussians.means, gaussians.quats, gaussians.scales, gaussians.opacities,
            gaussians.colors, intrinsic, torch.eye(4, device="cuda")[None], 32, 32,
        )
        self.assertTrue((alpha > 0).any())
        self.assertTrue(torch.isfinite(rendered).all())
        (rendered - 0.3).square().mean().backward()
        for head in (topology.policy.action_head, topology.policy.quality_head,
                     topology.policy.pair_head[-1], topology.residual[-1]):
            self.assertIsNotNone(head.weight.grad)
            self.assertTrue(torch.isfinite(head.weight.grad).all())
            self.assertGreater(float(head.weight.grad.abs().sum()), 0)


if __name__ == "__main__":
    unittest.main()