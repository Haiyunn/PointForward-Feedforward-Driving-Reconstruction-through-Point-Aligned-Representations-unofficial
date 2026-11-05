"""CPU-only long-context contracts; no dataset, timm weights, gsplat or CUDA.

Run with unittest discovery from the project directory. Only the image backbone
and RGB refiner are replaced; query geometry, fusion and decoding are real.
"""

import copy
import unittest
from unittest.mock import patch

import torch
from torch import nn

from pointforward.models.pointforward import PointForward, SceneInput
from pointforward.models.query import build_dense_queries, query_xyz
from pointforward.models.scene_graph import InstanceBoxes, assign_instances, sample_queries


MODEL_MODULE = "pointforward.models.pointforward"


class DummyDepth(nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = []

    def forward(self, images, height, width, sparse_depth=None, sparse_mask=None):
        self.calls.append((images.shape[0], height, width, sparse_depth, sparse_mask))
        # View-unique constant depth exposes order and slice errors.
        return (images[:, 0, 0, 0] + 2).view(-1, 1, 1).expand(-1, height, width)


class DummyBackbone(nn.Module):
    def __init__(self, out_channels=4, **kwargs):
        super().__init__()
        self.conv = nn.Conv2d(3, out_channels, 1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.calls = []

    def forward(self, images):
        self.calls.append(images.shape[0])
        return self.bn(self.conv(images)).sin()


def make_model(**kwargs):
    options = dict(feat_dim=4, color_dim=8, hidden=8, num_static=3,
                   num_dynamic=2, num_views=12, pretrained_backbone=False)
    options.update(kwargs)
    with patch(MODEL_MODULE + ".FeatureExtractor", DummyBackbone), patch(
        MODEL_MODULE + ".FeatureToRGB", return_value=nn.Identity()
    ):
        return PointForward(DummyDepth(), **options)


def make_scene(views=5, height=2, width=4, boxes=False):
    images = torch.linspace(0, 1, views).view(views, 1, 1, 1).expand(
        views, 3, height, width
    ).clone()
    poses = torch.eye(4).repeat(views, 1, 1)
    poses[:, 0, 3] = torch.arange(views) * 10
    scene = SceneInput(
        images=images, intrinsics=torch.eye(3).repeat(views, 1, 1),
        cam_to_world=poses, times=torch.linspace(0, 1, views),
        frame_index=torch.arange(views),
    )
    if boxes:
        centers = torch.zeros(3, views, 3)
        centers[0, :, 0] = poses[:, 0, 3] + 1
        centers[1, :, 0] = poses[:, 0, 3] + 3
        centers[2, :, 0] = 10000  # valid box but no observed pixels
        centers[:, :, 1:] = torch.tensor([1., 2.])
        valid = torch.ones(3, views, dtype=torch.bool)
        valid[0, 0] = False  # canonical frame must be the earliest VALID slot
        valid[1, 1:] = False  # one candidate, exercises replacement
        scene.boxes = InstanceBoxes(
            centers=centers, dims=torch.full_like(centers, 0.2),
            rots=torch.eye(3).repeat(3, views, 1, 1),
            times=scene.times.repeat(3, 1), valid=valid,
        )
    return scene


class LongContextModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def setUp(self):
        torch.manual_seed(17)

    def test_constructor_defaults_and_validation(self):
        model = make_model()
        self.assertEqual(model.depth_view_chunk, 0)
        self.assertEqual(model.backbone_view_chunk, 0)
        self.assertFalse(model.stream_queries)
        self.assertFalse(model.freeze_backbone_bn)
        for keyword in ("depth_view_chunk", "backbone_view_chunk"):
            with self.assertRaises(ValueError):
                make_model(**{keyword: -1})

    def test_default_twelve_views_matches_legacy_sampling_and_backbone(self):
        model = make_model().train()
        scene = make_scene(12, boxes=True)
        depth = torch.full((12, 2, 4), 2.)
        dense, views = build_dense_queries(
            scene.images, depth, scene.intrinsics, scene.cam_to_world
        )
        ids = assign_instances(query_xyz(dense), scene.boxes, scene.frame_index[views])
        # Ignore the third, empty instance: its removal is the explicit bug fix.
        torch.manual_seed(81)
        idx = sample_queries(ids, views, 3, 2, 12, 2)
        expected = model._finish_queries(scene, dense[idx], views[idx], ids[idx])
        torch.manual_seed(81)
        actual = model.build_queries(scene, depth)
        for name in expected:
            torch.testing.assert_close(actual[name], expected[name], rtol=0, atol=0)
        reference = copy.deepcopy(model.backbone)
        torch.testing.assert_close(model.extract_features(scene.images), reference(scene.images))
        self.assertEqual(model.backbone.calls, [12])
        model.predict_depth(scene)
        self.assertEqual([call[0] for call in model.depth_model.calls], [12])

    def test_depth_chunks_preserve_all_views_sparse_inputs_and_scale(self):
        scene = make_scene(180, height=4, width=6)
        scene.images.requires_grad_()
        scene.sparse_depth = torch.arange(180.).view(180, 1, 1, 1).expand(180, 1, 4, 6)
        scene.sparse_mask = scene.sparse_depth > 90
        model = make_model(depth_view_chunk=7, depth_scale=0.5)
        depth = model.predict_depth(scene)
        self.assertEqual(depth.shape, (180, 4, 6))
        self.assertFalse(depth.requires_grad)
        torch.testing.assert_close(depth[:, 0, 0], scene.images[:, 0, 0, 0] + 2)
        calls = model.depth_model.calls
        self.assertEqual([c[0] for c in calls], [7] * 25 + [5])
        for start, call in zip(range(0, 180, 7), calls):
            size, h, w, sparse, mask = call
            self.assertEqual((h, w), (2, 3))
            torch.testing.assert_close(sparse, scene.sparse_depth[start:start + size])
            torch.testing.assert_close(mask, scene.sparse_mask[start:start + size])
        model.depth_view_chunk = 0
        torch.testing.assert_close(depth, model.predict_depth(scene))
        scene.sparse_depth = scene.sparse_mask = None
        model.predict_depth(scene)
        self.assertIsNone(model.depth_model.calls[-1][3])
        self.assertIsNone(model.depth_model.calls[-1][4])

    def test_streaming_static_budget_all_180_views_and_no_grad(self):
        scene = make_scene(180)
        scene.images.requires_grad_()
        depth = torch.full((180, 2, 4), 2., requires_grad=True)
        for budget in (3, 10):  # without and with replacement
            model = make_model(num_static=budget, stream_queries=True)
            with patch(MODEL_MODULE + ".build_dense_queries", wraps=build_dense_queries) as build:
                result = model.build_queries(scene, depth)
            self.assertEqual(len(build.call_args_list), 180)
            self.assertTrue(all(call.args[0].shape[0] == 1 for call in build.call_args_list))
            self.assertEqual(result["queries"].shape, (180 * budget, 12))
            self.assertEqual(torch.bincount(result["view_id"]).tolist(), [budget] * 180)
            self.assertTrue((result["instance_id"] == -1).all())
            self.assertTrue(all(not value.requires_grad for value in result.values()))

    def test_streaming_identity_canonicalisation_normalisation_and_reproducibility(self):
        scene = make_scene(boxes=True)
        depth = torch.full((5, 2, 4), 2.)
        model = make_model(stream_queries=True)
        torch.manual_seed(19)
        result = model.build_queries(scene, depth)
        torch.manual_seed(19)
        repeated = model.build_queries(scene, depth)
        for name in result:
            torch.testing.assert_close(result[name], repeated[name], rtol=0, atol=0)
        ids, views = result["instance_id"], result["view_id"]
        self.assertEqual(result["queries"].shape, (5 * 3 + 2 * 2, 12))
        self.assertEqual(torch.bincount(views[ids < 0]).tolist(), [3] * 5)
        self.assertEqual(int((ids == 0).sum()), 2)
        self.assertEqual(int((ids == 1).sum()), 2)
        self.assertFalse((ids == 2).any())
        actual_ids = assign_instances(result["world_xyz"], scene.boxes, result["box_index"])
        torch.testing.assert_close(ids, actual_ids)
        torch.testing.assert_close(result["base_xyz"][ids == 0], torch.tensor([[11., 1., 2.]]).repeat(2, 1))
        torch.testing.assert_close(result["base_xyz"][ids == 1], torch.tensor([[3., 1., 2.]]).repeat(2, 1))
        torch.testing.assert_close(result["queries"][:, :3], result["base_xyz"])
        torch.testing.assert_close(result["times"], scene.times[views])
        torch.testing.assert_close(result["box_index"], scene.frame_index[views])
        direction, moment = result["queries"][:, 6:9], result["queries"][:, 9:12]
        torch.testing.assert_close(direction.norm(dim=-1), torch.ones(ids.numel()))
        torch.testing.assert_close((direction * moment).sum(-1), torch.zeros(ids.numel()), atol=1e-5, rtol=0)
        posed, valid = model._pose_points(result["base_xyz"], ids, scene.boxes, scene.frame_index)
        rows = torch.arange(ids.numel())
        torch.testing.assert_close(posed[views, rows], result["world_xyz"])
        self.assertTrue(valid[views, rows].all())
        self.assertFalse(valid[0, ids == 0].any())

    def test_reservoir_equals_global_random_topk_for_unequal_view_pools(self):
        scene = make_scene(3, height=1, width=8)
        centers = torch.tensor([[[1., 1., 2.], [11., 1., 2.], [21., 1., 2.]]])
        dims = torch.tensor([[[0.5, 0.5, 0.5], [4.5, 0.5, 0.5], [12.5, 0.5, 0.5]]])
        scene.boxes = InstanceBoxes(centers, dims, torch.eye(3).repeat(1, 3, 1, 1),
                                    scene.times[None], torch.ones(1, 3, dtype=torch.bool))
        depth = torch.full((3, 1, 8), 2.)
        dense, views = build_dense_queries(scene.images, depth, scene.intrinsics, scene.cam_to_world)
        ids = assign_instances(query_xyz(dense), scene.boxes, views)
        self.assertEqual(torch.bincount(views[ids == 0]).tolist(), [1, 2, 4])
        torch.manual_seed(212)
        keys = torch.cat([torch.rand(n, dtype=torch.float64) for n in (1, 2, 4)])
        selected = keys.topk(3).indices
        expected_xyz = query_xyz(dense[ids == 0])[selected]
        model = make_model(num_static=0, num_dynamic=3, stream_queries=True)
        torch.manual_seed(212)
        result = model.build_queries(scene, depth)
        torch.testing.assert_close(result["world_xyz"], expected_xyz)
        self.assertEqual(torch.unique(result["world_xyz"], dim=0).shape[0], 3)

    def test_empty_static_dynamic_and_zero_budgets_never_borrow_index_zero(self):
        scene = make_scene(boxes=True)
        # First instance covers all pixels in every view; other pools are empty.
        scene.boxes.valid[0] = True
        scene.boxes.dims[0] = 1000
        for streaming in (False, True):
            model = make_model(stream_queries=streaming)
            result = model.build_queries(scene, torch.full((5, 2, 4), 2.))
            self.assertEqual(result["queries"].shape, (2, 12))
            self.assertTrue((result["instance_id"] == 0).all())
            model.num_static = model.num_dynamic = 0
            empty = model.build_queries(scene, torch.full((5, 2, 4), 2.))
            self.assertEqual(empty["queries"].shape, (0, 12))
            self.assertEqual(empty["world_xyz"].shape, (0, 3))

    def test_checkpoint_gradients_and_bn_statistics_update_once(self):
        for frozen in (False, True):
            for momentum in (0.1, None):
                with self.subTest(frozen=frozen, momentum=momentum):
                    model = make_model(backbone_view_chunk=2, freeze_backbone_bn=frozen).train()
                    model.backbone.bn.momentum = momentum
                    reference = copy.deepcopy(model.backbone)
                    if frozen:
                        reference.bn.eval()
                    images = torch.randn(5, 3, 3, 4)  # no input gradients required
                    expected = torch.cat([reference(x) for x in images.split(2)])
                    actual = model.extract_features(images)
                    self.assertEqual(model.backbone.calls, [2, 2, 1])
                    torch.testing.assert_close(actual, expected)
                    actual.square().sum().backward()
                    expected.square().sum().backward()
                    self.assertEqual(model.backbone.calls[3:], [1, 2, 2])
                    for param, ref in zip(model.backbone.parameters(), reference.parameters()):
                        self.assertIsNotNone(param.grad)
                        self.assertTrue(torch.isfinite(param.grad).all())
                        self.assertGreater(float(param.grad.abs().sum()), 0)
                        torch.testing.assert_close(param.grad, ref.grad)
                    for name, buffer in model.backbone.named_buffers():
                        torch.testing.assert_close(buffer, dict(reference.named_buffers())[name])
                    self.assertEqual(int(model.backbone.bn.num_batches_tracked), 0 if frozen else 3)
                    self.assertTrue(model.backbone.bn.training)

    def test_frozen_bn_survives_train_eval_and_uncheckpointed_mode(self):
        model = make_model(freeze_backbone_bn=True)
        images = torch.randn(5, 3, 2, 4, requires_grad=True)
        for chunk in (0, 2):
            model.backbone_view_chunk = chunk
            for training in (True, False):
                model.train(training)
                model.extract_features(images).sum().backward()
                self.assertEqual(model.backbone.bn.training, training)
                self.assertEqual(int(model.backbone.bn.num_batches_tracked), 0)
        self.assertTrue(torch.isfinite(images.grad).all())
        self.assertGreater(float(images.grad.abs().sum()), 0)

    def test_checkpoint_eval_no_grad_and_exception_cleanup(self):
        model = make_model(backbone_view_chunk=2).eval()
        images = torch.randn(5, 3, 2, 4)
        reference = copy.deepcopy(model.backbone)
        with torch.no_grad():
            actual = model.extract_features(images)
        torch.testing.assert_close(actual, reference(images))
        self.assertEqual(model.backbone.calls, [2, 2, 1])
        bn = model.backbone.bn
        original = bn.running_mean
        with self.assertRaisesRegex(RuntimeError, "synthetic"):
            with model._backbone_bn_context(recompute=True):
                self.assertIsNot(bn.running_mean, original)
                raise RuntimeError("synthetic")
        self.assertIs(bn.running_mean, original)
        self.assertFalse(bn.training)

    def test_all_180_views_reach_real_fusion_and_gaussian_count_with_gradients(self):
        self._check_full_context_fusion(3)

    def test_all_300_views_reach_real_fusion_and_gaussian_count_with_gradients(self):
        self._check_full_context_fusion(5)

    def _check_full_context_fusion(self, cameras):
        views = 60 * cameras
        model = make_model(num_views=views, num_static=1, stream_queries=True,
                           depth_view_chunk=7, backbone_view_chunk=7,
                           freeze_backbone_bn=True, fusion_chunk=4096).train()
        scene = make_scene(views)
        # Co-located cameras keep all sampled points visible for this tiny test.
        scene.cam_to_world[:, 0, 3] = 0
        scene.frame_index = torch.arange(60).repeat_interleave(cameras)
        scene.times = scene.frame_index.float() / 59
        depth = model.predict_depth(scene)
        sampled = model.build_queries(scene, depth)
        with patch.object(model.fusion, "forward", wraps=model.fusion.forward) as fuse:
            gaussians = model.decode(scene, depth, sampled)
        kwargs = fuse.call_args.kwargs
        for name in ("features", "points_per_view", "depth_maps", "intrinsics", "view_times"):
            self.assertEqual(kwargs[name].shape[0], views, name)
        self.assertEqual(kwargs["points_per_view"].shape, (views, views, 3))
        self.assertEqual(gaussians.count, views)
        self.assertEqual(model.fusion._phi_v_views, views)
        self.assertEqual(model.fusion.chunk_size, 4096)
        loss = gaussians.colors.square().mean() + gaussians.means.square().mean() + gaussians.sigma.mean()
        loss.backward()
        for module in (model.backbone, model.fusion, model.decoder):
            grads = [p.grad for p in module.parameters() if p.grad is not None]
            self.assertTrue(grads)
            self.assertTrue(all(torch.isfinite(g).all() for g in grads))
            self.assertGreater(sum(float(g.abs().sum()) for g in grads), 0)
        self.assertLessEqual(max(model.backbone.calls), 7)
        self.assertEqual(int(model.backbone.bn.num_batches_tracked), 0)


if __name__ == "__main__":
    unittest.main()