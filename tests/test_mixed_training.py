import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from PIL import Image

from pointforward.data.argoverse import ArgoverseDriveStudioClips
from pointforward.data.mixed import MixedDataset, MixedDistributedSampler
from scripts.train import build_dataset


class FakeDataset:
    scenes = [{"name": "001"}]

    def __init__(self, count):
        self.count = count

    def __len__(self):
        return self.count

    def __getitem__(self, index):
        return {"scene": "001"}


class MixedTrainingTests(unittest.TestCase):
    def test_balanced_sharded_sampling(self):
        dataset = MixedDataset([FakeDataset(1000), FakeDataset(200)], ["waymo", "argoverse"])
        self.assertEqual(dataset[-1]["scene"], "argoverse/001")
        torch.testing.assert_close(dataset.sample_weights[:1000].sum(), dataset.sample_weights[1000:].sum())
        shards = [list(MixedDistributedSampler(dataset, num_replicas=2, rank=rank, seed=42)) for rank in range(2)]
        draw = list(MixedDistributedSampler(dataset, num_replicas=1, rank=0, seed=42))
        self.assertEqual([value for group in zip(*shards) for value in group], draw)
        self.assertTrue(500 < sum(index >= 1000 for index in draw) < 700)

    def test_source_splits_and_argoverse_dispatch(self):
        cfg = {"data": {"split_file": "must-not-inherit.json", "sources": [
            {"name": "waymo", "root": "/waymo", "train_scenes": ["001"], "val_scenes": ["002"]},
            {"name": "argoverse", "root": "/argo", "layout": "argoverse_drivestudio",
             "train_scenes": ["003"], "val_scenes": ["004"]},
        ]}}
        with patch("scripts.train.WaymoDriveStudioClips", return_value=FakeDataset(10)) as waymo, \
                patch("scripts.train.ArgoverseDriveStudioClips", return_value=FakeDataset(2)) as argo:
            build_dataset(cfg, "train")
            self.assertEqual(waymo.call_args.kwargs["scenes"], ["001"])
            self.assertEqual(argo.call_args.kwargs["scenes"], ["003"])
            self.assertIn("REGULAR_VEHICLE", argo.call_args.kwargs["rigid_classes"])
            build_dataset(cfg, "val")
            self.assertEqual(argo.call_args.kwargs["scenes"], ["004"])

    def test_principal_point_crop_adjusts_intrinsics_without_mutation(self):
        dataset = object.__new__(ArgoverseDriveStudioClips)
        intrinsic = np.array([[100., 0., 50.], [0., 100., 100.], [0., 0., 1.]])
        with tempfile.TemporaryDirectory() as directory:
            Image.new("RGB", (100, 200), "white").save(directory + "/image.png")
            image, adjusted = dataset._load_calibrated_image(
                {"path": directory, "intrinsics": {0: intrinsic}}, "image.png", (50, 50), 0
            )
        self.assertEqual(tuple(image.shape), (3, 50, 50))
        np.testing.assert_allclose(adjusted, [[50, 0, 25], [0, 50, 25], [0, 0, 1]])
        self.assertEqual(intrinsic[1, 2], 100.)


if __name__ == "__main__":
    unittest.main()