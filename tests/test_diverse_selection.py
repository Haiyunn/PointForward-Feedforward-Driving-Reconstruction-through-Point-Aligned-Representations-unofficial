"""Focused synthetic tests: no dataset, GPU, inference, or file writes required."""
import unittest
from types import SimpleNamespace

import numpy as np

from scripts.select_diverse_scenes import (
    diverse_select, ego_path_distance, motion_evidence, road_mask_support,
    select_global_ids,
)


class DiverseSelectionTests(unittest.TestCase):
    def test_motion_threshold_and_prefix(self):
        frames = np.arange(40)
        xyz = np.column_stack((frames * 0.2, np.zeros((40, 2))))
        self.assertTrue(motion_evidence(frames, xyz, frames[:20])["eligible"])
        slow = motion_evidence(frames, xyz * 0.5, frames[:20])
        self.assertIn("below_median_speed", slow["reasons"])
        late = motion_evidence(frames[15:], xyz[15:], frames[:20])
        self.assertIn("insufficient_prefix_observations", late["reasons"])

    def test_stationary_never_selected_even_if_ego_moves(self):
        frames = np.arange(180)
        xyz = np.tile([100, 20, 0], (180, 1))
        motion = motion_evidence(frames, xyz, frames[:20])
        self.assertFalse(motion["eligible"])
        evidence = [{"global_instance_id": 17, **motion}]
        self.assertEqual(select_global_ids(evidence, [17, 2], [17, 2]), [])

    def test_stationary_prefix_rejected(self):
        frames = np.arange(180)
        xyz = np.column_stack((np.maximum(0, frames - 20) * 0.2, np.zeros((180, 2))))
        result = motion_evidence(frames, xyz, frames[:20])
        self.assertIn("below_prefix_displacement", result["reasons"])

    def test_gaps_and_nonfinite_rejected(self):
        frames = np.r_[np.arange(20), np.arange(30, 50)]
        xyz = np.column_stack((frames * 0.2, np.zeros((40, 2))))
        self.assertIn("track_gap", motion_evidence(frames, xyz, range(20))["reasons"])
        xyz[0, 0] = np.nan
        self.assertFalse(motion_evidence(frames, xyz, range(20))["eligible"])

    def test_actual_segment_distance_and_roadside(self):
        frames = np.arange(100)
        ego = np.column_stack((frames, np.zeros((100, 2))))
        result = ego_path_distance([15.4, 1.8, 0], 15, frames, ego, [1, 0, 0])
        self.assertAlmostEqual(result["distance_m"], 1.8)
        self.assertTrue(result["supported"])
        self.assertFalse(ego_path_distance([15, 3, 0], 15, frames, ego, [1, 0, 0])["supported"])
        # Nearby to the infinite extension, but far beyond the actual swept path.
        self.assertFalse(ego_path_distance([130, 0, 0], 99, frames, ego, [1, 0, 0])["supported"])

    def test_time_local_and_crossing_direction(self):
        frames = np.arange(100)
        ego = np.column_stack((frames, np.zeros((100, 2))))
        self.assertFalse(ego_path_distance([80, 0, 0], 0, frames, ego, [1, 0, 0])["supported"])
        self.assertFalse(ego_path_distance([10, 0, 0], 10, frames, ego, [0, 1, 0])["supported"])
        self.assertFalse(ego_path_distance([10, 0, 0], 10, frames, ego, [-1, 0, 0])["supported"])

    def test_global_keys_and_stable_loader_ties(self):
        from scripts.render_vehicle_removal import chosen_instance_keys
        pose = np.eye(4)
        pose[0, 3] = 10
        meta = {"instances": {42: {0: (pose, np.ones(3))},
                              7: {0: (pose.copy(), np.ones(3))},
                              99: {0: (pose.copy(), np.ones(3))}}}
        dataset = SimpleNamespace(max_instances=2, _ego_pose=lambda m, f: np.eye(4))
        keys = chosen_instance_keys(dataset, meta, [0])
        self.assertEqual(keys, [42, 7])
        evidence = [{"global_instance_id": key, "eligible": True} for key in [99, 7, 42]]
        self.assertEqual(select_global_ids(evidence, keys, [42, 7, 99]), [42, 7])
        self.assertEqual(select_global_ids(evidence, keys, [42]), [])
        self.assertEqual(select_global_ids(evidence, keys, [42, 7]), [42])

    def test_rare_scenes_retained_without_edit_and_determinism(self):
        scenes = [{"name": str(i), "attributes": {
            "time_of_day": "Night" if i == 0 else "Dawn/Dusk" if i == 1 else "Day",
            "location": str(i % 2), "ego_motion_layout_proxy": "straight"},
            "remove_instance_ids": [] if i < 2 else [10]} for i in range(8)]
        result = diverse_select(scenes, 5, 3)
        self.assertEqual([s["name"] for s in result[:2]], ["0", "1"])
        self.assertEqual(result, diverse_select(list(reversed(scenes)), 5, 3))
        self.assertEqual(len(result), 5)
        with self.assertRaises(ValueError):
            diverse_select(scenes, 9, 3)
        with self.assertRaises(ValueError):
            diverse_select(scenes, 1, 3)

    def test_mask_probes_outside_occluded_bottom_footprint(self):
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[61:, :] = 255
        audit = SimpleNamespace(mask=lambda frame, camera: mask)
        projected = {"bottom_uv": np.array([[30, 50], [30, 60], [70, 50], [70, 60]]),
                     "image_size": (100, 100)}
        result = road_mask_support(audit, 0, 0, projected)
        self.assertTrue(result["supported"])
        self.assertEqual(result["points"], 15)
        self.assertTrue(all(v > 60 for u, v in result["mask_pixels"]))
        mask[:] = 0
        self.assertFalse(road_mask_support(audit, 0, 0, projected)["supported"])
        audit.mask = lambda frame, camera: None
        self.assertIsNone(road_mask_support(audit, 0, 0, projected))

    def test_insufficient_visible_mask_probes_are_not_road_support(self):
        audit = SimpleNamespace(mask=lambda frame, camera: np.full((100, 100), 255, dtype=np.uint8))
        projected = {"bottom_uv": np.array([[30, 98], [30, 99], [70, 98], [70, 99]]),
                     "image_size": (100, 100)}
        result = road_mask_support(audit, 0, 0, projected)
        self.assertFalse(result["supported"])
        self.assertEqual(result["points"], 0)


if __name__ == "__main__":
    unittest.main()