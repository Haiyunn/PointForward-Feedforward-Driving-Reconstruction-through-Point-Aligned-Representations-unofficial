import json
from pathlib import Path
from unittest.mock import patch

import pytest

from scripts.regression_sample import (
    EXPECTED_SCENES,
    build_concat_command,
    checkpoint_candidates,
    parse_ffprobe_json,
    validate_sampling_manifest,
)
from scripts.render_continuous import decode_render_gaussians


def canonical_manifest():
    return {
        "schema_version": 1,
        "base_config": "configs/variants/01_base.yaml",
        "base_checkpoint": "/tmp/ckpt_020000.pt",
        "split_file": "splits/waymo_ds.json",
        "split": "val",
        "camera": 0,
        "fps": 10,
        "num_frames": 180,
        "scenes": [{"scene": scene, "start_frame": 0} for scene in EXPECTED_SCENES],
    }


def test_manifest_validation_accepts_only_canonical_scene_order():
    validate_sampling_manifest(canonical_manifest())
    invalid = canonical_manifest()
    invalid["scenes"] = list(reversed(invalid["scenes"]))
    with pytest.raises(ValueError, match="canonical"):
        validate_sampling_manifest(invalid)


def test_checkpoint_candidates_filter_and_rank_strict_names_through_symlink(tmp_path):
    target = tmp_path / "target"
    target.mkdir()
    for name in ("ckpt_000010.pt", "ckpt_000200.pt", "ckpt_000200.gs.pt", "ckpt_latest.pt"):
        (target / name).write_bytes(b"checkpoint")
    index = tmp_path / "index"
    index.mkdir()
    (index / "run").symlink_to(target, target_is_directory=True)

    candidates = checkpoint_candidates(index)

    assert [path.name for path in candidates] == ["ckpt_000200.pt", "ckpt_000010.pt"]


def test_parse_ffprobe_json_normalizes_video_fields():
    payload = json.dumps({
        "streams": [{
            "codec_name": "h264", "pix_fmt": "yuv420p", "width": 768, "height": 576,
            "avg_frame_rate": "10/1", "nb_read_frames": "180",
        }]
    })
    assert parse_ffprobe_json(payload) == {
        "frames": 180, "codec": "h264", "pix_fmt": "yuv420p",
        "fps": 10.0, "width": 768, "height": 576,
    }


def test_concat_command_preserves_input_order_and_encodes_h264(tmp_path):
    inputs = [tmp_path / f"scene_{index}.mp4" for index in range(3)]
    output = tmp_path / "all.mp4"
    command = build_concat_command(inputs, output, 10)
    input_values = [command[index + 1] for index, value in enumerate(command[:-1]) if value == "-i"]
    assert input_values == [str(path) for path in inputs]
    assert "[0:v][1:v][2:v]concat=n=3:v=1:a=0[outv]" in command
    assert command[command.index("-c:v") + 1] == "libx264"
    assert command[command.index("-pix_fmt") + 1] == "yuv420p"


def test_rec_decode_helper_uses_reconstruction_cache_and_render_gaussians():
    class FakeRec:
        def reconstruct_scene(self, scene, cache):
            assert scene == "scene"
            assert cache == "old-cache"
            return "depth", "current", "history-aware", "new-cache"

    with patch("scripts.render_continuous.PointForwardRec", FakeRec):
        result = decode_render_gaussians(FakeRec(), "scene", "old-cache")

    assert result == ("depth", "current", "history-aware", "new-cache")