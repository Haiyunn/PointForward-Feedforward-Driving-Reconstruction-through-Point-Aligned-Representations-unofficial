"""Deterministic ten-scene Waymo visual regression sampling."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path
from typing import Any

import torch
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SPEC = PROJECT_ROOT / "configs/regression_sampling/waymo_val10.json"
DEFAULT_CONFIG = PROJECT_ROOT / "configs/variants/01_base.yaml"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "results/regression_sampling"
CHECKPOINT_RE = re.compile(r"^ckpt_(\d+)\.pt$")
EXPECTED_SCENES = ("005", "025", "049", "082", "102", "146", "213", "245", "314", "438")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False) as handle:
        json.dump(value, handle, indent=2)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve_project_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def load_json(path: Path) -> dict:
    with path.open("r") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object in {path}")
    return value


def load_yaml(path: Path) -> dict:
    with path.open("r") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"expected YAML mapping in {path}")
    return value


def validate_sampling_manifest(spec: dict) -> None:
    required = {
        "schema_version", "base_config", "base_checkpoint", "split_file", "split",
        "camera", "fps", "num_frames", "scenes",
    }
    missing = sorted(required - spec.keys())
    if missing:
        raise ValueError(f"sampling manifest missing keys: {missing}")
    if spec["schema_version"] != 1:
        raise ValueError(f"unsupported sampling schema_version: {spec['schema_version']}")
    if spec["split"] != "val" or spec["camera"] != 0 or spec["fps"] != 10:
        raise ValueError("sampling manifest must use val split, camera 0, and 10 fps")
    if spec["num_frames"] != 180:
        raise ValueError("sampling manifest must use exactly 180 frames per scene")
    scenes = spec["scenes"]
    if not isinstance(scenes, list) or len(scenes) != 10:
        raise ValueError("sampling manifest must contain exactly 10 scenes")
    names = tuple(item.get("scene") for item in scenes if isinstance(item, dict))
    starts = tuple(item.get("start_frame") for item in scenes if isinstance(item, dict))
    if names != EXPECTED_SCENES or starts != (0,) * 10:
        raise ValueError("sampling manifest scenes/order/positional starts are not canonical")


def checkpoint_step(path: Path) -> int | None:
    match = CHECKPOINT_RE.fullmatch(path.name)
    return int(match.group(1)) if match else None


def checkpoint_candidates(root: Path) -> list[Path]:
    if not root.exists():
        raise FileNotFoundError(f"variant result index does not exist: {root}")
    found: dict[str, Path] = {}
    for directory, _, filenames in os.walk(root, followlinks=True):
        for filename in filenames:
            if CHECKPOINT_RE.fullmatch(filename):
                path = Path(directory, filename).resolve()
                found[str(path)] = path
    return sorted(
        found.values(),
        key=lambda path: (checkpoint_step(path), path.stat().st_mtime_ns),
        reverse=True,
    )


def validate_checkpoint(path: Path) -> None:
    if not path.is_file() or path.stat().st_size == 0 or not os.access(path, os.R_OK):
        raise ValueError(f"checkpoint is missing, empty, or unreadable: {path}")
    try:
        value = torch.load(path, map_location="cpu")
    except Exception as exc:
        raise ValueError(f"checkpoint cannot be loaded: {path}: {exc}") from exc
    if not isinstance(value, dict) or not isinstance(value.get("model"), dict) or not value["model"]:
        raise ValueError(f"checkpoint has no nonempty model key: {path}")


def discover_latest_checkpoint(config: dict, explicit: str) -> Path:
    if explicit != "auto":
        path = resolve_project_path(explicit)
        if checkpoint_step(path) is None:
            raise ValueError(f"checkpoint filename must match ckpt_N.pt exactly: {path.name}")
        validate_checkpoint(path)
        return path
    else:
        variant = config.get("variant", {}).get("name")
        if not variant:
            raise ValueError("selected config has no variant.name for checkpoint discovery")
        candidates = checkpoint_candidates(PROJECT_ROOT / "results/by_version" / variant)
        if not candidates:
            raise FileNotFoundError(f"no strict ckpt_N.pt files found for variant {variant}")
        failures = []
        for path in candidates:
            try:
                validate_checkpoint(path)
            except ValueError as exc:
                failures.append(str(exc))
                continue
            return path
        raise ValueError(f"no valid checkpoint found for variant {variant}: {failures}")


def parse_ffprobe_json(text: str) -> dict:
    streams = json.loads(text).get("streams", [])
    if len(streams) != 1:
        raise ValueError(f"expected exactly one video stream, found {len(streams)}")
    stream = streams[0]
    frame_value = stream.get("nb_read_frames", stream.get("nb_frames"))
    if frame_value in (None, "N/A"):
        raise ValueError("ffprobe did not report a frame count")
    rate_value = stream.get("avg_frame_rate") or stream.get("r_frame_rate")
    if not rate_value or rate_value == "0/0":
        raise ValueError("ffprobe did not report a frame rate")
    return {
        "frames": int(frame_value),
        "codec": stream.get("codec_name"),
        "pix_fmt": stream.get("pix_fmt"),
        "fps": float(Fraction(rate_value)),
        "width": int(stream["width"]),
        "height": int(stream["height"]),
    }


def probe_video(path: Path) -> dict:
    result = subprocess.run(
        [
            "ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0",
            "-show_entries",
            "stream=codec_name,pix_fmt,width,height,avg_frame_rate,r_frame_rate,nb_frames,nb_read_frames",
            "-of", "json", str(path),
        ],
        check=True, capture_output=True, text=True,
    )
    return parse_ffprobe_json(result.stdout)


def require_video(path: Path, frames: int, fps: int) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"missing video: {path}")
    info = probe_video(path)
    if info["frames"] != frames or info["codec"] != "h264" or info["pix_fmt"] != "yuv420p":
        raise ValueError(f"invalid video {path}: {info}, expected {frames} H.264/yuv420p frames")
    if abs(info["fps"] - fps) > 1e-6:
        raise ValueError(f"invalid video frame rate {path}: {info['fps']}, expected {fps}")
    return info


def video_is_valid(path: Path, frames: int, fps: int) -> bool:
    try:
        require_video(path, frames, fps)
    except (OSError, ValueError, subprocess.SubprocessError, json.JSONDecodeError):
        return False
    return True


def build_concat_command(inputs: list[Path], output: Path, fps: int) -> list[str]:
    command = ["ffmpeg", "-y"]
    for path in inputs:
        command.extend(["-i", str(path)])
    joined = "".join(f"[{index}:v]" for index in range(len(inputs)))
    command.extend([
        "-filter_complex", f"{joined}concat=n={len(inputs)}:v=1:a=0[outv]",
        "-map", "[outv]", "-r", str(fps), "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-movflags", "+faststart", str(output), "-loglevel", "error",
    ])
    return command


def build_comparison_command(baseline: Path, current: Path, output: Path) -> list[str]:
    video_filter = (
        "[0:v]drawtext=text='BASELINE':x=12:y=12:fontsize=24:fontcolor=white:"
        "box=1:boxcolor=black@0.7[left];"
        "[1:v]drawtext=text='CURRENT':x=12:y=12:fontsize=24:fontcolor=white:"
        "box=1:boxcolor=black@0.7[right];[left][right]hstack=inputs=2[outv]"
    )
    return [
        "ffmpeg", "-y", "-i", str(baseline), "-i", str(current),
        "-filter_complex", video_filter, "-map", "[outv]", "-c:v", "libx264",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(output),
        "-loglevel", "error",
    ]


def run_command(command: list[str], log_path: Path | None = None, env: dict | None = None) -> None:
    if log_path is None:
        subprocess.run(command, cwd=PROJECT_ROOT, env=env, check=True)
        return
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w") as log:
        subprocess.run(
            command, cwd=PROJECT_ROOT, env=env, check=True,
            stdout=log, stderr=subprocess.STDOUT, text=True,
        )


def frame_ids(scene_dir: Path, cameras: tuple[int, ...]) -> list[int]:
    image_dir = scene_dir / "images"
    if not image_dir.is_dir():
        raise FileNotFoundError(f"missing scene image directory: {image_dir}")
    by_frame: dict[int, set[int]] = {}
    for path in image_dir.iterdir():
        if path.suffix.lower() not in {".jpg", ".jpeg", ".png"} or "_" not in path.stem:
            continue
        frame_text, camera_text = path.stem.rsplit("_", 1)
        try:
            by_frame.setdefault(int(frame_text), set()).add(int(camera_text))
        except ValueError:
            continue
    return sorted(frame for frame, present in by_frame.items() if set(cameras) <= present)


def scene_windows(config: dict, spec: dict) -> list[dict]:
    root = resolve_project_path(config["data"]["root"])
    cameras = tuple(int(value) for value in config["data"].get("cameras", (0, 1, 2)))
    if spec["camera"] >= len(cameras) or cameras[spec["camera"]] != spec["camera"]:
        raise ValueError(
            f"camera position {spec['camera']} does not map to physical camera {spec['camera']}: {cameras}"
        )
    windows = []
    for item in spec["scenes"]:
        scene_dir = root / item["scene"]
        if not scene_dir.is_dir():
            raise FileNotFoundError(f"missing scene directory: {scene_dir}")
        frames = frame_ids(scene_dir, cameras)
        start = item["start_frame"]
        selected = frames[start:start + spec["num_frames"]]
        if len(selected) != spec["num_frames"]:
            raise ValueError(f"scene {item['scene']} has fewer than 180 frames at positional start {start}")
        if any(right != left + 1 for left, right in zip(selected, selected[1:])):
            raise ValueError(f"scene {item['scene']} selected raw frame IDs are not contiguous")
        for raw_frame in selected:
            matches = list((scene_dir / "images").glob(f"{raw_frame:03d}_{spec['camera']}.*"))
            if not matches:
                raise FileNotFoundError(
                    f"missing camera-{spec['camera']} image for scene {item['scene']} frame {raw_frame}"
                )
        windows.append({
            "scene": item["scene"], "positional_start": start,
            "raw_frame_ids": selected, "raw_frame_start": selected[0], "raw_frame_end": selected[-1],
        })
    return windows


def check_tools() -> None:
    for tool in ("ffmpeg", "ffprobe"):
        if shutil.which(tool) is None:
            raise RuntimeError(f"required executable not found: {tool}")
    result = subprocess.run(
        ["ffmpeg", "-hide_banner", "-encoders"], check=True, capture_output=True, text=True,
    )
    if "libx264" not in result.stdout:
        raise RuntimeError("ffmpeg does not provide the libx264 encoder")


def git_metadata() -> dict:
    def capture(arguments: list[str]) -> str:
        result = subprocess.run(
            ["git", *arguments], cwd=PROJECT_ROOT, capture_output=True, text=True, check=False,
        )
        return result.stdout.strip() if result.returncode == 0 else f"unavailable: {result.stderr.strip()}"
    return {
        "commit": capture(["rev-parse", "HEAD"]),
        "status": capture(["status", "--short", "--", "."]),
    }


def build_plan(args: argparse.Namespace) -> dict:
    check_tools()
    spec_path = resolve_project_path(args.spec)
    spec = load_json(spec_path)
    validate_sampling_manifest(spec)
    config_path = resolve_project_path(args.config)
    base_config_path = resolve_project_path(spec["base_config"])
    split_path = resolve_project_path(spec["split_file"])
    split = load_json(split_path)
    missing = [item["scene"] for item in spec["scenes"] if item["scene"] not in split.get("val", [])]
    if missing:
        raise ValueError(f"canonical scenes missing from val split {split_path}: {missing}")
    config = load_yaml(config_path)
    base_config = load_yaml(base_config_path)
    for label, candidate in (("selected", config), ("baseline", base_config)):
        configured_split = candidate.get("data", {}).get("split_file")
        if not configured_split or resolve_project_path(configured_split) != split_path:
            raise ValueError(f"{label} config does not use canonical split file {split_path}")
    current_windows = scene_windows(config, spec)
    base_windows = scene_windows(base_config, spec)
    if current_windows != base_windows:
        raise ValueError("selected config and frozen baseline do not resolve identical scene framing")
    base_checkpoint = resolve_project_path(args.base_ckpt or spec["base_checkpoint"])
    validate_checkpoint(base_checkpoint)
    current_checkpoint = discover_latest_checkpoint(config, args.ckpt)
    variant = config.get("variant", {}).get("name", config_path.stem)
    step = checkpoint_step(current_checkpoint)
    base_sha = sha256_file(base_checkpoint)
    current_sha = sha256_file(current_checkpoint)
    run_id = f"{variant}_step{step:06d}_{current_sha[:12]}"
    return {
        "schema_version": 1,
        "created_at": utc_now(),
        "spec_path": str(spec_path),
        "spec_sha256": sha256_file(spec_path),
        "variant": variant,
        "run_id": run_id,
        "camera": spec["camera"],
        "fps": spec["fps"],
        "num_frames": spec["num_frames"],
        "split": spec["split"],
        "split_path": str(split_path),
        "base": {
            "config_path": str(base_config_path), "config_sha256": sha256_file(base_config_path),
            "checkpoint_path": str(base_checkpoint), "checkpoint_sha256": base_sha,
        },
        "current": {
            "config_path": str(config_path), "config_sha256": sha256_file(config_path),
            "checkpoint_path": str(current_checkpoint), "checkpoint_sha256": current_sha,
            "checkpoint_step": step,
        },
        "scenes": current_windows,
        "git": git_metadata(),
    }


def expected_render_manifest(plan: dict, role: str, scene: dict) -> dict:
    identity = plan[role]
    return {
        "scene": scene["scene"],
        "positional_start": scene["positional_start"],
        "raw_frame_ids": scene["raw_frame_ids"],
        "camera": plan["camera"],
        "fps": plan["fps"],
        "checkpoint_path": identity["checkpoint_path"],
        "config_path": identity["config_path"],
    }


def scene_artifact_valid(directory: Path, expected: dict, num_frames: int) -> bool:
    manifest_path = directory / "render_manifest.json"
    video_path = directory / "continuous.mp4"
    if not manifest_path.is_file() or not video_path.is_file():
        return False
    try:
        manifest = load_json(manifest_path)
        if any(manifest.get(key) != value for key, value in expected.items()):
            return False
        require_video(video_path, num_frames, expected["fps"])
    except (OSError, ValueError, subprocess.SubprocessError, json.JSONDecodeError):
        return False
    return True


def render_command(plan: dict, role: str, scene: dict, output: Path) -> list[str]:
    identity = plan[role]
    return [
        sys.executable, "-u", "-m", "scripts.render_continuous",
        "--config", identity["config_path"], "--ckpt", identity["checkpoint_path"],
        "--split", plan["split"], "--out", str(output), "--scene", scene["scene"],
        "--start-frame", str(scene["positional_start"]), "--num-frames", str(plan["num_frames"]),
        "--camera", str(plan["camera"]), "--fps", str(plan["fps"]),
    ]


def frozen_baseline_compatible(path: Path, plan: dict) -> bool:
    manifest_path = path / "baseline_manifest.json"
    if not manifest_path.is_file():
        return False
    try:
        manifest = load_json(manifest_path)
    except (OSError, ValueError, json.JSONDecodeError):
        return False
    keys = ("config_path", "config_sha256", "checkpoint_path", "checkpoint_sha256")
    return all(manifest.get(key) == plan["base"][key] for key in keys) and (
        manifest.get("camera") == plan["camera"]
        and manifest.get("fps") == plan["fps"]
        and manifest.get("num_frames") == plan["num_frames"]
        and manifest.get("scenes") == plan["scenes"]
    )


def run_sampling(args: argparse.Namespace, plan: dict) -> Path:
    output_root = resolve_project_path(args.output_root)
    baseline_root = output_root / "baselines" / plan["base"]["checkpoint_sha256"]
    run_root = output_root / "runs" / plan["run_id"]
    status_path = run_root / "status.json"
    status = {"state": "running", "started_at": utc_now(), "commands": [], "completed_scenes": []}
    atomic_write_json(status_path, status)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    env.setdefault("HF_HUB_OFFLINE", "1")
    env.setdefault("TRANSFORMERS_OFFLINE", "1")

    try:
        baseline_manifest_exists = (baseline_root / "baseline_manifest.json").exists()
        baseline_reusable = frozen_baseline_compatible(baseline_root, plan)
        if baseline_manifest_exists and not baseline_reusable and not args.force:
            raise RuntimeError(
                f"frozen baseline manifest is incompatible: {baseline_root}; use --force to rebuild explicitly"
            )
        for role, root in (("base", baseline_root), ("current", run_root)):
            for scene in plan["scenes"]:
                scene_dir = root / "scenes" / scene["scene"]
                expected = expected_render_manifest(plan, role, scene)
                may_reuse = role != "base" or baseline_reusable
                if may_reuse and not args.force and scene_artifact_valid(
                    scene_dir, expected, plan["num_frames"]
                ):
                    continue
                command = render_command(plan, role, scene, scene_dir)
                status["commands"].append(command)
                atomic_write_json(status_path, status)
                run_command(command, run_root / "logs" / f"{role}_{scene['scene']}.log", env)
                if not scene_artifact_valid(scene_dir, expected, plan["num_frames"]):
                    raise RuntimeError(f"renderer produced an invalid {role} artifact for scene {scene['scene']}")
                status["completed_scenes"].append({"role": role, "scene": scene["scene"]})
                atomic_write_json(status_path, status)

        baseline_manifest = {
            **plan["base"], "camera": plan["camera"], "fps": plan["fps"],
            "num_frames": plan["num_frames"], "scenes": plan["scenes"], "frozen_at": utc_now(),
        }
        atomic_write_json(baseline_root / "baseline_manifest.json", baseline_manifest)

        baseline_scene_videos = [baseline_root / "scenes" / item["scene"] / "continuous.mp4" for item in plan["scenes"]]
        current_scene_videos = [run_root / "scenes" / item["scene"] / "continuous.mp4" for item in plan["scenes"]]
        baseline_infos = [require_video(path, 180, 10) for path in baseline_scene_videos]
        current_infos = [require_video(path, 180, 10) for path in current_scene_videos]
        framing = {(info["width"], info["height"]) for info in baseline_infos + current_infos}
        if len(framing) != 1:
            raise ValueError(f"scene videos do not share matching resolution/framing: {sorted(framing)}")

        baseline_all = baseline_root / "baseline_all_scenes.mp4"
        current_all = run_root / "current_all_scenes.mp4"
        comparison = run_root / "comparison_two_column.mp4"
        for inputs, output in ((baseline_scene_videos, baseline_all), (current_scene_videos, current_all)):
            if args.force or not video_is_valid(output, 1800, 10):
                command = build_concat_command(inputs, output, plan["fps"])
                status["commands"].append(command)
                run_command(command)
            require_video(output, 1800, 10)
        baseline_all_info = require_video(baseline_all, 1800, 10)
        current_all_info = require_video(current_all, 1800, 10)
        if (baseline_all_info["width"], baseline_all_info["height"]) != (
            current_all_info["width"], current_all_info["height"]
        ):
            raise ValueError("combined baseline/current videos have incompatible framing")
        comparison_valid = video_is_valid(comparison, 1800, 10)
        if comparison_valid:
            comparison_info = require_video(comparison, 1800, 10)
            comparison_valid = comparison_info["width"] == baseline_all_info["width"] * 2
        if args.force or not comparison_valid:
            command = build_comparison_command(baseline_all, current_all, comparison)
            status["commands"].append(command)
            run_command(command)
        comparison_info = require_video(comparison, 1800, 10)
        if comparison_info["width"] != baseline_all_info["width"] * 2:
            raise ValueError(f"comparison width is not two columns: {comparison_info}")

        run_manifest = {
            **plan, "completed_at": utc_now(), "commands": status["commands"],
            "outputs": {
                "baseline_all_scenes": str(baseline_all),
                "current_all_scenes": str(current_all),
                "comparison_two_column": str(comparison),
            },
        }
        atomic_write_json(run_root / "run_manifest.json", run_manifest)
        status.update({"state": "complete", "completed_at": utc_now(), "outputs": run_manifest["outputs"]})
        atomic_write_json(status_path, status)
        return run_root
    except Exception as exc:
        status.update({"state": "failed", "failed_at": utc_now(), "error": str(exc)})
        atomic_write_json(status_path, status)
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", default=str(DEFAULT_SPEC))
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in ("preflight", "run"):
        child = subparsers.add_parser(name)
        child.add_argument("--config", default=str(DEFAULT_CONFIG))
        child.add_argument("--ckpt", default="auto", help="auto or an explicit strict ckpt_N.pt path")
        child.add_argument(
            "--base-ckpt",
            help="explicit frozen version-1 ckpt_N.pt; defaults to the path in the sampling spec",
        )
        child.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
        if name == "run":
            child.add_argument("--gpu", default="0")
            child.add_argument("--force", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    plan = build_plan(args)
    output_root = resolve_project_path(args.output_root)
    if args.command == "preflight":
        plan_path = output_root / "preflight_plan.json"
        atomic_write_json(plan_path, plan)
        print(json.dumps(plan, indent=2))
        print(f"preflight plan written to {plan_path}")
        return
    run_root = run_sampling(args, plan)
    print(f"regression sampling complete: {run_root}")


if __name__ == "__main__":
    main()