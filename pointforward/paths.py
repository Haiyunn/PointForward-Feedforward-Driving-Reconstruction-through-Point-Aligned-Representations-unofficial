"""Command-line overrides for the filesystem paths in a config.

Configs ship with repository-relative defaults (``data/...``, ``third_party/InfiniDepth``).
Every entry script that loads a config accepts the same flags, so a local layout never
requires editing YAML:

    --data-root        DriveStudio-processed Waymo root (or the single source of a config)
    --argoverse-root   DriveStudio-processed Argoverse 2 root (mixed-source configs)
    --depth-repo       cloned InfiniDepth repository
    --depth-ckpt       InfiniDepth checkpoint
    --warm-start       checkpoint to warm-start from (sets ``train.warm_start``)
"""

from __future__ import annotations

import argparse


def add_path_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    group = parser.add_argument_group("paths")
    group.add_argument("--data-root", default=None, help="override data.root (Waymo / single source)")
    group.add_argument("--argoverse-root", default=None, help="override the root of Argoverse sources")
    group.add_argument("--depth-repo", default=None, help="override depth.repo_root")
    group.add_argument("--depth-ckpt", default=None, help="override depth.checkpoint")
    group.add_argument("--warm-start", default=None, help="override train.warm_start")
    return parser


def apply_path_args(cfg: dict, args: argparse.Namespace) -> dict:
    """Write the path flags present on ``args`` into ``cfg`` in place and return it."""
    get = lambda name: getattr(args, name, None)
    data = cfg.get("data") or {}
    if "sources" in data:
        for source in data["sources"]:
            argo = source.get("layout") == "argoverse_drivestudio"
            root = get("argoverse_root") if argo else get("data_root")
            if root:
                source["root"] = root
    elif get("data_root") and data:
        data["root"] = get("data_root")
    depth = cfg.get("depth")
    if depth is not None:
        if get("depth_repo"):
            depth["repo_root"] = get("depth_repo")
        if get("depth_ckpt"):
            depth["checkpoint"] = get("depth_ckpt")
    if get("warm_start"):
        cfg.setdefault("train", {})["warm_start"] = get("warm_start")
    return cfg
