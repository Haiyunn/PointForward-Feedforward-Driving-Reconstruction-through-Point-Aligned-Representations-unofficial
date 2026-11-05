"""Dataset loaders.

Two Waymo preprocessing layouts are supported:

* :class:`WaymoDriveStudioClips` -- the DriveStudio / OmniRe layout, which
  carries per-frame LiDAR and world-space instance boxes.  This is the primary
  loader; it supports the paper's LiDAR setting.
* :class:`WaymoClips` -- the StreetGaussian layout, whose boxes are ego-frame
  with a yaw angle and which has no per-frame LiDAR.

Both yield the same item structure, so :func:`to_scene_input` works with either.
"""

from .drivestudio import WaymoDriveStudioClips, WaymoDriveStudioWindows, collate_windows
from .drivestudio import ALL_CAMERAS as DS_ALL_CAMERAS
from .drivestudio import FRONT_CAMERAS as DS_FRONT_CAMERAS
from .drivestudio import RIGID_CLASSES as DS_RIGID_CLASSES
from .waymo import (
    ALL_CAMERAS,
    FRONT_CAMERAS,
    RIGID_CLASSES,
    WaymoClips,
    collate_clip,
    load_track_info,
    to_scene_input,
)

__all__ = [
    "WaymoDriveStudioClips",
    "WaymoDriveStudioWindows",
    "collate_windows",
    "DS_ALL_CAMERAS",
    "DS_FRONT_CAMERAS",
    "DS_RIGID_CLASSES",
    "ALL_CAMERAS",
    "FRONT_CAMERAS",
    "RIGID_CLASSES",
    "WaymoClips",
    "collate_clip",
    "load_track_info",
    "to_scene_input",
]
