# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause
"""Helpers to adjust VR point-of-view by moving XR anchor Xforms (USD / pxr).

Isaac Lab OpenXRDevice sets carb ``/xrstage/profile/ar/customAnchor`` to the **world-space** anchor
(usually ``/World/XRAnchor``) created from :class:`isaaclab.devices.openxr.XrCfg` ``anchor_pos`` /
``anchor_rot``.

Kit’s XR stage also exposes an internal root under ``/_xr/stage/`` (e.g. ``xrCamera``). The
``/_xr/stage/xrAnchor`` Xform (when present) is the **stage-local** root for the headset rig; you
can translate/rotate it to shift POV without changing ``XrCfg``.

Use either path depending on whether you want to follow Isaac Lab’s custom anchor (recommended with
``teleop_med7_vr.py``) or drive the Kit XR stage directly.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pxr import Usd

# Isaac Lab default when ``anchor_prim_path`` is None (see OpenXRDevice).
WORLD_CUSTOM_ANCHOR_PATH = "/World/XRAnchor"
# Omniverse Kit XR stage root (headset rig); may appear after XR session starts.
XR_STAGE_ANCHOR_PATH = "/_xr/stage/xrAnchor"


def _matrix_from_translate_quat_wxyz(
    pos: tuple[float, float, float],
    quat_wxyz: tuple[float, float, float, float],
):
    from pxr import Gf

    w, x, y, z = quat_wxyz
    q = Gf.Quatd(w, Gf.Vec3d(x, y, z))
    q = q.GetNormalized()
    rot = Gf.Rotation(q)
    mat = Gf.Matrix4d()
    mat.SetRotate(rot)
    mat.SetTranslateOnly(Gf.Vec3d(float(pos[0]), float(pos[1]), float(pos[2])))
    return mat


def set_xform_matrix_from_pose(
    stage: "Usd.Stage",
    prim_path: str,
    pos: tuple[float, float, float],
    quat_wxyz: tuple[float, float, float, float],
) -> bool:
    """Set an Xform’s transform to ``pos`` + ``quat_wxyz`` (w,x,y,z) using a single matrix op."""
    from pxr import UsdGeom

    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        return False
    xf = UsdGeom.Xformable(prim)
    mat = _matrix_from_translate_quat_wxyz(pos, quat_wxyz)
    xf.MakeMatrixXform().Set(mat)
    return True


def set_world_custom_anchor_pose(
    stage: "Usd.Stage",
    pos: tuple[float, float, float],
    quat_wxyz: tuple[float, float, float, float],
) -> bool:
    """Update ``/World/XRAnchor`` (same prim Isaac Lab’s OpenXRDevice + carb customAnchor use)."""
    return set_xform_matrix_from_pose(stage, WORLD_CUSTOM_ANCHOR_PATH, pos, quat_wxyz)


def set_xr_stage_anchor_pose(
    stage: "Usd.Stage",
    pos: tuple[float, float, float],
    quat_wxyz: tuple[float, float, float, float],
) -> bool:
    """Update ``/_xr/stage/xrAnchor`` if that prim exists (Kit XR stage root)."""
    return set_xform_matrix_from_pose(stage, XR_STAGE_ANCHOR_PATH, pos, quat_wxyz)


def try_both_anchor_poses(
    stage: "Usd.Stage",
    pos: tuple[float, float, float],
    quat_wxyz: tuple[float, float, float, float],
) -> tuple[bool, bool]:
    """Apply the same pose to ``/World/XRAnchor`` and ``/_xr/stage/xrAnchor`` when valid.

    Returns:
        (world_anchor_ok, stage_anchor_ok)
    """
    return (
        set_world_custom_anchor_pose(stage, pos, quat_wxyz),
        set_xr_stage_anchor_pose(stage, pos, quat_wxyz),
    )
