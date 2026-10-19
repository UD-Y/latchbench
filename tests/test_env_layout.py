"""RJ45 env layout: in-hand plug pose, start pose and reachability of every port under randomization (no Isaac Sim).

Needs ``pxr`` (patchbench.assets) and ``isaaclab_so101`` (pure-Python ee_pose/constants): run in the Arena venv,
    cd ~/work/IsaacLab-Arena && .venv/bin/python -m pytest -q ~/work/patchbench/tests/test_env_layout.py
"""

import itertools
import math

import numpy as np
import pytest

pytest.importorskip("pxr")
ee_pose = pytest.importorskip("isaaclab_so101.ee_pose")

from isaaclab_so101.constants import JOINT_LIMITS_RAD, PAN_AXIS_XY, SIM_JOINT_NAMES  # noqa: E402

from patchbench.environment import start_joint_pose  # noqa: E402
from patchbench.objects import (  # noqa: E402
    DIMS,
    PANEL_POS,
    PANEL_POS_RANGE,
    PANEL_ROT_RANGE,
    PLUG_GRASP_BEHIND,
    PLUG_IN_HAND_ROT_XYZW,
    TCP_IN_GRIPPER,
    plug_in_hand_pos,
    wrist_roll_for,
)


def rot(q):
    x, y, z, w = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])  # fmt: skip


def test_in_hand_rotation():
    r = rot(PLUG_IN_HAND_ROT_XYZW)  # columns: plug axes in the gripper frame
    np.testing.assert_allclose(r[:, 0], [0, 1, 0], atol=1e-12)  # insertion = gripper +Y (wrist camera looks along it)
    np.testing.assert_allclose(r[:, 1], [1, 0, 0], atol=1e-12)  # plug width across the jaws
    np.testing.assert_allclose(r[:, 2], [0, 0, -1], atol=1e-12)  # latch side along the jaws (down when jaws point down)
    assert np.isclose(np.linalg.det(r), 1.0)


def test_in_hand_position():
    tip = np.array(plug_in_hand_pos())
    grasp = tip - (DIMS.plug_length + PLUG_GRASP_BEHIND) * rot(PLUG_IN_HAND_ROT_XYZW)[:, 0]
    np.testing.assert_allclose(grasp, TCP_IN_GRIPPER, atol=1e-12)


def _ik(tip_env, yaw=0.0, pitch=0.0):
    """Joint angles for the plug tip at ``tip_env`` with its axis at (yaw, pitch) from env +X; None if unreachable."""
    L = DIMS.plug_length + PLUG_GRASP_BEHIND
    axis = np.array([math.cos(pitch) * math.cos(yaw), math.cos(pitch) * math.sin(yaw), -math.sin(pitch)])
    tcp = np.array(tip_env) - L * axis
    xb, yb, zb = tcp[1], -tcp[0], tcp[2]  # env -> base frame (base yawed +90 deg)
    psi = math.atan2(yb - PAN_AXIS_XY[1], xb - PAN_AXIS_XY[0])
    return ee_pose.natural_ik(xb, yb, zb, pitch, wrist_roll_for(psi, -math.pi / 2 + yaw), wrist_floor=0.0)


def _margin(q):
    return min(min(v - lo, hi - v) for v, (lo, hi) in zip(q, JOINT_LIMITS_RAD))


def test_start_pose_holds_plug_in_front_of_middle_port():
    q = start_joint_pose(PANEL_POS)
    assert list(q) == list(SIM_JOINT_NAMES)
    ref = _ik((PANEL_POS[0] - 0.030, PANEL_POS[1], PANEL_POS[2]))
    np.testing.assert_allclose([q[n] for n in SIM_JOINT_NAMES[:5]], ref, atol=1e-9)
    # gripper straight down + roll => plug axis horizontal along env +X: base -Y.
    tcp_b = (PANEL_POS[1], -(PANEL_POS[0] - 0.030 - DIMS.plug_length - PLUG_GRASP_BEHIND))
    psi = math.atan2(tcp_b[1] - PAN_AXIS_XY[1], tcp_b[0] - PAN_AXIS_XY[0])
    r = rot(ee_pose.natural_ee_quat_xyzw(*tcp_b, 0.0, wrist_roll_for(psi))) @ rot(PLUG_IN_HAND_ROT_XYZW)
    np.testing.assert_allclose(r[:, 0], [0, -1, 0], atol=1e-9)
    np.testing.assert_allclose(r[:, 2], [0, 0, -1], atol=1e-9)


def test_every_port_reachable_under_randomization():
    worst = math.inf
    corners = (-PANEL_POS_RANGE, PANEL_POS_RANGE)
    angles = (-PANEL_ROT_RANGE, PANEL_ROT_RANGE)
    for dx, dy, dz, yaw, pitch, port, depth in itertools.product(
        corners, corners, corners, angles, angles, range(DIMS.n_ports), (-0.015, DIMS.seat_depth + 0.003)
    ):
        y_port = (1 - port) * DIMS.pitch
        mouth = np.array(PANEL_POS) + [dx, dy, dz] + [-y_port * math.sin(yaw), y_port * math.cos(yaw), 0.0]
        axis = np.array([math.cos(pitch) * math.cos(yaw), math.cos(pitch) * math.sin(yaw), -math.sin(pitch)])
        q = _ik(mouth + depth * axis, yaw, pitch)
        assert q is not None, (dx, dy, dz, yaw, pitch, port, depth)
        worst = min(worst, _margin(q))
    assert worst > 0.30, worst
