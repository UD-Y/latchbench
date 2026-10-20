"""Render-only trailing cable for camera runs (dataset, videos): no physics, no forces on the robot.

The cable is ``CABLE_SEGMENTS`` cylinders per env (spawned by ``objects.attach_plug``). Every control step,
:func:`cable_follow` lays them along a cubic Bezier curve from the boot end (leaving along the plug's -X axis,
through the gripper jaws) to a fixed point on the table behind the robot base, entering it vertically. Poses are
written for all envs at once through Isaac Lab's ``FrameView`` (Fabric on PhysX).

Physical cable dynamics are out of scope on PhysX (CLAUDE.md 1.4, deformable cable); the Newton showcase simulates a
real cable (VBD). The Bezier tangent lengths below are cosmetic choices, not physics.
"""

from __future__ import annotations

import re

import torch

from patchbench.objects import CABLE_SEGMENTS, DIMS

BOOT_END_PLUG = (-DIMS.plug_length - 0.014, 0.0, 0.0)
"""m, plug frame: where the trailing cable starts (end of the look layer's boot)."""

ANCHOR_ENV = (-0.06, 0.15, 0.003)
"""m, env frame: where the cable meets the table (behind and to the left of the SO-101 base)."""

TANGENT_OUT, TANGENT_IN = 0.07, 0.10
"""m: Bezier handle lengths at the boot (along plug -X) and at the table (vertical)."""

OVERLAP = 1.06
"""Segment length scale so neighbouring cylinders overlap at bends."""


class _CableState:
    def __init__(self, env):
        from isaaclab.sim.views import FrameView

        from patchbench.task import PLUG_BODY, ROBOT

        self.robot = env.scene[ROBOT]
        self.body_idx = self.robot.find_bodies(PLUG_BODY)[0][0]
        self.view = FrameView("/World/envs/env_[0-9]+/Robot/cable_vis/seg_[0-9]+", device=env.device)
        paths = self.view.prim_paths
        E, K = env.num_envs, CABLE_SEGMENTS
        assert len(paths) == E * K, (len(paths), E, K)
        # view order -> (env, segment): sorted prim paths put env_10 before env_2
        order = [(int(re.search(r"env_(\d+)/", p).group(1)), int(p.rsplit("_", 1)[1])) for p in paths]
        self.flat = torch.tensor([e * K + k for e, k in order], device=env.device)
        dev = env.device
        self.boot = torch.tensor(BOOT_END_PLUG, device=dev)
        self.anchor = env.scene.env_origins + torch.tensor(ANCHOR_ENV, device=dev)
        self.t = torch.linspace(0.0, 1.0, K + 1, device=dev)


def _quat_x_to(d: torch.Tensor) -> torch.Tensor:
    """xyzw quaternion rotating +X onto unit vectors ``d`` (..., 3)."""
    x = torch.zeros_like(d)
    x[..., 0] = 1.0
    v = torch.cross(x, d, dim=-1)
    w = 1.0 + (d * x).sum(-1, keepdim=True)
    q = torch.cat([v, w], -1)
    flip = w[..., 0] < 1e-6  # d = -X: rotate pi about Z
    q[flip] = torch.tensor([0.0, 0.0, 1.0, 0.0], device=d.device)
    return q / q.norm(dim=-1, keepdim=True)


def cable_follow(env, env_ids=None):
    """Event term (modes ``reset`` and ``interval``): pose every env's cable segments along the boot-to-table curve."""
    import warp as wp
    from isaaclab.utils.math import matrix_from_quat

    st = getattr(env, "_pb_cable", None)
    if st is None:
        st = env._pb_cable = _CableState(env)
    pose = st.robot.data.body_link_pose_w
    pose = (pose.torch if hasattr(pose, "torch") else pose)[:, st.body_idx]  # (E, 7) pos + quat xyzw
    rot = matrix_from_quat(pose[:, 3:7])
    p0 = pose[:, :3] + rot @ st.boot
    p1 = p0 - TANGENT_OUT * rot[:, :, 0]
    p3 = st.anchor
    p2 = p3 + torch.tensor([0.0, 0.0, TANGENT_IN], device=p3.device)
    t = st.t[None, :, None]
    pts = ((1 - t) ** 3) * p0[:, None] + 3 * ((1 - t) ** 2) * t * p1[:, None] + 3 * (1 - t) * t**2 * p2[:, None] \
        + t**3 * p3[:, None]  # (E, K + 1, 3)  # fmt: skip
    d = pts[:, 1:] - pts[:, :-1]
    length = d.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    quat = _quat_x_to(d / length)
    E, K = d.shape[:2]
    scale = torch.cat([length * OVERLAP, torch.ones(E, K, 2, device=d.device)], -1)
    pos, quat, scale = (x.reshape(E * K, -1)[st.flat].contiguous() for x in (pts[:, :-1], quat, scale))
    with st.view.xform_world_space_writer() as w:
        w.set_poses(wp.from_torch(pos), wp.from_torch(quat))
        w.set_scales(wp.from_torch(scale))
