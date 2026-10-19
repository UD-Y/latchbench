"""Batched differential IK for the 5-DOF SO-101 (position rows weight 1.0, orientation soft-weighted).

Three layers, all pure torch, batched over environments ``(E, ...)``, no Isaac imports:

- :class:`SO101Kinematics`: forward kinematics and the geometric Jacobian of the arm (``Rotation`` ... ``Wrist_Roll``)
  up to the ``gripper`` link, in the robot ``base`` frame, plus an optional tool transform (e.g. the plug tip held in
  the gripper). Joint origins are copied from ``isaaclab_so101``'s URDF, which Isaac Lab generated from the same USD
  the sim loads (``SO-ARM101-USD.usd``); ``tests/test_controllers_ik.py`` checks the TCP at the home pose against
  ``isaaclab_so101.ee_pose.HOME_NATURAL_POSE``.
- :func:`dls_step`: one weighted damped-least-squares step from a Jacobian and a pose error supplied by the caller
  (works equally with a PhysX Jacobian).
- :class:`CartesianServo`: what the controllers use. It solves the IK *kinematically* (iterating on its own joint
  goal, not on the measured joints), so the goal it sends is the joint vector whose FK is the target, exactly like a
  real LeRobot ``Goal_Position`` stream. Servo sag under gravity is removed by a slow task-space integral on the
  measured error that the controllers enable in free space and freeze in contact, so a target placed past a surface
  produces a push proportional to the servo's position stiffness rather than a winding-up integral.

Rotations are 3x3 matrices; quaternion conventions stay in the caller (Isaac Lab here is (x, y, z, w)).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

ARM_JOINTS = ("Rotation", "Pitch", "Elbow", "Wrist_Pitch", "Wrist_Roll")

# URDF joint origins (xyz [m], rpy [rad], parent -> child, axis +Z after the origin), from
# isaaclab_so101/embodiments/data/curobo/urdf/SO-ARM101-USD.urdf @ 12a3ecc (generated from SO-ARM101-USD.usd).
_URDF_JOINTS = (
    ((0.0207909, -0.0230745, 0.0948817), (-3.14159012, 0.0, 1.57079996)),  # Rotation: base -> shoulder
    ((-0.0303992, -0.0182778, -0.05420001), (1.56253029, -1.57079272, -3.13333017)),  # Pitch -> upper_arm
    ((-0.11257003, -0.028, 0.0), (0.00000005, 0.00000001, 1.57079976)),  # Elbow -> lower_arm
    ((-0.13490003, 0.00519998, -0.00000002), (-0.00000008, -0.00000023, -1.57079975)),  # Wrist_Pitch -> wrist
    ((0.0, -0.06110004, 0.01810001), (1.57080003, -0.0000006, 3.14159021)),  # Wrist_Roll -> gripper
)

# isaaclab_so101.constants (USD limits, degrees), arm joints only.
ARM_LIMITS_DEG = ((-110.0, 110.0), (-100.0, 100.0), (-100.0, 90.0), (-95.0, 95.0), (-160.0, 160.0))
# isaaclab_so101.constants.TCP_OFFSET: between the closed jaw tips, gripper frame (-Z runs along the jaws).
TCP_OFFSET = (-0.007, 0.0, -0.102)


# ----------------------------------------------------------------------------------------------- rotation helpers
def rpy_matrix(r: float, p: float, y: float) -> torch.Tensor:
    """URDF convention: R = Rz(y) Ry(p) Rx(r)."""
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    return torch.tensor(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ],
        dtype=torch.float64,
    )


def rot_z(q: torch.Tensor) -> torch.Tensor:
    c, s = torch.cos(q), torch.sin(q)
    o, z = torch.ones_like(q), torch.zeros_like(q)
    return torch.stack([torch.stack([c, -s, z], -1), torch.stack([s, c, z], -1), torch.stack([z, z, o], -1)], -2)


def skew(v: torch.Tensor) -> torch.Tensor:
    x, y, z = v.unbind(-1)
    o = torch.zeros_like(x)
    return torch.stack([torch.stack([o, -z, y], -1), torch.stack([z, o, -x], -1), torch.stack([-y, x, o], -1)], -2)


def rot_log(R: torch.Tensor) -> torch.Tensor:
    """Axis-angle vector of a rotation matrix ``(..., 3, 3) -> (..., 3)`` (angles below ~179 deg)."""
    cos = ((R.diagonal(dim1=-2, dim2=-1).sum(-1) - 1.0) / 2.0).clamp(-1.0, 1.0)
    angle = torch.acos(cos)
    v = torch.stack([R[..., 2, 1] - R[..., 1, 2], R[..., 0, 2] - R[..., 2, 0], R[..., 1, 0] - R[..., 0, 1]], -1)
    sin = torch.sin(angle)
    scale = torch.where(sin > 1e-6, angle / (2.0 * sin.clamp(min=1e-6)), torch.full_like(sin, 0.5))
    return v * scale.unsqueeze(-1)


def rot_exp(w: torch.Tensor) -> torch.Tensor:
    """Rodrigues: axis-angle ``(..., 3) -> (..., 3, 3)``."""
    angle = w.norm(dim=-1, keepdim=True)
    K = skew(w / angle.clamp(min=1e-12))
    a = angle.unsqueeze(-1)
    eye = torch.eye(3, dtype=w.dtype, device=w.device).expand(K.shape)
    return eye + torch.sin(a) * K + (1.0 - torch.cos(a)) * (K @ K)


def quat_xyzw_to_matrix(q: torch.Tensor) -> torch.Tensor:
    q = q / q.norm(dim=-1, keepdim=True)
    x, y, z, w = q.unbind(-1)
    return torch.stack(
        [
            torch.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], -1),
            torch.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], -1),
            torch.stack([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], -1),
        ],
        -2,
    )


def pose_error(p: torch.Tensor, R: torch.Tensor, p_t: torch.Tensor, R_t: torch.Tensor) -> torch.Tensor:
    """``(E, 6)`` = (position error, orientation error as a base-frame axis-angle) from current to target."""
    return torch.cat([p_t - p, rot_log(R_t @ R.transpose(-1, -2))], dim=-1)


# ------------------------------------------------------------------------------------------------ kinematics
class SO101Kinematics:
    """FK + geometric Jacobian of the SO-101 arm in the ``base`` frame. ``q``: ``(E, 5)`` arm joints [rad]."""

    def __init__(self, device: str | torch.device = "cpu", dtype: torch.dtype = torch.float32):
        self.device, self.dtype = torch.device(device), dtype
        self.origin_p = torch.tensor([j[0] for j in _URDF_JOINTS], dtype=dtype, device=self.device)
        self.origin_R = torch.stack([rpy_matrix(*j[1]) for j in _URDF_JOINTS]).to(self.device, dtype)
        lim = torch.tensor(ARM_LIMITS_DEG, dtype=dtype, device=self.device) * (math.pi / 180.0)
        self.q_min, self.q_max = lim[:, 0], lim[:, 1]

    def forward(
        self, q: torch.Tensor, tool_p: torch.Tensor | None = None, tool_R: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Tool pose and joint frames.

        Args:
            q: ``(E, 5)`` arm joint angles.
            tool_p, tool_R: tool frame in the ``gripper`` link frame, ``(E, 3)`` / ``(E, 3, 3)`` or unbatched;
                default: the gripper link origin.

        Returns:
            ``(p, R, axes, origins)``: tool position ``(E, 3)``, rotation ``(E, 3, 3)``, joint axes ``(E, 5, 3)``
            and joint positions ``(E, 5, 3)``, all in the base frame.
        """
        E = q.shape[0]
        R = torch.eye(3, dtype=q.dtype, device=q.device).expand(E, 3, 3)
        p = torch.zeros(E, 3, dtype=q.dtype, device=q.device)
        axes, origins = [], []
        Rz = rot_z(q)
        for i in range(5):
            p = p + (R @ self.origin_p[i].to(q.dtype).unsqueeze(-1)).squeeze(-1)
            R = R @ self.origin_R[i].to(q.dtype)
            axes.append(R[..., :, 2])
            origins.append(p)
            R = R @ Rz[:, i]
        if tool_p is not None:
            p = p + (R @ tool_p.to(q.dtype).expand(E, 3).unsqueeze(-1)).squeeze(-1)
        if tool_R is not None:
            R = R @ tool_R.to(q.dtype).expand(E, 3, 3)
        return p, R, torch.stack(axes, 1), torch.stack(origins, 1)

    def jacobian(self, p: torch.Tensor, axes: torch.Tensor, origins: torch.Tensor) -> torch.Tensor:
        """``(E, 6, 5)``: linear rows ``z_i x (p - o_i)``, angular rows ``z_i``."""
        lin = torch.cross(axes, p.unsqueeze(1) - origins, dim=-1)
        return torch.cat([lin, axes], dim=-1).transpose(1, 2)

    def clamp(self, q: torch.Tensor) -> torch.Tensor:
        return torch.maximum(torch.minimum(q, self.q_max), self.q_min)


# ------------------------------------------------------------------------------------------------------ IK
def dls_step(
    J: torch.Tensor, err: torch.Tensor, orient_weight: float | torch.Tensor, damping: float
) -> torch.Tensor:
    """Weighted damped least squares: ``dq = Jw^T (Jw Jw^T + lambda^2 I)^-1 W e``.

    Args:
        J: ``(E, 6, n)`` Jacobian (linear rows first).
        err: ``(E, 6)`` pose error (position [m], orientation axis-angle [rad]).
        orient_weight: weight of the orientation rows relative to position (1 rad counts like ``w`` metres).
        damping: lambda [m].
    """
    w = torch.ones(6, dtype=J.dtype, device=J.device)
    w[3:] = orient_weight
    Jw = J * w.view(1, 6, 1)
    ew = err * w
    A = Jw @ Jw.transpose(1, 2) + (damping**2) * torch.eye(6, dtype=J.dtype, device=J.device)
    return (Jw.transpose(1, 2) @ torch.linalg.solve(A, ew.unsqueeze(-1))).squeeze(-1)


def priority_step(J: torch.Tensor, err: torch.Tensor, damping: float, orient_damping: float) -> torch.Tensor:
    """Task-priority DLS: position rows exact (primary), orientation rows in the position null space (secondary).

    The 5-DOF SO-101 has 2 joint DOF left after position, so of the 3 orientation errors it fixes the best 2-D
    projection; an unreachable orientation never pulls the plug tip off its position target (a weighted solve with
    orientation weight 0.1 moved it 15 mm for a 34 deg yaw error, tests/test_controllers_ik.py).
    """
    Jp, Jo = J[:, :3], J[:, 3:]
    eye = torch.eye(3, dtype=J.dtype, device=J.device)
    Jp_pinv = Jp.transpose(1, 2) @ torch.linalg.inv(Jp @ Jp.transpose(1, 2) + damping**2 * eye)
    dq1 = (Jp_pinv @ err[:, :3].unsqueeze(-1)).squeeze(-1)
    # Exact projector (SVD pinv): a damped one leaks the orientation step into position (18 mm in that test).
    N = torch.eye(J.shape[-1], dtype=J.dtype, device=J.device) - torch.linalg.pinv(Jp) @ Jp
    Jt = Jo @ N
    rhs = err[:, 3:] - (Jo @ dq1.unsqueeze(-1)).squeeze(-1)
    dq2 = (Jt.transpose(1, 2) @ torch.linalg.solve(Jt @ Jt.transpose(1, 2) + orient_damping**2 * eye, rhs.unsqueeze(-1)))
    return dq1 + dq2.squeeze(-1)


@dataclass
class IKCfg:
    mode: str = "priority"
    """"priority" (position primary, orientation in its null space) or "weighted" (one DLS with soft orientation rows)."""
    orient_damping: float = 0.05
    """Priority mode: DLS lambda of the orientation sub-task [rad]."""
    orient_weight: float = 0.1
    """Weighted mode: orientation rows soft-weighted, 1 rad of error counts like 0.1 m (CLAUDE.md section 7). A 5-DOF arm cannot
    hold every orientation; inside the reachable insertion cone both converge, outside it position wins."""
    damping: float = 0.005
    """DLS lambda [m]."""
    iterations: int = 8
    """Kinematic IK iterations per control step (warm-started from the previous goal)."""
    max_step: float = 0.2
    """Max joint change per iteration [rad]."""


def solve_ik(
    kin: SO101Kinematics,
    q0: torch.Tensor,
    target_p: torch.Tensor,
    target_R: torch.Tensor,
    tool_p: torch.Tensor | None = None,
    tool_R: torch.Tensor | None = None,
    cfg: IKCfg | None = None,
    iterations: int | None = None,
) -> torch.Tensor:
    """Kinematic IK: iterate DLS on the FK model from ``q0`` (``(E, 5)``) toward the target pose."""
    cfg = cfg or IKCfg()
    q = q0.clone()
    for _ in range(cfg.iterations if iterations is None else iterations):
        p, R, axes, origins = kin.forward(q, tool_p, tool_R)
        J, err = kin.jacobian(p, axes, origins), pose_error(p, R, target_p, target_R)
        if cfg.mode == "priority":
            dq = priority_step(J, err, cfg.damping, cfg.orient_damping)
        else:
            dq = dls_step(J, err, cfg.orient_weight, cfg.damping)
        q = kin.clamp(q + dq.clamp(-cfg.max_step, cfg.max_step))
    return q


@dataclass
class ServoCfg:
    ik: IKCfg = None  # type: ignore[assignment]
    sag_gain: float = 4.0
    """1/s, task-space integral on the measured position error (free space only): removes the servo's gravity sag
    (hold test: Pitch 0.6 deg, Elbow 1.1 deg at P=16 -> mm at the plug tip)."""
    sag_limit: float = 0.010
    """m, clamp on the integral offset."""

    def __post_init__(self):
        if self.ik is None:
            self.ik = IKCfg()


class CartesianServo:
    """Turns a tool-frame target pose (base frame) into arm joint goals; one instance per controller, batched."""

    def __init__(self, num_envs: int, device: str | torch.device = "cpu", cfg: ServoCfg | None = None, dt: float = 1 / 30):
        self.cfg = cfg or ServoCfg()
        self.kin = SO101Kinematics(device)
        self.dt = dt
        self.goal = torch.zeros(num_envs, 5, device=self.kin.device)
        self.offset = torch.zeros(num_envs, 3, device=self.kin.device)

    def reset(self, env_ids: torch.Tensor, q_meas: torch.Tensor) -> None:
        self.goal[env_ids] = q_meas[env_ids, :5]
        self.offset[env_ids] = 0.0

    def measured_pose(self, q_meas: torch.Tensor, tool_p, tool_R) -> tuple[torch.Tensor, torch.Tensor]:
        p, R, _, _ = self.kin.forward(q_meas[:, :5], tool_p, tool_R)
        return p, R

    def step(
        self,
        target_p: torch.Tensor,
        target_R: torch.Tensor,
        q_meas: torch.Tensor,
        tool_p: torch.Tensor,
        tool_R: torch.Tensor,
        integrate: torch.Tensor,
    ) -> torch.Tensor:
        """Arm joint goals ``(E, 5)``. ``integrate``: ``(E,)`` bool, update the sag integral (free space)."""
        p_meas, _ = self.measured_pose(q_meas, tool_p, tool_R)
        upd = self.offset + self.cfg.sag_gain * self.dt * (target_p - p_meas)
        n = upd.norm(dim=-1, keepdim=True)
        upd = upd * (self.cfg.sag_limit / n.clamp(min=self.cfg.sag_limit))
        self.offset = torch.where(integrate.unsqueeze(-1), upd, self.offset)
        self.goal = solve_ik(self.kin, self.goal, target_p + self.offset, target_R, tool_p, tool_R, self.cfg.ik)
        return self.goal
