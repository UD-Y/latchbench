"""Open-loop insertion: go to the estimated port pose and push (CLAUDE.md section 7).

Also holds what every scripted controller shares: the observation struct, the noisy port-pose estimate and the
batched phase machine up to the approach. :class:`~patchbench.controllers.spiral.SpiralController` extends it with
contact sensing; nothing else differs between the controllers (same motion, same IK, same speeds).

Geometry: the target port's ``mouth`` frame (columns: insertion, width, height axes) and the plug frame (origin at
the plug tip, +X insertion, +Z latch side) coincide at an aligned insertion of depth 0 (assets note,
``patchbench/assets/build_assets.py``). The controller therefore commands the plug frame to
``mouth_est + depth * x_axis + lat[0] * y_axis + lat[1] * z_axis`` with the mouth's orientation.

All tensors are batched over environments; there are no per-env Python loops.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from .ik import CartesianServo, ServoCfg, rot_exp, rot_log

# Phases (shared numbering; the spiral controller adds SEARCH and INSERT).
INIT, TO_STANDOFF, TO_PRE, SETTLE, APPROACH, SEARCH, INSERT, HOLD, DONE = range(9)
PHASES = ("init", "to_standoff", "to_pre", "settle", "approach", "search", "insert", "hold", "done")


@dataclass
class ControllerObs:
    """What a scripted controller sees, batched over ``E`` envs, robot ``base`` frame, SI units.

    The port pose is the TRUE pose; the controller adds its own estimate noise (``pose_noise_mm``).
    ``load`` and ``force`` are only read by the telemetry / oracle spiral respectively.
    """

    joint_pos: torch.Tensor  # (E, 6) measured joints, sim order Rotation, Pitch, Elbow, Wrist_Pitch, Wrist_Roll, Jaw
    port_pos: torch.Tensor  # (E, 3) target mouth centre
    port_rot: torch.Tensor  # (E, 3, 3) columns = insertion, width, height axes
    tool_p: torch.Tensor  # (E, 3) plug frame origin (tip) in the gripper link frame
    tool_R: torch.Tensor  # (E, 3, 3) plug frame in the gripper link frame
    seat_depth: torch.Tensor | float = 0.016  # m (build_assets.jack_geometry: seat at x = 16 mm)
    load: torch.Tensor | None = None  # (E, 6) signed Present_Load [counts], sim joint order
    force: torch.Tensor | None = None  # (E, 3) plug + latch contact force from the panel [N]


@dataclass
class OpenLoopCfg:
    dt: float = 1 / 30
    """Control period (LeRobot 30 Hz; patchbench.physics DECIMATION)."""
    pose_noise_mm: float = 1.0
    """Per-axis std of the port-position estimate error [mm] (CLAUDE.md 6.2: 1 or 2; 0 for the expert)."""
    seed: int = 0
    standoff: float = 0.040
    """m in front of the estimated mouth: first waypoint, reached on a straight line from the start pose."""
    pre_insert: float = 0.015
    """m in front of the estimated mouth (CLAUDE.md section 7)."""
    move_speed: float = 0.08
    """m/s to the standoff."""
    turn_speed: float = 1.5
    """rad/s, orientation interpolation to the standoff."""
    pre_speed: float = 0.03
    """m/s from the standoff to the pre-insert pose."""
    settle_s: float = 0.6
    """s at the pre-insert pose (sag integral running) before the approach ..."""
    settle_tol: float = 0.0004
    """... once the measured tip is within this distance of the target [m] (or after 3x settle_s)."""
    approach_speed: float = 0.010
    """m/s along the insertion axis (CLAUDE.md section 7)."""
    push_past_seat: float = 0.003
    """m beyond the seat depth for the final commanded depth (CLAUDE.md section 7: seat depth + 3 mm)."""
    hold_s: float = 1.0
    servo: ServoCfg = field(default_factory=ServoCfg)


class OpenLoopController:
    """Batched phase machine: standoff -> pre-insert -> settle -> approach and push to seat + 3 mm -> hold."""

    def __init__(self, num_envs: int, device: str | torch.device = "cpu", cfg: OpenLoopCfg | None = None):
        self.cfg = cfg or OpenLoopCfg()
        self.num_envs, self.device = num_envs, torch.device(device)
        self.servo = CartesianServo(num_envs, device, self.cfg.servo, self.cfg.dt)
        self._gen = torch.Generator().manual_seed(self.cfg.seed)
        E, dev = num_envs, self.device
        self.phase = torch.full((E,), INIT, dtype=torch.long, device=dev)
        self.t_phase = torch.zeros(E, device=dev)  # time in the current phase [s]
        self.noise = torch.zeros(E, 3, device=dev)
        self.est_p = torch.zeros(E, 3, device=dev)
        self.est_R = torch.eye(3, device=dev).repeat(E, 1, 1)
        self.start_p = torch.zeros(E, 3, device=dev)
        self.start_R = torch.eye(3, device=dev).repeat(E, 1, 1)
        self.move_T = torch.ones(E, device=dev)
        self.depth = torch.zeros(E, device=dev)  # commanded depth along the estimated axis [m]
        self.lat = torch.zeros(E, 2, device=dev)  # commanded lateral offset (width, height) [m]
        self.jaw_goal = torch.zeros(E, device=dev)
        self.meas_depth = torch.zeros(E, device=dev)  # measured tip depth along the estimated axis [m]
        self.meas_lat = torch.zeros(E, 2, device=dev)

    # ---------------------------------------------------------------------------------------------- lifecycle
    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        """Mark envs for re-initialisation from their next observation."""
        ids = self._ids(env_ids)
        self.phase[ids] = INIT

    def _ids(self, env_ids) -> torch.Tensor:
        if env_ids is None:
            return torch.arange(self.num_envs, device=self.device)
        return torch.as_tensor(env_ids, device=self.device, dtype=torch.long).flatten()

    def _init_envs(self, ids: torch.Tensor, obs: ControllerObs) -> None:
        cfg = self.cfg
        n = ids.numel()
        noise = torch.randn(n, 3, generator=self._gen) * (cfg.pose_noise_mm * 1e-3)
        self.noise[ids] = noise.to(self.device)
        self.servo.reset(ids, obs.joint_pos)
        p, R = self.servo.measured_pose(obs.joint_pos, obs.tool_p, obs.tool_R)
        self.start_p[ids], self.start_R[ids] = p[ids], R[ids]
        self.est_p[ids] = obs.port_pos[ids] + self.noise[ids]
        self.est_R[ids] = obs.port_rot[ids]
        stand = self.est_p[ids] - cfg.standoff * self.est_R[ids][..., 0]
        dist = (stand - p[ids]).norm(dim=-1)
        ang = rot_log(self.est_R[ids] @ R[ids].transpose(-1, -2)).norm(dim=-1)
        self.move_T[ids] = torch.maximum(torch.maximum(dist / cfg.move_speed, ang / cfg.turn_speed), torch.tensor(0.5))
        self.depth[ids] = -cfg.standoff
        self.lat[ids] = 0.0
        self.jaw_goal[ids] = obs.joint_pos[ids, 5]
        self.t_phase[ids] = 0.0
        self.phase[ids] = TO_STANDOFF
        self._init_extra(ids)

    def _init_extra(self, ids: torch.Tensor) -> None:
        """Hook for subclasses."""

    # --------------------------------------------------------------------------------------------------- step
    def _goto(self, mask: torch.Tensor, phase: int) -> None:
        self.phase = torch.where(mask, torch.full_like(self.phase, phase), self.phase)
        self.t_phase = torch.where(mask, torch.zeros_like(self.t_phase), self.t_phase)

    def _port_target(self) -> torch.Tensor:
        R = self.est_R
        return self.est_p + self.depth.unsqueeze(-1) * R[..., 0] + self.lat[:, :1] * R[..., 1] + self.lat[:, 1:] * R[..., 2]

    def compute(self, obs: ControllerObs) -> torch.Tensor:
        """Joint goal positions ``(E, 6)`` (sim order, rad) for one control step."""
        cfg = self.cfg
        init = self.phase == INIT
        if init.any():
            self._init_envs(init.nonzero().flatten(), obs)
        self.t_phase += cfg.dt

        # Measured plug pose in the estimated port frame (joint encoders are available to every controller).
        p_meas, _ = self.servo.measured_pose(obs.joint_pos, obs.tool_p, obs.tool_R)
        rel = torch.einsum("eij,ei->ej", self.est_R, p_meas - self.est_p)
        self.meas_depth, self.meas_lat = rel[:, 0], rel[:, 1:]

        ph = self.phase
        # standoff -> pre-insert -> settle (shared)
        m = ph == TO_PRE
        self.depth = torch.where(m, torch.clamp(self.depth + cfg.pre_speed * cfg.dt, max=-cfg.pre_insert), self.depth)
        self._goto(m & (self.depth >= -cfg.pre_insert), SETTLE)
        m = (ph == TO_STANDOFF) & (self.t_phase >= self.move_T)
        self._goto(m, TO_PRE)
        err = (p_meas - self._port_target()).norm(dim=-1)
        m = (ph == SETTLE) & (((self.t_phase >= cfg.settle_s) & (err < cfg.settle_tol)) | (self.t_phase >= 3 * cfg.settle_s))
        self._goto(m, APPROACH)

        self._step_contact_phases(obs)

        # target pose
        tgt_p, tgt_R = self._port_target(), self.est_R
        mv = self.phase == TO_STANDOFF
        if mv.any():
            s = (self.t_phase / self.move_T).clamp(0.0, 1.0).unsqueeze(-1)
            p_lin = self.start_p + s * (tgt_p - self.start_p)
            R_lin = self.start_R @ rot_exp(s * rot_log(self.start_R.transpose(-1, -2) @ self.est_R))
            tgt_p = torch.where(mv.unsqueeze(-1), p_lin, tgt_p)
            tgt_R = torch.where(mv.view(-1, 1, 1), R_lin, tgt_R)
        integrate = self.phase <= SETTLE
        arm = self.servo.step(tgt_p, tgt_R, obs.joint_pos, obs.tool_p, obs.tool_R, integrate)
        return torch.cat([arm, self.jaw_goal.unsqueeze(-1)], dim=-1)

    def _step_contact_phases(self, obs: ControllerObs) -> None:
        """Open loop: approach and push to seat + 3 mm, hold, done."""
        cfg = self.cfg
        seat = torch.as_tensor(obs.seat_depth, device=self.device, dtype=self.depth.dtype).expand(self.num_envs)
        final = seat + cfg.push_past_seat
        m = self.phase == APPROACH
        self.depth = torch.where(m, torch.minimum(self.depth + cfg.approach_speed * cfg.dt, final), self.depth)
        self._goto(m & (self.depth >= final), HOLD)
        self._goto((self.phase == HOLD) & (self.t_phase >= cfg.hold_s), DONE)
