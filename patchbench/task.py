"""RJ45InsertTask(TaskBase): stage objectives (success = latched), failures, reset randomization, language string.

The per-env episode state lives in one :class:`RJ45State` per env instance (``env._rj45``), created lazily on first use
from ``env.cfg.rj45_params`` (set by the environment's ``env_cfg_callback``). It is updated once per control step,
the first time any term asks for it after physics (the failure terms run every step), and feeds
``patchbench.predicates.StageTracker`` (pure torch; the stage logic lives there, not here).

Frames. World quantities are what Isaac Lab reports. "Base" (``*_b``) is the SO-101 ``base`` link frame, the frame of
``ee_frame`` / ``eef_pos`` and of the IK action: the arm faces base -Y, i.e. env +X. Quaternions are (x, y, z, w).

Contact attribution. The panel is one rigid body, so the plug(+latch)-vs-panel force cannot be split by jack with a
filtered contact sensor. The whole force is assigned to the port whose mouth is laterally nearest the plug tip.
"""

from __future__ import annotations

import functools
import math
from dataclasses import dataclass, field

import torch
from isaaclab.managers import EventTermCfg, SceneEntityCfg, TerminationTermCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils.configclass import configclass
from isaaclab.utils.math import (
    combine_frame_transforms,
    matrix_from_quat,
    quat_from_euler_xyz,
    quat_mul,
    subtract_frame_transforms,
)
from isaaclab_arena.progress_tracking.progress_objective import ProgressObjective
from isaaclab_arena.tasks.task_base import TaskBase
from isaaclab_arena.tasks.task_termination_cfg import TaskTerminationCfg

from patchbench import physics
from patchbench.assets.build_assets import RJ45Dims, jack_geometry
from patchbench.objects import PANEL_POS, PANEL_POS_RANGE, PANEL_ROT_RANGE, PLUG_IN_HAND_ROT_XYZW, plug_in_hand_pos
from patchbench.predicates import STAGES, PortGeometry, StageTracker, port_coordinates

DIMS = RJ45Dims()
PORT_WORDS = ("left", "middle", "right")
"""Port index -> word, seen from the robot (port_0 is at panel +Y = left looking along the insertion axis)."""

ROBOT, PANEL, STAND = "robot", "rj45_panel", "rj45_panel_stand"
SENSOR_BODY, SENSOR_LATCH = "plug_body_contact", "plug_latch_contact"
PLUG_BODY, LATCH_JOINT = "body", "latch_joint"


def task_string(port: int) -> str:
    return f"insert the cable into the {PORT_WORDS[port]} port"


@dataclass
class RJ45Params:
    """Episode randomization and task settings (CLAUDE.md 6.2). Lengths in m, angles in rad."""

    tier: str = "medium"
    port: str = "random"  # "0" | "1" | "2" | "random"
    pose_noise: float = 0.001  # sigma of the port-position estimate, per axis [m]
    randomize: bool = True  # panel pose and friction (servo params: actuator cfg ``randomize``)
    panel_pos: tuple[float, float, float] = PANEL_POS  # nominal panel frame in the env frame
    panel_pos_range: float = PANEL_POS_RANGE  # +- per axis
    panel_rot_range: float = PANEL_ROT_RANGE  # +- yaw and pitch
    friction_scale: tuple[float, float] = (0.7, 1.3)
    stand_offset: tuple[float, float, float] = (0.0, 0.0, 0.0)  # stand centre in the panel frame (set by the env)
    seed: int = 0
    timeout_s: float = 20.0
    extra: dict = field(default_factory=dict)


class RJ45State:
    """Per-env RJ45 episode state: target port, pose-estimate noise, panel pose, stage tracker, cached signals."""

    def __init__(self, env, p: RJ45Params):
        from so101_contact.isaaclab.actuator import STS3215Actuator

        self.p, self.num_envs, self.device = p, env.num_envs, env.device
        E, dev = self.num_envs, self.device
        self.robot, self.panel, self.stand = env.scene[ROBOT], env.scene[PANEL], env.scene[STAND]
        self.sensors = [env.scene[SENSOR_BODY], env.scene[SENSOR_LATCH]]
        self.body_idx = self.robot.find_bodies(PLUG_BODY)[0][0]
        self.latch_jid = self.robot.find_joints(LATCH_JOINT)[0][0]
        self.servo_groups = STS3215Actuator.bind_all(self.robot)
        g = jack_geometry(DIMS, p.tier)
        self.clearance, self.seat_depth = g.clearance, g.seat_depth
        self.num_ports = DIMS.n_ports
        self.port_pos_p = torch.tensor([[0.0, (1 - i) * DIMS.pitch, 0.0] for i in range(self.num_ports)], device=dev)
        self.flip = torch.diag(torch.tensor([1.0, -1.0, -1.0], device=dev))  # port frame in the panel frame
        self.in_hand_pos = torch.tensor(plug_in_hand_pos(), device=dev).expand(E, 3)
        self.in_hand_quat = torch.tensor(PLUG_IN_HAND_ROT_XYZW, device=dev).expand(E, 4)
        self.gripper_idx = self.robot.find_bodies("gripper")[0][0]
        self.tracker = StageTracker(E, dev)
        self.target_port = torch.full((E,), 1, dtype=torch.long, device=dev)
        self.noise = torch.zeros(E, 3, device=dev)
        self.gen = torch.Generator().manual_seed(p.seed)
        self.now = torch.zeros(E, len(STAGES) - 1, dtype=torch.bool, device=dev)
        self.contact_force_w = torch.zeros(E, 3, device=dev)
        self.port_force_w = torch.zeros(E, self.num_ports, 3, device=dev)
        self.invalid = torch.zeros(E, dtype=torch.bool, device=dev)
        self._step = -1

    # --- geometry -----------------------------------------------------------------------------------------------

    def ports(self) -> PortGeometry:
        pose = _t(self.panel.data.root_link_pose_w)
        rot = matrix_from_quat(pose[:, 3:7])  # (E, 3, 3)
        pos = pose[:, None, :3] + torch.einsum("eij,pj->epi", rot, self.port_pos_p)
        mouth_rot = (rot @ self.flip).unsqueeze(1).expand(-1, self.num_ports, 3, 3)
        return PortGeometry(pos, mouth_rot, self.clearance, self.seat_depth)

    def plug_pose_w(self) -> torch.Tensor:
        return _t(self.robot.data.body_link_pose_w)[:, self.body_idx]

    def latch_angle(self) -> torch.Tensor:
        return _t(self.robot.data.joint_pos)[:, self.latch_jid]

    def target_mouth_w(self) -> tuple[torch.Tensor, torch.Tensor]:
        """True target-port mouth position (E, 3) and orientation (E, 4) in the world frame."""
        pose = _t(self.panel.data.root_link_pose_w)
        e = torch.arange(self.num_envs, device=self.device)
        pos, _ = combine_frame_transforms(pose[:, :3], pose[:, 3:7], self.port_pos_p[self.target_port])
        flip = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).expand(len(e), 4)  # 180 deg about X
        return pos, quat_mul(pose[:, 3:7], flip)

    def to_base(self, pos_w: torch.Tensor, quat_w: torch.Tensor | None = None):
        root = _t(self.robot.data.root_link_pose_w)
        if quat_w is None:
            quat_w = torch.tensor([0.0, 0.0, 0.0, 1.0], device=self.device).expand(pos_w.shape[0], 4)
        return subtract_frame_transforms(root[:, :3], root[:, 3:7], pos_w, quat_w)

    # --- per step -----------------------------------------------------------------------------------------------

    def contact_force(self) -> torch.Tensor:
        """Normal + friction force of the panel on plug body + latch, world frame (E, 3) (ENV_AUDIT section 10)."""
        total = torch.zeros(self.num_envs, 3, device=self.device)
        for s in self.sensors:
            for name in ("force_matrix_w", "friction_forces_w"):
                fm = getattr(s.data, name)
                if fm is not None:
                    total = total + _t(fm).reshape(self.num_envs, -1, 3).sum(1)
        return total

    def update(self, env) -> None:
        if env.common_step_counter == self._step:
            return
        self._step = env.common_step_counter
        ports = self.ports()
        plug = self.plug_pose_w()
        tip, axis = plug[:, :3], matrix_from_quat(plug[:, 3:7])[:, :, 0]
        f = self.contact_force()
        coords = port_coordinates(tip, ports)
        nearest = coords[..., 1].abs().argmin(dim=-1)
        self.contact_force_w = f
        self.port_force_w = torch.zeros(self.num_envs, self.num_ports, 3, device=self.device)
        self.port_force_w[torch.arange(self.num_envs, device=self.device), nearest] = f
        overload = torch.cat([g.overload_active for g in self.servo_groups], dim=-1)
        self.now = self.tracker.update(
            tip_pos=tip, plug_axis=axis, ports=ports, target_port=self.target_port, contact_force=self.port_force_w,
            latch_angle=self.latch_angle(), episode_time=env.episode_length_buf.float() * env.step_dt,
            overload_active=overload,
        )  # fmt: skip
        # plug welded to the gripper: any drift means the solver blew up
        grip = _t(self.robot.data.body_link_pose_w)[:, self.gripper_idx]
        expect, _ = combine_frame_transforms(grip[:, :3], grip[:, 3:7], self.in_hand_pos)
        drift = (expect - tip).norm(dim=-1)
        q = _t(self.robot.data.joint_pos)
        self.invalid = ~torch.isfinite(plug).all(-1) | ~torch.isfinite(q).all(-1) | ~(drift < 0.005)

    def reset(self, env, env_ids: torch.Tensor) -> None:
        p, n = self.p, len(env_ids)
        if n == 0:
            return
        if p.port == "random":
            port = torch.randint(0, self.num_ports, (n,), generator=self.gen)
        else:
            port = torch.full((n,), int(p.port), dtype=torch.long)
        self.target_port[env_ids] = port.to(self.device)
        self.noise[env_ids] = (torch.randn(n, 3, generator=self.gen) * p.pose_noise).to(self.device)

        def u(lim, k):
            return ((torch.rand(n, k, generator=self.gen) * 2 - 1) * lim).to(self.device)

        dpos = u(p.panel_pos_range, 3) if p.randomize else torch.zeros(n, 3, device=self.device)
        rot = u(p.panel_rot_range, 2) if p.randomize else torch.zeros(n, 2, device=self.device)
        zero = torch.zeros(n, device=self.device)
        quat = quat_from_euler_xyz(zero, rot[:, 0], rot[:, 1])  # pitch about Y, yaw about Z
        pos = env.scene.env_origins[env_ids] + torch.tensor(p.panel_pos, device=self.device) + dpos
        self.panel.write_root_pose_to_sim_index(root_pose=torch.cat([pos, quat], -1), env_ids=env_ids)
        off = torch.tensor(p.stand_offset, device=self.device).expand(n, 3)
        s_pos, _ = combine_frame_transforms(pos, quat, off)
        self.stand.write_root_pose_to_sim_index(root_pose=torch.cat([s_pos, quat], -1), env_ids=env_ids)
        self.tracker.reset(env_ids)
        self.invalid[env_ids] = False

    def task_strings(self) -> list[str]:
        return [task_string(int(i)) for i in self.target_port.tolist()]


def _t(x):
    return x.torch if hasattr(x, "torch") else x


def get_state(env) -> RJ45State:
    """The env's RJ45State, created on first use (needs the scene to exist, i.e. after sim.reset)."""
    env = env.unwrapped if hasattr(env, "unwrapped") else env
    st = getattr(env, "_rj45", None)
    if st is None:
        st = RJ45State(env, env.cfg.rj45_params)
        env._rj45 = st
        env.target_port = st.target_port  # same tensor, updated in place on reset (patchbench.policies.EnvAdapter)
        env.get_language_instructions = st.task_strings  # per-env language (target port differs per env)
    return st


def updated_state(env) -> RJ45State:
    st = get_state(env)
    st.update(env)
    return st


# ------------------------------------------------------------------------------------------- terms (Isaac Lab MDP)


def stage_at_least(env, stage: int) -> torch.Tensor:
    """Progress predicate: the episode's furthest stage is >= ``stage`` (1 reached ... 5 latched)."""
    return updated_state(env).tracker.furthest >= stage


def wrong_port(env) -> torch.Tensor:
    """Failure: the plug tip entered a non-target opening (lateral inside, depth > 2 mm)."""
    return updated_state(env).tracker.wrong_port.clone()


def plug_invalid(env) -> torch.Tensor:
    """Failure: non-finite plug/joint state, or the welded plug drifted > 5 mm from its grasp pose (solver blow-up)."""
    return updated_state(env).invalid.clone()


def episode_record(env, env_id: int) -> dict:
    """Arena episode-recorder term (runs before the reset): the finished episode's StageTracker summary, flat JSON."""
    st = get_state(env)
    out = {}
    for k, v in st.tracker.summary().items():
        if k == "success":  # Arena's core term already writes "success" (from the success termination)
            continue
        x = v[int(env_id)].item()
        out[k] = None if isinstance(x, float) and math.isnan(x) else x
    out["tier"] = st.p.tier
    out["target_port"] = int(st.target_port[int(env_id)])
    out["task"] = task_string(out["target_port"])
    return out


def reset_rj45(env, env_ids: torch.Tensor | None) -> None:
    """Reset event: target port, pose-estimate noise, panel + stand pose, stage tracker."""
    st = get_state(env)
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    st.reset(env, torch.as_tensor(env_ids, device=env.device, dtype=torch.long))


@configclass
class RJ45SceneCfg:
    """Contact sensors on the plug links, filtered against the panel's single rigid body (ENV_AUDIT section 10)."""

    plug_body_contact: ContactSensorCfg = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/Plug/body", filter_prim_paths_expr=["{ENV_REGEX_NS}/" + PANEL],
        **physics.CONTACT_SENSOR,
    )  # fmt: skip
    plug_latch_contact: ContactSensorCfg = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/Plug/latch", filter_prim_paths_expr=["{ENV_REGEX_NS}/" + PANEL],
        **physics.CONTACT_SENSOR,
    )  # fmt: skip


@configclass
class RJ45EventsCfg:
    reset_rj45: EventTermCfg = EventTermCfg(func=reset_rj45, mode="reset")
    panel_friction: EventTermCfg | None = None  # set by RJ45InsertTask when randomizing


class RJ45InsertTask(TaskBase):
    """Insert the held RJ45 plug into the target port of the 3-port panel until it latches."""

    def __init__(self, params: RJ45Params):
        port = params.port
        desc = task_string(int(port)) if port != "random" else "insert the cable into the left/middle/right port"
        super().__init__(episode_length_s=params.timeout_s, task_description=desc)
        self.params = params

    def get_scene_cfg(self):
        return RJ45SceneCfg()

    def get_termination_cfg(self) -> TaskTerminationCfg:
        success = [
            ProgressObjective(
                name=STAGES[k], predicate_sequence=[functools.partial(stage_at_least, stage=k)],
                parent_subtask_idx=k - 1, description=f"stage '{STAGES[k]}' reached (CLAUDE.md 6.1)",
            )
            for k in range(1, len(STAGES))
        ]  # fmt: skip
        failures = {"wrong_port": TerminationTermCfg(func=wrong_port), "plug_invalid": TerminationTermCfg(func=plug_invalid)}
        return TaskTerminationCfg(
            timeout_s=self.params.timeout_s, success=success, failures=failures, subtasks_are_sequential=True
        )

    def get_events_cfg(self):
        cfg = RJ45EventsCfg()
        if self.params.randomize:
            from isaaclab.envs import mdp

            # PhysX averages the two materials' friction (default combine mode), so with the plug fixed at
            # CONTACT_FRICTION the panel range [2a - 1, 2b - 1] * f gives the pair friction f * [a, b].
            a, b = self.params.friction_scale
            f = physics.CONTACT_FRICTION
            rng = ((2 * a - 1) * f, (2 * b - 1) * f)
            cfg.panel_friction = EventTermCfg(
                func=mdp.randomize_rigid_body_material, mode="reset",
                params={"static_friction_range": rng, "dynamic_friction_range": rng, "restitution_range": (0.0, 0.0),
                        "num_buckets": 64, "asset_cfg": SceneEntityCfg(PANEL), "make_consistent": True},
            )  # fmt: skip
        return cfg

    def get_mimic_env_cfg(self, arm_mode):
        raise NotImplementedError("RJ45InsertTask has no mimic config (scripted experts generate the data).")

    def get_metrics(self):
        from patchbench.metrics import (
            OverloadEventMetric,
            PeakAxialForceMetric,
            StageFunnelMetric,
            TimeToLatchMetric,
            WrongPortRateMetric,
        )

        return [StageFunnelMetric(), PeakAxialForceMetric(), TimeToLatchMetric(), OverloadEventMetric(), WrongPortRateMetric()]
