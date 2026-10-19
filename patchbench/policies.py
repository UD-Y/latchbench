"""@register_policy wrappers for the scripted controllers (open_loop, telemetry_spiral, oracle_spiral, expert).

Run through Arena's policy runner with the dotted class path (registration happens on import), e.g.::

    flock ~/work/.isaac.lock python isaaclab_arena/evaluation/policy_runner.py \
        --policy_type patchbench.policies.ExpertPolicy --num_episodes 20 --num_envs 8 \
        --external_environment_class_path patchbench.environment:RJ45PanelEnv rj45_panel --tier loose

Every env-specific lookup (scene entity names, where the target port index lives, the contact sensor, the plug's
in-hand pose) is in :class:`EnvAdapter` and nowhere else, so adapting to an env change is a one-place edit.

PolicyCfg fields become CLI flags (Arena adds ``--<field>`` for every field except ``device``/``num_envs``). ``--seed``
belongs to Arena's shared parser, so the noise RNG seed is ``policy_seed``. The env must not also define
``--pose_noise_mm`` (argparse would reject the duplicate flag): the port-pose estimate noise is the controller's.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import gymnasium as gym
import torch
from isaaclab_arena.assets.register import register_policy
from isaaclab_arena.policy.policy_base import PolicyBase, PolicyCfg

from patchbench.controllers import (
    ControllerObs,
    OpenLoopCfg,
    OpenLoopController,
    SpiralCfg,
    SpiralController,
    oracle_detect_cfg,
    telemetry_detect_cfg,
)
from patchbench.controllers.ik import quat_xyzw_to_matrix

SIM_JOINTS = ("Rotation", "Pitch", "Elbow", "Wrist_Pitch", "Wrist_Roll", "Jaw")


def _t(x) -> torch.Tensor:
    """Isaac Lab data fields are ProxyArrays at bb0c8e1 (``.torch`` view); plain tensors pass through."""
    return x.torch if hasattr(x, "torch") else x


# ------------------------------------------------------------------------------------------- observation mapping
@dataclass
class EnvKeys:
    robot: str = "robot"
    panel: str = "rj45_panel"
    contact_sensor: str = "plug_contact"
    """Plug (body + latch) vs panel contact sensor (normal + friction forces, patchbench.physics.CONTACT_SENSOR)."""


class EnvAdapter:
    """Reads a :class:`ControllerObs` from the RJ45 Arena env and writes joint goals in the action's joint order."""

    def __init__(self, env: gym.Env, keys: EnvKeys | None = None):
        self.keys = keys or EnvKeys()
        self.env = env.unwrapped
        self.robot = self.env.scene[self.keys.robot]
        names = list(self.robot.joint_names)
        self.joint_ids = torch.tensor([names.index(n) for n in SIM_JOINTS], device=self.env.device)
        term = self.env.action_manager.get_term("arm_action")
        act_names = list(getattr(term, "_joint_names", SIM_JOINTS))
        self.action_perm = torch.tensor([SIM_JOINTS.index(n) for n in act_names], device=self.env.device)
        from patchbench import objects  # Isaac imports; only after the app is up

        dims = objects.DIMS
        self.port_offsets = torch.tensor(objects.PANEL_CLASSES["loose"].port_positions(), device=self.env.device)
        self.seat_depth = dims.seat_depth
        # Port frames: rotated 180 deg about the panel's X (latch side down), build_assets.build_panel.
        self.port_rot_in_panel = torch.diag(torch.tensor([1.0, -1.0, -1.0], device=self.env.device))
        E = self.env.num_envs
        self.tool_p = torch.tensor(objects.plug_in_hand_pos(), device=self.env.device).expand(E, 3)
        self.tool_R = quat_xyzw_to_matrix(torch.tensor(objects.PLUG_IN_HAND_ROT_XYZW, device=self.env.device)).expand(E, 3, 3)

    def target_port(self) -> torch.Tensor:
        e = self.env
        for name in ("target_port", "target_ports"):
            if hasattr(e, name):
                return torch.as_tensor(getattr(e, name), device=e.device).long().expand(e.num_envs)
        task = getattr(getattr(e, "cfg", None), "isaaclab_arena_env", None)
        raise AttributeError(f"EnvAdapter: no target port index on the env ({type(e).__name__}, {task})")

    def observe(self) -> ControllerObs:
        e, k = self.env, self.keys
        d = self.robot.data
        root_p, root_R = _t(d.root_pos_w), quat_xyzw_to_matrix(_t(d.root_quat_w))
        RT = root_R.transpose(-1, -2)
        panel = e.scene[k.panel].data
        pan_p, pan_R = _t(panel.root_pos_w), quat_xyzw_to_matrix(_t(panel.root_quat_w))
        idx = self.target_port()
        port_w = pan_p + torch.einsum("eij,ej->ei", pan_R, self.port_offsets[idx])
        port_R_w = pan_R @ self.port_rot_in_panel
        load = force = None
        try:
            from so101_contact.isaaclab.observations import servo_load
            from isaaclab.managers import SceneEntityCfg

            load = servo_load(e, SceneEntityCfg(k.robot))[:, self.joint_ids]
        except Exception:  # noqa: BLE001  (telemetry unavailable: open-loop / oracle still work)
            load = None
        if k.contact_sensor in e.scene.keys():
            s = e.scene[k.contact_sensor].data
            f = _t(s.force_matrix_w) if s.force_matrix_w is not None else _t(s.net_forces_w).unsqueeze(2)
            if getattr(s, "friction_forces_w", None) is not None:
                f = f + _t(s.friction_forces_w)
            force = f.reshape(e.num_envs, -1, 3).sum(1)
        return ControllerObs(
            joint_pos=_t(d.joint_pos)[:, self.joint_ids],
            port_pos=torch.einsum("eij,ej->ei", RT, port_w - root_p),
            port_rot=RT @ port_R_w,
            tool_p=self.tool_p,
            tool_R=self.tool_R,
            seat_depth=self.seat_depth,
            load=load,
            force=force,
        )

    def to_action(self, goal_sim: torch.Tensor) -> torch.Tensor:
        return goal_sim[:, self.action_perm]


# ------------------------------------------------------------------------------------------------------ policies
@dataclass
class ScriptedPolicyCfg(PolicyCfg):
    pose_noise_mm: float = 1.0
    """Per-axis std of the port-position estimate error [mm] (CLAUDE.md 6.2)."""
    policy_seed: int = 0
    """Seed of the estimate-noise RNG (Arena owns ``--seed``)."""
    contact_threshold: float | None = None
    """Contact threshold in the sensor's units (telemetry: Present_Load counts; oracle: N). None = default."""
    latch_drop: float | None = None
    """Latch snap: fall below the peak by this much (sensor units). None = default."""
    backstop_threshold: float | None = None
    """Hard stop during insert (sensor units). None = default."""
    push_depth_mm: float = 4.0
    """Search push: commanded depth past the contact depth [mm]."""


class _ScriptedPolicy:
    """Shared get_action / reset for the controller-backed policies."""

    sensor: str | None = None

    def _make_controller(self, num_envs: int, device) -> OpenLoopController:
        c = self.config
        if self.sensor is None:
            return OpenLoopController(num_envs, device, OpenLoopCfg(pose_noise_mm=c.pose_noise_mm, seed=c.policy_seed))
        det = telemetry_detect_cfg() if self.sensor == "telemetry" else oracle_detect_cfg()
        if c.contact_threshold is not None:
            det.contact = c.contact_threshold
        if c.latch_drop is not None:
            det.latch_drop = c.latch_drop
        if c.backstop_threshold is not None:
            det.backstop = c.backstop_threshold
        cfg = SpiralCfg(pose_noise_mm=c.pose_noise_mm, seed=c.policy_seed, sensor=self.sensor, detect=det,
                        push_depth=c.push_depth_mm * 1e-3)
        return SpiralController(num_envs, device, cfg)

    def _lazy_init(self, env) -> None:
        if getattr(self, "adapter", None) is None:
            self.adapter = EnvAdapter(env)
            u = env.unwrapped
            self.controller = self._make_controller(u.num_envs, u.device)

    def get_action(self, env: gym.Env, observation) -> torch.Tensor:
        self._lazy_init(env)
        goal = self.controller.compute(self.adapter.observe())
        return self.adapter.to_action(goal)

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        if getattr(self, "controller", None) is not None:
            self.controller.reset(env_ids)


@dataclass
class OpenLoopPolicyCfg(ScriptedPolicyCfg):
    pass


@register_policy
class OpenLoopPolicy(_ScriptedPolicy, PolicyBase[OpenLoopPolicyCfg]):
    name = "open_loop"
    sensor = None

    def __init__(self, config: OpenLoopPolicyCfg):
        PolicyBase.__init__(self, config)
        self.adapter = None


@dataclass
class TelemetrySpiralPolicyCfg(ScriptedPolicyCfg):
    pass


@register_policy
class TelemetrySpiralPolicy(_ScriptedPolicy, PolicyBase[TelemetrySpiralPolicyCfg]):
    name = "telemetry_spiral"
    sensor = "telemetry"

    def __init__(self, config: TelemetrySpiralPolicyCfg):
        PolicyBase.__init__(self, config)
        self.adapter = None


@dataclass
class OracleSpiralPolicyCfg(ScriptedPolicyCfg):
    pass


@register_policy
class OracleSpiralPolicy(_ScriptedPolicy, PolicyBase[OracleSpiralPolicyCfg]):
    name = "oracle_spiral"
    sensor = "oracle"

    def __init__(self, config: OracleSpiralPolicyCfg):
        PolicyBase.__init__(self, config)
        self.adapter = None


@dataclass
class ExpertPolicyCfg(ScriptedPolicyCfg):
    pose_noise_mm: float = 0.0
    """Ground-truth port pose (CLAUDE.md section 7: expert for datagen, sigma = 0)."""
    expert_controller: Literal["open_loop", "oracle_spiral", "telemetry_spiral"] = "open_loop"
    """Which controller the expert runs (the best one on loose/medium with sigma = 0)."""


@register_policy
class ExpertPolicy(_ScriptedPolicy, PolicyBase[ExpertPolicyCfg]):
    name = "expert"

    def __init__(self, config: ExpertPolicyCfg):
        PolicyBase.__init__(self, config)
        self.adapter = None
        self.sensor = {"open_loop": None, "oracle_spiral": "oracle", "telemetry_spiral": "telemetry"}[
            config.expert_controller
        ]


__all__ = ["EnvAdapter", "EnvKeys", "ExpertPolicy", "OpenLoopPolicy", "OracleSpiralPolicy", "TelemetrySpiralPolicy"]
