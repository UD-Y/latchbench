"""@register_policy wrappers for the scripted controllers (open_loop, telemetry_spiral, oracle_spiral, expert).

Run through Arena's policy runner with the dotted class path (registration happens on import), e.g.::

    flock -o ~/work/.isaac.lock python isaaclab_arena/evaluation/policy_runner.py \
        --policy_type latchbench.policies.ExpertPolicy --num_episodes 20 --num_envs 8 \
        --external_environment_class_path latchbench.environment:RJ45PanelEnv rj45_panel --tier loose

Every env-specific lookup (scene entity names, the env's ``RJ45State``, action joint order) is in
:class:`EnvAdapter` and nowhere else, so adapting to an env change is a one-place edit.

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

from latchbench.controllers import (
    ControllerObs,
    OpenLoopCfg,
    OpenLoopController,
    SpiralCfg,
    SpiralController,
    oracle_detect_cfg,
    telemetry_detect_cfg,
)
from latchbench.controllers.ik import quat_xyzw_to_matrix

SIM_JOINTS = ("Rotation", "Pitch", "Elbow", "Wrist_Pitch", "Wrist_Roll", "Jaw")


def _t(x) -> torch.Tensor:
    """Isaac Lab data fields are ProxyArrays at bb0c8e1 (``.torch`` view); plain tensors pass through."""
    return x.torch if hasattr(x, "torch") else x


# ------------------------------------------------------------------------------------------- observation mapping
@dataclass
class EnvKeys:
    robot: str = "robot"
    action_term: str = "arm_action"


class EnvAdapter:
    """Reads a :class:`ControllerObs` from the RJ45 Arena env and writes joint goals in the action's joint order.

    Geometry, target port, plug grasp and contact force come from the env's ``latchbench.task.RJ45State``
    (``get_state(env)``): the TRUE target mouth pose (the controller adds its own estimate noise), the plug's in-hand
    pose, and the plug + latch vs panel force (normal + friction). Telemetry from servosense's ``servo_load``.
    """

    def __init__(self, env: gym.Env, keys: EnvKeys | None = None):
        from latchbench.task import get_state  # Isaac imports; only after the app is up

        self.keys = keys or EnvKeys()
        self.env = env.unwrapped
        self.state = get_state(self.env)
        self.robot = self.env.scene[self.keys.robot]
        dev = self.env.device
        names = list(self.robot.joint_names)
        self.joint_ids = torch.tensor([names.index(n) for n in SIM_JOINTS], device=dev)
        term = self.env.action_manager.get_term(self.keys.action_term)
        act_names = list(getattr(term, "_joint_names", SIM_JOINTS))
        self.action_perm = torch.tensor([SIM_JOINTS.index(n) for n in act_names], device=dev)
        E = self.env.num_envs
        self.tool_p = self.state.in_hand_pos.expand(E, 3)
        self.tool_R = quat_xyzw_to_matrix(self.state.in_hand_quat).expand(E, 3, 3)

    def observe(self) -> ControllerObs:
        from isaaclab.managers import SceneEntityCfg
        from servosense.isaaclab.observations import servo_load

        st, e = self.state, self.env
        pos_b, quat_b = st.to_base(*st.target_mouth_w())
        root_q = _t(self.robot.data.root_link_pose_w)[:, 3:7]
        root_R = quat_xyzw_to_matrix(root_q)
        force_b = torch.einsum("eji,ej->ei", root_R, st.contact_force())
        return ControllerObs(
            joint_pos=_t(self.robot.data.joint_pos)[:, self.joint_ids],
            port_pos=pos_b,
            port_rot=quat_xyzw_to_matrix(quat_b),
            tool_p=self.tool_p,
            tool_R=self.tool_R,
            seat_depth=float(st.seat_depth),
            load=servo_load(e, SceneEntityCfg(self.keys.robot))[:, self.joint_ids],
            force=force_b,
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
    push_depth_mm: float = 6.0
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
