"""RJ45PanelEnv(ExampleEnvironmentBase), name ``rj45_panel``; loaded via ``--external_environment_class_path``.

    cd ~/work/IsaacLab-Arena && source .venv/bin/activate && export OMNI_KIT_ACCEPT_EULA=YES ACCEPT_EULA=Y
    flock -o ~/work/.isaac.lock python isaaclab_arena/evaluation/policy_runner.py --headless --policy_type zero_action \
        --num_episodes 2 --external_environment_class_path patchbench.environment:RJ45PanelEnv \
        rj45_panel --tier loose --port random --port_est_noise_mm 1

Scene (env frame, metres; the robot base is at the env origin and faces +X):
  * SO-101 follower (isaaclab_so101 ``so101_abs_joint``: action = 6 absolute joint goals [rad], sim joint order
    Rotation, Pitch, Elbow, Wrist_Pitch, Wrist_Roll, Jaw), all six joints on so101-contact's ``STS3215Actuator``
    (LeRobot P=16/I=0/D=32, servo parameters randomized per reset unless ``--no_randomize``).
  * RJ45 plug welded to the ``gripper`` link (patchbench.objects): plug +X = gripper +Y, latch side down when the jaws
    point down. The plug's links ``body``/``latch`` and joint ``latch_joint`` are part of the robot articulation.
  * 3-port panel of the chosen tier (kinematic) on a stand; panel frame at ``RJ45Params.panel_pos`` = (0.22, 0, 0.05),
    insertion along env +X, port_0 (left) at +Y. Reachability: closed-form 5-DOF IK (isaaclab_so101.ee_pose) for the
    plug tip anywhere from 15 mm in front of each mouth to 3 mm past the seat, under the full +-10 mm / +-3 deg panel
    randomization, keeps every joint >= 0.35 rad (20 deg) inside its limits, gripper pointing straight down.
  * Cameras (``--enable_cameras``): ``camera_ego`` (wrist) and ``external_camera`` (front, on the robot base).

Physics from patchbench/physics.py (dt 1/480, decimation 16 -> 30 Hz, Factory-style PhysX scene, 32 position iterations
on the SO-101 articulation which now includes the plug).
"""

from __future__ import annotations

import argparse
import math

try:
    from isaaclab_arena_environments.example_environment_base import ExampleEnvironmentBase
except ImportError:  # CPU-only tests import start_joint_pose
    ExampleEnvironmentBase = object

from patchbench import physics

PRE_INSERT_GAP = 0.030
"""m: at reset the plug tip is this far in front of the nominal middle port's mouth, on its axis."""

FRONT_CAMERA_EYE, FRONT_CAMERA_TARGET = (0.12, -0.26, 0.20), (0.20, 0.0, 0.05)
"""External camera look-at, robot placement frame (arm faces +X): a 3/4 view of the plug and panel from the right."""


def _t(x):
    return x.torch if hasattr(x, "torch") else x


# ----------------------------------------------------------------------------------------- observation terms


def obs_plug_tip_pos(env):
    """Plug tip (frame origin) position in the base frame (E, 3); known on hardware from FK + the grasp."""
    from patchbench.task import get_state

    st = get_state(env)
    plug = st.plug_pose_w()
    return st.to_base(plug[:, :3], plug[:, 3:7])[0]


def obs_plug_tip_quat(env):
    """Plug frame orientation (x, y, z, w) in the base frame (E, 4); plug +X = insertion, +Z = latch side."""
    from patchbench.task import get_state

    st = get_state(env)
    plug = st.plug_pose_w()
    return st.to_base(plug[:, :3], plug[:, 3:7])[1]


def obs_port_pos_est(env):
    """Target-port mouth position estimate in the base frame (E, 3): truth + per-episode N(0, pose_noise) noise."""
    from patchbench.task import get_state

    st = get_state(env)
    pos, quat = st.target_mouth_w()
    return st.to_base(pos + st.noise, quat)[0]


def obs_port_quat_est(env):
    """Target-port mouth frame (x, y, z, w) in the base frame (E, 4): +X insertion, +Z latch side (down)."""
    from patchbench.task import get_state

    st = get_state(env)
    pos, quat = st.target_mouth_w()
    return st.to_base(pos, quat)[1]


def obs_target_port(env):
    """Target port index (E, 1) as float: 0 left (+Y), 1 middle, 2 right."""
    from patchbench.task import get_state

    return get_state(env).target_port.float().unsqueeze(-1)


def obs_port_pos_true(env):
    """Privileged: true target-port mouth position in the base frame (E, 3)."""
    from patchbench.task import get_state

    st = get_state(env)
    pos, quat = st.target_mouth_w()
    return st.to_base(pos, quat)[0]


def obs_contact_force(env):
    """Privileged (oracle): plug + latch vs panel force, normal + friction, in the base frame (E, 3) [N]."""
    from isaaclab.utils.math import quat_apply_inverse

    from patchbench.task import get_state

    st = get_state(env)
    root = _t(st.robot.data.root_link_pose_w)
    return quat_apply_inverse(root[:, 3:7], st.contact_force())


def obs_latch_angle(env):
    """Privileged: latch joint angle from rest (E, 1) [rad], negative = pressed down."""
    from patchbench.task import get_state

    return get_state(env).latch_angle().unsqueeze(-1)


def obs_furthest_stage(env):
    """Privileged: furthest stage this episode (E, 1): 0 none, 1 reached ... 5 latched."""
    from patchbench.task import get_state

    return get_state(env).tracker.furthest.float().unsqueeze(-1)


def make_observations_cfg():
    import isaaclab.envs.mdp as mdp
    from isaaclab.managers import ObservationGroupCfg as ObsGroup
    from isaaclab.managers import ObservationTermCfg as ObsTerm
    from isaaclab.managers import SceneEntityCfg
    from isaaclab.utils.configclass import configclass
    from isaaclab_so101.constants import SIM_JOINT_NAMES
    from isaaclab_so101.embodiments.so101 import ee_pos_in_base, ee_quat_in_base
    from so101_contact.isaaclab.observations import servo_current as _servo_current
    from so101_contact.isaaclab.observations import servo_load as _servo_load

    joints = SceneEntityCfg("robot", joint_names=list(SIM_JOINT_NAMES), preserve_order=True)

    @configclass
    class RJ45ObservationsCfg:
        @configclass
        class PolicyCfg(ObsGroup):
            """What a deployable controller may use (all available on a real SO-101 + a pose estimate)."""

            actions = ObsTerm(func=mdp.last_action)
            joint_pos = ObsTerm(func=mdp.joint_pos, params={"asset_cfg": joints})
            joint_vel = ObsTerm(func=mdp.joint_vel, params={"asset_cfg": joints})
            servo_load = ObsTerm(func=_servo_load, params={"asset_cfg": joints})
            servo_current = ObsTerm(func=_servo_current, params={"asset_cfg": joints})
            eef_pos = ObsTerm(func=ee_pos_in_base)
            eef_quat = ObsTerm(func=ee_quat_in_base)
            plug_tip_pos = ObsTerm(func=obs_plug_tip_pos)
            plug_tip_quat = ObsTerm(func=obs_plug_tip_quat)
            port_pos_est = ObsTerm(func=obs_port_pos_est)
            port_quat_est = ObsTerm(func=obs_port_quat_est)
            target_port = ObsTerm(func=obs_target_port)

            def __post_init__(self):
                self.enable_corruption = False
                self.concatenate_terms = False

        @configclass
        class PrivilegedCfg(ObsGroup):
            """Simulation-only signals: the oracle controller's force sensor and ground truth for analysis."""

            contact_force = ObsTerm(func=obs_contact_force)
            port_pos = ObsTerm(func=obs_port_pos_true)
            latch_angle = ObsTerm(func=obs_latch_angle)
            furthest_stage = ObsTerm(func=obs_furthest_stage)

            def __post_init__(self):
                self.enable_corruption = False
                self.concatenate_terms = False

        policy: PolicyCfg = PolicyCfg()
        privileged: PrivilegedCfg = PrivilegedCfg()

    return RJ45ObservationsCfg()


# ----------------------------------------------------------------------------------------------- environment


def start_joint_pose(panel_pos, gap: float = PRE_INSERT_GAP) -> dict[str, float]:
    """Joint angles holding the plug ``gap`` in front of the middle port's mouth, axis along env +X, latch down."""
    from isaaclab_so101.constants import HOME_JOINT_POS, PAN_AXIS_XY, SIM_JOINT_NAMES
    from isaaclab_so101.ee_pose import natural_ik

    from patchbench.objects import DIMS, PLUG_GRASP_BEHIND, wrist_roll_for

    tcp_x = panel_pos[0] - gap - DIMS.plug_length - PLUG_GRASP_BEHIND
    xb, yb, zb = panel_pos[1], -tcp_x, panel_pos[2]  # env -> base: base X = env Y, base Y = env -X
    psi = math.atan2(yb - PAN_AXIS_XY[1], xb - PAN_AXIS_XY[0])
    q = natural_ik(xb, yb, zb, 0.0, wrist_roll_for(psi))  # tilt 0: jaws down; plug axis along base -Y (env +X)
    assert q is not None, f"start pose unreachable for panel at {panel_pos}"
    out = dict(zip(SIM_JOINT_NAMES[:5], q))
    out["Jaw"] = HOME_JOINT_POS["Jaw"]
    return out


class RJ45PanelEnv(ExampleEnvironmentBase):
    name = "rj45_panel"

    def get_env(self, args_cli: argparse.Namespace):
        import isaaclab_so101
        from isaaclab.actuators import ImplicitActuatorCfg
        from isaaclab.assets import ArticulationCfg
        from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
        from isaaclab_arena.scene.scene import Scene
        from isaaclab_arena.utils.pose import Pose
        from isaaclab_so101.assets import SO101_CFG
        from so101_contact.isaaclab.actuator import so101_sts3215_cfg

        from patchbench.objects import PANEL_CLASSES, STAND_DEPTH, RJ45PanelStand, RJ45Plug, _RJ45Panel, stand_size
        from isaaclab_arena.recording.episode_recorder_manager import EpisodeRecorderTermCfg

        from patchbench.task import RJ45InsertTask, RJ45Params, episode_record

        isaaclab_so101.register()
        p = RJ45Params(
            tier=args_cli.tier, port=args_cli.port, pose_noise=args_cli.port_est_noise_mm * 1e-3,
            randomize=not args_cli.no_randomize, seed=args_cli.seed or 0,
        )  # fmt: skip

        panel = PANEL_CLASSES[p.tier]()
        panel.set_initial_pose(Pose(position_xyz=p.panel_pos), create_reset_event=False)
        z_bottom_p = _RJ45Panel.extent_z()[0]
        y0, y1 = _RJ45Panel.extent_y()
        size = stand_size(p.panel_pos[2] + z_bottom_p)
        p.stand_offset = (STAND_DEPTH / 2, (y0 + y1) / 2, z_bottom_p - size[2] / 2)
        stand = RJ45PanelStand(size=size)
        stand_pos = tuple(a + b for a, b in zip(p.panel_pos, p.stand_offset))
        stand.set_initial_pose(Pose(position_xyz=stand_pos), create_reset_event=False)
        ground = self.asset_registry.get_asset_by_name("ground_plane")()
        light = self.asset_registry.get_asset_by_name("light")()

        q0 = start_joint_pose(p.panel_pos)
        embodiment = self.asset_registry.get_asset_by_name("so101_abs_joint")(enable_cameras=args_cli.enable_cameras)
        servo = so101_sts3215_cfg(
            args_cli.servo_friction_mode, physics_dt=physics.PHYSICS_DT, randomize=p.randomize, seed=p.seed
        )
        latch = ImplicitActuatorCfg(joint_names_expr=["latch_joint"], stiffness=None, damping=None)  # USD spring
        robot = SO101_CFG.replace(
            prim_path="{ENV_REGEX_NS}/Robot",
            actuators={**servo, "latch": latch},
            init_state=ArticulationCfg.InitialStateCfg(
                pos=SO101_CFG.init_state.pos, rot=SO101_CFG.init_state.rot, joint_pos={**q0, "latch_joint": 0.0}
            ),
        )
        robot = RJ45Plug().attach_to(robot)
        robot.spawn.activate_contact_sensors = True
        embodiment.scene_config.robot = robot
        embodiment.observation_config = make_observations_cfg()
        if args_cli.enable_cameras:
            for cam in (embodiment.camera_config.camera_ego, embodiment.camera_config.external_camera):
                cam.height, cam.width = args_cli.camera_height, args_cli.camera_width
            embodiment.set_external_camera_view(FRONT_CAMERA_EYE, FRONT_CAMERA_TARGET)

        def env_cfg_callback(env_cfg):
            env_cfg.sim.dt = physics.PHYSICS_DT
            env_cfg.decimation = physics.DECIMATION
            env_cfg.sim.render_interval = physics.DECIMATION
            for k, v in physics.PHYSX_SCENE.items():
                setattr(env_cfg.sim.physics, k, v)
            env_cfg.rj45_params = p
            return env_cfg

        return IsaacLabArenaEnvironment(
            name=self.name,
            embodiment=embodiment,
            scene=Scene(assets=[ground, light, panel, stand]),
            task=RJ45InsertTask(p),
            env_cfg_callback=env_cfg_callback,
            episode_recorder_terms={"patchbench": EpisodeRecorderTermCfg(func=episode_record)},
        )

    @staticmethod
    def add_cli_args(parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--tier", choices=["loose", "medium", "spec"], default="medium")
        parser.add_argument("--port", choices=["0", "1", "2", "random"], default="random")
        # Not --pose_noise_mm: that is the scripted policies' PolicyCfg flag, and a subcommand flag of the same name
        # would silently overwrite it in the shared argparse namespace.
        parser.add_argument(
            "--port_est_noise_mm", type=float, default=1.0, help="sigma of obs port_pos_est (per axis, per episode)"
        )
        parser.add_argument("--no_randomize", action="store_true", help="nominal panel pose, friction and servo params")
        parser.add_argument(
            "--servo_friction_mode", choices=["physx_joint", "explicit"], default=physics.SERVO_FRICTION_MODE,
            help="STS3215 gearbox friction: PhysX joint friction (default, physics.py) or explicit torque (faster)",
        )
        parser.add_argument("--camera_height", type=int, default=480)
        parser.add_argument("--camera_width", type=int, default=640)
