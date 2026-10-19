# Environment audit (gate G0)

Written 19 Oct 2026 on the laptop. Every fact below was read from the installed source or measured; nothing is assumed.

## 1. Stack

| Component | Version / commit | How it is installed |
|---|---|---|
| IsaacLab-Arena | `aa36f19` (2026-09-23) | `~/work/IsaacLab-Arena`, `uv sync` (default groups `isaaclab-from-source`, `openpi`) |
| Isaac Lab | submodule `bb0c8e1` (2026-08-25); `isaaclab` 16.4.0, `isaaclab_physx` 5.1.0, `isaaclab_assets` 0.6.4 | editable from `submodules/IsaacLab` |
| Isaac Sim | 6.1.0.0 (pip wheels) | from Arena's `uv.lock` |
| isaaclab-so101 | 0.1.0, `art-e-fact/isaaclab-so101` @ `12a3ecc` (2026-10-08) | `~/work/isaaclab-so101`, `uv pip install -e` (not on PyPI) |
| Python / torch / CUDA | 3.12.14 / 2.11.0+cu128 / 12.8 | Arena venv |
| warp-lang / numpy | 1.16.0 / 2.3.1 | Arena overrides |
| Driver | 580.173.02 | |

Install recipe (what worked):
```bash
cd ~/work/IsaacLab-Arena && GIT_LFS_SKIP_SMUDGE=1 git checkout aa36f19
git submodule init -- submodules/IsaacLab
git config submodule.submodules/IsaacLab.url https://github.com/isaac-sim/IsaacLab.git   # .gitmodules uses SSH
GIT_LFS_SKIP_SMUDGE=1 git submodule update --depth 1 -- submodules/IsaacLab
uv sync                                   # ~10 GB download; venv 33 GB but hardlinked from the uv cache
uv pip install -e ~/work/isaaclab-so101   # repeat after every uv sync (sync removes it)
```
The `Isaac-GR00T` submodule is not fetched: it is only used by the `gr00t-client` group, and we use GR00T through LeRobot.

## 2. Machines

| | Laptop | Brev |
|---|---|---|
| GPU | RTX 4060 Laptop, **8 GB**, compute 8.9 | 1× L40S 48 GB, compute 8.9 |
| RAM | 15.3 GiB + 8 GiB swap | provider-dependent (check ≥ 64 GB) |
| CPU | Ryzen 9 7945HX, 32 threads | 12 vCPU (massedcompute, $1.06/hr) |
| Disk | 55 GB free after install (uv cache 41 GB is reusable / prunable) | 625 GB |

Isaac Lab's stated minimum is 32 GB RAM and 16 GB VRAM (`submodules/IsaacLab/docs/source/setup/installation/index.rst:87`). The laptop has half of each.

## 2a. Smoke tests (gate G0) — both pass

Measured on the laptop, one Isaac process at a time, idle baseline 1.7 GB RAM used / 0.5 GB VRAM:

| Test | Result | Wall | Peak RSS | Peak VRAM |
|---|---|---|---|---|
| Arena `policy_runner.py --policy_type zero_action --num_steps 20 cube_goal_pose` (Franka, 1 env, no cameras) | exit 0, 20/20 steps at 18.5 steps/s | 19 s | 4.4 GB | 2.1 GB |
| `isaaclab-so101/examples/arena/smoke_test.py` (3 embodiments, placement, **2 cameras 640×480**, IK, LeRobot recorder; 1 env) | exit 0, **all 10 checks OK** | 82 s | 9.8 GB | **6.0 GB** |

- First-ever launch adds ~3 min of RTX shader compilation; later launches use the cache.
- Physics dt 1/120 s, decimation 8 in `cube_goal_pose` (control at 15 Hz). Our env sets its own (30 Hz control).
- With cameras, **VRAM is the binding limit**: one env with two 640×480 cameras uses 6.0 of 8 GB. Hence the laptop rule
  "cameras: `num_envs ≤ 2` at ≤ 240×320, pipeline checks only".
- Without cameras one env costs ~2 GB VRAM / ~4.4 GB RSS; the `num_envs ≤ 32` state-only limit should be re-measured with the
  RJ45 scene in P3.
- Background load matters: VS Code's C/C++ extension used ~6 GB indexing the 33 GB venv and pushed the camera run to 15.0/15.3 GB
  RAM. Excluded in `~/work/.vscode/settings.json` (`C_Cpp.intelliSenseEngine: disabled`, venv/submodules excluded from watch/search).
- Two Isaac processes at once collided on `/tmp/isaaclab/logs/dataset_<timestamp>_rank0.hdf5`
  (`BlockingIOError: unable to lock file`). One Isaac process at a time, on both machines.

## 3. Isaac Lab actuator API (pinned `bb0c8e1`)

`source/isaaclab/isaaclab/actuators/actuator_base.py`:
```python
ActuatorBase.__init__(self, cfg, joint_names, joint_ids, num_envs, device,
                      actuator_effort_limit=None, actuator_velocity_limit=None,
                      effort_limit=None, velocity_limit=None)   # last two deprecated aliases
ActuatorBase.reset(self, env_ids: Sequence[int])
ActuatorBase.compute(self, control_action: ArticulationActions, joint_pos, joint_vel) -> ArticulationActions
```
`ActuatorBaseCfg` fields (`actuator_base_cfg.py`): `class_type`, `joint_names_expr`, `actuator_effort_limit`, `actuator_velocity_limit`,
`joint_effort_limit`, `joint_velocity_limit`, `effort_limit_sim` / `velocity_limit_sim` (deprecated), `stiffness`, `damping`, `armature`,
`friction`, `dynamic_friction`, `viscous_friction`, `effort_limit` / `velocity_limit` (deprecated).
`ArticulationActions` (`isaaclab/utils/types.py`): `joint_positions`, `joint_velocities`, `joint_efforts`, `joint_indices`.

**Matches CLAUDE.md section 2.** No API drift.

## 4. SO-101 and BAM

- `isaaclab_assets.robots.so101` exists in the pinned submodule: `SO101_CFG`, `SO101_HIGH_PD_CFG`.
- **BAM is not installed.** `BamActuatorCfg` / `BamMotorCfg` are not in `bb0c8e1` and not on Isaac Lab `main`. They are on the **`develop`** branch
  (`0ec7c29`, 2026-10-19): `source/isaaclab/isaaclab/actuators/actuator_bam_cfg.py`,
  `source/isaaclab_newton/isaaclab_newton/actuators/bam.py`, `bam_kernels.py`. `so101-contact` mirrors names and equations from there.

## 5. isaaclab-so101 embodiment

- Register after `SimulationApp` starts: `isaaclab_so101.register()` → embodiments `so101_abs_joint` (absolute joint positions),
  `so101_rel_joint`, `so101_ik` (relative SE(3) diff-IK + binary Jaw), `so101_abs_ik`, plus devices `so101_leader`, `so101_gamepad`.
- **Sim joint names differ from LeRobot's**: sim `Rotation, Pitch, Elbow, Wrist_Pitch, Wrist_Roll, Jaw`; LeRobot
  `shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper`. `isaaclab_so101.mapping` (`LEROBOT_JOINT_KEYS`,
  `motor_norm_to_sim_radians`, `sim_radians_to_motor_norm`) does the conversion. Datasets use LeRobot names (CLAUDE.md 6.3).
- Joint limits (deg): Rotation ±110, Pitch ±100, Elbow −100/+90, Wrist_Pitch ±95, Wrist_Roll ±160, Jaw −10/+100.
- `TCP_OFFSET = (-0.007, 0, -0.102)` m in the `gripper` link frame (−Z along the jaws). The plug's fixed attach parent is the `gripper` link.
- Cameras (`enable_cameras=True`): `camera_ego` (wrist, obs `camera_ego_rgb`) and `external_camera` (attached to the base, obs
  `external_camera_rgb`), both 640×480 RGB. Map to `observation.images.wrist` / `observation.images.front`.
- Default actuators (`isaaclab_so101/assets.py`, the embodiment we use): one `ImplicitActuatorCfg` per joint, **`effort_limit_sim=30` N·m on
  every joint** (≈ 15× the STS3215's 1.912 N·m stall torque), stiffness Rotation 55 / Pitch 30 / Elbow 25 / Wrist_Pitch 12 /
  Wrist_Roll 7 / Jaw 4, damping 0.7 / 0.8 / 0.7 / 0.5 / 0.5 / 0.3. The high-PD (IK) variant uses 400 / 80. Isaac Lab's own
  `SO101_CFG` instead uses `joint_effort_limit=10`, stiffness 17.8, damping 0.60 (LeIsaac values). Neither model has servo
  saturation, back-EMF or telemetry; `STS3215Actuator` replaces them on all six joints.
- USD `isaaclab_so101/embodiments/data/SO-ARM101-USD.usd`: articulation root `/so101_new_calib/base`; rigid links `base, shoulder,
  upper_arm, lower_arm, wrist, gripper, jaw`; revolute joints `Rotation … Jaw`. **Plug attach parent: `gripper`** (the fixed jaw; `jaw` moves).
- Arm faces +X; `init_state` yaws the base 90°. Newton cannot load this USD yet (README).

## 6. Arena APIs (aa36f19)

| Need | Where |
|---|---|
| `TaskBase` (abstract: `get_scene_cfg`, `get_termination_cfg`, `get_events_cfg`, `get_mimic_env_cfg`, `get_metrics`) | `isaaclab_arena/tasks/task_base.py` |
| `TaskTerminationCfg(timeout_s, success: list[ProgressObjective], failures: dict[str, TerminationTermCfg], subtasks_are_sequential, ...)` | `isaaclab_arena/tasks/task_termination_cfg.py` |
| `ProgressObjective(name, predicate_sequence=..., score, logical, ...)` | `isaaclab_arena/progress_tracking/progress_objective.py` |
| `MetricBase` (abstract: `get_recorder_term_cfg`, `get_metric_term_cfg`) | `isaaclab_arena/metrics/metric_base.py`; example: `metrics/object_moved.py` |
| `ExampleEnvironmentBase` (`get_env`, `build(cfg)`, `add_cli_args`) | `isaaclab_arena_environments/example_environment_base.py` |
| `@register_asset`, `@register_policy` | `isaaclab_arena/assets/register.py` |
| `PolicyBase(config)` (`get_action(env, obs)`, `reset(env_ids)`, `close`, `set_task_description`), `PolicyCfg` | `isaaclab_arena/policy/policy_base.py` |
| Shared spatial predicates `depth_in_range`, `lateral_in_proximity`, `velocity_below_threshold` | `isaaclab_arena/tasks/predicates/spatial.py` |
| Custom object asset example | `isaaclab_arena_environments/isaac_cap/usbc_insertion/assets.py` (also `isaaclab_arena/assets/object_library.py`) |

**Contradiction with the plan:** there is no `CompletionCriteria` class. Success is `TaskTerminationCfg.success`, a list of
`ProgressObjective`s; with `subtasks_are_sequential=True` the stages reach → contact → align → seat → latch map onto one
`predicate_sequence`. MONDAY.md P5 is updated accordingly.

Prior art: `isaaclab_arena_environments/isaac_cap/usbc_insertion/` is a USB-C insertion task, but **Newton/MJWarp only**
(contact via solref/solimp), so its physics settings do not transfer to our PhysX plan. Its task/predicate structure does.

## 7. LeRobot

Separate venv, `lerobot==0.6.1` (latest on PyPI). 0.6.1 has GR00T N1.7 (`nvidia/GR00T-N1.7-3B`, backbone
`nvidia/Cosmos-Reason2-2B`) and the `use_relative_actions`, `relative_exclude_joints`, `use_bf16` flags used in CLAUDE.md section 10.
Defaults: `tune_llm=False`, `tune_visual=False`, `tune_projector=True`, `tune_diffusion_model=True`; images resized to 256×256.
`~/r2s/lerobot` (0.5.2) is too old.

## 8. Contradictions with CLAUDE.md / MONDAY.md (all now corrected in those files)

1. Laptop GPU is an RTX 4060 8 GB, not an RTX 4090 16 GB. Work split revised (CLAUDE.md section 2).
2. `isaaclab-so101` is not on PyPI; install from git.
3. Arena's submodules use SSH URLs; use HTTPS (required on Brev).
4. BAM is on Isaac Lab `develop`, not `main` and not the pinned submodule.
5. No `CompletionCriteria`; use `TaskTerminationCfg` + `ProgressObjective`.
6. Isaac Sim MCP is not configured on the laptop; visual checks are headless camera captures.
7. LeRobot must be 0.6.1 for GR00T N1.7.

## 9. Explicit actuators on PhysX (P4, 19 Oct)

Read from `isaaclab/actuators/actuator_collection.py` and `isaaclab/actuators/actuator_control.py` at `bb0c8e1`:

- `ActuatorCollection._build_groups` constructs explicit groups with
  `cfg, joint_names, joint_ids, num_envs, device, stiffness, damping, actuator_velocity_limit, actuator_effort_limit`
  (so the actuator `__init__` must accept `stiffness`/`damping`).
- For explicit groups it writes the cfg's `armature`, `friction`, `dynamic_friction`, `viscous_friction`,
  `joint_effort_limit` and `joint_velocity_limit` to PhysX, and **zero** drive stiffness/damping.
- PhysX joint `friction` is a static friction **effort** [N·m] on Isaac Sim ≥ 5 (`ArticulationData.joint_friction_coeff`).
  `write_joint_friction_coefficient_to_sim_index` goes through a CPU copy; writing it every physics step from the
  actuator cost ~3 % wall time with 1 env (not yet measured at 16+ envs).
- Commands: `robot.actuators.target_command.set_position_index(value=..., joint_ids=...)`, then
  `write_data_to_sim()`; `compute()` gets `control_action.joint_positions` as the goal.
- `data.joint_acc` is finite-differenced from `joint_vel`; `data.mass_matrix` (6×6 for the fixed-base SO-101) and
  `data.gravity_compensation_forces` come from PhysX. `M·q̈ − τ_applied` reproduces the gravity torque within 5 %
  in a static hold (shoulder 0.191 vs 0.183 N·m, elbow 0.328 vs 0.337 N·m), so it is used as BAM's external torque.
- SO-101 USD: the default prim `/so101_new_calib` is referenced at the robot prim path, so links are
  `<prim_path>/gripper`, `<prim_path>/jaw`. Contact-sensor filters must name one link each (a `.*` filter matched
  10 prims and failed).
- `simulation_app.close()` in a `finally` hides exceptions raised before it: print the traceback first.

Hold test (`scripts/hold_test.py`, 1 env, no cameras, 30 s wall, 4.2 GB RSS): see `results/hold_test*/summary.json`.

Friction-mode cost at 16 envs (`scripts/bench_friction_mode.py`, 16 SO-101s on STS3215 actuators, random goals,
dt 1/240, laptop RTX 4060, no cameras): `explicit` 251 physics steps/s (4022 env-steps/s, 6.5 GB RSS),
`physx_joint` 222 physics steps/s (3547 env-steps/s, 5.0 GB RSS), i.e. −12 %. `physx_joint` is the env default
(`patchbench/physics.py`). Re-measure at 64–256 envs on Brev, where the per-step CPU copy grows with num_envs.

## 10. Contact sensing for the plug (P3, 19 Oct)

- `ContactSensor.data.force_matrix_w` is the filtered **normal** force only. Along the RJ45 latch most of the axial
  insertion load is friction: with normal forces alone the axial force read 0.1 N while the wrist spring carried
  1.1 N. Set `track_friction_forces=True` and add `data.friction_forces_w`.
- `friction_forces_w` is silently ~0 with the default `max_contact_data_count_per_prim=4` (the plug body has more
  contacts than that); with 64 the normal + friction axial force matches the wrist spring force within ~5 %.
- Filter on the panel's single kinematic rigid body (`/World/envs/env_.*/Panel`): one filter entry per env.

## 11. Arena env `rj45_panel` (P5, 19 Oct)

Read from Arena `aa36f19` / Isaac Lab `bb0c8e1` while building `patchbench/environment.py`:

- External env factories still use the argparse path: `ExampleEnvironmentBase.get_env(args_cli)` +
  `add_cli_args(subparser)`; flags go after the `rj45_panel` subcommand. A subcommand flag with the same name as a
  main-parser flag (e.g. a PolicyCfg field) silently overwrites it in the shared namespace, so the env's estimate-noise
  flag is `--port_est_noise_mm`, not `--pose_noise_mm` (the scripted policies' flag). Headless is the default
  (there is no `--headless`; `--viz kit` opens a GUI).
- `IsaacLabArenaEnvironment(env_cfg_callback=...)` is where dt / decimation / PhysX scene settings are set
  (`physics.py`); the callback runs after Arena has put its own `PhysxCfg()` on `env_cfg.sim.physics`.
- Scene, observations, events, metrics are merged from Scene + embodiment + task (`arena_env_builder.compose_manager_cfg`).
  The SO-101 embodiment's `observation_config` is replaced by the env's own (`policy` + `privileged` groups); Arena
  adds `camera_obs` when `--enable_cameras`.
- Success is `TaskSuccessTerm` over `ProgressObjective`s. With `subtasks_are_sequential=True` the objectives need
  `parent_subtask_idx` (normally set by `CompositeTaskBase`; set by hand here, one per stage), and the tracker advances
  at most one subtask per step, so success fires up to 4 control steps after the latch predicate holds.
- `record_post_step` runs before resets and `record_pre_reset` before the reset events, so per-episode metrics are
  recorded in `record_pre_reset` (as `metrics/success_rate.py` does) from the stage tracker.
- Startup events run after `load_managers`; observation terms are first called inside `load_managers` (shape
  inference), so shared per-env state is created lazily by whichever term asks first (`patchbench.task.get_state`).
- Spawners: `@clone` calls the spawn function once for env_0 and copies the prim spec to the other envs, then applies
  `activate_contact_sensors` to every rigid body under the prim. The plug is added inside the robot's spawn function
  (`objects.spawn_robot_with_plug`): `plug.usd` referenced at `<Robot>/Plug`, `UsdPhysics.FixedJoint`
  `<Robot>/plug_grasp` from `gripper` to `Plug/body`. PhysX makes `body` and `latch` links of the SO-101 articulation
  (robot joint `latch_joint`, driven by an `ImplicitActuatorCfg(stiffness=None, damping=None)` that keeps the USD spring).
- `pxr` (core) imports without Kit in the Arena venv, `PhysxSchema` does not; the CPU venv has no `pxr` at all.

Layout (reachability with `isaaclab_so101.ee_pose.natural_ik`, `tests/test_env_layout.py`): a 5-DOF arm can hold a
horizontal plug at any yaw only in the gimbal case, jaws pointing straight down (wrist roll then absorbs the pan
azimuth). So the plug is held across the jaws (plug +X = gripper +Y, latch side along the jaws; the mirror mounting
leaves the wrist camera looking at the cable) and the panel faces the robot: port_1 mouth at env (0.22, 0, 0.05) m. Over the full randomization (+-10 mm, +-3 deg yaw/pitch), all three ports,
tip from -15 mm to seat + 3 mm, the worst joint is 0.35 rad (20 deg) inside its limit (Wrist_Pitch).

Runs (laptop RTX 4060, headless, one Isaac process at a time; `results/arena_env_v0/`):

| Run | Result | Wall | Peak RSS |
|---|---|---|---|
| policy_runner zero_action, 2 envs, 2 episodes, `loose` / `medium` / `spec` | exit 0 on all three; episodes time out at 20 s (zero joint goals swing the plug into the panel: peak axial 11 N, one overload onset) | 2:33 / 2:03 / 2:43 | 6.6 / 6.1 / 6.1 GB |
| check_env, 1 env, cameras 240x320, `medium` | `medium_front.png`, `medium_wrist.png` | 2:46 | 9.8 GB |
| check_env `--bench 300`, 16 envs, state-only, `physx_joint` | **168 physics steps/s = 0.35x real time** (2691 env-physics-steps/s) | 2:52 | 8.2 GB |
| same, `--servo_friction_mode explicit` | 190 physics steps/s = 0.40x real time (3042 env-steps/s) | | 7.1 GB |
| check_env `--push`, `loose`, nominal pose, joint-space IK + tip-error integral | plug enters the middle jack to 12.1 mm (stage `aligned`), latch cammed to -4 deg, then stalls at 4.8 N axial with the command at seat + 3 mm | | |

Throughput is under the ~0.5x mark at 16 envs. The `explicit` friction fallback buys only 13 %: the cost is dt 1/480 x
32 position iterations on a 9-link articulation (the G1 contact settings), not the per-step friction write. Kept
`physx_joint`. A 20-episode sweep at ~10 s per episode is ~200 env-seconds, about 36 s wall with 16 envs. That is fine
on the laptop; full n=200 x 3 tiers x 3 policies sweeps belong on Brev.

Wrist camera: with the isaaclab_so101 mount, the camera looks along the plug, and the plug body hides the panel
(30 mm beyond the tip at reset). The front (`external_camera`, on the base) view shows plug, panel and stand.

Joint-space push (`--push`): when the tip blocks on the lead-in, the axial load times the 0.16 m wrist-pitch-to-TCP
lever rotates the P-only wrist servo, and the tip slides down the panel face (13 mm in one run with fast alignment).
Controllers must align before pushing and keep the push force low until the tip is inside.
