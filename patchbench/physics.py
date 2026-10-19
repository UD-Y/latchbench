"""PhysX settings chosen by contact tuning (scripts/check_contact.py). Each value cites its tuning run id.

Chosen run: ``c480_it32_co0.5`` (results/contact_tuning/REPORT.md, 19 Oct). Reduced grid around the CLAUDE.md
centre (dt 1/480, 16 iterations, contact offset 0.5 mm) +- one step per axis, plus Isaac Lab's Factory settings.
Passing G1 (aligned plug latches with a rise -> peak -> drop axial force on loose and medium in all 5 seeds,
0 explosions, 0 tunnels, penetration < 0.1 mm) at lowest cost: ``c480_it16_co0.2``/``c480_it16_co0.5``. Chosen
instead: 32 position iterations, because under a 32 N jam load the plug sank 0.73 mm at 16 iterations and 0.12 mm
at 32 (the spiral controller pushes into faces). Cost 2.5 wall s per simulated s for 24 envs on the laptop (vs
1.8 at 16 iterations). Rejected: dt 1/240 and 8 iterations (loose plug loses the signature), contact offset 1 mm
(spurious 14.6 N peak on loose), Factory 1/120 / 192 it / 5 mm (2 explosions). Spec tier passes with the same
settings (``spec_c480_it32_co0.5``, peak 2.5 N).
"""

PHYSICS_DT = 1 / 480
"""s; run c480_it32_co0.5. 1/240 lost the loose latch signature (run c240_it16_co0.5)."""

DECIMATION = 16
"""Physics steps per 30 Hz control step (LeRobot rate), 1/480 * 16 = 1/30 s."""

SOLVER_POSITION_ITERATIONS = 32
"""Articulation (robot + plug) TGS position iterations; run c480_it32_co0.5 (jam penetration 0.12 vs 0.73 mm)."""

SOLVER_VELOCITY_ITERATIONS = 1
"""Isaac Lab Factory (isaaclab_tasks/contrib/factory/factory_env_cfg.py)."""

CONTACT_OFFSET = 0.5e-3
"""m, on every plug and jack collider; 0.2 mm gave identical results (c480_it16_co0.2), 1.0 mm a spurious peak."""

REST_OFFSET = 0.0
"""m; clearances are geometric (CLAUDE.md section 5)."""

MAX_DEPENETRATION_VELOCITY = 5.0
"""m/s on the plug links; Isaac Lab Factory."""

CONTACT_FRICTION = 0.35
"""Static = dynamic friction, plastic on plastic, for the plug/jack physics material (randomized x[0.7, 1.3])."""

PHYSX_SCENE = dict(
    solver_type=1,
    max_velocity_iteration_count=1,
    bounce_threshold_velocity=0.2,
    friction_offset_threshold=0.01,
    friction_correlation_distance=0.00625,
    gpu_max_rigid_contact_count=2**23,
    gpu_max_rigid_patch_count=2**23,
    gpu_collision_stack_size=2**28,
    gpu_max_num_partitions=1,
)
"""``isaaclab_physx.physics.PhysxCfg`` kwargs; Isaac Lab Factory scene settings."""

CONTACT_SENSOR = dict(track_friction_forces=True, max_contact_data_count_per_prim=64)
"""Plug-jack force = normal + friction; friction is ~0 with the default 4 contact slots (docs/ENV_AUDIT.md 10)."""

SERVO_FRICTION_MODE = "explicit"
"""STS3215 gearbox friction mode for the RJ45 env (so101_contact ``friction_mode``).

``explicit`` (all friction in the actuator torque, smooth tanh, no stiction). Switched from ``physx_joint`` on
19 Oct ~21:30: in ``physx_joint`` the actuator's external-torque estimate ``M*qdd - tau`` also contains PhysX's own
friction reaction, so a stuck joint's load-dependent budget grows with the command and the joint never restarts
(BAM subtracts its own friction; we cannot read it from PhysX). P6 expert on loose, 20 episodes, sigma 0:
explicit 16/20 latched (G2 passes), physx_joint 7/20. ``physx_joint`` stays available (env flag
--servo_friction_mode) until the estimate is fixed in so101-contact. Earlier physx_joint results (hold_test_physx_joint:
elbow Present_Load flat until ~2 N) are affected by the same feedback and overstate stiction.
Cost at 16 envs (dt 1/240): explicit 251 vs physx_joint 222 physics steps/s (scripts/bench_friction_mode.py).
"""
