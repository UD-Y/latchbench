"""PhysX settings chosen by contact tuning (scripts/check_contact.py). Each value cites its tuning run id."""

SERVO_FRICTION_MODE = "physx_joint"
"""STS3215 gearbox friction mode for the RJ45 env (so101_contact ``friction_mode``).

``physx_joint`` writes BAM's Coulomb budget to PhysX joint friction every physics step, so a joint that is not
moving holds small contact torques in stiction, as the real gearbox does (results/hold_test_physx_joint: elbow
Present_Load flat until ~2 N). Decision 19 Oct: default ``physx_joint``; fall back to ``explicit`` if the per-step
CPU friction write is too slow at 16 envs. Measured 19 Oct (scripts/bench_friction_mode.py, 16 SO-101s, laptop):
222 physics steps/s vs 251 for ``explicit`` (-12 %), both ~real time -> keep ``physx_joint``.
"""
