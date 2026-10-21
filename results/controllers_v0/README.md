# controllers v0: integration results (19 Oct 2026, laptop, headless, no cameras)

Runner: `run_controllers.py` (Arena env built exactly as `policy_runner`, policy instantiated directly, stage tracker
snapshotted before each reset). 8 envs, 20 episodes per row, `--port random`, panel randomization on, servo params
randomized, **`--servo_friction_mode explicit`** (see below). n = 20: 95 % CIs are roughly +-20 points.

| policy | tier | sigma [mm] | reached | contacted | aligned | seated | latched | mean time to latch [s] |
|---|---|---|---|---|---|---|---|---|
| expert (open loop, true pose) | loose | 0 | 1.00 | 1.00 | 0.80 | 0.80 | **0.80** | 5.1 |
| open_loop | loose | 1 | 1.00 | 1.00 | 0.55 | 0.55 | **0.55** | 5.3 |
| telemetry_spiral | loose | 1 | 1.00 | 1.00 | 0.80 | 0.80 | **0.80** | 6.2 |
| oracle_spiral | loose | 1 | 1.00 | 1.00 | 0.80 | 0.80 | **0.80** | 5.5 |
| open_loop | medium | 1 | 1.00 | 1.00 | 0.35 | 0.35 | **0.35** | 5.4 |
| telemetry_spiral | medium | 1 | 1.00 | 1.00 | 0.70 | 0.70 | **0.70** | 7.6 |
| oracle_spiral | medium | 1 | 1.00 | 1.00 | 0.90 | 0.90 | **0.90** | 5.7 |

gap_closure = (telemetry - open_loop) / (oracle - open_loop): loose (0.80-0.55)/(0.80-0.55) = 1.0, medium
(0.70-0.35)/(0.90-0.35) = 0.64. Preliminary (n = 20); the headline needs n = 200 on Brev.

**G2 (expert latches >= 70 % of 20 on loose): passes with `explicit` servo friction (16/20), fails with the env default
`physx_joint` (7/20, `expert_loose_n20_s0_physx_joint.json`, run before the final integral gains).** With
`physx_joint`, joints that stop do not restart even for a 10 mm commanded lead (3/20 never reached the port): the
STS3215 friction budget uses `ext = M*qdd - tau_applied`, which absorbs the PhysX friction reaction while a joint is
stuck, so the budget grows with the command. Decision needed on `latchbench/physics.py` `SERVO_FRICTION_MODE`.

Remaining failure modes (all policies): the plug reaches the face ~1 mm off the axis (servo sag changes as the arm
extends; the sag integral does not fully remove it in the 3 s entry gate) and then slides along the face. Seated
always means latched (the push lead gives enough force on loose and medium).
