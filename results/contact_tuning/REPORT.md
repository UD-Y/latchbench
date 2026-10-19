# Contact tuning report

| run | dt | pos it | contact offset mm | wall s / sim s | aligned latch+signature (loose, medium) | peak axial N (loose, medium) | explosions | tunnels | max pen. aligned / all mm | latch rate all variants (loose, medium) | G1 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| c240_it16_co0.5 | 1/240 | 16 | 0.5 | 0.90 | 0.0, 1.0 | 0.29, 0.55 | 0 | 0 | 0.009 / 0.721 | 0.95, 0.73 | fail |
| c480_it8_co0.5 | 1/480 | 8 | 0.5 | 1.41 | 0.0, 1.0 | 0.22, 0.55 | 0 | 0 | 0.005 / 0.729 | 0.97, 0.78 | fail |
| c480_it16_co0.2 | 1/480 | 16 | 0.2 | 1.55 | 1.0, 1.0 | 0.48, 0.88 | 0 | 0 | 0.001 / 0.740 | 0.93, 0.75 | PASS |
| c480_it16_co1.0 | 1/480 | 16 | 1 | 1.81 | 1.0, 1.0 | 14.59, 4.46 | 0 | 0 | 0.016 / 0.726 | 0.97, 0.75 | PASS |
| c480_it16_co0.5 | 1/480 | 16 | 0.5 | 1.82 | 1.0, 1.0 | 0.48, 0.88 | 0 | 0 | 0.001 / 0.726 | 0.92, 0.77 | PASS |
| c480_it32_co0.5 | 1/480 | 32 | 0.5 | 2.50 | 1.0, 1.0 | 0.49, 1.37 | 0 | 0 | 0.000 / 0.117 | 0.93, 0.75 | PASS |
| factory_120_it192_co5 | 1/120 | 192 | 5 | 2.97 | 1.0, 1.0 | 0.40, 1.10 | 2 | 0 | 0.006 / 0.654 | 0.98, 0.77 | fail |
| c960_it16_co0.5 | 1/960 | 16 | 0.5 | 3.47 | 1.0, 1.0 | 0.92, 1.55 | 0 | 0 | 0.000 / 0.118 | 0.93, 0.75 | PASS |

Cheapest passing config: **c480_it16_co0.2** (1.55 wall s per sim s, 24 envs).
Beam check: peak axial / beam tip force = 0.50.

Spec tier, spec_c480_it32_co0.5: aligned latch 1.0, signature 1.0, peak axial 2.46 N, explosions 0, tunnels 0, max penetration all variants 0.115 mm.

Chosen for patchbench/physics.py: **c480_it32_co0.5** (32 iterations: jam penetration 0.12 mm vs 0.73 mm at 16 under a 32 N load; see the physics.py docstring).
