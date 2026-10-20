# PatchBench

An [Isaac Lab-Arena](https://github.com/isaac-sim/IsaacLab-Arena) benchmark for contact-rich insertion with a
$120 arm: an SO-101 inserts an RJ45 plug into a 3-port panel at three clearance tiers. Success is staged
(reach → contact → align → seat → latch) and reported as a funnel per tier.

The SO-101 has no force sensor. Its STS3215 servos do report their PWM duty (`Present_Load`) and current
(`Present_Current`); [so101-contact](https://github.com/UD-Y/so101-contact) simulates those registers exactly as
LeRobot reads them on a real arm. PatchBench asks how much of a force sensor's benefit that telemetry recovers.

## Result

Three scripted controllers that differ only in what they sense, n = 200 episodes per cell, port-pose estimate
noise σ = 1 mm, PhysX:

| | loose (+0.50 mm) | medium (+0.25 mm) | spec (+0.115 mm) |
|---|---|---|---|
| **open-loop**: push along the estimated port axis | 0.35 [0.29, 0.42] | 0.32 [0.25, 0.38] | 0.17 [0.13, 0.23] |
| **telemetry spiral**: spiral search, contact and drop-in from simulated `Present_Load` | 0.81 [0.76, 0.86] | 0.69 [0.63, 0.75] | 0.56 [0.49, 0.63] |
| **oracle spiral**: same search, true plug–jack force | 0.82 [0.76, 0.87] | 0.72 [0.65, 0.78] | 0.56 [0.49, 0.63] |
| **gap closure** (telemetry − open-loop) / (oracle − open-loop) | **0.99** [0.84, 1.17] | **0.94** [0.74, 1.18] | **1.00** [0.77, 1.29] |

Latch rate with Wilson 95 % intervals; gap-closure intervals from Jeffreys posteriors. Full funnels, time to latch
and provenance: [results/LEADERBOARD.md](results/LEADERBOARD.md).

![leaderboard](results/leaderboard.png)

Within this search strategy, the servo's own load signal does about as well as a perfect force sensor. What that
does and does not show:

- The oracle uses the same spiral search, so gap closure measures **sensing**, not search: a better searcher could
  raise the oracle line.
- The servo model is a hardware-identified fit (Rhoban BAM) of one STS3215. The benchmark default
  `explicit` servo friction has no gearbox stiction, which makes the simulated telemetry cleaner than a real
  gearbox. The stiction mode (`--servo_friction_mode physx_joint`) latched 41/60 vs 45/60 for the expert on
  loose, within noise; the sweep above was not repeated in it.
- The plug is rigidly fixed in the gripper (no in-hand slip), and nothing here has run on a real arm yet: the
  [calibration protocol](https://github.com/UD-Y/so101-contact/blob/main/docs/CALIBRATION_PROTOCOL.md) is how
  the sim-to-real check would be done.

## What is in the box

| Path | What |
|---|---|
| `patchbench/assets/build_assets.py` | parametric RJ45 plug (body + sprung latch on a revolute joint) and 3-port panel, primitives only, one USD per tier |
| `patchbench/environment.py`, `task.py`, `metrics.py` | Arena environment `rj45_panel`: SO-101 on STS3215 servo actuators, panel, plug in hand, staged task, metrics |
| `patchbench/predicates.py` | pure-torch stage logic (reach → contact → align → seat → latch), wrong-port detection |
| `patchbench/physics.py` | PhysX settings chosen by contact tuning, each with its run id |
| `patchbench/controllers/`, `policies.py` | IK, open-loop and spiral (telemetry / oracle) controllers, Arena policies |
| `scripts/evaluate.py`, `leaderboard.py` | evaluator (results JSON per policy × tier) and leaderboard + gap closure |
| `patchbench/policy_server.py` | ZMQ protocol to evaluate LeRobot policies (e.g. GR00T N1.7) from a separate venv |
| `scripts/gen_data.py`, `to_lerobot.py` | expert rollouts with cameras → LeRobot v3 dataset (real SO-101 conventions) |
| `scripts/check_contact.py`, `hold_test.py` | contact tuning (virtual wrist) and the SO-101 servo hold / press test |
| `scripts/brev/` | setup and overnight chain for a 1× L40S VM |
| `docs/` | [BENCHMARK.md](docs/BENCHMARK.md) (protocol, formats, policy server), [ENV_AUDIT.md](docs/ENV_AUDIT.md) (versions, APIs, pitfalls), DATA.md |

## Install

Linux, an RTX GPU (cameras need RT cores: L40S yes, H100/A100 no), Python 3.12, [uv](https://docs.astral.sh/uv/).
Pinned: IsaacLab-Arena `aa36f19` (Isaac Lab submodule `bb0c8e1`, Isaac Sim 6.1.0.0), isaaclab-so101 `12a3ecc`.

```bash
mkdir -p ~/work && cd ~/work
GIT_LFS_SKIP_SMUDGE=1 git clone https://github.com/isaac-sim/IsaacLab-Arena.git && cd IsaacLab-Arena
GIT_LFS_SKIP_SMUDGE=1 git checkout aa36f19
git submodule init -- submodules/IsaacLab
git config submodule.submodules/IsaacLab.url https://github.com/isaac-sim/IsaacLab.git
GIT_LFS_SKIP_SMUDGE=1 git submodule update --depth 1 -- submodules/IsaacLab
uv sync && source .venv/bin/activate && export OMNI_KIT_ACCEPT_EULA=YES ACCEPT_EULA=Y
cd ~/work
git clone https://github.com/art-e-fact/isaaclab-so101 && git -C isaaclab-so101 checkout 12a3ecc
git clone https://github.com/UD-Y/so101-contact && git clone https://github.com/UD-Y/patchbench
uv pip install -e isaaclab-so101 -e so101-contact -e patchbench pyzmq   # repeat after every `uv sync`
```

## Run

```bash
cd ~/work/patchbench
python scripts/evaluate.py --policy telemetry_spiral --tiers loose,medium,spec --n 200 --num_envs 32
python scripts/leaderboard.py
```

On a shared machine, wrap every Isaac Sim launch in `flock -o ~/work/.isaac.lock …` (one Isaac process at a time;
`-o` matters, see ENV_AUDIT section 11). The full sweep (3 policies × 3 tiers × 200) took about 75 min on an
RTX 4060 laptop GPU, state-only. Evaluating your own LeRobot policy: `--policy lerobot --policy.path <hf repo>`
(see [BENCHMARK.md](docs/BENCHMARK.md) section 5).

Watch it: `--viz kit` opens the viewer on `scripts/hold_test.py` and `scripts/check_contact.py`, e.g.

```bash
python scripts/check_contact.py --viz kit --tiers medium --offsets_mm 0 --tilts_deg 0 --seeds 1 --pos_iters 32 --cutaway --hold_open
```

## How the numbers were made

- **Contact physics (gate G1):** a compliant "virtual wrist" pushed the plug into each tier with offsets and tilts;
  a reduced grid of 8 PhysX configs chose dt 1/480 s, 32 position iterations, 0.5 mm contact offset: the latch gives a
  repeatable rise → peak → drop force on all tiers, no explosions or tunnelling
  ([results/contact_tuning/REPORT.md](results/contact_tuning/REPORT.md)). Contact force must include friction
  (`track_friction_forces=True`, 64 contact slots); normal-only force under-reads the latch load ~10×.
- **Servo:** STS3215 firmware loop → PWM duty → DC motor → BAM m5 gearbox friction, LeRobot gains (P = 16); hold
  test: sag ≤ 1.8°, 30° step settles in 0.33 s ([results/hold_test](results/hold_test)).
- **Controllers:** the expert (σ = 0) latched 16/20 on loose (gate G2).

## Status

v0.1 (21 Oct 2026). Done: assets, environment, contact tuning, servo actuator, three controllers, n = 200 sweep,
evaluator, LeRobot v3 data pipeline (laptop pilot), Brev scripts. Pending: the GR00T N1.7 baseline (dataset at
scale, fine-tune and evaluation on a 1× L40S), a real-arm calibration run.

## License

Apache-2.0.
