# PatchBench: evaluation protocol, results format, policy server

## 1. What is measured

Each episode, the SO-101 holds an RJ45 plug (rigidly attached to the fixed jaw) in front of a 3-port panel and must
insert it into the target port. Stages are tracked per episode by `patchbench.predicates.StageTracker`
(CLAUDE.md section 6.1): `reached → contacted → aligned → seated → latched`. Stages are sequential and monotone; an
episode reports its furthest stage. **Success = latched.**

Clearance tiers (lateral clearance per side): `loose` +0.50 mm, `medium` +0.25 mm, `spec` +0.115 mm.

## 2. Running an evaluation

Use the Arena venv's python. On the laptop, every Isaac Sim run holds the shared lock and stays state-only.

```bash
cd ~/work/patchbench
flock -o ~/work/.isaac.lock python scripts/evaluate.py --policy open_loop --tiers loose --n 16 --num_envs 16
```

| flag | default | meaning |
|---|---|---|
| `--policy` | — | `open_loop`, `telemetry_spiral`, `oracle_spiral`, `expert` (Arena-registered, `patchbench/policies.py`) or `lerobot` |
| `--policy.path` | — | `lerobot` only: HF repo id or local checkpoint (`random` = dummy server policy) |
| `--tiers` | `loose,medium,spec` | comma-separated |
| `--n` | 100 | episodes per tier (200 for headline numbers) |
| `--num_envs` | 16 | parallel envs (laptop: ≤ 32 state-only, ≤ 2 with cameras) |
| `--seed` | 0 | Arena `--seed`, also passed to the policy's noise RNG if its cfg has `noise_seed`/`policy_seed` |
| `--pose_noise_mm` | 1.0 (expert 0.0) | σ of the port-pose estimate given to scripted controllers: passed to the policy cfg (`pose_noise_mm`) if it has that field, else to the env as `--port_est_noise_mm` (never both) |
| `--port` | `random` | target port `0,1,2` or `random` |
| `--out` | `results/` | output root |
| `--policy_arg` | | extra policy cfg flag, repeatable, e.g. `--policy_arg=--expert_controller=oracle_spiral`, `--policy_arg=--contact_threshold=40` |
| `--dry_run` | | print the per-tier worker configs and Arena argv, start nothing |

Each tier runs in its own Isaac Sim process (one tier = one panel USD), sequentially. The worker uses Arena's own
pieces (`get_arena_builder_from_cli`, the policy registry and `build_policy_from_cli`, the per-episode
`EpisodeRecorderManager`), so only the step loop is ours.

**Episode schedule.** Global episode `i < n` is episode `i // num_envs` of env `i % num_envs`. Envs keep running
until all slots are filled; surplus episodes are discarded. Taking "the first n episodes to finish" would favour
short episodes (early terminations) and bias the success rate. The same `--seed` and `--num_envs` give the same
episodes; a different `--num_envs` gives a different (equally valid) sample.

The env records `StageTracker.summary()` for each finishing episode through an Arena episode recorder term (it runs
before reset, so the episode is intact). If the env adds no recorder term, `evaluate.py` adds one that reads the
env's `StageTracker`.

## 3. Outputs

```
results/<policy>/<tier>.json            # summary, schema below
results/<policy>/<tier>_episodes.csv    # one row per scheduled episode
results/<policy>/funnel.png             # stage funnel per tier
results/<policy>/raw/<tier>.jsonl       # Arena per-episode records (all episodes, incl. surplus), worker log
results/LEADERBOARD.md, results/leaderboard.png
```

`<tier>.json` (CLAUDE.md section 6.2; the first 13 keys are fixed, the rest are provenance):

```json
{"policy": "telemetry_spiral", "tier": "medium", "n": 200, "seed": 0,
 "funnel": {"reached": 0.99, "contacted": 0.97, "aligned": 0.80, "seated": 0.62, "latched": 0.58},
 "success": 0.58, "ci95": [0.51, 0.65], "mean_time_to_latch_s": 6.1, "wrong_port": 0.0,
 "overload_events": 0.03, "git_sha": "abc1234", "so101_contact_version": "0.1.0", "physics": "physx",
 "mean_peak_axial_force_n": 4.2, "pose_noise_mm": 1.0, "num_envs": 16, "port": "random", "cameras": false,
 "wall_time_s": 812.0, "sim_steps": 3600}
```

| key | definition |
|---|---|
| `funnel[s]` | fraction of episodes whose furthest stage is ≥ `s` |
| `success` | `funnel["latched"]` |
| `ci95` | Wilson score 95 % interval of `success` |
| `mean_time_to_latch_s` | mean time from reset to latch over latched episodes; `null` if none latched |
| `wrong_port` | fraction of episodes in which the plug tip entered a non-target port (aligned predicate) |
| `overload_events` | mean number of servo overload onsets (any joint, rising edges) per episode |
| `git_sha` | patchbench commit, `+dirty` if tracked files were modified |
| `mean_peak_axial_force_n` | mean over episodes of the peak plug–jack force along the insertion axis |

## 4. Leaderboard and gap closure

`python scripts/leaderboard.py [--results results/]` (also run at the end of `evaluate.py`) reads every
`results/*/*.json` and writes the success table, the per-tier stage funnels and

```
gap_closure = (telemetry_spiral − open_loop) / (oracle_spiral − open_loop)
```

for every tier where all three scripted policies have results. Its 95 % interval comes from 20 000 Monte Carlo
draws of the three success rates from independent Jeffreys posteriors Beta(k + ½, n − k + ½). When
`oracle_spiral − open_loop < 0.05` the ratio is not reported: there is no gap to close, and the ratio would be noise.

## 5. LeRobot policies: ZMQ policy server

Isaac Sim and LeRobot need different torch builds, so a LeRobot policy (e.g. the GR00T N1.7 fine-tune) runs in the
separate `lerobot==0.6.1` venv and talks to the simulator over ZMQ. `--policy lerobot` starts the server itself
(`$LEROBOT_PYTHON`, default `~/work/.venv-lerobot/bin/python`) and turns cameras on. It needs a GPU for the policy
next to Isaac Sim: **Brev only**.

```bash
# Brev
python scripts/evaluate.py --policy lerobot --policy.path $HF_USER/patchbench-groot-n17 --tiers loose --n 100 --num_envs 8
python scripts/evaluate.py --policy lerobot --policy.path random --tiers loose --n 4 --num_envs 2   # protocol check
# or run the server by hand and pass --no_server:
~/work/.venv-lerobot/bin/python ~/work/patchbench/patchbench/policy_server.py --policy.path <repo> --endpoint tcp://127.0.0.1:5555
```

### Protocol v1

ZMQ REQ (simulator) / REP (policy server), one outstanding request. Each message is multipart:

```
frame 0     UTF-8 JSON header  {"v": 1, "cmd" | "ok": ..., "arrays": [{"key", "dtype", "shape"}, ...], ...}
frame 1..k  raw C-contiguous array buffers, one per entry of "arrays", in order (dtype = numpy dtype.str, e.g. "<f4")
```

| request `cmd` | request fields / arrays | reply fields / arrays |
|---|---|---|
| `hello` | — | `policy`, `chunk_size` H, `action_dim` 6, `state_dim` 6, `image_keys`, `joint_names` |
| `infer` | `task`: list of B strings, `env_ids`: list of B ints; arrays `observation.state` (B, 6) float32 [deg], `observation.images.front` and `observation.images.wrist` (B, H, W, 3) uint8 | `infer_s`; array `action` (B, H, 6) float32 [deg] (joint goal positions) |
| `reset` | `env_ids`: list of ints or `null` (all) | — |
| `shutdown` | — | — (the server then exits) |

Every reply has `"ok": true`, or `"ok": false, "error": "<Type>: <message>"`; the server stays up after an error.
The client re-sends `hello` until the server answers (model loading can take minutes) and raises `TimeoutError`
when a reply takes longer than `timeout_s` (default 60 s).

**Batching and chunks.** The client keeps an action queue per env. Each sim step it sends **one** `infer` request
holding only the envs whose queue is empty, executes `n_action_steps` actions of each returned chunk (default: the
full chunk H) and then re-plans. An env reset clears that env's queue and sends `reset`.

**Units and conventions** (match the dataset, CLAUDE.md section 6.3): joint order `shoulder_pan, shoulder_lift,
elbow_flex, wrist_flex, wrist_roll, gripper`; state and actions in degrees for all six joints; images RGB uint8 HWC
(the LeRobot server converts them to float CHW in [0, 1] before its preprocessor). The sim side converts rad ↔ deg.
If the dataset uses a different gripper convention, the same conversion must be applied in
`scripts/evaluate.py:_make_remote_lerobot_policy`.

**Server side.** `LeRobotPolicy` loads `PreTrainedConfig.from_pretrained(path)`, the policy class from
`get_policy_class(cfg.type)`, and the saved pre/post-processors via `make_pre_post_processors(..., pretrained_path=path)`
(the same calls as LeRobot's async inference server), then calls `predict_action_chunk` once per request and
post-processes each step of the chunk. `RandomPolicy` (state + uniform noise) is the dummy used by the tests.

Tested: protocol, batching, chunk replay, per-env reset, error reporting and timeouts
(`tests/test_policy_server.py`, both ends in one process), and a cross-venv round trip (server in the so101-contact
venv, client in the Arena venv). The `LeRobotPolicy` loader is not yet tested against a real checkpoint: that runs on
Brev once the lerobot venv and a fine-tuned checkpoint exist.
