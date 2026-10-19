# Data: raw episodes → LeRobot v3

Two steps in two venvs, because Isaac Sim's torch and LeRobot's torch conflict:

| Step | Script | Venv | Output |
|---|---|---|---|
| expert rollouts with cameras | `scripts/gen_data.py` | Arena (`~/work/IsaacLab-Arena/.venv`) | raw dir: `manifest.json` + `episodes/ep_NNNNNN.npz` |
| conversion (+ optional push) | `scripts/to_lerobot.py` | LeRobot (`~/work/.venv-lerobot`, `lerobot==0.6.1`) | LeRobot v3 dataset at `$HF_LEROBOT_HOME/<repo_id>` |

The raw format lives in `patchbench/raw_format.py`. It imports only numpy and the standard library, so both venvs can use it.

## Raw format (version 1)

One compressed npz per **successful (latched)** episode. Failed attempts are counted in `manifest.info.runs` but not saved.

| key | shape | dtype | meaning |
|---|---|---|---|
| `front` | (T, H, W, 3) | uint8 | static front camera RGB |
| `wrist` | (T, H, W, 3) | uint8 | wrist camera RGB |
| `state` | (T, 6) | float32 | joint positions [deg] |
| `action` | (T, 6) | float32 | absolute joint position targets the sim received at step t [deg] |
| `servo_load` | (T, 6) | int32 | `Present_Load` decoded to signed: 0.1 % PWM duty, ±1000 |
| `servo_current` | (T, 6) | int32 | `Present_Current`: 6.5 mA per count, unsigned (as LeRobot reads it) |
| `meta` | () | str (JSON) | `task`, `tier`, `port`, `success`, `seed`, `num_frames`, `image_hw`, `extra` |

- Joint order is the LeRobot SO-101 order `shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper`. It is the
  same as the sim order `Rotation, Pitch, Elbow, Wrist_Pitch, Wrist_Roll, Jaw` (ENV_AUDIT section 5).
- Frame t holds the observation **before** `env.step` and the target sent in that step, at 30 Hz.
  The last step of an episode is dropped: the env auto-resets inside that step, so its target read-back may already be the reset pose.
- Telemetry comes from `so101_contact.isaaclab.observations.servo_load/servo_current`. These go through the `TelemetryBus`,
  which applies read rate, one-step latency, quantization and noise.
- `task` = `"insert the cable into the {left|middle|right} port"` (port 0 = left).
- Files are written atomically (temp file, then rename), and `manifest.json` is rewritten after each episode lands.
  Because of this, `gen_data.py` can resume: per (tier, port) it generates only the successes that are still missing.

## LeRobot v3 dataset

`robot_type = so101_follower`, 30 fps, videos encoded with `libsvtav1` (the LeRobot default; `--vcodec h264` is faster).

| feature | dtype | shape | names |
|---|---|---|---|
| `observation.state` | float32 | (6,) | `shoulder_pan.pos` … `gripper.pos` |
| `action` | float32 | (6,) | `shoulder_pan.pos` … `gripper.pos` |
| `observation.images.front` | video | (H, W, 3) | |
| `observation.images.wrist` | video | (H, W, 3) | |
| `observation.servo_load` | int32 | (6,) | `<joint>.load` |
| `observation.servo_current` | int32 | (6,) | `<joint>.current` |
| `task` | language | | |

The telemetry features are kept separate from `observation.state`, so GR00T can be trained with or without them.
The GR00T processor in LeRobot 0.6.1 reads only `observation.state` and the image features.
Per-episode metadata that LeRobot has no column for (tier, port, seed, raw file, time to latch, …) is stored in
`meta/patchbench_episodes.jsonl`, keyed by `episode_index`. The generator settings are stored in `meta/patchbench_raw_info.json`.
Both files are uploaded with the dataset.

### Sim → real caveats (not faked)

- **Joint zero.** A real SO-101 with `use_degrees=True` reports degrees relative to the midpoint of its own calibration range.
  The sim reports degrees relative to the USD joint zero. Mapping one onto the other needs the real arm's calibration file;
  `isaaclab_so101.mapping` documents the USD limits.
- **Gripper.** Real LeRobot reports the gripper in 0–100 even with `use_degrees=True`. We record the sim jaw angle in degrees.
- **`Present_Current` sign.** The real sign handling still needs to be verified on hardware. We record the magnitude (CLAUDE.md 4.1).

## Commands

```bash
# laptop pilot (Arena venv, cameras on, 1 env, 240x320, one Isaac process at a time)
cd ~/work/patchbench
flock ~/work/.isaac.lock ~/work/IsaacLab-Arena/.venv/bin/python scripts/gen_data.py \
  --tiers loose --episodes_per_port 1 --num_envs 1 --height 240 --width 320 --out ~/data/raw/pilot
~/work/.venv-lerobot/bin/python scripts/to_lerobot.py --raw ~/data/raw/pilot --repo_id local/patchbench-pilot
# tests
~/work/IsaacLab-Arena/.venv/bin/python -m pytest -q tests/test_raw_format.py
~/work/.venv-lerobot/bin/python -m pytest -q -s tests/test_to_lerobot.py   # 3 synthetic episodes, end to end
```

Overnight on Brev, `scripts/brev/chain_overnight.sh` runs both steps: 480×640, `num_envs` sized from the L40S pilot,
and a push to `$HF_USER/patchbench-so101-v0`.
