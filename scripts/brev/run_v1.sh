#!/usr/bin/env bash
# GR00T v1 end to end on the Brev 1x L40S (as run 20-21 Oct 2026, after setup_brev.sh): telemetry-spiral expert with
# 1.5 mm port-pose noise -> UD77/latchbench-so101-v1 (observation.state = 6 positions + 6 Present_Load) ->
# GR00T N1.7 fine-tune UD77/latchbench-groot-n17-load -> offline check, n=50 x 3 tiers, 16-env camera run.
# Measured: datagen 130 min, encode+push 84 min, train 97 min (1.14 step/s), eval ~45 min. Run inside tmux:
#   tmux new -s v1 'bash ~/work/latchbench/scripts/brev/run_v1.sh; exec bash'
# Stages are resumable (chain_overnight.sh skips finished ones; STATE_DIR holds the markers).
set -euo pipefail
PB="$HOME/work/latchbench"
export HF_USER="${HF_USER:-UD77}" EPISODES_PER_PORT=100 DATAGEN_NUM_ENVS=16 SKIP_STAGES=3 NO_POWEROFF=1 VCODEC=h264 \
  EXPERT_CONTROLLER=telemetry_spiral EXPERT_POSE_NOISE_MM=1.5 STATE_WITH_LOAD=1 \
  DATASET_NAME=latchbench-so101-v1 MODEL_NAME=latchbench-groot-n17-load RAW="$HOME/data/raw/latchbench_v1" \
  GROOT_RUN=groot_latchbench_v1 STATE_DIR="$HOME/out/chain_state_v1"
mkdir -p "$HOME/logs"
bash "$PB/scripts/brev/chain_overnight.sh" 2>&1 | tee "$HOME/logs/chain_v1.log"
[ -f "$STATE_DIR/groot_done" ] || { echo "GR00T training did not finish; see ~/logs/chain_v1.log" >&2; exit 1; }

CKPT="$HOME/work/outputs/$GROOT_RUN/checkpoints/last/pretrained_model"   # same weights as the Hub repo
ARENA_PY="$HOME/work/IsaacLab-Arena/.venv/bin/python"
cd "$PB" && export OMNI_KIT_ACCEPT_EULA=YES ACCEPT_EULA=Y
"$HOME/work/.venv-lerobot/bin/python" scripts/check_policy_offline.py --policy.path "$CKPT" --raw "$RAW" \
  --episodes 0,250,500 --frames 0,30,60,90,120,150 2>&1 | tail -8 | tee "$HOME/logs/offline_v1.log"
flock -o "$HOME/work/.isaac.lock" "$ARENA_PY" scripts/evaluate.py --policy lerobot --policy.path "$CKPT" \
  --name lerobot_latchbench-groot-n17-load --tiers loose,medium,spec --n 50 --num_envs 16 --out "$HOME/eval_v1" \
  --no_leaderboard 2>&1 | tee "$HOME/logs/eval_v1.log"
flock -o "$HOME/work/.isaac.lock" "$ARENA_PY" scripts/evaluate.py --policy lerobot --policy.path "$CKPT" \
  --name lerobot_latchbench-groot-n17-load --tiers loose --n 16 --num_envs 16 --out "$HOME/eval_v1_viz" \
  --no_leaderboard --save_frames_every 2 2>&1 | tee "$HOME/logs/eval_v1_viz.log"
echo "fetch with (laptop): rsync -a --exclude '*.log' pb:eval_v1/ ~/work/latchbench/results/"
