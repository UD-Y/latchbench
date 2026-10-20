#!/usr/bin/env bash
# Overnight chain on the Brev 1x L40S VM (after setup_brev.sh printed READY). Run it inside tmux:
#   tmux new -s chain
#   export HF_TOKEN=... HF_USER=...          # or a cached `hf auth login` token + HF_USER
#   bash ~/work/patchbench/scripts/brev/chain_overnight.sh            # Ctrl-b d to detach
#   bash ~/work/patchbench/scripts/brev/chain_overnight.sh --dry-run  # plan + time estimates only
#
# Stages (each is resumable and skipped when its output already exists):
#   1  gen_data      expert rollouts, cameras on, TIERS x 3 ports x EPISODES_PER_PORT latched episodes  [Isaac]
#   2  to_lerobot    raw -> LeRobot v3 at $HF_LEROBOT_HOME/$DATASET_REPO, then push        [CPU+net, background]
#   3  sweeps        evaluate.py open_loop / telemetry_spiral / oracle_spiral on loose,medium,spec, n=200,
#                    state-only (runs while stage 2 encodes)                                             [Isaac]
#   4  groot         lerobot-train GR00T N1.7 (CLAUDE.md section 10), push to $HF_USER/patchbench-groot-n17 [GPU]
#   5  pack          results tarball in ~/out; then `sudo poweroff` unless NO_POWEROFF=1
# A failed stage is logged and the chain carries on with every stage that does not depend on it (sweeps do not
# need data; GR00T needs the converted dataset), then packs and powers off, so a failure never burns the night.
# Isaac Sim launches run one at a time (flock -o on ~/work/.isaac.lock, as on the laptop; -o so the Omniverse Hub
# daemon that Isaac Sim spawns does not inherit the lock and hold it after the run).
#
# Knobs (environment variables; defaults in brackets):
#   EPISODES_PER_PORT [150]  DATAGEN_TIERS [loose,medium]  DATAGEN_NUM_ENVS [8]  IMG_H/IMG_W [480/640]
#   SWEEP_N [200]  SWEEP_TIERS [loose,medium,spec]  SWEEP_NUM_ENVS [64]  SWEEP_POSE_NOISE_MM [1]
#   EXPERT_CONTROLLER [policy default: open_loop | oracle_spiral | telemetry_spiral]  EXPERT_POSE_NOISE_MM [policy default 0]
#   STATE_WITH_LOAD [0; 1 = observation.state = 6 positions + 6 Present_Load, to_lerobot.py --state_with_load]
#   A second dataset/model next to v0: DATASET_NAME [patchbench-so101-v0]  MODEL_NAME [patchbench-groot-n17]
#   RAW [~/data/raw/patchbench_v0]  GROOT_RUN [groot_patchbench]  STATE_DIR [~/out/chain_state]
#   GROOT_STEPS [6000]  GROOT_BATCH [32]  NO_POWEROFF [0]  SKIP_STAGES (e.g. "1,4")  VCODEC [libsvtav1]
# Time-estimate inputs (measure them on the L40S: gen_data prints env-frames/s, to_lerobot prints frames/s):
#   DATAGEN_FPS [60, placeholder]  EP_SECONDS [6]  FAIL_SECONDS [20]  DATAGEN_SUCCESS [0.5]  ENCODE_FPS [60, placeholder]
#   SWEEP_FPS [1500, placeholder]  SWEEP_EP_SECONDS [12]  GROOT_SEC_PER_STEP [0.6, placeholder]  STARTUP_S [120]
set -euo pipefail

DRY_RUN=false
for arg in "$@"; do
  case "$arg" in
    --dry-run) DRY_RUN=true ;;
    -h|--help) awk 'NR > 1 && !/^#/ {exit} NR > 1 {sub(/^# ?/, ""); print}' "$0"; exit 0 ;;
    *) echo "chain_overnight.sh: unknown option $arg (try --help)" >&2; exit 2 ;;
  esac
done

WORK="${WORK:-$HOME/work}"
PB="$WORK/patchbench"
ARENA="$WORK/IsaacLab-Arena"
ARENA_PY="$ARENA/.venv/bin/python"
LEROBOT_VENV="$WORK/.venv-lerobot"
ISAAC_LOCK="$WORK/.isaac.lock"
export PATH="$HOME/.local/bin:$PATH"
export OMNI_KIT_ACCEPT_EULA=YES ACCEPT_EULA=Y

EPISODES_PER_PORT="${EPISODES_PER_PORT:-150}"
DATAGEN_TIERS="${DATAGEN_TIERS:-loose,medium}"
DATAGEN_NUM_ENVS="${DATAGEN_NUM_ENVS:-8}"
IMG_H="${IMG_H:-480}"
IMG_W="${IMG_W:-640}"
SWEEP_N="${SWEEP_N:-200}"
SWEEP_TIERS="${SWEEP_TIERS:-loose,medium,spec}"
SWEEP_NUM_ENVS="${SWEEP_NUM_ENVS:-64}"
SWEEP_POSE_NOISE_MM="${SWEEP_POSE_NOISE_MM:-1}"
SWEEP_POLICIES=(open_loop telemetry_spiral oracle_spiral)
GROOT_STEPS="${GROOT_STEPS:-6000}"
GROOT_BATCH="${GROOT_BATCH:-32}"
NO_POWEROFF="${NO_POWEROFF:-0}"
SKIP_STAGES="${SKIP_STAGES:-}"
VCODEC="${VCODEC:-libsvtav1}"

DATAGEN_FPS="${DATAGEN_FPS:-60}"
EP_SECONDS="${EP_SECONDS:-6}"      # laptop pilot: latched episodes 154-182 frames
FAIL_SECONDS="${FAIL_SECONDS:-20}"  # failed attempts run to the timeout
DATAGEN_SUCCESS="${DATAGEN_SUCCESS:-0.5}"  # laptop pilot: expert latched 3 of 7 on loose
ENCODE_FPS="${ENCODE_FPS:-60}"
SWEEP_FPS="${SWEEP_FPS:-1500}"
SWEEP_EP_SECONDS="${SWEEP_EP_SECONDS:-12}"
GROOT_SEC_PER_STEP="${GROOT_SEC_PER_STEP:-0.6}"
STARTUP_S="${STARTUP_S:-120}"

HF_USER="${HF_USER:-}"
DATASET_REPO="${HF_USER:-HF_USER}/${DATASET_NAME:-patchbench-so101-v0}"
MODEL_REPO="${HF_USER:-HF_USER}/${MODEL_NAME:-patchbench-groot-n17}"
RAW="${RAW:-$HOME/data/raw/patchbench_v0}"
export HF_LEROBOT_HOME="${HF_LEROBOT_HOME:-$HOME/.cache/huggingface/lerobot}"
DATASET_ROOT="$HF_LEROBOT_HOME/$DATASET_REPO"
GROOT_RUN="${GROOT_RUN:-groot_patchbench}"
GROOT_OUT="$WORK/outputs/$GROOT_RUN"   # CLAUDE.md section 10 uses outputs/groot_patchbench, run from $WORK
RESULTS="$PB/results"
STATE="${STATE_DIR:-$HOME/out/chain_state}"    # stage markers
TS="$(date +%Y%m%d_%H%M%S)"
IFS=, read -r -a TIER_LIST <<< "$DATAGEN_TIERS"
IFS=, read -r -a SWEEP_TIER_LIST <<< "$SWEEP_TIERS"

# ------------------------------------------------------------------------------------------------- estimates
estimate() {
  python3 - "$@" <<'PY'
import sys
(eps_per_port, n_tiers, ep_s, fail_s, success, dfps, efps, sweep_n, n_sweep_tiers, n_pol, sweep_ep_s, sfps,
 steps, sps, startup) = map(float, sys.argv[1:])
fps = 30
ok_eps = eps_per_port * 3 * n_tiers
frames = ok_eps * ep_s * fps
sim_frames = frames + ok_eps * (1 / success - 1) * fail_s * fps  # failures run to the timeout
gen = sim_frames / dfps + startup * 3 * n_tiers                  # one Isaac launch per (tier, port)
enc = frames / efps
sweep = n_pol * n_sweep_tiers * sweep_n * sweep_ep_s * fps / sfps + startup * n_pol
groot = steps * sps
crit = gen + max(sweep, enc) + groot                               # stage 2 runs alongside stage 3
h = lambda s: f"{s / 3600:5.2f} h"
print(f"  1 gen_data   {int(ok_eps):5d} latched eps, {frames / 1e3:.0f} k frames, {sim_frames / 1e3:.0f} k sim frames @ {dfps:g} fps  {h(gen)}")
print(f"  2 to_lerobot {frames / 1e3:.0f} k frames @ {efps:g} fps encode (background, overlaps 3)            {h(enc)}")
print(f"  3 sweeps     {int(n_pol)} policies x {int(n_sweep_tiers)} tiers x n={int(sweep_n)} @ {sfps:g} env-fps            {h(sweep)}")
print(f"  4 groot      {int(steps)} steps x {sps:g} s/step                                      {h(groot)}")
print(f"  total (critical path gen + max(sweeps, encode) + groot)                    {h(crit)}")
print(f"  cost at $1.06/hr: ${crit / 3600 * 1.06:.2f}")
if crit > 7 * 3600:
    print("  WARNING: > 7 h. Lower EPISODES_PER_PORT or GROOT_STEPS, or drop the spec sweep (SWEEP_TIERS=loose,medium).")
PY
}

print_plan() {
  cat <<EOF
chain_overnight.sh plan ($(hostname), $(date '+%a %d %b %H:%M'))
  raw data      $RAW
  dataset       $DATASET_REPO  (local $DATASET_ROOT)
  model         $MODEL_REPO    (local $GROOT_OUT)
  results       $RESULTS  -> tarball in ~/out
  poweroff      $([[ "$NO_POWEROFF" == 1 ]] && echo "no (NO_POWEROFF=1)" || echo "yes, at the end")
  skip stages   ${SKIP_STAGES:-none}
Stages:
  1 gen_data    expert ${EXPERT_CONTROLLER:-default}, tiers $DATAGEN_TIERS, $EPISODES_PER_PORT latched eps/port, num_envs $DATAGEN_NUM_ENVS, ${IMG_H}x${IMG_W}, cameras on  $(stage_status 1)
  2 to_lerobot  $VCODEC, push to $DATASET_REPO (background)  $(stage_status 2)
  3 sweeps      ${SWEEP_POLICIES[*]} on $SWEEP_TIERS, n=$SWEEP_N, num_envs $SWEEP_NUM_ENVS, pose noise ${SWEEP_POSE_NOISE_MM} mm, state-only  $(stage_status 3)
  4 groot       GR00T N1.7, $GROOT_STEPS steps, batch $GROOT_BATCH, push to $MODEL_REPO  $(stage_status 4)
  5 pack        results tarball, then poweroff
Estimates (inputs: DATAGEN_FPS=$DATAGEN_FPS ENCODE_FPS=$ENCODE_FPS SWEEP_FPS=$SWEEP_FPS GROOT_SEC_PER_STEP=$GROOT_SEC_PER_STEP
           EP_SECONDS=$EP_SECONDS FAIL_SECONDS=$FAIL_SECONDS DATAGEN_SUCCESS=$DATAGEN_SUCCESS; placeholders until measured on the L40S):
EOF
  estimate "$EPISODES_PER_PORT" "${#TIER_LIST[@]}" "$EP_SECONDS" "$FAIL_SECONDS" "$DATAGEN_SUCCESS" "$DATAGEN_FPS" "$ENCODE_FPS" \
    "$SWEEP_N" "${#SWEEP_TIER_LIST[@]}" "${#SWEEP_POLICIES[@]}" "$SWEEP_EP_SECONDS" "$SWEEP_FPS" \
    "$GROOT_STEPS" "$GROOT_SEC_PER_STEP" "$STARTUP_S"
}

# ------------------------------------------------------------------------------------------------- outputs
datagen_complete() {
  "$ARENA_PY" - "$RAW" "$EPISODES_PER_PORT" "${TIER_LIST[@]}" <<'PY' 2>/dev/null
import sys
from patchbench import raw_format as rf
root, n, tiers = sys.argv[1], int(sys.argv[2]), sys.argv[3:]
m = rf.load_manifest(root)
sys.exit(0 if all(m.count(tier=t, port=p) >= n for t in tiers for p in range(3)) else 1)
PY
}
sweeps_complete() {
  local p t
  for p in "${SWEEP_POLICIES[@]}"; do
    for t in "${SWEEP_TIER_LIST[@]}"; do [[ -f "$RESULTS/$p/$t.json" ]] || return 1; done
  done
}
policy_complete() {
  local t
  for t in "${SWEEP_TIER_LIST[@]}"; do [[ -f "$RESULTS/$1/$t.json" ]] || return 1; done
}
stage_status() {
  case "$1" in
    1) if datagen_complete; then echo "[done]"; else echo "[todo]"; fi ;;
    2) if [[ -f "$STATE/dataset_pushed" ]]; then echo "[done]"
       elif [[ -f "$DATASET_ROOT/meta/info.json" ]]; then echo "[converted, push todo]"
       else echo "[todo]"; fi ;;
    3) if sweeps_complete; then echo "[done]"; else echo "[todo]"; fi ;;
    4) if [[ -f "$STATE/groot_done" ]]; then echo "[done]"
       elif [[ -d "$GROOT_OUT/checkpoints/last" ]]; then echo "[resume]"
       else echo "[todo]"; fi ;;
  esac
}

if [[ "$DRY_RUN" == true ]]; then
  print_plan
  exit 0
fi

# ------------------------------------------------------------------------------------------------- real run
[[ -n "${TMUX:-}" || "${ALLOW_NO_TMUX:-0}" == 1 ]] || {
  echo "Run this inside tmux (tmux new -s chain), or set ALLOW_NO_TMUX=1." >&2; exit 2; }
mkdir -p "$HOME/logs" "$HOME/out" "$STATE"
LOG="$HOME/logs/chain_$TS.log"
exec > >(tee -a "$LOG") 2>&1
log() { printf '[%s] %s\n' "$(date '+%H:%M:%S')" "$*"; }
skipped() { [[ ",$SKIP_STAGES," == *",$1,"* ]]; }

declare -A RC=()
run_stage() {   # run_stage <id> <function>: run in a subshell with set -e, record the exit code, never abort the chain
  local id="$1" fn="$2" t0=$SECONDS rc
  if skipped "$id"; then log "stage $id ($fn): SKIPPED (SKIP_STAGES)"; RC[$id]=skip; return 0; fi
  log "stage $id ($fn): start"
  set +e
  ( set -euo pipefail; "$fn" )
  rc=$?
  set -e
  RC[$id]=$rc
  log "stage $id ($fn): exit $rc after $(( (SECONDS - t0) / 60 )) min"
}

preflight() {
  log "preflight"
  print_plan
  [[ -x "$ARENA_PY" ]] || { echo "Arena venv missing: run setup_brev.sh" >&2; exit 1; }
  [[ -x "$LEROBOT_VENV/bin/lerobot-train" ]] || { echo "lerobot venv missing: run setup_brev.sh" >&2; exit 1; }
  [[ -n "$HF_USER" ]] || { echo "export HF_USER first" >&2; exit 1; }
  "$LEROBOT_VENV/bin/python" -c "from huggingface_hub import get_token; import sys; sys.exit(get_token() is None)" \
    || { echo "no Hugging Face token: export HF_TOKEN or run setup_brev.sh with HF_TOKEN set" >&2; exit 1; }
  for f in scripts/gen_data.py scripts/to_lerobot.py scripts/evaluate.py; do
    [[ $(wc -l < "$PB/$f") -gt 5 ]] || { echo "$PB/$f is still a stub" >&2; exit 1; }
  done
  nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader
  df -h "$HOME" | tail -1
}

stage_gen_data() {
  if datagen_complete; then log "gen_data: $RAW complete; skipped"; return 0; fi
  cd "$PB"
  flock -o "$ISAAC_LOCK" "$ARENA_PY" scripts/gen_data.py --tiers "$DATAGEN_TIERS" \
    --episodes_per_port "$EPISODES_PER_PORT" --num_envs "$DATAGEN_NUM_ENVS" --height "$IMG_H" --width "$IMG_W" \
    --out "$RAW" ${EXPERT_CONTROLLER:+--expert_controller "$EXPERT_CONTROLLER"} \
    ${EXPERT_POSE_NOISE_MM:+--pose_noise_mm "$EXPERT_POSE_NOISE_MM"}  # expert pose_noise_mm defaults to 0
}

stage_to_lerobot() {
  if [[ -f "$STATE/dataset_pushed" ]]; then log "to_lerobot: already pushed; skipped"; return 0; fi
  cd "$PB"
  "$LEROBOT_VENV/bin/python" scripts/to_lerobot.py --raw "$RAW" --repo_id "$DATASET_REPO" --root "$DATASET_ROOT" \
    --vcodec "$VCODEC" --push $([[ "${STATE_WITH_LOAD:-0}" == 1 ]] && echo --state_with_load)
  touch "$STATE/dataset_pushed"
}

stage_sweeps() {
  cd "$PB"
  local p
  for p in "${SWEEP_POLICIES[@]}"; do
    if policy_complete "$p"; then log "sweeps: $p complete; skipped"; continue; fi
    flock -o "$ISAAC_LOCK" "$ARENA_PY" scripts/evaluate.py --policy "$p" --tiers "$SWEEP_TIERS" --n "$SWEEP_N" \
      --num_envs "$SWEEP_NUM_ENVS" --seed 0 --pose_noise_mm "$SWEEP_POSE_NOISE_MM" --out "$RESULTS/"
  done
  "$ARENA_PY" scripts/leaderboard.py || log "leaderboard.py failed (non-fatal)"
}

stage_groot() {
  if [[ -f "$STATE/groot_done" ]]; then log "groot: done; skipped"; return 0; fi
  [[ -f "$DATASET_ROOT/meta/info.json" ]] || { echo "dataset $DATASET_ROOT missing" >&2; return 1; }
  # shellcheck disable=SC1091
  source "$LEROBOT_VENV/bin/activate"
  cd "$WORK"
  local resume=()
  if [[ -f "$GROOT_OUT/checkpoints/last/pretrained_model/train_config.json" ]]; then
    resume=(--resume=true --config_path="$GROOT_OUT/checkpoints/last/pretrained_model/train_config.json")
    log "groot: resuming from $GROOT_OUT/checkpoints/last"
  elif [[ -d "$GROOT_OUT" ]]; then
    mv "$GROOT_OUT" "$GROOT_OUT.stale_$TS"   # lerobot-train refuses an existing output_dir without a checkpoint
  fi
  # CLAUDE.md section 10, verbatim apart from $HF_USER, the step/batch knobs and the resume flags.
  "$LEROBOT_VENV/bin/lerobot-train" --dataset.repo_id="$DATASET_REPO" --policy.type=groot \
    --policy.base_model_path=nvidia/GR00T-N1.7-3B --policy.embodiment_tag=new_embodiment \
    --policy.chunk_size=16 --policy.n_action_steps=16 --policy.use_relative_actions=true \
    --policy.relative_exclude_joints='["gripper"]' --policy.use_bf16=true \
    --batch_size="$GROOT_BATCH" --steps="$GROOT_STEPS" --save_freq=2000 --output_dir="outputs/$GROOT_RUN" \
    --policy.push_to_hub=true --policy.repo_id="$MODEL_REPO" "${resume[@]}"
  touch "$STATE/groot_done"
}

stage_pack() {
  local tarball="$HOME/out/patchbench_results_$TS.tar.gz"
  local items=("$RESULTS" "$HOME/logs")
  [[ -f "$RAW/manifest.json" ]] && items+=("$RAW/manifest.json")
  [[ -d "$DATASET_ROOT/meta" ]] && items+=("$DATASET_ROOT/meta")
  [[ -d "$GROOT_OUT" ]] && items+=("$GROOT_OUT/train_config.json" "$GROOT_OUT/checkpoints/last/pretrained_model/train_config.json")
  local existing=() x
  for x in "${items[@]}"; do [[ -e "$x" ]] && existing+=("$x"); done
  tar -czf "$tarball" "${existing[@]}" 2>/dev/null || true
  {
    echo "chain $TS on $(hostname)"
    for k in 1 2 3 4; do echo "stage $k: ${RC[$k]:-not run}"; done
    echo "tarball: $tarball"
  } | tee "$HOME/out/chain_$TS.summary"
  log "fetch with: rsync -avz pb:out/ ~/work/patchbench/results/brev_$TS/"
}

# ------------------------------------------------------------------------------------------------- sequence
preflight
run_stage 1 stage_gen_data

CONVERT_PID=""
if [[ "${RC[1]}" == 0 || "${RC[1]}" == skip ]] && ! skipped 2; then
  log "stage 2 (stage_to_lerobot): start in background, log $HOME/logs/to_lerobot_$TS.log"
  ( set -euo pipefail; stage_to_lerobot ) > "$HOME/logs/to_lerobot_$TS.log" 2>&1 &
  CONVERT_PID=$!
else
  if skipped 2; then RC[2]=skip; else RC[2]=blocked; log "stage 2: not started (gen_data failed)"; fi
fi

run_stage 3 stage_sweeps

if [[ -n "$CONVERT_PID" ]]; then
  log "waiting for stage 2 (pid $CONVERT_PID)"
  set +e; wait "$CONVERT_PID"; RC[2]=$?; set -e
  log "stage 2 (stage_to_lerobot): exit ${RC[2]}"
  tail -n 40 "$HOME/logs/to_lerobot_$TS.log"
fi

# GR00T needs the converted dataset; a failed push alone does not block training from the local copy.
if [[ -f "$DATASET_ROOT/meta/info.json" ]]; then
  run_stage 4 stage_groot
else
  log "stage 4: not started (no dataset at $DATASET_ROOT)"; RC[4]=blocked
fi

run_stage 5 stage_pack

if [[ "$NO_POWEROFF" == 1 ]]; then
  log "NO_POWEROFF=1: leaving the VM on"
else
  log "powering off in 120 s (Ctrl-C to cancel)"
  sleep 120
  sudo poweroff
fi
