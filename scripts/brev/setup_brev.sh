#!/usr/bin/env bash
# Bootstrap a Brev 1x L40S Ubuntu VM for patchbench. Idempotent: re-run it after every rsync.
#
# Before running (from the laptop):
#   rsync -avz --exclude .venv --exclude '*.egg-info' ~/work/so101-contact ~/work/patchbench pb:~/work/
# On the VM:
#   export HF_TOKEN=... HF_USER=...        # optional here; needed for the chain's pushes and GR00T weights
#   bash ~/work/patchbench/scripts/brev/setup_brev.sh [--force-sync] [--skip-tests]
#
# Result (same layout as the laptop, CLAUDE.md section 3):
#   ~/work/IsaacLab-Arena   aa36f19 + Isaac Lab submodule bb0c8e1 (HTTPS, shallow, no LFS), uv venv at .venv
#   ~/work/isaaclab-so101   art-e-fact clone @ 12a3ecc, editable in the Arena venv
#   ~/work/so101-contact, ~/work/patchbench   (rsynced) editable in the Arena venv
#   ~/work/.venv-lerobot    lerobot[groot,training]==0.6.1 (separate: torch conflicts with Isaac Sim)
# Log: ~/logs/setup_<timestamp>.log
#
# Options:
#   --force-sync   re-run `uv sync` in Arena even if its venv exists (editable installs are then redone)
#   --skip-tests   skip the Arena smoke test and the pytest runs
set -euo pipefail

ARENA_REV=aa36f191d89b1ea65fbba62b3350d6fc7d0e9b9f      # ENV_AUDIT section 1
ISAACLAB_REV=bb0c8e1b9af381bf13064ec3303e17db79e4b6ef   # Arena's submodule pin (checked below)
SO101_REV=12a3eccd6d798864afc862bc849cc09fd8610224      # art-e-fact/isaaclab-so101 (not on PyPI)
LEROBOT_SPEC="lerobot[groot,training]==0.6.1"           # CLAUDE.md section 2
WORK="${WORK:-$HOME/work}"
ARENA="$WORK/IsaacLab-Arena"
SO101="$WORK/isaaclab-so101"
LEROBOT_VENV="$WORK/.venv-lerobot"
ISAAC_LOCK="$WORK/.isaac.lock"   # one Isaac Sim process at a time (same rule as the laptop)

FORCE_SYNC=false
SKIP_TESTS=false
for arg in "$@"; do
  case "$arg" in
    --force-sync) FORCE_SYNC=true ;;
    --skip-tests) SKIP_TESTS=true ;;
    -h|--help) awk 'NR > 1 && !/^#/ {exit} NR > 1 {sub(/^# ?/, ""); print}' "$0"; exit 0 ;;
    *) echo "setup_brev.sh: unknown option $arg (try --help)" >&2; exit 2 ;;
  esac
done

mkdir -p "$HOME/logs"
LOG="$HOME/logs/setup_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG") 2>&1
step() { printf '\n=== [%s] %s\n' "$(date +%H:%M:%S)" "$*"; }
die() { echo "setup_brev.sh: ERROR: $*" >&2; exit 1; }

for d in "$WORK/so101-contact" "$WORK/patchbench"; do
  [[ -f "$d/pyproject.toml" ]] || die "$d missing: rsync so101-contact and patchbench to $WORK first"
done

step "apt packages"
APT_PKGS=(libglu1-mesa libxt6 ffmpeg tmux git git-lfs rsync curl build-essential)
missing=()
for p in "${APT_PKGS[@]}"; do dpkg -s "$p" >/dev/null 2>&1 || missing+=("$p"); done
if ((${#missing[@]})); then
  sudo DEBIAN_FRONTEND=noninteractive apt-get update -q
  sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -q "${missing[@]}"
else
  echo "all present"
fi

step "uv"
export PATH="$HOME/.local/bin:$PATH"
if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi
uv --version

step "IsaacLab-Arena @ ${ARENA_REV:0:7}"
export GIT_LFS_SKIP_SMUDGE=1   # Arena's LFS files are docs images and test data
if [[ ! -d "$ARENA/.git" ]]; then
  git clone --filter=blob:none --no-checkout https://github.com/isaac-sim/IsaacLab-Arena.git "$ARENA"
fi
if [[ "$(git -C "$ARENA" rev-parse HEAD 2>/dev/null || true)" != "$ARENA_REV" ]]; then
  git -C "$ARENA" fetch -q origin "$ARENA_REV" || true
  git -C "$ARENA" checkout -q "$ARENA_REV"
  FORCE_SYNC=true
fi
# Only the Isaac Lab submodule (Isaac-GR00T is not needed: we use GR00T through LeRobot). .gitmodules uses SSH; the VM
# has no GitHub key, so point it at HTTPS.
if [[ ! -f "$ARENA/submodules/IsaacLab/pyproject.toml" ]]; then
  git -C "$ARENA" submodule init -- submodules/IsaacLab
  git -C "$ARENA" config submodule.submodules/IsaacLab.url https://github.com/isaac-sim/IsaacLab.git
  git -C "$ARENA" submodule update --depth 1 -- submodules/IsaacLab
  FORCE_SYNC=true
fi
got="$(git -C "$ARENA/submodules/IsaacLab" rev-parse HEAD)"
[[ "$got" == "$ISAACLAB_REV" ]] || die "Isaac Lab submodule at $got, expected $ISAACLAB_REV"
echo "Arena $(git -C "$ARENA" rev-parse --short HEAD), Isaac Lab $(git -C "$ARENA/submodules/IsaacLab" rev-parse --short HEAD)"

step "isaaclab-so101 @ ${SO101_REV:0:7}"
if [[ ! -d "$SO101/.git" ]]; then
  git clone --filter=blob:none --no-checkout https://github.com/art-e-fact/isaaclab-so101.git "$SO101"
fi
if [[ "$(git -C "$SO101" rev-parse HEAD 2>/dev/null || true)" != "$SO101_REV" ]]; then
  git -C "$SO101" fetch -q origin "$SO101_REV" || true
  git -C "$SO101" checkout -q "$SO101_REV"
fi
git -C "$SO101" rev-parse --short HEAD

step "uv sync (Arena venv; first run downloads Isaac Sim, ~10 GB)"
if [[ "$FORCE_SYNC" == true || ! -x "$ARENA/.venv/bin/python" ]]; then
  (cd "$ARENA" && uv sync)
else
  echo "venv exists; skipped (use --force-sync to re-sync)"
fi

step "editable installs into the Arena venv (always redone: uv sync removes them)"
uv pip install --python "$ARENA/.venv/bin/python" -q \
  -e "$SO101" -e "$WORK/so101-contact" -e "$WORK/patchbench" pytest
"$ARENA/.venv/bin/python" -c \
  "import isaaclab_so101, so101_contact, patchbench; print('isaaclab_so101, so101_contact', so101_contact.__version__, ', patchbench OK')"

step "LeRobot venv ($LEROBOT_SPEC) at $LEROBOT_VENV"
if ! "$LEROBOT_VENV/bin/python" -c "import importlib.metadata as m, sys; sys.exit(m.version('lerobot') != '0.6.1')" 2>/dev/null; then
  [[ -x "$LEROBOT_VENV/bin/python" ]] || uv venv --python 3.12 "$LEROBOT_VENV"
  uv pip install --python "$LEROBOT_VENV/bin/python" "$LEROBOT_SPEC" pytest
fi
# patchbench's raw-format module is numpy-only; the converter imports it from the repo path.
uv pip install --python "$LEROBOT_VENV/bin/python" -q --no-deps -e "$WORK/patchbench"
"$LEROBOT_VENV/bin/python" -c "import lerobot, importlib.metadata as m; print('lerobot', m.version('lerobot'))"
"$LEROBOT_VENV/bin/python" -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"

step "Hugging Face login"
if [[ -n "${HF_TOKEN:-}" ]]; then
  # Read from the environment inside Python: the token never appears on a command line or in the log.
  "$LEROBOT_VENV/bin/python" - <<'PY'
import os
from huggingface_hub import HfApi, login
login(token=os.environ["HF_TOKEN"], add_to_git_credential=False)
api = HfApi()
print("logged in as", api.whoami()["name"])
for repo in ("nvidia/GR00T-N1.7-3B", "nvidia/Cosmos-Reason2-2B"):
    try:
        api.auth_check(repo)
        print(f"  access OK: {repo}")
    except Exception as e:  # gated and not granted -> GR00T stage would fail later
        print(f"  WARNING no access to {repo}: {type(e).__name__}")
PY
  [[ -n "${HF_USER:-}" ]] || echo "WARNING: HF_USER not set; the chain needs it for dataset/model repo ids"
else
  echo "HF_TOKEN not set: skipped (export HF_TOKEN and HF_USER, then re-run, before starting the chain)"
fi

if [[ "$SKIP_TESTS" == false ]]; then
  step "CPU tests: so101-contact"
  (cd "$WORK/so101-contact" && "$ARENA/.venv/bin/python" -m pytest -q)
  step "CPU tests: patchbench"
  (cd "$WORK/patchbench" && "$ARENA/.venv/bin/python" -m pytest -q)

  step "Arena smoke test (headless, Franka cube_goal_pose, 20 steps; first launch compiles shaders ~3 min)"
  (
    cd "$ARENA"
    export OMNI_KIT_ACCEPT_EULA=YES ACCEPT_EULA=Y
    flock "$ISAAC_LOCK" "$ARENA/.venv/bin/python" isaaclab_arena/evaluation/policy_runner.py \
      --policy_type zero_action --num_steps 20 cube_goal_pose
  )
fi

step "READY"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
nvidia-smi
free -g | head -2
df -h "$HOME" | tail -1
cat <<EOF

  ######################################################################
  #  READY  ($(hostname), $(date '+%a %d %b %H:%M %Z'))
  #  Arena venv:   source $ARENA/.venv/bin/activate
  #  LeRobot venv: source $LEROBOT_VENV/bin/activate
  #  Next: tmux new -s chain; bash $WORK/patchbench/scripts/brev/chain_overnight.sh --dry-run
  #  Log:  $LOG
  ######################################################################
EOF
