#!/usr/bin/env bash
# agent_smoke_run.sh
# 用于 AI 代理或开发者的快速 smoke-run / 环境检查脚本。
# 放在仓库 scripts/ 下。支持 --dry-run（只打印将执行的命令）与 --run（实际执行）

set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DRY_RUN=false
DO_RUN=false
PYTHON_EXEC=python

usage(){
  cat <<EOF
Usage: $0 [--dry-run] [--run] [--task TASK]

Options:
  --dry-run      Print the sequence of commands the script would run (default)
  --run          Actually run the smoke steps (may execute pip installs / python)
  --task TASK    Task name to use for smoke run (default: anymal_c_flat)
EOF
}

TASK="anymal_c_flat"
while [ $# -gt 0 ]; do
  case "$1" in
    --dry-run) DRY_RUN=true ; shift ;;
    --run) DO_RUN=true ; DRY_RUN=false ; shift ;;
    --task) TASK="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown arg: $1"; usage; exit 2 ;;
  esac
done

# Helper: echo or run
run_cmd(){
  if [ "$DRY_RUN" = true ]; then
    echo "+ $*"
  else
    echo "-> $*"
    eval "$@"
  fi
}

# 1) Basic checks
run_cmd echo "ROOT_DIR=$ROOT_DIR"
run_cmd echo "Checking existence of key files..."
run_cmd test -d "$ROOT_DIR/legged_gym" || echo "Warning: legged_gym/ not found at $ROOT_DIR/legged_gym"
run_cmd test -f "$ROOT_DIR/legged_gym/scripts/train.py" || echo "Warning: train.py not found"
run_cmd test -f "$ROOT_DIR/legged_gym/scripts/play.py" || echo "Warning: play.py not found"

# 2) Optional: pip install editable (local)
run_cmd echo "Install local package in editable mode (recommended in dev env):"
run_cmd echo "pip install -e $ROOT_DIR/legged_gym"

# 3) Optional: install rsl_rl (if present). If --run provided, perform install; otherwise print steps.
if [ -d "$ROOT_DIR/rsl_rl" ]; then
  run_cmd echo "rsl_rl found — example install command:"
  if [ "$DO_RUN" = true ]; then
    # actually install
    run_cmd bash -c "cd $ROOT_DIR/rsl_rl && git checkout v1.0.2 || true && pip install -e ."
  else
    # dry-run prints the command to run
    run_cmd echo "cd $ROOT_DIR/rsl_rl && git checkout v1.0.2 || true && pip install -e ."
  fi
else
  if [ "$DO_RUN" = true ]; then
    # when running, clone and install (use https remote placeholder)
    run_cmd echo "rsl_rl not present: cloning and installing from origin (example):"
    run_cmd bash -c "git clone https://github.com/example/rsl_rl.git $ROOT_DIR/rsl_rl && cd $ROOT_DIR/rsl_rl && git checkout v1.0.2 || true && pip install -e ."
  else
    run_cmd echo "rsl_rl not present in repo root; ensure external dependency is installed as README suggests. Example: git clone https://github.com/example/rsl_rl.git && pip install -e rsl_rl"
  fi
fi

# 4) Smoke training run (num_envs=1, short)
run_cmd echo "Smoke training command (small, headless, short):"
TRAIN_CMD="${PYTHON_EXEC} $ROOT_DIR/legged_gym/scripts/train.py --task=${TASK} --num_envs=1 --headless --max_iterations=10"
run_cmd echo "$TRAIN_CMD"

# 5) Smoke playback
run_cmd echo "Smoke playback command (headless):"
PLAY_CMD="${PYTHON_EXEC} $ROOT_DIR/legged_gym/scripts/play.py --task=${TASK} --headless"
run_cmd echo "$PLAY_CMD"

# 6) sim2sim dry-run (if script exists)
if [ -f "$ROOT_DIR/legged_gym/scripts/run_sim2sim.sh" ]; then
  run_cmd echo "sim2sim script exists — example: $ROOT_DIR/legged_gym/scripts/run_sim2sim.sh"
  run_cmd echo "cd $ROOT_DIR/legged_gym/scripts && ./run_sim2sim.sh --help || true"
else
  run_cmd echo "run_sim2sim.sh not found in legged_gym/scripts"
fi

# Final note
run_cmd echo "Smoke-run script finished. If running with --run, check outputs/logs under legged_gym/runs and onnx/ as needed."

if [ "$DRY_RUN" = true ]; then
  echo "(Note: this was a dry-run. Re-run with --run to execute commands.)"
fi
