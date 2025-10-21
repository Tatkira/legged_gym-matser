#!/usr/bin/env bash
set -euo pipefail

# run_sim2sim.sh
# Safe launcher for sim2sim.py:
# - clears common ROS2 env vars to avoid mixing ROS1/ROS2
# - sources ROS1 (Noetic) setup
# - activates conda environment 'xdy' (adjust if your env name differs)
# - forces conda libstdc++ to be loaded first to avoid GLIBCXX issues
# - runs sim2sim.py with any additional args forwarded

echo "[run_sim2sim] Clearing ROS2-related environment variables to avoid distro mixing..."
unset ROS_DISTRO ROS_VERSION AMENT_PREFIX_PATH COLCON_PREFIX_PATH RMW_IMPLEMENTATION ROS_DOMAIN_ID ROS_ROOT ROS_PACKAGE_PATH || true

echo "[run_sim2sim] Sourcing ROS1 (Noetic) setup..."
if [ -f "/opt/ros/noetic/setup.bash" ]; then
    # Some ROS setup scripts assume certain vars exist; temporarily ensure ROS_DISTRO
    export ROS_DISTRO=noetic
    # disable nounset while sourcing to avoid failing on scripts that reference unset vars
    set +u
    # shellcheck disable=SC1091
    source /opt/ros/noetic/setup.bash
    set -u
else
    echo "/opt/ros/noetic/setup.bash not found. Please install or adjust the path." >&2
    exit 1
fi

echo "[run_sim2sim] Initializing conda..."
# Ensure conda is available in non-interactive shells
if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    # shellcheck disable=SC1091
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
elif [ -f "/opt/conda/etc/profile.d/conda.sh" ]; then
    # shellcheck disable=SC1091
    source "/opt/conda/etc/profile.d/conda.sh"
else
    echo "conda initialization script not found under HOME or /opt. Make sure conda is installed." >&2
fi

CONDA_ENV_NAME="xdy"
echo "[run_sim2sim] Activating conda environment: ${CONDA_ENV_NAME}"
conda activate "${CONDA_ENV_NAME}"

echo "[run_sim2sim] Ensuring conda's libstdc++ is used to avoid GLIBCXX errors..."
if [ -n "${CONDA_PREFIX-}" ] && [ -f "${CONDA_PREFIX}/lib/libstdc++.so.6" ]; then
    export LD_PRELOAD="${CONDA_PREFIX}/lib/libstdc++.so.6${LD_PRELOAD+-}${LD_PRELOAD-}" || true
    export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH-}"
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PYTHON:-python}"

echo "[run_sim2sim] Running sim2sim.py (cwd=${SCRIPT_DIR})"
# Ensure we do NOT force a particular MuJoCo GL backend here; allow the environment or sim2sim.py
# to decide. Unset MUJOCO_GL if it exists so the launcher stays "pure" and does not impose GLFW.
unset MUJOCO_GL || true
exec "$PY" "$SCRIPT_DIR/sim2sim.py" "$@"
