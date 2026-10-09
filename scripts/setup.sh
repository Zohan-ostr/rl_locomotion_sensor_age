#!/usr/bin/env bash
# Install into a project venv only. Never invokes sudo or changes system packages.
set -euo pipefail
PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1
export PIP_CACHE_DIR="$PROJECT_DIR/.cache/pip"
export MPLCONFIGDIR="$PROJECT_DIR/.cache/matplotlib"
mkdir -p "$PIP_CACHE_DIR" "$MPLCONFIGDIR"
OFFLINE=0
case "${1:-}" in
  --offline) OFFLINE=1 ;;
  --help) echo 'Usage: ./scripts/setup.sh [--offline] (PYTHON=python3.10 by default)'; exit 0 ;;
  '') ;;
  *) echo 'Unknown option. Use --help.' >&2; exit 2 ;;
esac
mkdir -p results/setup
LOG="results/setup/setup_$(date -u +%Y%m%dT%H%M%SZ)_$$.log"
exec > >(tee "$LOG") 2>&1
set -x
PYTHON_BIN="${PYTHON:-python3.10}"
if [[ ! -x .venv/bin/python ]]; then
  "$PYTHON_BIN" -c 'import sys; assert sys.version_info[:2] == (3, 10), "Use Python 3.10"'
  "$PYTHON_BIN" -m venv .venv
fi
.venv/bin/python -I -c 'import sys; assert sys.version_info[:2] == (3, 10), "Project venv must use Python 3.10"'
if [[ "$OFFLINE" == 0 ]]; then
  .venv/bin/python -I -m pip install -r requirements.txt
fi
UPSTREAM=vendor/unitree_rl_gym
COMMIT=276801e46c5d433564f24658bac64f254b7d2d4b
if [[ ! -d "$UPSTREAM" ]]; then
  if [[ "$OFFLINE" == 1 ]]; then echo 'Missing upstream; rerun without --offline.'; exit 1; fi
  mkdir -p vendor
  git clone https://github.com/unitreerobotics/unitree_rl_gym.git "$UPSTREAM"
  git -C "$UPSTREAM" checkout --detach "$COMMIT"
fi
# Existing checkouts are verified, never reset or overwritten.
[[ "$(git -C "$UPSTREAM" rev-parse HEAD)" == "$COMMIT" ]] || {
  echo 'Existing upstream uses another commit. Move it aside and rerun.'; exit 1;
}
.venv/bin/python -I -m pip check
.venv/bin/python -I scripts/verify_setup.py
SMOKE_DIR="results/setup/smoke_$(date -u +%Y%m%dT%H%M%SZ)_$$"
.venv/bin/python -I scripts/run_h1_baseline.py --seconds 5 --threads 1 --output "$SMOKE_DIR"
.venv/bin/python -I -c 'import json,sys; from pathlib import Path; s=json.loads((Path(sys.argv[1])/"summary.json").read_text()); assert s["fall_time_s"] is None and s["displacement_xy_m"][0] > 1, "Baseline did not walk successfully"' "$SMOKE_DIR"
set +x
echo "Готово. Лог: $LOG"
echo 'Интерактивный запуск: ./scripts/launch_h1_sandbox.sh'
