#!/usr/bin/env bash
# Run an original experiment of this repository with the defaults in configs/.
#   ./run_original.sh             window experiment, full supervision
#   ./run_original.sh sparse      window experiment, sparse supervision
#   ./run_original.sh baseline    pure EWAM baseline
# More flags go to `python -m rsi_framework run`, e.g. ./run_original.sh --layout 11
# First run renders configs/*.local.json from the templates; a missing EWAM service is
# started and stopped afterwards. Change the task in configs/envs/robodojo.json.
set -euo pipefail
cd "$(dirname "$0")"
export PROJECTS_ROOT="${PROJECTS_ROOT:-$PWD/projects}"
export EWAM_PYTHON="${EWAM_PYTHON:-$PROJECTS_ROOT/ewam_comparison/.venv-ewam/bin/python}"
if [ ! -x "$EWAM_PYTHON" ]; then
  echo "EWAM python not found at $EWAM_PYTHON; set EWAM_PYTHON (see README.md)" >&2
  exit 1
fi
mode=full
if [ $# -gt 0 ] && [[ "$1" != -* ]]; then mode=$1; shift; fi
case "$mode" in
  full)     set -- --env robodojo --vla ewam "$@" ;;
  sparse)   set -- --env robodojo --vla ewam --protocol window_sparse "$@" ;;
  baseline) set -- --env robodojo --vla ewam --controller none "$@" ;;
  *)        echo "usage: $0 [full|sparse|baseline] [flags]" >&2; exit 2 ;;
esac
exec "$EWAM_PYTHON" -m rsi_framework run "$@"
