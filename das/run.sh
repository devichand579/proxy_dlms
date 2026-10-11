#!/bin/bash
# usage: run.sh <bench args...>   (logs to $OUT_ROOT/logs/<name>.log)
# Copied Strace modules come from das/vendor (or the Strace repo when das sits inside it).
# Only the model weights are external: --model models/llada resolves against MODEL_ROOT.
DAS_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT=$(dirname "$DAS_DIR")
cd "$ROOT"
export PYTHONPATH=$ROOT TRACING_ENABLED=false MODEL_ROOT=${MODEL_ROOT:-/root/Strace}
PY=${PYTHON:-/root/miniconda3/envs/strace/bin/python}
LOGS=${OUT_ROOT:-$ROOT/outputs}/logs
mkdir -p "$LOGS"
name=$(echo "$@" | sed -n 's/.*--name \([^ ]*\).*/\1/p')
exec "$PY" -u -m ${MODULE:-das.bench} "$@" > "$LOGS/$name.log" 2>&1
