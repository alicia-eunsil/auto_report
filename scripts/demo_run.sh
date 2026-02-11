#!/usr/bin/env bash
set -euo pipefail

mkdir -p artifacts

{
  echo "[1/3] run collector"
  python scripts/collect_stub.py

  echo "[2/3] validate latest snapshot"
  python scripts/validate_snapshot.py

  echo "[3/3] month-end gate dry run"
  FORCE_MONTH_END=1 bash scripts/run_if_month_end.sh
} | tee artifacts/demo_run.log

echo "Saved log: artifacts/demo_run.log"
