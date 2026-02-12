#!/usr/bin/env bash
set -euo pipefail

mkdir -p artifacts

{
  echo "[1/4] run collector"
  python scripts/collect_stub.py

  echo "[2/4] validate latest snapshot"
  python scripts/validate_snapshot.py

  echo "[3/4] month-end gate dry run"
  FORCE_MONTH_END=1 bash scripts/run_if_month_end.sh

  echo "[4/4] build cumulative outputs and report"
  python scripts/postprocess_data.py
} | tee artifacts/demo_run.log

echo "Saved log: artifacts/demo_run.log"
