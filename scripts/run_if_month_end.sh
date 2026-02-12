#!/usr/bin/env bash
set -euo pipefail

today_day="$(date +%d)"
next_day_month="$(date -d tomorrow +%m)"
this_month="$(date +%m)"
collector_script="${COLLECTOR_SCRIPT:-scripts/collect_stub.py}"
run_marker="${RUN_MARKER_PATH:-artifacts/collector_ran.flag}"

mkdir -p "$(dirname "${run_marker}")"
rm -f "${run_marker}"

run_collector() {
  echo "Running collector: ${collector_script}"
  python "${collector_script}"
  touch "${run_marker}"
  echo "Collector run marker created: ${run_marker}"
}

if [[ "${FORCE_MONTH_END:-0}" == "1" ]]; then
  echo "FORCE_MONTH_END=1 set. Running collector regardless of date."
  run_collector
  exit 0
fi

if [[ "$next_day_month" != "$this_month" ]]; then
  echo "Month-end detected (day=$today_day). Running collector..."
  run_collector
else
  echo "Not month-end (day=$today_day). Skip collector."
fi
