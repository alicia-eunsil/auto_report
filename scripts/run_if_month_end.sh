#!/usr/bin/env bash
set -euo pipefail

today_day="$(date +%d)"
next_day_month="$(date -d tomorrow +%m)"
this_month="$(date +%m)"
collector_script="${COLLECTOR_SCRIPT:-scripts/collect_stub.py}"

run_collector() {
  echo "Running collector: ${collector_script}"
  python "${collector_script}"
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
