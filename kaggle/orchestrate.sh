#!/usr/bin/env bash
# Pipeline manager: runs every Kaggle stage in order, downloads outputs, logs progress.
# Usage: bash kaggle/orchestrate.sh <stage>   stages: main | wave2
set -u
cd "$(dirname "$0")/.."
K=./.venv/Scripts/kaggle
U=samiulalom01
LOG=logs/orchestrate.log
mkdir -p logs results/kaggle
log() { echo "$(date '+%F %T') $*" | tee -a "$LOG"; }

status() { $K kernels status "$U/$1" 2>&1 | sed 's/.*status "//;s/".*//'; }

wait_done() {  # wait until kernel is no longer queued/running; echo final status
  local k=$1 s
  while true; do
    s=$(status "$k")
    case "$s" in *RUNNING*|*QUEUED*) sleep 120;; *) log "$k -> $s"; echo "$s"; return;; esac
  done
}

push() {  # push a kernel, retrying while Kaggle refuses (e.g. concurrent GPU session limit)
  local k=$1 out
  for i in $(seq 1 60); do
    out=$($K kernels push -p "E:\\Claude\\ecg-paper\\kaggle\\build\\$k" 2>&1)
    if echo "$out" | grep -q "successfully pushed"; then log "pushed $k"; return 0; fi
    log "push $k refused (try $i): $(echo "$out" | tail -1)"; sleep 300
  done
  return 1
}

fetch() {
  local k=$1
  mkdir -p "results/kaggle/$k"
  $K kernels output "$U/$k" -p "E:\\Claude\\ecg-paper\\results\\kaggle\\$k" -o >/dev/null 2>&1
  log "fetched $k: $(ls results/kaggle/$k/results/*.json 2>/dev/null | wc -l) result files"
}

case "${1:-main}" in
  main)
    s=$(wait_done ecg-prep)
    case "$s" in *COMPLETE*) ;; *) log "prep failed: $s"; exit 2;; esac
    $K kernels output "$U/ecg-prep" -p "E:\\Claude\\ecg-paper\\results\\kaggle\\prep" -o --file-pattern ".*\.(csv|log|txt)$" >/dev/null 2>&1
    log "prep outputs fetched (csv/log only)"
    push ecg-run-main-a && push ecg-run-main-b
    wait_done ecg-run-main-a >/dev/null; fetch ecg-run-main-a
    wait_done ecg-run-main-b >/dev/null; fetch ecg-run-main-b
    log "STAGE main DONE"
    ;;
  wave2)
    push ecg-run-e4 && push ecg-merl
    wait_done ecg-run-e4 >/dev/null; fetch ecg-run-e4
    push ecg-run-seeds
    wait_done ecg-merl >/dev/null; fetch ecg-merl
    wait_done ecg-run-seeds >/dev/null; fetch ecg-run-seeds
    log "STAGE wave2 DONE"
    ;;
esac
