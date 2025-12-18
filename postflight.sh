#!/usr/bin/env bash
# Postflight check for metrics.py results
# Usage: ./postflight.sh <NUM_VMS>
# Output: YES or NO + problems

set -u

if [[ $# -ne 1 ]]; then
  echo "NO"
  echo "Problems found:"
  echo " - Usage: $0 <NUM_VMS>"
  exit 1
fi

NUM_VMS="$1"
OUT_ROOT="${OUT_ROOT:-/mnt/veera/experiments}"

FAIL=0
PROBLEMS=()

need_free_root_gb=3

gb_free() {
  df -BG "$1" | awk 'NR==2 {gsub("G","",$4); print $4}'
}

exists_nonempty() {
  local f="$1"
  if [[ ! -f "$f" ]]; then
    FAIL=1
    PROBLEMS+=("MISSING file: $f")
    return
  fi
  if [[ ! -s "$f" ]]; then
    FAIL=1
    PROBLEMS+=("EMPTY file: $f")
    return
  fi
}

# 1) Root disk still healthy
root_free=$(gb_free /)
if (( root_free < need_free_root_gb )); then
  FAIL=1
  PROBLEMS+=("DISK / free ${root_free}GB < ${need_free_root_gb}GB after run")
fi

# 2) Ensure results were written to OUT_ROOT (not to ~/vm-project/experiments on /)
if [[ -d "$HOME/vm-project/experiments" ]]; then
  FAIL=1
  PROBLEMS+=("DATA results found on root: ~/vm-project/experiments (should be under $OUT_ROOT)")
fi

# 3) Check expected result files
modes=("sequential" "pipeline" "microbatch")
for mode in "${modes[@]}"; do
  f="$OUT_ROOT/$mode/Results_VM_${NUM_VMS}.csv"
  exists_nonempty "$f"

  # Check header includes core columns
  if [[ -f "$f" ]]; then
    header=$(head -n 1 "$f")
    for col in "Pipeline_Method" "No_Of_VMs" "Response_Time_ms" "Throughput_tok_s"; do
      if ! echo "$header" | grep -q "$col"; then
        FAIL=1
        PROBLEMS+=("CSV header missing '$col' in $f")
      fi
    done

    # Check at least 2 lines (header + >=1 row)
    lines=$(wc -l < "$f" | tr -d ' ')
    if (( lines < 2 )); then
      FAIL=1
      PROBLEMS+=("CSV has no data rows (lines=$lines): $f")
    fi
  fi
done

# 4) Optional: if you changed to JSON-per-mode agent metrics, verify presence
# (This is a WARN only, not a FAIL)
for mode in "${modes[@]}"; do
  j="$OUT_ROOT/$mode/AgentMetrics_VM_${NUM_VMS}_${mode}.json"
  if [[ ! -f "$j" ]]; then
    PROBLEMS+=("WARN agent metrics JSON missing (ok if not implemented): $j")
  fi
done

# 5) Print file sizes (informational)
SIZES=()
for mode in "${modes[@]}"; do
  f="$OUT_ROOT/$mode/Results_VM_${NUM_VMS}.csv"
  if [[ -f "$f" ]]; then
    SIZES+=("SIZE $mode: $(du -h "$f" | awk '{print $1}') -> $f")
  fi
done

# Result
if (( FAIL == 0 )); then
  echo "YES"
  for s in "${SIZES[@]}"; do echo "$s"; done
  exit 0
else
  echo "NO"
  echo "Problems found:"
  for p in "${PROBLEMS[@]}"; do
    # show WARN lines too, but they don't necessarily mean failure
    echo " - $p"
  done
  for s in "${SIZES[@]}"; do echo "$s"; done
  exit 1
fi
