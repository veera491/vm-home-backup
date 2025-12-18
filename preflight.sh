#!/usr/bin/env bash
# Preflight check for metrics.py
# Output: YES or NO
# If NO, prints actionable problems

set -u

FAIL=0
PROBLEMS=()

need_free_root_gb=5
need_free_mnt_gb=5

# Provide safe defaults if env vars are missing
OUT_ROOT="${OUT_ROOT:-/mnt/veera/experiments}"
HF_HOME="${HF_HOME:-/mnt/veera/hf_home}"
PETALS_CACHE_DIR="${PETALS_CACHE_DIR:-/mnt/veera/petals_cache}"

# ---------- helpers ----------
gb_free() {
  df -BG "$1" | awk 'NR==2 {gsub("G","",$4); print $4}'
}

check_expected() {
  local name="$1"
  local actual="$2"
  local expected="$3"
  if [[ "$actual" != "$expected" ]]; then
    FAIL=1
    PROBLEMS+=("ENV $name='$actual' (expected '$expected')")
  fi
}

# ---------- checks ----------

# 1) Required env vars (we validate values, but we also safely default above)
check_expected "OUT_ROOT" "$OUT_ROOT" "/mnt/veera/experiments"
check_expected "HF_HOME" "$HF_HOME" "/mnt/veera/hf_home"

# TRANSFORMERS_CACHE must be unset or empty
if [[ -n "${TRANSFORMERS_CACHE:-}" ]]; then
  FAIL=1
  PROBLEMS+=("ENV TRANSFORMERS_CACHE is set ('${TRANSFORMERS_CACHE}') — should be unset")
fi

# PETALS_CACHE_DIR is recommended; warn only
if [[ -z "${PETALS_CACHE_DIR:-}" ]]; then
  PROBLEMS+=("WARN PETALS_CACHE_DIR not set (recommended)")
fi

# 2) Disk space
root_free=$(gb_free /)
mnt_free=$(gb_free /mnt)

if (( root_free < need_free_root_gb )); then
  FAIL=1
  PROBLEMS+=("DISK / free ${root_free}GB < ${need_free_root_gb}GB")
fi

if (( mnt_free < need_free_mnt_gb )); then
  FAIL=1
  PROBLEMS+=("DISK /mnt free ${mnt_free}GB < ${need_free_mnt_gb}GB")
fi

# 3) OUT_ROOT exists and writable
if [[ ! -d "$OUT_ROOT" ]]; then
  FAIL=1
  PROBLEMS+=("PATH OUT_ROOT does not exist: $OUT_ROOT")
else
  touch "$OUT_ROOT/.preflight_write_test" 2>/dev/null || {
    FAIL=1
    PROBLEMS+=("PERM Cannot write to OUT_ROOT: $OUT_ROOT")
  }
  rm -f "$OUT_ROOT/.preflight_write_test" 2>/dev/null || true
fi

# 4) Ensure no experiments on root
if [[ -d "$HOME/vm-project/experiments" ]]; then
  FAIL=1
  PROBLEMS+=("DATA experiments directory exists on root: ~/vm-project/experiments")
fi

# 5) Ensure metrics.py uses OUT_ROOT
if ! grep -q 'OUT_ROOT = os.environ.get("OUT_ROOT"' metrics.py 2>/dev/null; then
  FAIL=1
  PROBLEMS+=("CODE metrics.py does not read OUT_ROOT from env")
fi

# ---------- result ----------
if (( FAIL == 0 )); then
  echo "YES"
  exit 0
else
  echo "NO"
  echo "Problems found:"
  for p in "${PROBLEMS[@]}"; do
    echo " - $p"
  done
  exit 1
fi
