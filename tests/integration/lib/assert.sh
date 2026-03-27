#!/usr/bin/env bash
# Lightweight assertion helpers for shell-based integration tests

PASS=0
FAIL=0
SKIP=0

pass() {
  echo "  ✓ $1"
  PASS=$((PASS + 1))
}

fail() {
  echo "  ✗ $1"
  echo "    Expected: $2"
  echo "    Got:      $3"
  FAIL=$((FAIL + 1))
}

skip() {
  echo "  ○ SKIP: $1"
  SKIP=$((SKIP + 1))
}

assert_contains() {
  local description="$1"
  local haystack="$2"
  local needle="$3"
  if echo "$haystack" | grep -qF "$needle"; then
    pass "$description"
  else
    fail "$description" "contains '$needle'" "not found in output"
  fi
}

assert_not_contains() {
  local description="$1"
  local haystack="$2"
  local needle="$3"
  if echo "$haystack" | grep -qF "$needle"; then
    fail "$description" "does not contain '$needle'" "found in output"
  else
    pass "$description"
  fi
}

assert_json_path() {
  local description="$1"
  local json="$2"
  local path="$3"
  local expected="$4"
  local actual
  actual=$(echo "$json" | python3 -c "
import sys, json
try:
    data = json.load(sys.stdin)
    parts = '$path'.split('.')
    val = data
    for p in parts:
        if p.isdigit():
            val = val[int(p)]
        else:
            val = val[p]
    print(val)
except Exception as e:
    print('ERROR: ' + str(e))
" 2>/dev/null)
  if [ "$actual" = "$expected" ]; then
    pass "$description"
  else
    fail "$description" "$expected" "$actual"
  fi
}

assert_json_count_gte() {
  local description="$1"
  local json="$2"
  local path="$3"
  local min="$4"
  local actual
  actual=$(echo "$json" | python3 -c "
import sys, json
try:
    data = json.load(sys.stdin)
    parts = '$path'.split('.')
    val = data
    for p in parts:
        val = val[p]
    print(len(val))
except Exception as e:
    print(0)
" 2>/dev/null)
  if [ "$actual" -ge "$min" ] 2>/dev/null; then
    pass "$description (got $actual >= $min)"
  else
    fail "$description" ">= $min items" "$actual items"
  fi
}

print_summary() {
  echo ""
  echo "Results: ${PASS} passed, ${FAIL} failed, ${SKIP} skipped"
  if [ "$FAIL" -gt 0 ]; then
    return 1
  fi
  return 0
}
