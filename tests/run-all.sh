#!/usr/bin/env bash
# Runs the full test suite: unit tests + integration tests
#
# Usage:
#   bash tests/run-all.sh              # all tests
#   bash tests/run-all.sh --unit-only  # skip integration tests
#   bash tests/run-all.sh --no-live    # skip live site tests
#   SKIP_LIVE=1 bash tests/run-all.sh  # same as --no-live

set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

UNIT_ONLY=0
SKIP_LIVE="${SKIP_LIVE:-0}"

for arg in "$@"; do
  case "$arg" in
    --unit-only) UNIT_ONLY=1 ;;
    --no-live)   SKIP_LIVE=1 ;;
  esac
done

TOTAL_PASS=0
TOTAL_FAIL=0

run_step() {
  local name="$1"
  shift
  echo ""
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  echo " $name"
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  if "$@"; then
    TOTAL_PASS=$((TOTAL_PASS + 1))
    return 0
  else
    TOTAL_FAIL=$((TOTAL_FAIL + 1))
    return 1
  fi
}

cd "$ROOT_DIR"

# ─── Step 0: Install deps if needed ───────────────────────────────────────
if [ ! -d node_modules ]; then
  echo "Installing dependencies..."
  npm install --silent
fi

# ─── Step 1: Unit tests (Vitest, no browser needed) ───────────────────────
run_step "Unit Tests (detector.js layers)" \
  npx vitest run tests/unit --reporter=verbose

if [ "$UNIT_ONLY" = "1" ]; then
  echo ""
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  echo " Skipping integration tests (--unit-only)"
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  echo ""
  echo "Final: $TOTAL_PASS step(s) passed, $TOTAL_FAIL failed"
  [ "$TOTAL_FAIL" -eq 0 ]
  exit $?
fi

# ─── Step 2: Download fixtures if missing ─────────────────────────────────
AI_FIXTURES_DIR="$ROOT_DIR/tests/fixtures/images/ai"
if [ ! -f "$AI_FIXTURES_DIR/chatgpt-image.png" ]; then
  echo ""
  echo "Downloading fixture images (first run)..."
  node tests/fixtures/download-fixtures.js
fi

# ─── Step 3: Start fixture server ─────────────────────────────────────────
echo ""
echo "Starting fixture server on :3456..."
npx serve tests/fixtures -l 3456 --cors --no-clipboard &
SERVER_PID=$!

# Wait for server to be ready
for i in $(seq 1 10); do
  if curl -s http://localhost:3456/ >/dev/null 2>&1; then
    echo "  Fixture server ready"
    break
  fi
  sleep 0.5
done

cleanup_server() {
  kill "$SERVER_PID" 2>/dev/null || true
}
trap cleanup_server EXIT

# ─── Step 4: Integration tests ────────────────────────────────────────────
run_step "Integration: Extension loads" \
  bash tests/integration/test-extension-loads.sh

run_step "Integration: Badge accuracy" \
  bash tests/integration/test-badge-accuracy.sh

if [ "$SKIP_LIVE" = "0" ]; then
  run_step "Integration: Live sites" \
    bash tests/integration/test-live-sites.sh
else
  echo ""
  echo "  Skipping live site tests (SKIP_LIVE=1)"
fi

# ─── Summary ──────────────────────────────────────────────────────────────
echo ""
echo "════════════════════════════════════════"
echo " Final: $TOTAL_PASS step(s) passed, $TOTAL_FAIL failed"
echo "════════════════════════════════════════"
[ "$TOTAL_FAIL" -eq 0 ]
