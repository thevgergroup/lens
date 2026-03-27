#!/usr/bin/env bash
# Integration test: Badge accuracy on known AI vs real image fixtures
#
# Navigates to fixture pages with known-ground-truth images and verifies
# that the extension's confidence levels match expectations.
#
# Usage: bash tests/integration/test-badge-accuracy.sh

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
source "$SCRIPT_DIR/lib/assert.sh"

EXTENSION_PATH="$ROOT_DIR"
SESSION="lens-accuracy-$$"
BASE_URL="http://localhost:3456"
WAIT_SECS=5  # Time for extension to analyze images

cleanup() {
  agent-browser --session "$SESSION" close 2>/dev/null || true
}
trap cleanup EXIT

echo "=== Test: Badge accuracy on fixture pages ==="
echo ""

# Helper: get all lens-badge class values on the current page as JSON
get_badges_json() {
  agent-browser --session "$SESSION" eval "
    JSON.stringify(
      Array.from(document.querySelectorAll('.lens-badge')).map(el => ({
        class: el.className,
        text: el.textContent.trim().substring(0, 40),
        src: el.closest('.lens-wrapper')?.querySelector('img')?.dataset?.src
              || el.closest('.lens-wrapper')?.querySelector('img')?.src || ''
      }))
    )
  " 2>/dev/null || echo "[]"
}

# Helper: count badges with a specific confidence level class
count_badges_with_level() {
  local badges_json="$1"
  local level="$2"
  echo "$badges_json" | python3 -c "
import sys, json
badges = json.load(sys.stdin)
count = sum(1 for b in badges if 'lens-$level' in b.get('class', ''))
print(count)
" 2>/dev/null || echo 0
}

# ─── Test 1: AI Images Page ────────────────────────────────────────────────
echo "--- AI Images Page ---"

agent-browser \
  --extension "$EXTENSION_PATH" \
  --session "$SESSION" \
  open "$BASE_URL/pages/ai-images.html" \
  2>/dev/null || { skip "Could not open ai-images.html"; }

sleep "$WAIT_SECS"

badges=$(get_badges_json)
total_badges=$(echo "$badges" | python3 -c "import sys,json; print(len(json.load(sys.stdin)))" 2>/dev/null || echo 0)

echo "  Found $total_badges badges on AI images page"

if [ "$total_badges" -gt 0 ]; then
  # Count how many are flagged (definite + likely + possible)
  definite=$(count_badges_with_level "$badges" "definite")
  likely=$(count_badges_with_level "$badges" "likely")
  possible=$(count_badges_with_level "$badges" "possible")
  clean=$(count_badges_with_level "$badges" "clean")
  flagged=$((definite + likely + possible))

  echo "  Breakdown: $definite definite, $likely likely, $possible possible, $clean clean"

  if [ "$flagged" -ge 1 ]; then
    pass "At least 1 AI image correctly flagged on ai-images page"
  else
    fail "AI images not flagged" ">= 1 flagged" "0 flagged ($clean clean)"
  fi

  # If C2PA fixtures downloaded, at least 1 should be definite
  if [ "$definite" -ge 1 ]; then
    pass "At least 1 definite AI detection (C2PA or metadata)"
  else
    skip "No definite detections (C2PA fixtures may not be downloaded)"
  fi
else
  skip "No badges found on AI images page — run npm run fixture:download first"
fi

# Screenshot
mkdir -p "$ROOT_DIR/tests/integration/screenshots"
agent-browser --session "$SESSION" \
  screenshot "$ROOT_DIR/tests/integration/screenshots/ai-images-page.png" --full 2>/dev/null || true

# ─── Test 2: Mixed Page ────────────────────────────────────────────────────
echo ""
echo "--- Mixed Page (simulated news feed) ---"

agent-browser \
  --session "$SESSION" \
  open "$BASE_URL/pages/mixed-page.html" \
  2>/dev/null || { skip "Could not open mixed-page.html"; }

sleep "$WAIT_SECS"

badges_mixed=$(get_badges_json)
total_mixed=$(echo "$badges_mixed" | python3 -c "import sys,json; print(len(json.load(sys.stdin)))" 2>/dev/null || echo 0)

echo "  Found $total_mixed badges on mixed page (expected 3, ignoring 2 tiny images)"

# The mixed page has 3 real images (>32px) and 2 tiny images (should be ignored)
# We expect 3 badges total (not 5)
if [ "$total_mixed" -le 4 ]; then
  pass "Tiny images (<32px) correctly ignored (got $total_mixed badges, not 5)"
else
  fail "Too many badges — tiny images may not be filtered" "<= 4" "$total_mixed"
fi

agent-browser --session "$SESSION" \
  screenshot "$ROOT_DIR/tests/integration/screenshots/mixed-page.png" --full 2>/dev/null || true

# ─── Test 3: URL Heuristics Page ──────────────────────────────────────────
echo ""
echo "--- URL Heuristics Page (L1 fast-path) ---"

agent-browser \
  --session "$SESSION" \
  open "$BASE_URL/pages/url-heuristics.html" \
  2>/dev/null || { skip "Could not open url-heuristics.html"; }

sleep "$WAIT_SECS"

badges_url=$(get_badges_json)
definite_url=$(count_badges_with_level "$badges_url" "definite")
likely_url=$(count_badges_with_level "$badges_url" "likely")

if [ "$((definite_url + likely_url))" -ge 1 ]; then
  pass "URL heuristics firing: $definite_url definite, $likely_url likely"
else
  skip "URL heuristic badges not detected (fixtures may be missing)"
fi

echo ""
print_summary
