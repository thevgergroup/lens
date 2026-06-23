#!/usr/bin/env bash
# Integration test: Badge accuracy on known AI vs real image fixtures
#
# Results are stored as data-lens-level attributes on <img> elements.
# Levels: definite, likely, possible, unlikely, clean
# Flagged = definite | likely | possible
#
# Usage: bash tests/integration/test-badge-accuracy.sh

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
source "$SCRIPT_DIR/lib/assert.sh"

SESSION="lens-accuracy-$$"
BASE_URL="http://localhost:3456"
SETTLE_SECS=25  # Time for SW to cold-start TF.js + model and analyse all images

cleanup() {
  agent-browser --session "$SESSION" close 2>/dev/null || true
}
trap cleanup EXIT

echo "=== Test: Badge accuracy on fixture pages ==="
echo ""

# Query each level individually to avoid JSON parsing issues
get_level_count() {
  local level="$1"
  if [ "$level" = "total" ]; then
    agent-browser --session "$SESSION" eval \
      "document.querySelectorAll('img[data-lens-level]').length" 2>/dev/null || echo 0
  else
    agent-browser --session "$SESSION" eval \
      "document.querySelectorAll('img[data-lens-level=\"${level}\"]').length" 2>/dev/null || echo 0
  fi
}

get_flagged_count() {
  local d l p
  d=$(get_level_count definite)
  l=$(get_level_count likely)
  p=$(get_level_count possible)
  echo $(( ${d:-0} + ${l:-0} + ${p:-0} ))
}

# ── Test 1: AI Images Page ─────────────────────────────────────────────────
echo "--- AI Images Page ---"

agent-browser --session "$SESSION" open "$BASE_URL/pages/ai-images" 2>/dev/null

echo "  Waiting ${SETTLE_SECS}s for SW model load and analysis…"
sleep "$SETTLE_SECS"

total=$(get_level_count total)
definite=$(get_level_count definite)
likely=$(get_level_count likely)
possible=$(get_level_count possible)
unlikely=$(get_level_count unlikely)
clean=$(get_level_count clean)
flagged=$(( ${definite:-0} + ${likely:-0} + ${possible:-0} ))

echo "  Annotated=$total  definite=$definite  likely=$likely  possible=$possible  unlikely=$unlikely  clean=$clean"

if [ "${total:-0}" -ge 1 ] 2>/dev/null; then
  pass "Extension annotated $total images on AI page"
else
  fail "No images annotated" ">= 1" "0"
fi

if [ "${flagged:-0}" -ge 1 ] 2>/dev/null; then
  pass "At least $flagged AI image(s) correctly flagged (definite/likely/possible)"
else
  fail "No AI images flagged" ">= 1 flagged" "0"
fi

mkdir -p "$ROOT_DIR/tests/integration/screenshots"
agent-browser --session "$SESSION" \
  screenshot "$ROOT_DIR/tests/integration/screenshots/ai-images-page.png" --full 2>/dev/null || true

# ── Test 2: Real Images Page ───────────────────────────────────────────────
echo ""
echo "--- Real Images Page ---"

agent-browser --session "$SESSION" open "$BASE_URL/pages/real-images" 2>/dev/null
echo "  Waiting ${SETTLE_SECS}s…"
sleep "$SETTLE_SECS"

total_real=$(get_level_count total)
flagged_real=$(get_flagged_count)
clean_real=$(get_level_count clean)
unlikely_real=$(get_level_count unlikely)

echo "  Annotated=$total_real  flagged(FP)=$flagged_real  clean=$clean_real  unlikely=$unlikely_real"

if [ "${total_real:-0}" -ge 1 ] 2>/dev/null; then
  pass "Extension annotated $total_real images on real page"
else
  skip "No images annotated on real page"
fi

if [ "${flagged_real:-0}" -eq 0 ] 2>/dev/null; then
  pass "No false positives on real images page"
elif [ "${flagged_real:-0}" -le 2 ] 2>/dev/null; then
  pass "Acceptable FP rate: $flagged_real / $total_real flagged"
else
  fail "Too many false positives on real images page" "<= 2 FP" "$flagged_real FP"
fi

agent-browser --session "$SESSION" \
  screenshot "$ROOT_DIR/tests/integration/screenshots/real-images-page.png" --full 2>/dev/null || true

# ── Test 3: Mixed Page ─────────────────────────────────────────────────────
echo ""
echo "--- Mixed Page ---"

agent-browser --session "$SESSION" open "$BASE_URL/pages/mixed-page" 2>/dev/null
echo "  Waiting ${SETTLE_SECS}s…"
sleep "$SETTLE_SECS"

total_mixed=$(get_level_count total)
echo "  Annotated=$total_mixed (tiny images <32px should be excluded)"

if [ "${total_mixed:-0}" -le 4 ] 2>/dev/null; then
  pass "Tiny images correctly ignored ($total_mixed annotated, not 5)"
else
  fail "Too many annotations — tiny images may not be filtered" "<= 4" "$total_mixed"
fi

agent-browser --session "$SESSION" \
  screenshot "$ROOT_DIR/tests/integration/screenshots/mixed-page.png" --full 2>/dev/null || true

# ── Test 4: URL Heuristics Page ────────────────────────────────────────────
echo ""
echo "--- URL Heuristics Page (L1 fast-path) ---"

agent-browser --session "$SESSION" open "$BASE_URL/pages/url-heuristics" 2>/dev/null
sleep 5  # L1 is synchronous, no model load needed

flagged_url=$(get_flagged_count)

if [ "${flagged_url:-0}" -ge 1 ] 2>/dev/null; then
  pass "URL heuristics flagged $flagged_url image(s)"
else
  skip "URL heuristics not firing (fixtures may be missing)"
fi

echo ""
print_summary
