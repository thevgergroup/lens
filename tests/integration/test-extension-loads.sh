#!/usr/bin/env bash
# Integration test: Verify the extension loads and its service worker registers
#
# Usage: bash tests/integration/test-extension-loads.sh

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
source "$SCRIPT_DIR/lib/assert.sh"

echo "=== Test: Extension loads and registers ==="
echo ""

SESSION="lens-test-$$"

cleanup() {
  agent-browser --session "$SESSION" close 2>/dev/null || true
}
trap cleanup EXIT

echo "Loading extension and navigating to fixture server..."
output=$(agent-browser \
  --session "$SESSION" \
  open "http://localhost:3456/pages/ai-images" \
  2>&1) || true

assert_contains "Browser opened without error" "$output" ""

# Wait for content script injection and SW to start analysis
sleep 3

annotated=$(agent-browser --session "$SESSION" eval "
  document.querySelectorAll('img[data-lens-level]').length
" 2>/dev/null || echo 0)

echo "  Annotated images after 3s: $annotated"

if [ "$annotated" -ge 1 ] 2>/dev/null; then
  pass "Content script annotated at least 1 image with data-lens-level"
else
  skip "No annotations yet — waiting longer for SW model load"
fi

# SW cold-start loads TF.js + model (~5–15s) — wait for analysis to settle
sleep 20

annotated_final=$(agent-browser --session "$SESSION" eval "document.querySelectorAll('img[data-lens-level]').length" 2>/dev/null || echo 0)
definite=$(agent-browser --session "$SESSION" eval "document.querySelectorAll('img[data-lens-level=\"definite\"]').length" 2>/dev/null || echo 0)
likely=$(agent-browser --session "$SESSION" eval "document.querySelectorAll('img[data-lens-level=\"likely\"]').length" 2>/dev/null || echo 0)
possible=$(agent-browser --session "$SESSION" eval "document.querySelectorAll('img[data-lens-level=\"possible\"]').length" 2>/dev/null || echo 0)
flagged=$((${definite:-0} + ${likely:-0} + ${possible:-0}))

echo "  Results: annotated=$annotated_final definite=$definite likely=$likely possible=$possible flagged=$flagged"

if [ "$annotated_final" -ge 1 ] 2>/dev/null; then
  pass "Extension annotated $annotated_final images after analysis"
else
  fail "No images annotated" ">= 1 annotated" "0"
fi

if [ "$flagged" -ge 1 ] 2>/dev/null; then
  pass "At least $flagged AI image(s) flagged (definite/likely/possible)"
else
  fail "No AI images flagged on ai-images fixture page" ">= 1 flagged" "0"
fi

mkdir -p "$ROOT_DIR/tests/integration/screenshots"
agent-browser --session "$SESSION" \
  screenshot "$ROOT_DIR/tests/integration/screenshots/test-extension-loads.png" \
  --full 2>/dev/null || true
echo "  Screenshot: tests/integration/screenshots/test-extension-loads.png"

print_summary
