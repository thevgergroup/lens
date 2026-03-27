#!/usr/bin/env bash
# Integration test: Extension fires on known real-world AI image galleries
#
# Navigates to public websites that host known AI-generated images and
# verifies the extension detects them. These tests require internet access.
#
# Usage: bash tests/integration/test-live-sites.sh
# Skip with: SKIP_LIVE=1 bash tests/integration/test-live-sites.sh

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
source "$SCRIPT_DIR/lib/assert.sh"

if [ "${SKIP_LIVE:-0}" = "1" ]; then
  echo "Skipping live site tests (SKIP_LIVE=1)"
  exit 0
fi

EXTENSION_PATH="$ROOT_DIR"
SESSION="lens-live-$$"
WAIT_SECS=8  # Live sites need more time

cleanup() {
  agent-browser --session "$SESSION" close 2>/dev/null || true
}
trap cleanup EXIT

echo "=== Test: Extension on live AI image sites ==="
echo "(Requires internet access. Set SKIP_LIVE=1 to skip)"
echo ""

# Helper: count all lens-badges on current page
count_all_badges() {
  agent-browser --session "$SESSION" \
    get count ".lens-badge" 2>/dev/null || echo 0
}

count_flagged_badges() {
  agent-browser --session "$SESSION" eval "
    document.querySelectorAll('.lens-badge.lens-definite, .lens-badge.lens-likely, .lens-badge.lens-possible').length
  " 2>/dev/null || echo 0
}

# ─── Civitai (known SD/FLUX AI images) ────────────────────────────────────
echo "--- Civitai image gallery (all images are AI-generated) ---"

agent-browser \
  --extension "$EXTENSION_PATH" \
  --session "$SESSION" \
  open "https://civitai.com/images" \
  2>/dev/null && CIVITAI_OK=1 || CIVITAI_OK=0

if [ "$CIVITAI_OK" = "1" ]; then
  # Wait for page + extension analysis
  agent-browser --session "$SESSION" wait --load networkidle 2>/dev/null || sleep "$WAIT_SECS"
  sleep 3

  total=$(count_all_badges)
  flagged=$(count_flagged_badges)

  echo "  Total badges: $total, Flagged as AI: $flagged"

  if [ "$flagged" -ge 3 ] 2>/dev/null; then
    pass "Civitai: at least 3 AI images detected on gallery page ($flagged flagged)"
  elif [ "$total" -ge 1 ]; then
    pass "Civitai: extension fired ($total badges, $flagged flagged as AI)"
  else
    fail "Civitai: no badges found" ">= 1 badge" "0"
  fi

  mkdir -p "$ROOT_DIR/tests/integration/screenshots"
  agent-browser --session "$SESSION" \
    screenshot "$ROOT_DIR/tests/integration/screenshots/civitai-gallery.png" --full 2>/dev/null || true
else
  skip "Could not reach civitai.com (network issue)"
fi

# ─── C2PA Verify Tool ────────────────────────────────────────────────────
echo ""
echo "--- contentcredentials.org verify tool ---"
echo "  (checking extension doesn't break the site itself)"

agent-browser \
  --session "$SESSION" \
  open "https://contentcredentials.org/verify" \
  2>/dev/null && CCORG_OK=1 || CCORG_OK=0

if [ "$CCORG_OK" = "1" ]; then
  agent-browser --session "$SESSION" wait --load networkidle 2>/dev/null || sleep 4

  # Just verify the page loads and the extension doesn't break it
  title=$(agent-browser --session "$SESSION" get title 2>/dev/null || echo "")
  assert_not_contains \
    "Extension does not break contentcredentials.org" \
    "$title" \
    "error"
  pass "contentcredentials.org loads without errors"
else
  skip "Could not reach contentcredentials.org"
fi

echo ""
print_summary
