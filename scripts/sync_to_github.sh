#!/usr/bin/env bash
set -euo pipefail

# One-command helper for: add -> commit -> push
# Usage examples:
#   bash scripts/sync_to_github.sh
#   COMMIT_MSG="feat: update collector" bash scripts/sync_to_github.sh
#   REMOTE_URL="https://github.com/<user>/<repo>.git" bash scripts/sync_to_github.sh

COMMIT_MSG="${COMMIT_MSG:-chore: sync local changes}"
BRANCH="${BRANCH:-$(git branch --show-current)}"
REMOTE_NAME="${REMOTE_NAME:-origin}"
REMOTE_URL="${REMOTE_URL:-}"

if [[ -n "$REMOTE_URL" ]]; then
  if git remote get-url "$REMOTE_NAME" >/dev/null 2>&1; then
    git remote set-url "$REMOTE_NAME" "$REMOTE_URL"
  else
    git remote add "$REMOTE_NAME" "$REMOTE_URL"
  fi
fi

if ! git remote get-url "$REMOTE_NAME" >/dev/null 2>&1; then
  echo "[ERROR] Remote '$REMOTE_NAME' is not configured."
  echo "Set REMOTE_URL and run again, e.g.:"
  echo "  REMOTE_URL=https://github.com/<user>/<repo>.git bash scripts/sync_to_github.sh"
  exit 1
fi

if [[ -z "$BRANCH" ]]; then
  echo "[ERROR] Could not detect current branch."
  exit 1
fi

if [[ -z "$(git status --porcelain)" ]]; then
  echo "[INFO] No local changes to commit."
else
  git add .
  git commit -m "$COMMIT_MSG"
fi

git push -u "$REMOTE_NAME" "$BRANCH"
echo "[DONE] Synced to $REMOTE_NAME/$BRANCH"
