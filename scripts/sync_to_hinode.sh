#!/usr/bin/env bash
# Ship a COMMITTED tree to the hinode test host (see docs/HINODE_HOST_SETUP.md).
#
# Why `git archive` and not rsync: hinode has no `rsync` binary and there is no
# passwordless sudo to install one, so rsync-over-ssh cannot work at all. Shipping
# a committed ref instead of the working tree is also the stronger choice for
# attribution -- every result measured on the host maps to exactly one SHA, and a
# teammate mid-edit in the shared worktree cannot leak WIP into a recorded run.
#
#   ./scripts/sync_to_hinode.sh          # ships HEAD
#   ./scripts/sync_to_hinode.sh <ref>    # ships any committed ref
#
# Extraction is staged then swapped, so a transfer that dies partway leaves the
# previous tree intact rather than a half-empty directory. `.venv/` and
# `outputs/` live in the destination and are deliberately never touched.
set -euo pipefail

REF="${1:-HEAD}"
HOST="${LTX25_HINODE_HOST:-wakeuser@hinode}"
DEST="${LTX25_HINODE_DIR:-/home/wakeuser/ltx25-studio}"
STAGE="/home/wakeuser/tmp/ltx25-sync-$$"

SHA="$(git rev-parse "$REF")"
if [ -n "$(git status --porcelain)" ]; then
  echo "note: working tree is dirty; shipping COMMITTED $REF ($SHA), not the working tree." >&2
fi

# Paths this script owns in the destination. Listed explicitly so the swap can
# delete what the new tree removed without ever reaching .venv/ or outputs/.
OWNED="app.py ltx25_studio tests docs README.md requirements.txt pyproject.toml .gitignore .claude scripts"

# ConnectTimeout so an unreachable box fails in ~20s rather than on ssh's ~75s
# system default. The failure itself is already safe -- extraction is staged and
# swapped remotely, so a box that never accepted the connection has been left
# entirely untouched -- but a slow failure invites someone to assume it hung and
# start a second one.
if ! ssh -o BatchMode=yes -o ConnectTimeout=20 "$HOST" 'exit 0' 2>/dev/null; then
  echo "ERROR: cannot reach $HOST. Nothing was sent and the box is untouched." >&2
  echo "  Check 'tailscale status' / connectivity, then retry." >&2
  exit 4
fi

git archive --format=tar "$REF" | gzip -6 | ssh -o BatchMode=yes -o ConnectTimeout=20 "$HOST" \
  "set -euo pipefail
   mkdir -p '$STAGE' '$DEST'
   tar xzf - -C '$STAGE'
   cd '$DEST'
   for p in $OWNED; do rm -rf -- \"\$p\"; done
   for p in $OWNED; do [ -e \"$STAGE/\$p\" ] && mv -- \"$STAGE/\$p\" \"$DEST/\$p\" || true; done
   rm -rf '$STAGE'
   printf '%s\n' '$SHA' > '$DEST/.synced_sha'
   echo \"synced \$(cat '$DEST/.synced_sha') to $DEST\"
   ls -A '$DEST'"
