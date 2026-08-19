#!/usr/bin/env bash
# Run one of scripts/host_*.py on the hinode GPU host, detached, with the HF token
# supplied safely. See docs/HINODE_HOST_SETUP.md.
#
#   ./scripts/hinode_run.sh host_download_weights.py
#   SMOKE_MODE=text_to_audio_video ./scripts/hinode_run.sh host_smoke.py
#
# The token is piped into the remote launcher's STDIN and exported into the
# environment of the detached process. Never written to a file, and never placed
# in argv -- /proc is world-readable on that shared box, so `env HF_TOKEN=... python`
# would publish the token to every other user via `ps`.
#
# Detached via nohup because an ssh drop must not kill a 72 GB download or a
# multi-minute generation.
set -euo pipefail

SCRIPT="${1:?usage: hinode_run.sh <host_script.py> [env assignments passed through]}"
shift || true

HOST="${LTX25_HINODE_HOST:-wakeuser@hinode}"
TOKEN_FILE="${LTX25_HF_TOKEN_FILE:-$HOME/.cache/huggingface/token}"
[ -r "$TOKEN_FILE" ] || { echo "no readable token at $TOKEN_FILE" >&2; exit 1; }

# Environment the caller wants forwarded (SMOKE_* and friends). Built here so the
# remote side receives assignments it can `export` rather than interpolating
# anything sensitive.
FORWARD=""
for var in SMOKE_MODE SMOKE_W SMOKE_H SMOKE_FRAMES SMOKE_PROMPT SMOKE_SEED \
           SMOKE_STAGE_2 SMOKE_OUTPUT_ROOT LTX25_DISK_ROOT "$@"; do
  case "$var" in
    *=*) FORWARD="$FORWARD export ${var%%=*}='${var#*=}';" ;;
    *)   [ -n "${!var:-}" ] && FORWARD="$FORWARD export $var='${!var}';" ;;
  esac
done

LOG_NAME="$(basename "$SCRIPT" .py)"

# shellcheck disable=SC2029  # deliberate: the remote script is composed here.
cat "$TOKEN_FILE" | ssh -o BatchMode=yes -o ConnectTimeout=20 "$HOST" "
set -euo pipefail
# A token file with no trailing newline makes read return non-zero at EOF even
# though it assigned the value; set -e would then kill us with no output at all.
IFS= read -r TOK || true
[ -n \"\${TOK:-}\" ] || { echo 'no token on stdin' >&2; exit 1; }
source /home/wakeuser/ltx25-env.sh
cd \"\$LTX25_STUDIO_DIR\"
export HF_TOKEN=\"\$TOK\"
$FORWARD
LOG=/home/wakeuser/ltx25-logs/$LOG_NAME.log
nohup ./.venv/bin/python scripts/$SCRIPT > \"\$LOG\" 2>&1 &
echo \"started pid \$! -> \$LOG\"
"
