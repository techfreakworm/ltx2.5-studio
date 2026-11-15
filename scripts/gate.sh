#!/bin/sh
# The offline gate: pytest, then ruff when this environment has it.
#
#   PYTHON=.venv/bin/python scripts/gate.sh
#
# It names the interpreter and whether torch imports, because the result means
# nothing without them: this suite ran green in torch-less venvs for its whole
# history while failing 34 tests and erroring 20 with requirements.txt
# installed. Run it in the environment README.md's install lines build.
set -u
PYTHON=${PYTHON:-python3}
cd "$(dirname "$0")/.." || exit 2
export PYTHONDONTWRITEBYTECODE=1 # leave no __pycache__ behind

where=$("$PYTHON" -c 'import sys; print(sys.executable, sys.version.split()[0])') || {
    echo "gate: cannot run PYTHON=$PYTHON" >&2
    exit 2
}
echo "gate: interpreter $where"
torch=$("$PYTHON" -c 'import torch; print(torch.__version__)' 2>/dev/null) ||
    torch="NOT importable (torch/diffusers fidelity tests will skip)"
echo "gate: torch $torch"

status=0
"$PYTHON" -m pytest -q -p no:cacheprovider || status=1
if "$PYTHON" -c 'import ruff' 2>/dev/null; then
    "$PYTHON" -m ruff check --no-cache . || status=1
else
    echo "gate: ruff is not installed here, so lint was NOT checked"
fi

if [ "$status" -eq 0 ]; then echo "gate: PASS ($where)"; else echo "gate: FAIL ($where)"; fi
exit "$status"
