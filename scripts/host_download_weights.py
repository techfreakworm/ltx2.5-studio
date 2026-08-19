"""Fetch the weights on a GPU host through the app's OWN resolver.

Run on hinode; see docs/HINODE_HOST_SETUP.md.

Calling ``preflight.ensure_weights`` -> ``model_files.download_allow_list()`` is
the whole point. It is the code path the Space runs, so running it on a second
host **independently proves the resolver** (PLAN §5.2). Re-deriving the file list
here would prove only that a list can be typed twice, and would not catch a
resolver that drops ``scheduler/scheduler_config.json`` or picks the dead shard
set.

Reports free disk before and after, and the wall clock. ``/home/wakeuser/.cache``
is shared with other projects on that box, so "did it fit" is a measurement, not
an assumption.
"""

import logging
import os
import shutil
import sys
import time

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logging.getLogger("httpx").setLevel(logging.WARNING)

from ltx25_studio import model_files, preflight  # noqa: E402

DISK_ROOT = os.environ.get("LTX25_DISK_ROOT", "/home/wakeuser")


def main() -> int:
    token = os.environ.get("HF_TOKEN")
    if not token:
        # Refuse rather than fall through: an unauthenticated request to a gated
        # repo answers 401, and a 401 three layers down reads like a network
        # fault rather than a missing secret.
        print("HF_TOKEN absent; refusing to try a gated repo unauthenticated", file=sys.stderr)
        return 2

    allow = model_files.download_allow_list()
    print(
        f"resolver returned {len(allow)} files, {model_files.total_size_gb():.3f} GB, "
        f"revision {model_files.REVISION}",
        flush=True,
    )
    before = shutil.disk_usage(DISK_ROOT).free
    print(f"disk free BEFORE: {before} bytes ({before / 1e9:.3f} GB)", flush=True)

    start = time.time()
    fetched = preflight.ensure_weights(token=token)
    elapsed = time.time() - start

    after = shutil.disk_usage(DISK_ROOT).free
    print(
        f"fetched {len(fetched)}/{len(allow)} files in {elapsed:.1f}s "
        f"({elapsed / 60:.2f} min)",
        flush=True,
    )
    print(f"disk free AFTER : {after} bytes ({after / 1e9:.3f} GB)", flush=True)
    print(f"disk CONSUMED   : {before - after} bytes ({(before - after) / 1e9:.3f} GB)", flush=True)

    missing = preflight.missing_cached_files()
    print(f"missing_cached_files() after download: {len(missing)} {missing[:3]}", flush=True)
    preflight.require_cache_complete()
    print("preflight.require_cache_complete(): PASS", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
