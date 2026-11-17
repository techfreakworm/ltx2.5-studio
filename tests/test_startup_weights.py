"""The start-up check of the weights (#23): the image brings them; start-up fetches only gaps.

The README front matter preloads ``model_files.download_allow_list()`` into the
image, so a start whose cache already holds every file must make no download
call at all and say, in one line, that the weights came from the image. A
preload that is missing files must not break the Space: start-up fetches exactly
the missing ones, at their pinned revisions, and says loudly how many and how
large. ``preflight.ensure_weights()`` is what ``app.py`` runs for this, so it is
driven here as ``app.py`` drives it -- the cache resolved from the environment,
``huggingface_hub`` replaced by a recorder so nothing is ever downloaded.

Before a fallback download, start-up checks there is room for it (the missing
bytes, ZeroGPU's packed copy still to be written, a margin) and refuses with the
remedy when there is not: a download that fills the disk part-way is the worst
failure a public Space can have. ``shutil.disk_usage`` is replaced here too, so the
answer never depends on this machine's own disk.
"""

from __future__ import annotations

import logging
import shutil
import sys
import types
from pathlib import Path

import pytest

from ltx25_studio import config, model_files, pipelines, preflight
from ltx25_studio.errors import MissingModelFileError, ModelFileError

DECODER_WEIGHTS = "diffusion_decoder/diffusion_pytorch_model.safetensors"
GB = 10**9


@pytest.fixture(autouse=True)
def free_disk(monkeypatch):
    """``shutil.disk_usage`` for the cache's filesystem: 500 GB free unless a test sets it."""
    state = {"free": 500 * GB, "asked": []}

    def disk_usage(path):
        state["asked"].append(Path(path))
        return types.SimpleNamespace(total=2000 * GB, used=0, free=state["free"])

    monkeypatch.setattr(shutil, "disk_usage", disk_usage)
    return state


@pytest.fixture
def cache(tmp_path, monkeypatch):
    """An HF cache of the test's own, empty, read the way the app reads it."""
    monkeypatch.delenv(config.ENV_MODEL_SNAPSHOT_DIR, raising=False)
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path / "hub"))
    return preflight.cache_repo_dir()


def place(repo_cache, paths):
    """Put ``paths`` in the cache where a preload or a download leaves them."""
    for path in paths:
        target = model_files.local_path(path, repo_cache=repo_cache)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"w")


@pytest.fixture
def hub(monkeypatch, cache):
    """A stand-in ``huggingface_hub`` that records each download and lands the file."""
    calls: list[dict] = []

    def hf_hub_download(**kwargs):
        calls.append(kwargs)
        place(cache, [kwargs["filename"]])
        return str(model_files.local_path(kwargs["filename"], repo_cache=cache))

    module = types.ModuleType("huggingface_hub")
    module.hf_hub_download = hf_hub_download
    monkeypatch.setitem(sys.modules, "huggingface_hub", module)
    return calls


def start(caplog, *, on_zerogpu=False):
    """``app.py``'s call, with its log."""
    with caplog.at_level(logging.INFO, logger="ltx25_studio"):
        paths = preflight.ensure_weights(token="hf_test", on_zerogpu=on_zerogpu)
    return paths, [r for r in caplog.records if r.name == "ltx25_studio"]


def test_a_complete_cache_downloads_nothing_and_says_the_weights_came_from_the_image(
    cache, hub, caplog
):
    place(cache, model_files.download_allow_list())

    paths, records = start(caplog)

    assert hub == [], f"{len(hub)} download call(s) with every file already in the cache"
    assert len(records) == 1, [r.getMessage() for r in records]
    (line,) = records
    assert line.levelno == logging.INFO, line.getMessage()
    message = line.getMessage()
    assert "image" in message and "nothing" in message, message
    assert "35 files" in message and "71.92 GB" in message, message
    assert len(paths) == len(model_files.download_allow_list())
    assert all(path.exists() for path in paths)


def test_one_missing_file_is_downloaded_alone_at_its_revision_and_said_loudly(
    cache, hub, caplog
):
    place(cache, [p for p in model_files.download_allow_list() if p != DECODER_WEIGHTS])

    _paths, records = start(caplog)

    assert [call["filename"] for call in hub] == [DECODER_WEIGHTS]
    (call,) = hub
    assert call["revision"] == model_files.DIFFUSION_DECODER_REVISION
    assert call["repo_id"] == model_files.REPO_ID
    assert call["token"] == "hf_test"
    warnings = [r.getMessage() for r in records if r.levelno == logging.WARNING]
    assert len(warnings) == 1, [r.getMessage() for r in records]
    assert "1 of 35 files" in warnings[0] and "0.83 GB" in warnings[0], warnings[0]
    assert DECODER_WEIGHTS in warnings[0], warnings[0]


def test_an_empty_cache_downloads_every_file_as_the_first_deploy_did(cache, hub, caplog):
    _paths, records = start(caplog)

    asked = {call["filename"]: call["revision"] for call in hub}
    assert set(asked) == set(model_files.download_allow_list())
    assert len(hub) == len(asked), "a file was asked for twice"
    assert all(asked[path] == model_files.revision_of_path(path) for path in asked)
    warnings = [r.getMessage() for r in records if r.levelno == logging.WARNING]
    assert len(warnings) == 1 and "35 of 35 files (71.92 GB)" in warnings[0], warnings


def test_a_gap_that_cannot_be_fetched_still_stops_start_up_naming_the_file(
    cache, monkeypatch
):
    """The contract the start-up had before the preload: a named file and its URL."""
    place(cache, [p for p in model_files.download_allow_list() if p != DECODER_WEIGHTS])

    def refuse(**kwargs):
        raise PermissionError(13, "Permission denied")

    module = types.ModuleType("huggingface_hub")
    module.hf_hub_download = refuse
    monkeypatch.setitem(sys.modules, "huggingface_hub", module)

    with pytest.raises(MissingModelFileError) as caught:
        preflight.ensure_weights(token=None)
    message = str(caught.value)
    assert DECODER_WEIGHTS in message and "PermissionError" in message, message
    assert model_files.DIFFUSION_DECODER_REVISION in message, message


def test_the_disk_line_counts_the_preloaded_snapshot_once(cache, monkeypatch):
    """With the preload the snapshot is part of the image. The disk line measures
    it as the snapshot term, so its ~10 GB image term must be the rest of the image,
    or the line would read as if the weights still sat outside it."""
    from ltx25_studio import pipelines

    place(cache, model_files.download_allow_list())

    line = pipelines.disk_footprint()

    assert "rest of the image" in line, line
    assert "FLOOR" in line and "[OK]" in line, line


# --------------------------------------------------------------------------
# room for a fallback download: refused before a byte is written when it does not fit
# --------------------------------------------------------------------------


def test_a_complete_cache_needs_no_room_check(cache, hub, free_disk, caplog):
    place(cache, model_files.download_allow_list())

    start(caplog)

    assert free_disk["asked"] == [], "a complete cache asked for the free space"


def test_a_small_gap_with_room_is_downloaded(cache, hub, free_disk, caplog):
    free_disk["free"] = 12 * GB  # 0.83 GB missing + the 10 GB margin fit
    place(cache, [p for p in model_files.download_allow_list() if p != DECODER_WEIGHTS])

    start(caplog)

    assert [call["filename"] for call in hub] == [DECODER_WEIGHTS]
    assert free_disk["asked"], "the room was not checked before downloading"


def test_a_large_gap_without_room_is_refused_and_nothing_is_downloaded(
    cache, hub, free_disk, caplog
):
    free_disk["free"] = 50 * GB  # 71.92 GB missing + 10 GB margin do not fit

    with pytest.raises(ModelFileError) as caught:
        start(caplog)

    assert hub == [], "a download was started that the disk could not hold"
    message = str(caught.value)
    for part in (str(cache), "35 weight files", "71.92 GB", "50.0 GB", "preload_from_hub"):
        assert part in message, (part, message)
    assert "\n" not in message, message


def test_off_zerogpu_no_packed_copy_is_counted(cache, hub, free_disk, caplog):
    """Only ZeroGPU writes a packed copy to disk; a CUDA box with room for the set fetches it."""
    free_disk["free"] = 85 * GB  # 71.92 + 10 fit; with a 42.1 GB packed copy they would not

    start(caplog)

    assert len(hub) == len(model_files.download_allow_list())


def test_on_zerogpu_the_first_deploys_shape_still_starts(cache, hub, free_disk, caplog):
    """No preload at all (the remedy): 150 GB cap - ~10 GB image = 140 GB of room for the
    71.92 GB download, the 42.1 GB packed copy and the margin. It must not be refused."""
    free_disk["free"] = 2000 * GB  # ZeroGPU's filesystem is the host's: terabytes

    start(caplog, on_zerogpu=True)

    assert len(hub) == len(model_files.download_allow_list())


def test_on_zerogpu_the_cap_bounds_the_room_not_the_host_filesystem(
    cache, hub, free_disk, caplog, monkeypatch
):
    """A 72 GB copy already in the HF cache (a preload start-up does not find) leaves
    68 GB under the cap: downloading the set again would be evicted, so it is refused --
    though the host's filesystem reports terabytes free."""
    free_disk["free"] = 2000 * GB
    monkeypatch.setattr(pipelines, "_directory_bytes", lambda path: 72 * GB)

    with pytest.raises(ModelFileError) as caught:
        start(caplog, on_zerogpu=True)

    assert hub == []
    message = str(caught.value)
    assert "150 GB cap" in message and "68.0 GB" in message, message
    assert "42.1 GB packed copy" in message, message
