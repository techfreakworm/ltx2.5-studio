"""The start-up check of the weights (#23): the image brings them; start-up fetches only gaps.

The README front matter preloads ``model_files.download_allow_list()`` into the
image, so a start whose cache already holds every file must make no download
call at all and say, in one line, that the weights came from the image. A
preload that is missing files must not break the Space: start-up fetches exactly
the missing ones, at their pinned revisions, and says loudly how many and how
large. ``preflight.ensure_weights()`` is what ``app.py`` runs for this, so it is
driven here as ``app.py`` drives it -- the cache resolved from the environment,
``huggingface_hub`` replaced by a recorder so nothing is ever downloaded.
"""

from __future__ import annotations

import logging
import sys
import types

import pytest

from ltx25_studio import config, model_files, preflight
from ltx25_studio.errors import MissingModelFileError

DECODER_WEIGHTS = "diffusion_decoder/diffusion_pytorch_model.safetensors"


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


def start(caplog):
    """``app.py``'s call, with its log."""
    with caplog.at_level(logging.INFO, logger="ltx25_studio"):
        paths = preflight.ensure_weights(token="hf_test")
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
