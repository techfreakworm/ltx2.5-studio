"""``app.main()`` lets Gradio serve the results wherever the output folder is.

Gradio serves a file a step returns only from the working directory, the system temp dir,
its own cache or ``launch(allowed_paths=...)``. With ``LTX25_OUTPUT_DIR`` outside the first
two, the lead's run on real weights (2026-11-16) finished and Gradio refused its file.
``main()`` passes that folder, and only it, as ``allowed_paths``; where the folder is inside
(the default ``outputs`` under the working directory, and the Space) the launch is exactly as
before. No server starts here: ``Blocks.launch`` is replaced and its arguments recorded.
"""

from __future__ import annotations

import importlib
import logging
import sys
import tempfile

import pytest

from ltx25_studio import config, zerogpu
from ltx25_studio import generate as generate_module

gr = pytest.importorskip("gradio", reason="gradio is a declared dev dependency")


@pytest.fixture
def places(tmp_path, monkeypatch):
    """A working directory and a temp dir of the test's own; Gradio reads both when it
    checks a file, so ``main()`` must judge the folder against the same two."""
    cwd, temp = tmp_path / "cwd", tmp_path / "temp"
    for folder in (cwd, temp):
        folder.mkdir()
    monkeypatch.chdir(cwd)
    monkeypatch.setattr(tempfile, "tempdir", str(temp))
    return tmp_path


@pytest.fixture
def launch(monkeypatch):
    """``import app`` on a UI-only host, then ``app.main()``: what ``launch`` received."""
    monkeypatch.setattr(zerogpu, "host_has_cuda", lambda: False)
    for name in (config.ENV_DEVICE, config.ENV_QUANTIZATION, config.ENV_MODEL_SNAPSHOT_DIR):
        monkeypatch.delenv(name, raising=False)
    calls: list = []
    monkeypatch.setattr(
        gr.Blocks, "launch", lambda self, *a, **k: calls.append((a, k)), raising=False
    )
    previous = generate_module.gpu_dispatch()

    def main(output_dir: str | None):
        if output_dir is None:
            monkeypatch.delenv(config.ENV_OUTPUT_DIR, raising=False)
        else:
            monkeypatch.setenv(config.ENV_OUTPUT_DIR, output_dir)
        sys.modules.pop("app", None)
        module = importlib.import_module("app")
        module.main()
        assert len(calls) == 1, "main() launches once"
        return module, calls[0]

    yield main
    generate_module.set_gpu_dispatch(previous)
    sys.modules.pop("app", None)


def allowed_path_lines(caplog) -> list[str]:
    return [r.getMessage() for r in caplog.records if "allowed_paths" in r.getMessage()]


def test_the_default_output_folder_launches_exactly_as_before(places, launch, caplog):
    caplog.set_level(logging.INFO, logger="ltx25_studio")
    app, (args, kwargs) = launch(None)
    assert args == () and kwargs == {"max_file_size": app.MAX_UPLOAD_BYTES}
    assert allowed_path_lines(caplog) == []


def test_a_folder_under_the_temp_dir_launches_exactly_as_before(places, launch, caplog):
    caplog.set_level(logging.INFO, logger="ltx25_studio")
    app, (args, kwargs) = launch(str(places / "temp" / "ltx25-outputs"))
    assert args == () and kwargs == {"max_file_size": app.MAX_UPLOAD_BYTES}
    assert allowed_path_lines(caplog) == []


def test_a_folder_outside_both_is_passed_as_the_only_allowed_path(places, launch, caplog):
    caplog.set_level(logging.INFO, logger="ltx25_studio")
    outside = places / "elsewhere" / "outputs"
    app, (args, kwargs) = launch(str(outside))
    assert args == ()
    assert kwargs == {
        "max_file_size": app.MAX_UPLOAD_BYTES,
        "allowed_paths": [str(outside.resolve())],
    }
    (line,) = allowed_path_lines(caplog)
    assert str(outside.resolve()) in line


def test_a_relative_folder_is_judged_where_it_resolves(places, launch):
    app, (_args, kwargs) = launch("../elsewhere")
    assert kwargs["allowed_paths"] == [str((places / "elsewhere").resolve())]


def test_a_folder_reached_through_a_link_into_the_temp_dir_is_inside(places, launch):
    """Gradio resolves links before it compares folders (``utils.is_in_or_equal``), as the
    macOS temp dir is reached through ``/var`` -> ``/private/var``."""
    link = places / "link-to-temp"
    link.symlink_to(places / "temp", target_is_directory=True)
    app, (_args, kwargs) = launch(str(link / "outputs"))
    assert kwargs == {"max_file_size": app.MAX_UPLOAD_BYTES}
