"""``scripts/local_generate.py``'s arguments and its ``--json`` report.

The subprocess runs of the script (exit codes, the dry run) are in
``tests/test_local_runtime.py``. Here the script runs in-process through its own
argument parser and ``main()``, with the local runner replaced by a fake that
writes real frames, so no weights are loaded: the parser's values must reach
``generate()`` and the run record must reach the ``--json`` file.
"""

from __future__ import annotations

import json

import pytest
from conftest import video_file
from local_fakes import tiny_snapshot
from test_encoder_contract import FRAME_COUNT, VocoderPipe

from ltx25_studio import audio_conditioning


def _local_generate_module():
    import importlib.util

    from conftest import REPO_ROOT

    spec = importlib.util.spec_from_file_location(
        "local_generate_under_test", REPO_ROOT / "scripts" / "local_generate.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_cli_leaves_the_level_to_the_mode_unless_one_is_given(tmp_path):
    """--noise-scale defaulted to 0.5, which snapped mode 04 to 0.42 instead of its 0.91."""
    cli = _local_generate_module()
    common = ["--mode", "video_to_video", "--snapshot-dir", str(tmp_path)]

    assert cli._parser().parse_args(common).noise_scale is None
    assert cli._parser().parse_args([*common, "--noise-scale", "0.725"]).noise_scale == 0.725
    assert "read by no mode" in cli._parser().format_help()


class _Runner(VocoderPipe):
    """What the script's factory hands ``generate()``: records the call, writes real frames."""

    timings = ()


def _main(tmp_path, monkeypatch, *flags):
    cli = _local_generate_module()
    runners = []

    def build_local_runner(**kwargs):
        runners.append(_Runner())
        runners[-1].device = kwargs["device"]
        return runners[-1]

    monkeypatch.setattr(cli.local_runtime, "build_local_runner", build_local_runner)
    for name in (cli.memory.HIGH_WATERMARK_ENV, cli.memory.LOW_WATERMARK_ENV):
        monkeypatch.delenv(name, raising=False)
    report = tmp_path / "run.json"
    code = cli.main(
        [
            "--mode",
            "video_to_video",
            "--video",
            video_file(frames=17),
            "--width",
            "64",
            "--height",
            "64",
            "--device",
            "cpu",
            "--dtype",
            "float32",
            "--snapshot-dir",
            str(tiny_snapshot()),
            "--out",
            str(tmp_path / "out"),
            "--json",
            str(report),
            *flags,
        ]
    )
    data = json.loads(report.read_text())
    assert code == cli.EXIT_OK, data.get("error")
    (runner,) = runners
    return runner.calls[-1], data


def test_a_run_without_the_flag_varies_the_clip_at_the_modes_own_level(tmp_path, monkeypatch):
    """Measured by the lead: a run without --noise-scale came out at 0.42, not mode 04's 0.91."""
    sent, data = _main(tmp_path, monkeypatch)

    assert sent["video_variation"].level == audio_conditioning.DEFAULT_VIDEO_LEVEL == 0.909375
    assert data["record"]["video_level"] == 0.909375


def test_the_flag_still_sets_the_level(tmp_path, monkeypatch):
    sent, data = _main(tmp_path, monkeypatch, "--noise-scale", "0.725")

    assert sent["video_variation"].level == 0.725
    assert data["record"]["video_level"] == 0.725


def test_the_json_carries_the_run_record_beside_the_artifacts(tmp_path, monkeypatch):
    sent, data = _main(tmp_path, monkeypatch)

    record, artifacts = data["record"], data["artifacts"]
    assert sent["num_frames"] == 17, "the clip's 17 frames set the length"
    assert record["mode"] == "video_to_video" and record["length_source"] == "video"
    assert record["seed"] == artifacts["seed"] and record["frames"] == FRAME_COUNT
    assert record["device"] == "cpu", record
    assert artifacts["output"].endswith(".mp4") and artifacts["warnings"]


@pytest.mark.parametrize("record", [None, "not a dataclass"])
def test_a_result_without_a_record_writes_none(record):
    from types import SimpleNamespace

    assert _local_generate_module()._record(SimpleNamespace(record=record)) is None
