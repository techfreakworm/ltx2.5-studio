"""Spec: an API caller can see what ran (issue #18, S6).

``/on_generate`` returns the files, the notes and the diagnostics; the page's
Done card adds the adapter and the time from the run record, which an API
caller never sees. ``record.diagnostics`` now carries, on every run, one line
per attached adapter ("adapter <label> at <weight>: attached to <n> layers") and
one line for the time, split into the phases the run reports to the progress
bar, in the bar's own words:

    took 23.4 s on cuda: loading the model 1.2 s, preparing the inputs 0.3 s,
    sampling 14.0 s, decoding the frames 4.3 s, writing the file 1.1 s

Each phase is timed with a monotonic clock from where it starts to where the
next one starts; the last ends where the run's own time ends, so the phases add
up to the total. The same time line is written once per run to the log at INFO
(logger ``ltx25_studio.generate``), so the Space's own log carries per-phase
times.
"""

from __future__ import annotations

import logging
import re
import time

import pytest
from test_generate import FakePipe, LoRAFakePipe, _staged, a_request, good_encoder
from test_run_record import Output, run

from ltx25_studio import generate as generate_module

TOOK = re.compile(r"^took (?P<total>\d+\.\d) s(?: on (?P<device>\w+))?(?:: (?P<split>.+))?$")


class Bar:
    """A ``gr.Progress``-shaped handle that keeps every label it is shown."""

    def __init__(self):
        self.labels: list[str] = []

    def __call__(self, position, desc=None):
        if desc and (not self.labels or self.labels[-1] != desc):
            self.labels.append(str(desc))


class PhasedPipe(FakePipe):
    """Announces its phases as the real runner does, and takes measurable time in each."""

    def __call__(self, *args, progress=None, **kwargs):
        if progress is not None:
            progress.sampling()
        time.sleep(0.12)
        if progress is not None:
            progress.decoding()
        time.sleep(0.06)
        return super().__call__(*args, **kwargs)


def took_lines(result) -> list[str]:
    return [line for line in result.record.diagnostics if line.startswith("took ")]


def adapter_lines(result) -> list[str]:
    return [line for line in result.record.diagnostics if line.startswith("adapter ")]


def phases(line: str) -> list[tuple[str, float]]:
    match = TOOK.match(line)
    assert match and match["split"], f"not a time line with phases: {line!r}"
    out = []
    for part in match["split"].split(", "):
        label, seconds, unit = part.rsplit(" ", 2)
        assert unit == "s", part
        out.append((label, float(seconds)))
    return out


# --------------------------------------------------------------------------
# the lines
# --------------------------------------------------------------------------


def test_an_adapter_run_carries_the_adapter_line_and_the_time_line(tmp_path):
    result = run(
        tmp_path,
        a_request(
            width=96, height=64, num_frames=9, loras=(_staged(tmp_path / "Bts.safetensors", 0.8),)
        ),
        pipe=LoRAFakePipe(output=Output()),
    )

    assert adapter_lines(result) == ["adapter Bts at 0.8: attached to 1 layer"]
    assert len(took_lines(result)) == 1, result.record.diagnostics


def test_an_adapter_less_run_carries_only_the_time_line(tmp_path):
    result = run(tmp_path)

    assert adapter_lines(result) == []
    assert len(took_lines(result)) == 1, result.record.diagnostics
    assert TOOK.match(took_lines(result)[0]), took_lines(result)


def test_the_lines_are_diagnostics_not_notes(tmp_path):
    result = run(
        tmp_path,
        a_request(width=96, height=64, num_frames=9, loras=(_staged(tmp_path / "a.safetensors"),)),
        pipe=LoRAFakePipe(output=Output()),
    )

    lines = took_lines(result) + adapter_lines(result)
    assert lines and not any(
        note.startswith(("took ", "adapter ")) for note in result.warnings
    ), result.warnings
    assert "adapter a at 1.0: attached to 1 layer" in lines


def test_the_time_is_the_records_own_and_names_its_device(tmp_path):
    from test_decode_policy import OK, runner_on

    runner, _pipeline, _built = runner_on("cpu")
    result = generate_module.generate(
        mode="text_to_video",
        seed=7,
        prompt="p",
        **OK,
        decoder="conv",
        pipeline_factory=lambda **_: runner,
        encoder=good_encoder,
        output_root=tmp_path,
        retention_bytes=10**9,
    )

    match = TOOK.match(took_lines(result)[0])
    assert match["device"] == "cpu" == result.record.device
    assert float(match["total"]) == round(result.record.elapsed_seconds, 1)


# --------------------------------------------------------------------------
# the split: the bar's own words, timed where each phase starts and ends
# --------------------------------------------------------------------------


def test_the_phases_are_the_bars_own_words_in_the_bars_order(tmp_path):
    bar = Bar()
    result = run(tmp_path, pipe=PhasedPipe(output=Output()), progress=bar)

    assert [label for label, _ in phases(took_lines(result)[0])] == bar.labels
    assert bar.labels == [
        "loading the model",
        "preparing the inputs",
        "sampling",
        "decoding the frames",
        "writing the file",
    ]


def test_the_phase_seconds_add_up_to_the_total(tmp_path):
    result = run(tmp_path, pipe=PhasedPipe(output=Output()), progress=Bar())

    line = took_lines(result)[0]
    split = phases(line)
    total = float(TOOK.match(line)["total"])
    assert abs(sum(seconds for _, seconds in split) - total) <= 0.05 * (len(split) + 1), line
    assert dict(split)["sampling"] >= 0.1 and dict(split)["decoding the frames"] >= 0.05, line


def test_the_real_runner_times_its_phases_with_nobody_listening(tmp_path):
    """No progress handle (a script, a direct caller): the runner still names its phases."""
    from test_decode_policy import OK, runner_on

    runner, pipeline, _built = runner_on("cuda")
    result = generate_module.generate(
        mode="text_to_video",
        seed=7,
        prompt="p",
        **OK,
        pipeline_factory=lambda **_: runner,
        encoder=good_encoder,
        output_root=tmp_path,
        retention_bytes=10**9,
    )

    labels = [label for label, _ in phases(took_lines(result)[0])]
    assert labels == [
        "loading the model",
        "preparing the inputs",
        "sampling",
        "decoding the frames",
        "writing the file",
    ], labels
    assert "progress" not in pipeline.kwargs and "callback_on_step_end" not in pipeline.kwargs


def test_a_fake_with_nobody_listening_is_handed_what_it_was_before(tmp_path):
    pipe = FakePipe(output=Output())
    run(tmp_path, pipe=pipe)

    (_args, kwargs), *_ = pipe.calls
    assert "progress" not in kwargs, sorted(kwargs)


# --------------------------------------------------------------------------
# the log: once per run, at INFO, on ltx25_studio.generate
# --------------------------------------------------------------------------


def _logged(caplog) -> list[str]:
    return [
        record.getMessage()
        for record in caplog.records
        if record.name == "ltx25_studio.generate" and record.getMessage().startswith("took ")
    ]


def test_the_time_line_is_logged_once_per_run_at_info(tmp_path, caplog):
    caplog.set_level(logging.INFO, logger="ltx25_studio.generate")
    first = run(tmp_path / "one", pipe=PhasedPipe(output=Output()), progress=Bar())
    second = run(tmp_path / "two")

    assert len(_logged(caplog)) == 2, _logged(caplog)
    assert _logged(caplog) == took_lines(first) + took_lines(second)
    assert all(
        record.levelno == logging.INFO
        for record in caplog.records
        if record.getMessage().startswith("took ")
    )


def test_a_run_through_the_gpu_dispatch_logs_its_time_once(tmp_path, caplog, monkeypatch):
    """The web side dispatches, the worker runs: the line comes from the worker alone."""

    def worker(**params):
        token = generate_module._IN_GPU_WORKER.set(True)
        try:
            return generate_module.generate(**params)
        finally:
            generate_module._IN_GPU_WORKER.reset(token)

    monkeypatch.setattr(generate_module, "_GPU_DISPATCH", worker)
    caplog.set_level(logging.INFO, logger="ltx25_studio.generate")

    result = run(tmp_path)

    assert _logged(caplog) == took_lines(result) and len(took_lines(result)) == 1


@pytest.mark.parametrize("weight", [1.0, 0.85, 1.25])
def test_the_weight_is_written_as_the_page_writes_it(tmp_path, weight):
    result = run(
        tmp_path,
        a_request(
            width=96, height=64, num_frames=9, loras=(_staged(tmp_path / "b.safetensors", weight),)
        ),
        pipe=LoRAFakePipe(output=Output()),
    )

    shown = {1.0: "1.0", 0.85: "0.85", 1.25: "1.25"}[weight]
    assert adapter_lines(result) == [f"adapter b at {shown}: attached to 1 layer"]
