"""Spec: the model-chosen length note says what the range did (issue #16, S4).

With "Let the model pick the length" and a range of 5 to 10 s, the duration
head predicted 4.47 s; diffusers raised it to the shortest allowed length and
the page said "The model chose the length: 5.0 s.", so the user thinks the
model wanted 5 s. When the head's own figure lies outside the range the note
now says so: "The model chose 4.5 s; your shortest length is 5 s, so the video
is 5.0 s." (and the mirror image for the longest). Inside the range the note is
as it was.

The figure is read off the head the pipeline runs, by a forward hook that only
reads the output, so what is sampled does not change. No log text is parsed.
"""

from __future__ import annotations

import contextlib
import math

import numpy as np
import pytest
from test_exports_contract import factory_returning, one_muxed_file_encoder
from test_user_words import assert_in_the_uis_words

from ltx25_studio import generate as generate_module

RATE = 24.0


def _real_head():
    """diffusers' own head, small, with its output pinned by the test."""
    pytest.importorskip("torch")
    duration_head = pytest.importorskip("diffusers.pipelines.ltx2.duration_head")
    return duration_head.LTX2DurationHead(
        video_cross_attention_dim=8,
        audio_cross_attention_dim=8,
        pooler_hidden_dim=8,
        num_pooler_heads=2,
        mlp_hidden_dim=8,
    )


@contextlib.contextmanager
def predicting(head, seconds: float):
    """The head returns ``seconds`` for any prompt: its last layer is a constant log-duration."""
    import torch

    weight, bias = head.mlp_out.weight.detach().clone(), head.mlp_out.bias.detach().clone()
    with torch.no_grad():
        head.mlp_out.weight.zero_()
        head.mlp_out.bias.fill_(math.log(seconds))
    try:
        yield head
    finally:
        with torch.no_grad():
            head.mlp_out.weight.copy_(weight)
            head.mlp_out.bias.copy_(bias)


class _Output:
    def __init__(self, frames: int):
        self.frames = np.zeros((1, frames, 8, 8, 3), dtype=np.uint8)


class _Inner:
    def __init__(self, head):
        self.duration_head = head


class HeadPipe:
    """Calls the head exactly as diffusers' pipeline does when ``num_frames`` is left out."""

    def __init__(self, head):
        self.pipeline = _Inner(head)
        self.calls: list[dict] = []

    def __call__(self, **kwargs):
        import torch

        self.calls.append(kwargs)
        frames = kwargs.get("num_frames") or self.pipeline.duration_head.predict_num_frames(
            video_tokens=torch.zeros(1, 4, 8),
            frame_rate=RATE,
            temporal_compression_ratio=8,
            min_seconds=kwargs["min_seconds"],
            max_seconds=kwargs["max_seconds"],
        )
        return _Output(frames)


def run(head, *, low=5.0, high=10.0, auto=True, tmp_path):
    return generate_module.generate(
        mode="text_to_video",
        prompt="a man shouts across a canyon",
        seed=7,
        auto_duration=auto,
        min_seconds=low,
        max_seconds=high,
        pipeline_factory=factory_returning(HeadPipe(head)),
        encoder=one_muxed_file_encoder,
        output_root=tmp_path,
        retention_bytes=10**9,
    )


def chosen(result) -> list[str]:
    return [note for note in result.warnings if note.startswith("the model chose")]


# --------------------------------------------------------------------------
# the head diffusers ships, called the way its pipeline calls it
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("predicted", "note"),
    [
        (4.47, "the model chose 4.5 s; your shortest length is 5 s, so the video is 5.0 s."),
        (7.3, "the model chose the length: 7.0 s."),
        # 240 frames at most, floored to the grid: 233 frames.
        (12.6, "the model chose 12.6 s; your longest length is 10 s, so the video is 9.7 s."),
    ],
    ids=["below", "inside", "above"],
)
def test_the_note_says_what_the_model_chose_and_what_the_range_did(predicted, note, tmp_path):
    with predicting(_real_head(), predicted) as head:
        result = run(head, tmp_path=tmp_path)

    assert chosen(result) == [note], result.warnings


def test_a_figure_one_decimal_would_round_onto_the_bound_is_shown_to_two(tmp_path):
    with predicting(_real_head(), 4.96) as head:
        result = run(head, tmp_path=tmp_path)

    assert chosen(result) == [
        "the model chose 4.96 s; your shortest length is 5 s, so the video is 5.0 s."
    ], result.warnings


def test_a_range_bound_with_a_fraction_keeps_it(tmp_path):
    with predicting(_real_head(), 1.2) as head:
        result = run(head, low=2.5, high=4.0, tmp_path=tmp_path)

    assert chosen(result) == [
        "the model chose 1.2 s; your shortest length is 2.5 s, so the video is 2.7 s."
    ], result.warnings


def test_the_new_notes_are_in_the_uis_words(tmp_path):
    notes = []
    for predicted in (4.47, 12.6):
        with predicting(_real_head(), predicted) as head:
            notes += chosen(run(head, tmp_path=tmp_path / str(predicted)))

    assert len(notes) == 2
    assert_in_the_uis_words(notes, "the model-chosen length note")


def test_the_reading_is_taken_off_the_head_after_the_run(tmp_path):
    head = _real_head()
    with predicting(head, 4.47):
        run(head, tmp_path=tmp_path)

    assert not head._forward_hooks, "the run left its hook on the duration head"


def test_a_length_the_user_set_gets_no_note_and_no_hook(tmp_path):
    head = _real_head()
    seen = []
    real = head.register_forward_hook
    head.register_forward_hook = lambda hook: seen.append(hook) or real(hook)

    result = generate_module.generate(
        mode="text_to_video",
        prompt="p",
        seed=7,
        num_frames=49,
        pipeline_factory=factory_returning(HeadPipe(head)),
        encoder=one_muxed_file_encoder,
        output_root=tmp_path,
        retention_bytes=10**9,
    )

    assert chosen(result) == [] and seen == []


# --------------------------------------------------------------------------
# the real pipeline: the tiny checkpoint's own duration head, through generate()
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def tiny():
    pytest.importorskip("torch")
    pytest.importorskip("diffusers")
    from test_real_pipeline_tiny import _tiny_snapshot_cached, load_smoke

    if not _tiny_snapshot_cached():
        pytest.skip("the tiny checkpoint is not in the HF cache")
    return load_smoke().tiny_stack(device="cpu")


def run_tiny_auto(tiny, tmp_path, *, low=1.0, high=2.0):
    from test_real_pipeline_tiny import run_tiny

    return run_tiny(
        tiny,
        "text_to_video",
        tmp_path,
        auto_duration=True,
        min_seconds=low,
        max_seconds=high,
    )


@pytest.mark.parametrize(
    ("predicted", "note"),
    [
        # 25 frames: the shortest the grid allows at or above 1 s.
        (0.6, "the model chose 0.6 s; your shortest length is 1 s, so the video is 1.0 s."),
        # round(1.5 * 24) = 36, floored to the grid: 33 frames.
        (1.5, "the model chose the length: 1.4 s."),
        # clamped to 48 frames, floored to the grid: 41 frames.
        (3.2, "the model chose 3.2 s; your longest length is 2 s, so the video is 1.7 s."),
    ],
    ids=["below", "inside", "above"],
)
def test_the_tiny_pipeline_gives_the_three_notes(tiny, predicted, note, tmp_path):
    with predicting(tiny["base"].duration_head, predicted):
        result = run_tiny_auto(tiny, tmp_path)

    assert chosen(result) == [note], result.warnings
    assert not tiny["base"].duration_head._forward_hooks


def test_reading_the_figure_does_not_change_what_is_sampled(tiny, tmp_path, monkeypatch):
    """The same seed, with and without the reading: the same frames, bit for bit."""
    import av

    def frames_of(result):
        with av.open(str(result.video_path)) as container:
            return np.stack([f.to_ndarray(format="rgb24") for f in container.decode(video=0)])

    with predicting(tiny["base"].duration_head, 0.6):
        read = run_tiny_auto(tiny, tmp_path / "read")

        class NotRead:
            seconds = None

            def __init__(self, pipe):
                pass

            def close(self):
                pass

        monkeypatch.setattr(generate_module, "_DurationReading", NotRead, raising=False)
        unread = run_tiny_auto(tiny, tmp_path / "unread")

    assert chosen(read) != chosen(unread), "the reading was not taken in the first run"
    assert np.array_equal(frames_of(read), frames_of(unread))
