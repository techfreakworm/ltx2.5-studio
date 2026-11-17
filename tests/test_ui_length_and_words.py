"""The words the round-4 review found untrue or in a builder's terms (minor 14).

* The Done card said "chosen by the model" for a model-picked length even when the range chose
  it: diffusers clamps the duration head's figure to the shortest and longest the user allowed,
  so a length at either end may be the range's, not the model's (issue #16: the head said
  4.47 s, the range made it 5.0 s, and the page said "chosen by the model"). Now the card says
  "chosen by the model" only for a length strictly inside the range, which only the model's
  own figure gives; at an end it says which end ("the shortest you allowed"), true whichever
  decided. The run's note says what the model chose.
* "How quota works" told a timed-out run to press Generate again; the timeout card told it to
  make the run shorter or smaller first. Both now give the card's step.
* "legal" (a frame count, the Snap button, the length slider) and "single pass" were a
  builder's words on the page.
"""

from __future__ import annotations

import re

import pytest
from test_exports_contract import factory_returning, one_muxed_file_encoder

from ltx25_studio import generate as generate_module
from ltx25_studio import modes
from ltx25_studio.ui import handlers, render
from ltx25_studio.ui.fields import FIELD_KEYS, FIELDS

#: What the fake pipeline renders: 121 frames, 5.04 s at 24 fps (a length on the 8k + 1 grid).
FRAMES = 121


class Rendered:
    def __init__(self):
        self.frames = [list(range(FRAMES))]
        self.audio = ("sample",)


class Pipe:
    def __call__(self, **kwargs):
        return Rendered()


@pytest.fixture
def wired(tmp_path, monkeypatch):
    real = generate_module.generate
    monkeypatch.setattr(
        generate_module,
        "generate",
        lambda **params: real(
            pipeline_factory=factory_returning(Pipe()),
            encoder=one_muxed_file_encoder,
            output_root=tmp_path / "runs",
            retention_bytes=10**9,
            **params,
        ),
    )


def visible(markup: str) -> str:
    return " ".join(re.sub(r"<[^>]+>", " ", markup).replace(" ", " ").split())


def length_row(shortest: float, longest: float) -> str:
    values = {spec.key: spec.default for spec in FIELDS} | {
        "prompt": "a lighthouse keeper climbs a spiral staircase",
        "auto_duration": True,
        "min_seconds": shortest,
        "max_seconds": longest,
    }
    outcome = handlers.run("text_to_video", *(values[k] for k in FIELD_KEYS))
    assert outcome.error is None, outcome.status
    assert outcome.result.record.frames == FRAMES
    assert outcome.result.record.length_source == generate_module.LENGTH_FROM_MODEL
    summary = visible(render.run_summary(handlers.run_facts(outcome)))
    return re.search(r"Length (.*?) Seed", summary).group(1)


@pytest.mark.parametrize(
    ("shortest", "longest", "row"),
    [
        # 121 frames lie strictly inside 73..233: only the model's own figure gives that.
        (3.0, 10.0, "5.0 s, chosen by the model"),
        # The shortest 5 s allows is 121 frames: the range may have raised the model's figure.
        (5.0, 10.0, "5.0 s, the shortest you allowed"),
        # The longest 5.1 s allows is 121 frames: the range may have lowered it.
        (2.0, 5.1, "5.0 s, the longest you allowed"),
    ],
)
def test_the_done_card_says_chosen_by_the_model_only_inside_the_range(
    wired, shortest, longest, row
):
    assert length_row(shortest, longest) == row


def test_the_ends_of_the_range_are_the_duration_heads_own():
    """The card reads the ends the way the pipeline clamps (``constraints``, which mirrors
    diffusers' ``predict_num_frames``)."""
    from ltx25_studio import constraints

    assert constraints.duration_head_frame_range(
        min_seconds=5.0, max_seconds=10.0, frame_rate=24.0
    ) == (121, 233)
    assert constraints.duration_head_frame_range(
        min_seconds=2.0, max_seconds=5.1, frame_rate=24.0
    ) == (49, 121)


def test_how_quota_works_gives_the_timeout_cards_step():
    timeout_step = render._CARDS["timeout"][1]
    assert timeout_step.startswith("Make the video shorter or smaller")
    explainer = render.quota_explainer(multiplier=2, example_reserve=60)
    assert timeout_step in explainer
    assert "press Generate again." not in explainer.replace(timeout_step, "")


def test_no_builder_word_is_on_the_page_in_any_mode(tmp_path, monkeypatch):
    pytest.importorskip("gradio", reason="gradio is a declared dev dependency")
    from test_ui_form import Form

    from ltx25_studio.ui import build_ui, showcase

    monkeypatch.setenv(showcase.EXAMPLES_ENV, str(tmp_path / "no-examples"))
    form = Form(build_ui())
    banned = re.compile(r"\blegal\b|single pass", re.IGNORECASE)
    for mode_id in modes.MODES:
        form.pick_mode(mode_id)
        for block in form.demo.blocks.values():
            props = form.props[block._id]
            words = [form.get(block), props.get("label"), props.get("info")]
            for text in (w for w in words if isinstance(w, str)):
                assert not banned.search(text), (mode_id, text)
    assert not banned.search(render.frame_hint(121, ratio=8, lower=121, upper=121, valid=True))
    assert render.frame_hint(121, ratio=8, lower=121, upper=121, valid=True) == (
        "121 frames runs (5.04 s)."
    )
