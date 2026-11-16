"""No inert levers (review round 2, F6; AC-15).

The decoder choice is a lever only while Upscale ×2 is off: stage 2 always finishes with the
model's diffusion decoder (``pipelines.StageRunner``), so under Upscale the choice is shown
but cannot be changed, and the line under it says why. Proved through the real handlers and,
for the reason itself, through a real ``StageRunner`` over fake stages.
"""

from __future__ import annotations

import pytest
from test_decode_policy import OK, runner_on
from test_ui_form import Form

from ltx25_studio import modes
from ltx25_studio.ui import render

gr = pytest.importorskip("gradio", reason="gradio is a declared dev dependency")

PICTURE_MODES = [m for m, s in modes.MODES.items() if "video" in s.exports and not s.skip_stage_2]


@pytest.fixture(scope="module")
def demo():
    from ltx25_studio.ui import build_ui

    return build_ui()


def test_stage_2_finishes_with_the_diffusion_decoder_whatever_the_choice():
    """Why the choice is not a lever under Upscale: the real stage runner ignores it."""
    ran = set()
    for policy in ("auto", "diffusion", "conv"):
        runner, _pipeline, _built = runner_on("cuda")
        runner(prompt="a", video_decoder=policy, enable_stage_2=True, **OK)
        ran.add(runner.last_run["decoder"])
    assert ran == {"diffusion"}
    single = set()
    for policy in ("diffusion", "conv"):
        runner, _pipeline, _built = runner_on("cuda")
        runner(prompt="a", video_decoder=policy, **OK)
        single.add(runner.last_run["decoder"])
    assert single == {"diffusion", "conv"}, "without Upscale the choice is a real lever"


@pytest.mark.parametrize("mode_id", PICTURE_MODES)
def test_the_decoder_choice_cannot_be_changed_while_upscale_is_on(demo, mode_id):
    form = Form(demo)
    decoder, line = form.fields["decoder"], form.by_id("decoder-line")
    form.pick_mode(mode_id)
    form.user(form.fields["decoder"], "conv")
    assert form.interactive(decoder) and form.get(line) == render.decoder_line("conv")

    form.user(form.fields["enable_stage_2"], True)
    assert not form.interactive(decoder), f"{mode_id}: the decoder is a dead lever under Upscale"
    assert form.get(line) == render.UPSCALE_DECODER_LINE

    form.user(form.fields["enable_stage_2"], False)
    assert form.interactive(decoder) and form.get(line) == render.decoder_line("conv")


def test_upscale_carried_into_another_mode_keeps_the_decoder_locked(demo):
    form = Form(demo)
    form.user(form.fields["enable_stage_2"], True)
    form.pick_mode("text_to_audio")  # no picture: no decoder, no upscale
    assert not form.visible(form.fields["decoder"])
    form.pick_mode("image_text_to_video")
    assert form.field("enable_stage_2") is True
    assert not form.interactive(form.fields["decoder"])
    assert form.get(form.by_id("decoder-line")) == render.UPSCALE_DECODER_LINE
