"""The decoder choices are named by what they do (#27).

On the Space the same seed and prompt through both decoders gave the same scene: the diffusion
decoder a little softer and steadier from frame to frame, the standard (conv) decoder sharper
frame by frame with more shimmer (``evidence/space/matrix/ab_decoder/measure.txt``). The page
called the softer one "Best quality" and the sharper one "Fast". Now the choices read "Auto",
"Steadier (diffusion decoder)" and "Sharper (standard decoder)" everywhere a user reads them:
the choice, its line, the Upscale lock line, the Done card and the backend's refusals. Auto stays
the vendor's default, and its line is true on every machine: off an NVIDIA GPU, Auto always
decodes with the standard decoder (``pipelines.StageRunner._single_stage_decoder``). The values
the API sends (``auto``, ``diffusion``, ``conv``) do not change.
"""

from __future__ import annotations

import pytest

from ltx25_studio import errors, modes
from ltx25_studio.config import DECODER_CHOICES
from ltx25_studio.ui import render

OLD_NAMES = ("Best quality", "Fast")
LABELS = ("Auto", "Steadier (diffusion decoder)", "Sharper (standard decoder)")


def test_the_choices_are_named_by_what_they_do_and_send_the_same_values():
    assert [label for label, _v, _w in render.DECODER_OPTIONS] == list(LABELS)
    assert [value for _l, value, _w in render.DECODER_OPTIONS] == list(DECODER_CHOICES)
    assert list(DECODER_CHOICES) == ["auto", "diffusion", "conv"]
    lines = {value: words for _l, value, words in render.DECODER_OPTIONS}
    assert lines["diffusion"] == (
        "A little softer, steadier from frame to frame; takes a few seconds longer."
    )
    assert lines["conv"] == "Sharper frame by frame, with a little more shimmer; faster."
    # Auto never takes the diffusion decoder off an NVIDIA (CUDA) GPU, and on one only where
    # it can run (loaded, and the video at least its minimum size).
    assert (
        lines["auto"]
        == "Diffusion decoder on an NVIDIA GPU that can run it, standard decoder elsewhere."
    )
    assert render.UPSCALE_DECODER_LINE == (
        "Upscale ×2 always finishes with the diffusion decoder, whichever you pick here."
    )


def test_the_done_card_names_the_decoder_that_ran_by_its_choice():
    for value, label in (("diffusion", LABELS[1]), ("conv", LABELS[2])):
        facts = render.RunFacts(
            mode=modes.get_mode("text_to_video"),
            seed=1,
            returned_size=(704, 480),
            seconds=5.0,
            stage_2=False,
            loras=(),
            took=3.0,
            decoder=value,
        )
        assert f"<dt>Decoder</dt><dd>{render._e(label)}</dd>" in render.run_summary(facts)


def test_the_backends_refusals_use_the_same_names():
    small = str(
        errors.DecoderTooSmallError(
            output=(192, 128, 9), minimum=(224, 224, 17), smallest=(224, 224, 17), stage_2=False
        )
    )
    assert small.startswith("Steadier (diffusion decoder) cannot decode a video this small")
    assert small.endswith("Make the video larger or longer, or choose Sharper (standard decoder).")
    upscale = str(
        errors.DecoderTooSmallError(
            output=(192, 128, 9), minimum=(224, 224, 17), smallest=(128, 128, 17), stage_2=True
        )
    )
    assert upscale.startswith("Upscale ×2 cannot finish") and upscale.endswith(
        "turn Upscale ×2 off."
    )
    unavailable = str(
        errors.DecoderUnavailableError(reason="it needs an NVIDIA (CUDA) GPU at this size")
    )
    assert unavailable == (
        "Steadier (diffusion decoder) was chosen but cannot run here: it needs an NVIDIA (CUDA) "
        "GPU at this size. Choose Sharper (standard decoder), or Auto, which picks the decoder "
        "that can run."
    )
    for text in (small, upscale, unavailable):
        for old in OLD_NAMES:
            assert old not in text, (old, text)


def test_the_built_page_offers_the_new_names_in_every_mode_and_never_the_old(tmp_path, monkeypatch):
    pytest.importorskip("gradio", reason="gradio is a declared dev dependency")
    import re

    from test_ui_form import Form

    from ltx25_studio.ui import build_ui, handlers, showcase

    monkeypatch.setenv(showcase.EXAMPLES_ENV, str(tmp_path / "no-examples"))
    form = Form(build_ui())
    decoder = form.fields["decoder"]
    assert [label for label, _value in form.props[decoder._id]["choices"]] == list(LABELS)
    old = re.compile(r"Best quality|\bFast\b")
    for mode_id in modes.MODES:
        form.pick_mode(mode_id)
        for block in form.demo.blocks.values():
            words = [form.get(block), *(form.props[block._id].get(k) for k in ("label", "info"))]
            words += [str(choice) for choice in form.props[block._id].get("choices") or ()]
            for text in (w for w in words if isinstance(w, str)):
                assert not old.search(text), (mode_id, text)
    _shown, free, line = handlers.decoder_view("text_to_video", True, "conv")
    assert not free and line == render.UPSCALE_DECODER_LINE
