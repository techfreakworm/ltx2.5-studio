"""A LoRA rejection gives one next step, and off ZeroGPU says nothing of spending (minor 3).

The rejection card said "Rejected: the adapter's shapes don't fit LTX-2.5. ltxv-2b-detail.safetensors
was not staged. Nothing was spent." and then the validator's words: no next step, against the
page's own rule (a failure says what happened and one step), and "Nothing was spent." on a
machine with nothing to spend. Now each rejection kind has its step ("Use an adapter made for
LTX-2.5.", "Download the file again, then upload it.", ...), and "Nothing was spent." is the
Space's sentence alone; "spent" joins the meter words the page may not use off ZeroGPU.
"""

from __future__ import annotations

import pytest
from lora_fixtures import wrong_dimension_adapter

from ltx25_studio import errors
from ltx25_studio.ui import handlers, render

KINDS = [cls for cls, _headline in render._LORA_HEADLINES]
#: The kinds whose remedy is another adapter, one made for this model.
WRONG_MODEL = (
    errors.LoRAVersionError,
    errors.LoRAConventionError,
    errors.LoRADimensionError,
    errors.LoRAModuleNotFoundError,
    errors.LoRANotAttachedError,
)


def made(cls, text: str = "the validator's own words") -> BaseException:
    exc = cls.__new__(cls)
    Exception.__init__(exc, text)
    return exc


def step_of(card: str) -> str:
    _head, body, _quote = card.split("\n\n")
    return body.split("was not staged.", 1)[1].replace("Nothing was spent.", "").strip()


@pytest.mark.parametrize("cls", KINDS, ids=lambda cls: cls.__name__)
def test_every_rejection_kind_gives_one_next_step(cls):
    card = render.lora_rejected("style.safetensors", made(cls), on_zerogpu=False)
    step = step_of(card)
    assert step and step.endswith("."), card
    assert step == render.lora_step(made(cls))
    if cls in WRONG_MODEL:
        assert step == "Use an adapter made for LTX-2.5."
    assert card.endswith("> the validator's own words"), "the validator's words stay verbatim"


def test_the_other_kinds_name_their_own_remedy():
    assert "Hub id" in render.lora_step(made(errors.LoRAFetchError))
    assert "IC LoRA" in render.lora_step(made(errors.LoRAInContextError))
    assert "again" in render.lora_step(made(errors.LoRATruncatedError))
    assert "report it" in render.lora_step(RuntimeError("an error the app has never seen"))


@pytest.mark.parametrize("on_zerogpu", [True, False])
def test_nothing_was_spent_is_the_spaces_sentence_alone(on_zerogpu):
    card = render.lora_rejected("style.safetensors", made(errors.LoRADimensionError), on_zerogpu=on_zerogpu)
    assert ("Nothing was spent." in card) is on_zerogpu


def test_the_handler_words_the_card_for_this_platform(tmp_path):
    verdict, staged, _summary = handlers.on_validate_lora("", str(wrong_dimension_adapter(tmp_path)), [])
    assert staged == [] and verdict.startswith("### Rejected: ")
    assert ("Nothing was spent." in verdict) is handlers.run_platform()[1]
    assert "Use an adapter made for LTX-2.5." in verdict


@pytest.mark.parametrize("on_zerogpu", [True, False])
def test_ready_says_nothing_of_spending_off_zerogpu(on_zerogpu):
    words = render.ready_line(on_zerogpu=on_zerogpu)
    assert words.startswith("Ready.")
    assert ("spent" in words) is on_zerogpu
