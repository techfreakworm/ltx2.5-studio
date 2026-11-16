"""The LoRA bay, through the real validator and the real staging module (B8).

* an accepted adapter is staged by ``lora.staged_entry``: strength 1.0, a
  content key, the validator's report;
* the same adapter checked twice is staged once (``lora.find_duplicate``);
* strength goes through ``lora.check_weight``, never a UI copy of the range;
* an upload is copied into an app-owned folder before it is staged, so
  Gradio's cache clean-up cannot remove it before Generate (F27);
* rejection is a plain headline with the validator's message verbatim.
"""

from __future__ import annotations

import os

import pytest
from lora_fixtures import OFFICIAL_METADATA, good_adapter, wrong_dimension_adapter

from ltx25_studio import lora
from ltx25_studio.ui import handlers, render, uploads


@pytest.fixture(autouse=True)
def upload_dir(tmp_path, monkeypatch):
    target = tmp_path / "owned"
    monkeypatch.setenv(uploads.UPLOADS_ENV, str(target))
    return target


def test_an_accepted_upload_is_staged_at_strength_one_from_an_app_owned_copy(tmp_path, upload_dir):
    adapter = good_adapter(tmp_path / "gradio-cache")
    verdict, staged, _summary = handlers.on_validate_lora("", str(adapter), [])

    assert len(staged) == 1, verdict
    entry = staged[0]
    assert entry["weight"] == lora.staging.DEFAULT_WEIGHT == 1.0
    assert entry["content_key"], "duplicates are found by content key"
    assert os.path.commonpath([entry["path"], str(upload_dir)]) == str(upload_dir), (
        "the staged path must be the app's own copy, not Gradio's cache file"
    )
    assert entry["path"] != str(adapter)
    adapter.unlink()  # Gradio's cache clean-up
    assert os.path.isfile(entry["path"]), "the staged adapter vanished with the upload"
    assert verdict.startswith("#### Accepted: ")


def test_the_same_adapter_checked_twice_is_staged_once(tmp_path):
    first = good_adapter(tmp_path / "a")
    renamed = good_adapter(tmp_path / "b", name="same-adapter-other-name.safetensors")
    _v, staged, _s = handlers.on_validate_lora("", str(first), [])
    verdict, staged, _s = handlers.on_validate_lora("", str(renamed), staged)

    assert len(staged) == 1
    assert verdict.startswith("#### Already staged")


def test_a_hub_id_is_fetched_through_the_lora_package(tmp_path, monkeypatch):
    adapter = good_adapter(tmp_path)
    monkeypatch.setattr(lora, "fetch_lora", lambda source: adapter)
    _v, staged, _s = handlers.on_validate_lora("org/my-lora", None, [])
    assert staged[0]["source"] == "org/my-lora"
    assert staged[0]["path"] == str(adapter)


def test_a_path_typed_into_the_hub_box_is_refused(tmp_path):
    adapter = good_adapter(tmp_path)
    verdict, staged, _s = handlers.on_validate_lora(str(adapter), None, [])
    assert staged == []
    assert verdict.startswith("### Rejected:")


def test_a_rejection_quotes_the_validator_verbatim_under_a_plain_headline(tmp_path):
    adapter = wrong_dimension_adapter(tmp_path)
    verdict, staged, _s = handlers.on_validate_lora("", str(adapter), [])

    assert staged == []
    copy = sorted(uploads.uploads_dir().glob("*/*.safetensors"))[-1]
    with pytest.raises(lora.LoRAError) as raised:
        lora.validate_lora(str(copy), upload=True)
    assert verdict.startswith("### Rejected: ")
    assert str(raised.value).strip() in verdict, "the validator's message, verbatim"
    assert "LoRA" + "DimensionError" not in verdict.split("\n")[0]
    assert "Nothing was spent." in verdict


def test_strength_goes_through_the_lora_modules_own_check(tmp_path):
    _v, staged, _s = handlers.on_validate_lora("", str(good_adapter(tmp_path)), [])
    staged = handlers.on_lora_weight(staged, 0, 0.65)
    assert staged[0]["weight"] == 0.65
    with pytest.raises(lora.LoRAWeightError):
        handlers.on_lora_weight(staged, 0, 0)
    with pytest.raises(lora.LoRAWeightError):
        handlers.on_lora_weight(staged, 0, lora.staging.MAX_WEIGHT + 0.5)


def test_remove_takes_out_exactly_one_adapter(tmp_path):
    staged = [{"source": "a", "weight": 1.0}, {"source": "b", "weight": 0.5}]
    assert handlers.on_remove_lora(staged, 0) == [{"source": "b", "weight": 0.5}]
    assert handlers.on_remove_lora(staged, 5) == staged


def test_no_more_than_the_bays_rows_are_staged(tmp_path):
    staged = []
    for index in range(handlers.MAX_STAGED_LORAS + 1):
        # Different metadata, so a different header and a distinct adapter.
        path = good_adapter(
            tmp_path / str(index), metadata=OFFICIAL_METADATA | {"ss_tag": f"adapter-{index}"}
        )
        staged_before = list(staged)
        verdict, staged, _s = handlers.on_validate_lora("", str(path), staged)
        if index >= handlers.MAX_STAGED_LORAS:
            assert staged == staged_before and verdict.startswith("#### Not added")


def test_the_lora_row_says_no_lora_in_words_and_lists_strengths():
    assert render.lora_summary_label([]) == "LoRA: none. Add an adapter"
    assert render.lora_idle() == "No LoRA: runs use the base model."
    two = [{"source": "a", "weight": 0.8}, {"source": "b", "weight": 1.0}]
    assert render.lora_summary_label(two) == "LoRA: 2 adapters at 0.8 and 1"


def test_the_staged_strengths_appear_in_the_cost_panel():
    facts = render.CostFacts(
        mode_name="m",
        state="ready",
        loras=(("style.safetensors", 0.8),),
        reserve=240,
        base=180,
        extras=(("LoRA", 60),),
    )
    panel = render.cost_panel(facts)
    assert "style.safetensors" in panel and ">0.8<" in panel
    assert "LoRA +60 s" in panel and "is-lit" in panel


def test_the_strength_slider_never_offers_more_than_the_lora_module_accepts():
    from ltx25_studio.ui import blocks

    assert 0 < blocks.LORA_SLIDER_MAX <= lora.staging.MAX_WEIGHT
