"""Spec for ``ltx25_studio/lora/validate.py`` — PLAN §4.3 stages 1-6.

Upstream's entire validation is one substring test. Everything below is the part
nobody else does for us.

The contract:

* ``validate_lora(source, *, max_bytes=DEFAULT_MAX_LORA_BYTES, fetcher=None)``
  accepts **either** a local path **or** a Hub repo id, and validates both
  identically. A repo id is resolved through ``fetcher`` — injected so the whole
  suite runs with no network.
* Returns a ``LoRAValidation`` report carrying ``path``, ``inspection``,
  ``convention``, ``model_version`` and ``warnings``.
* Stage ordering is observable: the size bound runs **before** any parse, so an
  oversized text file is rejected for its size, not its contents.
* An absent ``model_version`` is a **warning, not a rejection**. Most community
  LoRAs have no metadata at all; being strict here breaks the headline feature.

Note that a report is not permission to generate. Stage 7 (``lora/apply.py``) is
the check that actually protects the user, and it cannot run here.
"""

from __future__ import annotations

import pytest
from conftest import (
    DIFFUSERS_PREFIX,
    NATIVE_CONNECTOR_PREFIX,
    NATIVE_PREFIX,
    OFFICIAL_METADATA,
    TRANSFORMER_MODULE_PATHS,
    lora_pair,
    lora_tensors,
)

from ltx25_studio.errors import (
    LoRAConventionError,
    LoRAError,
    LoRAFormatError,
    LoRAPairingError,
    LoRATooLargeError,
    LoRAVersionError,
)
from ltx25_studio.lora import validate as lora_validate

HUB_REPO_ID = "techfreakworm/ltx25-test-lora"


@pytest.fixture(params=["upload", "hub"])
def validate_via(request):
    """Runs a validation through the direct-upload route or the Hub route.

    Every stage spec runs through both. PLAN §4.3: "Accepts either a Hub repo-id
    or a direct file upload" — one validator, not two code paths that drift.
    """
    route = request.param

    def _run(path, **kwargs):
        if route == "upload":
            return lora_validate.validate_lora(path, **kwargs)
        calls = []

        def fetcher(*args, **kw):
            calls.append((args, kw))
            return path

        unfetched = "the Hub route must resolve the repo id through the injected fetcher"
        try:
            report = lora_validate.validate_lora(HUB_REPO_ID, fetcher=fetcher, **kwargs)
        except LoRAError:
            # A rejection is a legitimate outcome — but it must come from parsing
            # the resolved file, not from refusing to resolve a repo id at all.
            if not calls:
                pytest.fail(unfetched)
            raise
        if not calls:
            pytest.fail(unfetched)
        return report

    _run.route = route
    return _run


# --------------------------------------------------------------------------
# the happy path
# --------------------------------------------------------------------------


def test_a_well_formed_adapter_validates(validate_via, good_lora):
    report = validate_via(good_lora)
    assert report.model_version == "2.5.0"
    assert report.convention is not None
    version_warnings = [w for w in report.warnings if "version" in w.lower()]
    assert version_warnings == [], (
        f"an adapter that declares model_version must not warn about it: {report.warnings}"
    )


def test_the_report_carries_the_resolved_path(validate_via, good_lora):
    report = validate_via(good_lora)
    assert str(report.path) == str(good_lora)


def test_the_hub_route_passes_the_repo_id_to_the_fetcher(good_lora):
    seen = {}

    def fetcher(*args, **kwargs):
        seen["args"] = args
        seen["kwargs"] = kwargs
        return good_lora

    lora_validate.validate_lora(HUB_REPO_ID, fetcher=fetcher)
    rendered = repr(seen)
    assert HUB_REPO_ID in rendered, f"fetcher never saw the repo id: {rendered}"


def test_a_local_file_is_not_sent_to_the_fetcher(good_lora):
    def fetcher(*args, **kwargs):  # pragma: no cover - must not run
        pytest.fail("an existing local file must not be fetched from the Hub")

    lora_validate.validate_lora(good_lora, fetcher=fetcher)


# --------------------------------------------------------------------------
# stage 1 — size bound
# --------------------------------------------------------------------------


def test_an_oversized_upload_is_rejected(validate_via, good_lora):
    with pytest.raises(LoRATooLargeError):
        validate_via(good_lora, max_bytes=8)


def test_the_size_error_names_the_limit_and_the_actual_size(good_lora):
    """The message states both numbers **in bytes** so the user can see the gap."""
    with pytest.raises(LoRATooLargeError) as excinfo:
        lora_validate.validate_lora(good_lora, max_bytes=64)
    message = str(excinfo.value)
    assert "64" in message, message
    assert str(good_lora.stat().st_size) in message, message


def test_the_size_bound_runs_before_any_parse(tmp_path):
    """Stage ordering, proved without a spy.

    The file is both oversized *and* not a safetensors container. If the parser
    ran first we would get a format error; PLAN §4.3 stage 1 says "refuse before
    any download/parse", so it must be the size error.
    """
    path = tmp_path / "huge-notes.txt"
    path.write_bytes(b"x" * 4096)
    with pytest.raises(LoRATooLargeError):
        lora_validate.validate_lora(path, max_bytes=128)


def test_a_file_at_exactly_the_limit_is_accepted(good_lora):
    report = lora_validate.validate_lora(good_lora, max_bytes=good_lora.stat().st_size)
    assert report is not None


def test_the_default_size_bound_does_not_reject_the_official_adapter():
    """PLAN §2.4: the official distilled LoRA is 9.699 GB. A default limit below
    that rejects the very adapter the model ships with."""
    limit = lora_validate.DEFAULT_MAX_LORA_BYTES
    assert limit > 9.699 * 1000**3, f"{limit} bytes would reject the official 9.699 GB adapter"
    assert limit < 78 * 1000**3, "PLAN §2.5 leaves ~78 GB of headroom in total"


# --------------------------------------------------------------------------
# stages 2-3 — format failures reach the caller as typed errors
# --------------------------------------------------------------------------


def test_a_text_file_is_rejected_by_both_routes(validate_via, tmp_path):
    path = tmp_path / "notes.txt"
    path.write_text("I meant to upload the lora but grabbed my notes\n")
    with pytest.raises(LoRAFormatError):
        validate_via(path)


def test_both_routes_reject_a_bad_file_with_the_same_error_type(tmp_path, good_lora):
    path = tmp_path / "notes.txt"
    path.write_text("nope\n")

    with pytest.raises(LoRAError) as direct:
        lora_validate.validate_lora(path)

    def fetcher(*args, **kwargs):
        return path

    with pytest.raises(LoRAError) as hub:
        lora_validate.validate_lora(HUB_REPO_ID, fetcher=fetcher)

    assert type(direct.value) is type(hub.value), (
        "the Hub route and the upload route must validate identically (PLAN §4.3)"
    )


# --------------------------------------------------------------------------
# stage 4 — model_version
# --------------------------------------------------------------------------


def test_the_declared_2_5_0_version_is_accepted(validate_via, good_lora):
    assert validate_via(good_lora).model_version == "2.5.0"


def test_a_2_5_patch_release_is_accepted(validate_via, write_safetensors):
    path = write_safetensors(
        name="v251.safetensors",
        tensors=lora_tensors(NATIVE_PREFIX, TRANSFORMER_MODULE_PATHS),
        metadata=OFFICIAL_METADATA | {"model_version": "2.5.1"},
    )
    assert validate_via(path).model_version == "2.5.1"


def test_a_missing_model_version_is_a_warning_not_a_rejection(validate_via, write_safetensors):
    """THE over-strictness trap. Most community LoRAs carry no ``__metadata__``.

    Rejecting them would break the headline feature for the majority of real
    uploads, so absence warns and proceeds (PLAN §4.3 stage 4: "warn on absent").
    """
    path = write_safetensors(
        name="no-metadata.safetensors",
        tensors=lora_tensors(NATIVE_PREFIX, TRANSFORMER_MODULE_PATHS),
        metadata=None,
    )
    report = validate_via(path)
    assert report.model_version is None
    assert report.warnings, "an absent model_version must produce a visible warning"
    assert any("model_version" in w for w in report.warnings), report.warnings


def test_metadata_present_but_without_a_version_also_only_warns(validate_via, write_safetensors):
    path = write_safetensors(
        name="partial-metadata.safetensors",
        tensors=lora_tensors(NATIVE_PREFIX, TRANSFORMER_MODULE_PATHS),
        metadata={"lora_rank": "64", "lora_alpha": "64"},
    )
    report = validate_via(path)
    assert report.warnings


def test_an_explicitly_incompatible_version_is_rejected(validate_via, write_safetensors):
    path = write_safetensors(
        name="v230.safetensors",
        tensors=lora_tensors(NATIVE_PREFIX, TRANSFORMER_MODULE_PATHS),
        metadata=OFFICIAL_METADATA | {"model_version": "2.3.0"},
    )
    with pytest.raises(LoRAVersionError):
        validate_via(path)


def test_the_version_error_names_the_found_and_the_expected_version(write_safetensors):
    path = write_safetensors(
        name="v230b.safetensors",
        tensors=lora_tensors(NATIVE_PREFIX, TRANSFORMER_MODULE_PATHS),
        metadata=OFFICIAL_METADATA | {"model_version": "2.3.0"},
    )
    with pytest.raises(LoRAVersionError) as excinfo:
        lora_validate.validate_lora(path)
    message = str(excinfo.value)
    assert "2.3.0" in message and "2.5" in message, message


# --------------------------------------------------------------------------
# stage 5 — convention
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "prefix", [NATIVE_PREFIX, DIFFUSERS_PREFIX, NATIVE_CONNECTOR_PREFIX, "connectors."]
)
def test_every_known_convention_is_accepted(prefix, validate_via, write_safetensors):
    path = write_safetensors(
        name="conv.safetensors",
        tensors=lora_tensors(prefix, ("layers.0.proj",)),
        metadata=OFFICIAL_METADATA,
    )
    assert validate_via(path).convention is not None


def test_an_unknown_convention_is_rejected(validate_via, write_safetensors):
    path = write_safetensors(
        name="sdxl-ish.safetensors",
        tensors=lora_tensors("unet.", ("down_blocks.0.resnets.0.conv1",)),
        metadata=OFFICIAL_METADATA,
    )
    with pytest.raises(LoRAConventionError):
        validate_via(path)


def test_the_convention_error_names_the_prefixes_actually_found(write_safetensors):
    tensors = lora_tensors("unet.", ("down_blocks.0.resnets.0.conv1",))
    tensors |= lora_tensors("controlnet.", ("input_blocks.0.0",))
    path = write_safetensors(name="mystery.safetensors", tensors=tensors, metadata=OFFICIAL_METADATA)
    with pytest.raises(LoRAConventionError) as excinfo:
        lora_validate.validate_lora(path)
    message = str(excinfo.value)
    assert "unet." in message and "controlnet." in message, (
        f"PLAN §4.3 stage 5: the error must name what it actually found; got {message!r}"
    )


def test_a_key_without_the_lora_substring_is_rejected(validate_via, write_safetensors):
    """Upstream's only check, done earlier and with a better message."""
    tensors = lora_tensors(NATIVE_PREFIX, ("transformer_blocks.0.attn1.to_q",))
    tensors["diffusion_model.transformer_blocks.0.attn1.to_q.bias"] = ("F32", [8])
    path = write_safetensors(name="stray-key.safetensors", tensors=tensors, metadata=OFFICIAL_METADATA)
    with pytest.raises(LoRAError):
        validate_via(path)


# --------------------------------------------------------------------------
# stage 6 — A/B pairing
# --------------------------------------------------------------------------


def test_a_correctly_paired_peft_adapter_passes(validate_via, good_lora):
    assert validate_via(good_lora) is not None


def test_a_correctly_paired_kohya_adapter_passes(validate_via, write_safetensors):
    path = write_safetensors(
        name="kohya.safetensors",
        tensors=lora_tensors(NATIVE_PREFIX, TRANSFORMER_MODULE_PATHS, style="kohya"),
        metadata=OFFICIAL_METADATA,
    )
    assert validate_via(path) is not None


def test_lora_a_without_its_b_is_rejected(validate_via, write_safetensors):
    tensors = lora_tensors(NATIVE_PREFIX, TRANSFORMER_MODULE_PATHS)
    del tensors[f"{NATIVE_PREFIX}{TRANSFORMER_MODULE_PATHS[0]}.lora_B.weight"]
    path = write_safetensors(name="unpaired-a.safetensors", tensors=tensors, metadata=OFFICIAL_METADATA)
    with pytest.raises(LoRAPairingError):
        validate_via(path)


def test_lora_b_without_its_a_is_rejected(validate_via, write_safetensors):
    tensors = lora_tensors(NATIVE_PREFIX, TRANSFORMER_MODULE_PATHS)
    del tensors[f"{NATIVE_PREFIX}{TRANSFORMER_MODULE_PATHS[0]}.lora_A.weight"]
    path = write_safetensors(name="unpaired-b.safetensors", tensors=tensors, metadata=OFFICIAL_METADATA)
    with pytest.raises(LoRAPairingError):
        validate_via(path)


def test_lora_down_without_its_up_is_rejected(validate_via, write_safetensors):
    tensors = lora_tensors(NATIVE_PREFIX, TRANSFORMER_MODULE_PATHS, style="kohya")
    del tensors[f"{NATIVE_PREFIX}{TRANSFORMER_MODULE_PATHS[0]}.lora_up.weight"]
    path = write_safetensors(name="unpaired-down.safetensors", tensors=tensors, metadata=OFFICIAL_METADATA)
    with pytest.raises(LoRAPairingError):
        validate_via(path)


def test_the_pairing_error_names_the_unpaired_key(write_safetensors):
    tensors = lora_pair(NATIVE_PREFIX, "transformer_blocks.0.attn1.to_q")
    tensors |= lora_pair(NATIVE_PREFIX, "transformer_blocks.1.attn1.to_v")
    orphan = f"{NATIVE_PREFIX}transformer_blocks.1.attn1.to_v.lora_B.weight"
    del tensors[orphan]
    path = write_safetensors(name="orphan.safetensors", tensors=tensors, metadata=OFFICIAL_METADATA)
    with pytest.raises(LoRAPairingError) as excinfo:
        lora_validate.validate_lora(path)
    message = str(excinfo.value)
    assert "transformer_blocks.1.attn1.to_v" in message, message
    assert "transformer_blocks.0.attn1.to_q" not in message, (
        f"only the unpaired key should be named; got {message!r}"
    )


# --------------------------------------------------------------------------
# the wrong-architecture adapter survives to stage 7 — PLAN §9.3.1
# --------------------------------------------------------------------------


def test_a_wrong_architecture_adapter_passes_stages_one_to_six(validate_via, wrong_architecture_lora):
    """This is not a bug in the validator — it is the point.

    With no transformer to compare against — which is the pre-load UI case, and
    the Hub case before anything is downloaded — nothing readable from the header
    tells you these module paths do not exist in *this* model. Stages 1-6 pass,
    stage 6.5 cannot run, and the last line of defence is stage 7 in
    ``lora/apply.py``. (Given a transformer, ``test_lora_compat.py`` requires 6.5
    to catch it earlier.)
    """
    report = validate_via(wrong_architecture_lora)
    assert report.model_version == "2.5.0"
    assert report.convention is not None
