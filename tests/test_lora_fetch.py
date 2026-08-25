"""Spec for ``ltx25_studio/lora/fetch.py`` — the Hub route, PLAN §4.6.

The route existed as an interface and nothing else: ``validate_lora`` carried
a ``fetcher`` seam, the UI carried a repo-id textbox, and no fetcher was ever
written — so every repo id anyone typed rejected with "cannot be read",
including the UI's own placeholder.

Three properties are the point, and each has its own tests:

* **the size bound binds BEFORE the download** — an oversized adapter is
  refused from the Hub's metadata listing, and the download seam is provably
  never called;
* **ambiguity is refused, convention is honoured** — a multi-file repo picks
  ``pytorch_lora_weights.safetensors`` because that is the file diffusers'
  own ``load_lora_weights(repo_id)`` would take (published convention, not a
  guess), and anything else ambiguous refuses naming the candidates;
* **the handler stages a LOCAL path** — the fetch happens in the web process,
  so the billed GPU worker only ever reads a file.

Everything runs offline: the listing and the download are injected seams, and
the Hub's failure classes are stood in for by name (the mapper matches on the
MRO's class names precisely so these tests need no huggingface_hub import).
"""

from __future__ import annotations

from pathlib import Path

import pytest
from conftest import (
    NATIVE_PREFIX,
    OFFICIAL_METADATA,
    TRANSFORMER_MODULE_PATHS,
    build_safetensors_bytes,
    lora_tensors,
)

from ltx25_studio.errors import LoRAFetchError, LoRATooLargeError
from ltx25_studio.lora import DIFFUSERS_LORA_WEIGHT_NAME, fetch
from ltx25_studio.lora.fetch import fetch_lora

CAP = 1000


def _fetcher(files, downloaded=None, cap=CAP):
    """A fetch_lora partial over injected seams, recording download calls."""
    calls = downloaded if downloaded is not None else []

    def download(repo_id, filename):
        calls.append((repo_id, filename))
        return f"/cache/{repo_id}/{filename}"

    def run(source):
        return fetch_lora(source, max_bytes=cap, repo_files=lambda repo: files, download=download)

    return run, calls


# --------------------------------------------------------------------------
# source parsing — exactly the two shapes the UI's helper text promises
# --------------------------------------------------------------------------


def test_a_bare_repo_id_resolves_its_single_safetensors_file():
    run, calls = _fetcher({"adapter.safetensors": 512, "README.md": 10})
    assert run("org/name") == Path("/cache/org/name/adapter.safetensors")
    assert calls == [("org/name", "adapter.safetensors")]


def test_a_repo_id_with_an_embedded_filename_fetches_that_file():
    run, calls = _fetcher({"sub/dir/mine.safetensors": 512, "other.safetensors": 512})
    assert run("org/name/sub/dir/mine.safetensors") == Path(
        "/cache/org/name/sub/dir/mine.safetensors"
    )
    assert calls == [("org/name", "sub/dir/mine.safetensors")]


def test_a_url_is_refused_naming_the_accepted_forms():
    """A pasted Hub URL carries revision segments this route does not honour;
    half-parsing it would fetch something other than what was pointed at."""
    run, calls = _fetcher({"adapter.safetensors": 512})
    with pytest.raises(LoRAFetchError) as excinfo:
        run("https://huggingface.co/org/name/blob/main/adapter.safetensors")
    assert "org/name" in str(excinfo.value)
    assert calls == []


def test_a_single_segment_source_is_refused():
    run, _ = _fetcher({"adapter.safetensors": 512})
    with pytest.raises(LoRAFetchError) as excinfo:
        run("just-a-name")
    assert "org/name" in str(excinfo.value)


# --------------------------------------------------------------------------
# file selection — convention honoured, ambiguity refused with the list
# --------------------------------------------------------------------------


def test_a_repo_with_no_safetensors_refuses_and_names_what_is_there():
    run, calls = _fetcher({"README.md": 10, "adapter.bin": 512})
    with pytest.raises(LoRAFetchError) as excinfo:
        run("org/name")
    assert "no .safetensors" in str(excinfo.value)
    assert "adapter.bin" in str(excinfo.value)
    assert calls == []


def test_multiple_files_prefer_the_diffusers_convention_name():
    """Not a guess: pytorch_lora_weights.safetensors is the file diffusers'
    own load_lora_weights(repo_id) defaults to (lora_base.py,
    LORA_WEIGHT_NAME_SAFE), so a bare repo id means the same file to us as it
    would to the loader itself."""
    run, calls = _fetcher({DIFFUSERS_LORA_WEIGHT_NAME: 512, "backup.safetensors": 512})
    run("org/name")
    assert calls == [("org/name", DIFFUSERS_LORA_WEIGHT_NAME)]


def test_multiple_files_without_the_convention_refuse_and_list_candidates():
    run, calls = _fetcher({"a.safetensors": 512, "b.safetensors": 512})
    with pytest.raises(LoRAFetchError) as excinfo:
        run("org/name")
    message = str(excinfo.value)
    assert "a.safetensors" in message and "b.safetensors" in message
    assert "org/name/<file>" in message, "the refusal must say HOW to disambiguate"
    assert calls == []


def test_an_explicit_filename_that_does_not_exist_refuses_and_lists():
    run, calls = _fetcher({"real.safetensors": 512})
    with pytest.raises(LoRAFetchError) as excinfo:
        run("org/name/imagined.safetensors")
    assert "real.safetensors" in str(excinfo.value)
    assert calls == []


def test_an_explicit_non_safetensors_filename_is_refused():
    run, calls = _fetcher({"adapter.bin": 512})
    with pytest.raises(LoRAFetchError) as excinfo:
        run("org/name/adapter.bin")
    assert "safetensors" in str(excinfo.value)
    assert calls == []


# --------------------------------------------------------------------------
# the size bound binds BEFORE the download
# --------------------------------------------------------------------------


def test_an_oversized_file_is_refused_from_metadata_without_downloading():
    run, calls = _fetcher({"adapter.safetensors": CAP + 1})
    with pytest.raises(LoRATooLargeError) as excinfo:
        run("org/name")
    assert calls == [], "the bound must come from the listing, not from downloading past it"
    assert str(CAP + 1) in str(excinfo.value) and str(CAP) in str(excinfo.value)
    assert "org/name/adapter.safetensors" in str(excinfo.value)


def test_a_listing_without_a_size_refuses_rather_than_downloading_unbounded():
    """'Could not check' must not become 'checked and fine': a sizeless
    listing gives the cap nothing to bind, and the cap is why we looked."""
    run, calls = _fetcher({"adapter.safetensors": None})
    with pytest.raises(LoRAFetchError) as excinfo:
        run("org/name")
    assert "unbounded" in str(excinfo.value)
    assert calls == []


def test_a_file_exactly_at_the_cap_downloads():
    run, calls = _fetcher({"adapter.safetensors": CAP})
    run("org/name")
    assert calls == [("org/name", "adapter.safetensors")]


# --------------------------------------------------------------------------
# the Hub's own failure shapes, mapped to actionable messages
# --------------------------------------------------------------------------

# Stand-ins matched by MRO class NAME, exactly as the mapper does — so the
# offline gate needs no huggingface_hub import and a hub module rename cannot
# crash the mapping.


class RepositoryNotFoundError(Exception):
    pass


class GatedRepoError(RepositoryNotFoundError):
    pass


def _failing_fetcher(exc):
    def raising_listing(repo_id):
        raise exc

    def run(source):
        return fetch_lora(
            source, max_bytes=CAP, repo_files=raising_listing, download=lambda *a: "/x"
        )

    return run


def test_a_missing_repo_names_the_private_repo_ambiguity():
    """The Hub deliberately reports private and nonexistent repos the same
    way; a message that only says 'not found' sends the user to check one of
    the two possible causes."""
    with pytest.raises(LoRAFetchError) as excinfo:
        _failing_fetcher(RepositoryNotFoundError("404"))("org/name")
    message = str(excinfo.value)
    assert "not found" in message and "private" in message


def test_a_gated_repo_names_the_action_not_just_the_state():
    """GatedRepoError subclasses RepositoryNotFoundError, so a naive
    order-of-checks reports every gated repo as missing — the gated message
    is the one with a step the user can actually take."""
    with pytest.raises(LoRAFetchError) as excinfo:
        _failing_fetcher(GatedRepoError("403"))("org/name")
    assert "gated" in str(excinfo.value)
    assert "terms" in str(excinfo.value)


def test_an_unrecognised_listing_failure_is_wrapped_verbatim():
    with pytest.raises(LoRAFetchError) as excinfo:
        _failing_fetcher(OSError("connection reset by peer"))("org/name")
    assert "connection reset by peer" in str(excinfo.value)


def test_a_failing_download_is_wrapped_verbatim():
    def download(repo_id, filename):
        raise OSError("disk full")

    with pytest.raises(LoRAFetchError) as excinfo:
        fetch_lora(
            "org/name",
            max_bytes=CAP,
            repo_files=lambda repo: {"adapter.safetensors": 512},
            download=download,
        )
    assert "disk full" in str(excinfo.value)


def test_the_default_seams_are_huggingface_hub(monkeypatch):
    """The injected seams must default to the real Hub machinery — a fetcher
    whose defaults are stubs is the §8 inert-guard shape with extra steps."""
    import inspect as _inspect

    source = _inspect.getsource(fetch)
    assert "hf_hub_download" in source
    assert "files_metadata=True" in source, (
        "the listing must request file metadata — sizes are what the bound binds on"
    )


# --------------------------------------------------------------------------
# the handler stages a LOCAL path — the fetch lives in the web process
# --------------------------------------------------------------------------


def _valid_adapter(tmp_path):
    adapter = tmp_path / "fetched.safetensors"
    adapter.write_bytes(
        build_safetensors_bytes(
            lora_tensors(NATIVE_PREFIX, TRANSFORMER_MODULE_PATHS), OFFICIAL_METADATA
        )
    )
    return adapter


def test_the_handler_resolves_a_repo_id_and_stages_the_local_path(tmp_path, monkeypatch):
    """The whole point of the placement: by the time anything is billed, the
    adapter is a file read. The staged entry carries the fetched local path
    while 'source' stays what the user typed, so the UI displays the repo id
    and the worker never sees it."""
    from ltx25_studio import lora as lora_package
    from ltx25_studio.ui import handlers

    adapter = _valid_adapter(tmp_path)
    fetched = []

    def fake_fetch(source):
        fetched.append(source)
        return adapter

    monkeypatch.setattr(lora_package, "fetch_lora", fake_fetch)
    verdict, staged, summary = handlers.on_validate_lora("org/my-lora", None, [])

    assert fetched == ["org/my-lora"], "the repo id must resolve through the fetcher"
    assert len(staged) == 1
    assert staged[0]["source"] == "org/my-lora"
    assert staged[0]["path"] == str(adapter)
    assert "org/my-lora" in verdict


def test_the_handler_surfaces_a_fetch_refusal_verbatim(tmp_path, monkeypatch):
    from ltx25_studio import lora as lora_package
    from ltx25_studio.ui import handlers

    def fake_fetch(source):
        raise LoRAFetchError(source=source, reason="'org/gone' was not found on the Hub")

    monkeypatch.setattr(lora_package, "fetch_lora", fake_fetch)
    verdict, staged, _ = handlers.on_validate_lora("org/gone", None, [])

    assert staged == []
    assert "was not found on the Hub" in verdict


def test_the_upload_direction_still_stages_its_own_path(tmp_path):
    """The two directions converge (§4.3): an upload is already local, the
    fetcher is never consulted, and 'path' equals 'source'."""
    from ltx25_studio.ui import handlers

    adapter = _valid_adapter(tmp_path)
    verdict, staged, _ = handlers.on_validate_lora("", str(adapter), [])

    assert len(staged) == 1
    assert staged[0]["path"] == str(adapter)
    assert staged[0]["source"] == str(adapter)
