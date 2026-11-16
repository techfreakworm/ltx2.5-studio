"""The fork-boundary contract: our errors must survive a ZeroGPU worker.

docs/SPACE_MODE_MATRIX.md §4 measured what happens without this. On ZeroGPU
every ``@spaces.GPU`` call runs in a forked worker, and ``spaces`` transmits
the raised exception *object* back to the parent only when ``isinstance(exc,
gr.Error)`` — everything else is reduced to its class name, so a fully
diagnosed ``DecodeError(reason=...)`` reached a live user as the eleven
characters ``'DecodeError'``. And the object travels over a
``multiprocessing.Queue``, so eligibility without picklability is *worse* than
nothing: the worker catches the ``PicklingError`` and ships that instead, and
the user loses even the class name.

So the fix is a pair, and this file pins both halves independently:

* every class in the hierarchy IS a ``gr.Error`` (eligibility);
* every class round-trips pickle preserving class, message and attributes
  (survival). Losing either half re-fails a test here by name.

It also pins what the pair must NOT change: ``str(exc)`` stays byte-identical
to the pre-gr.Error hierarchy (gr.Error's own ``__str__`` repr-quotes the
message — the single quotes around ``'DecodeError'`` on the Space were exactly
that), and ``ui.render.failure`` shows the message verbatim, before *and after* a
round trip — the after case being precisely what the parent process renders.

Discovery is programmatic. The classes are walked out of the module, split
into grouping bases and concrete leaves *structurally* (a base is a class
another discovered class subclasses), and construction recipes come from
``test_errors.ERROR_SPECS`` — whose own coverage test already refuses any new
class that lacks a recipe. A 28th error class therefore lands in this suite
automatically; it cannot dodge by not being hand-listed here.

Everything in this file is offline and costs zero GPU (PLAN §9.1).
"""

from __future__ import annotations

import html
import inspect
import pickle

import gradio as gr
import pytest
from test_errors import ERROR_SPECS

from ltx25_studio import errors
from ltx25_studio.ui import render

# --------------------------------------------------------------------------
# programmatic discovery — no hand-list for a new class to dodge
# --------------------------------------------------------------------------

DISCOVERED = {
    name: obj
    for name, obj in inspect.getmembers(errors, inspect.isclass)
    if obj.__module__ == errors.__name__
    and issubclass(obj, errors.LTXStudioError)
    and not name.startswith("_")
}

#: A grouping base is structural, not declared: some *other* discovered class
#: subclasses it. Concrete leaves are everything else, and every leaf must
#: have a construction recipe in ERROR_SPECS or ``_construct`` fails loudly.
GROUPING_BASES = {
    name
    for name, cls in DISCOVERED.items()
    if any(other is not cls and issubclass(other, cls) for other in DISCOVERED.values())
}


def _construct(name):
    cls = DISCOVERED[name]
    spec = ERROR_SPECS.get(name)
    if spec is not None:
        return cls(**spec[0])
    assert name in GROUPING_BASES, (
        f"{name} is a concrete error with no construction recipe — add it to "
        f"test_errors.ERROR_SPECS so the fork-boundary contract covers it"
    )
    # Bases carry no keyword contract; a plain message exercises the same
    # __init__/__str__/__reduce__ path the leaves inherit.
    return cls(f"synthetic {name} message for the fork-boundary suite")


def test_discovery_found_the_whole_exported_hierarchy():
    """The walk and ``__all__`` must agree in both directions: an exported name
    that is not in the hierarchy, or a hierarchy class that is not exported,
    is a class this suite would silently miss. The failure-category API is the
    only non-error export."""
    category_api = {*(name.upper() for name in errors.CATEGORIES), "CATEGORIES", "Failure"}
    category_api |= {"categorise", "is_out_of_memory"}
    assert set(DISCOVERED) == set(errors.__all__) - category_api


# --------------------------------------------------------------------------
# half one — eligibility: the object is only transmitted if it IS a gr.Error
# --------------------------------------------------------------------------


@pytest.mark.parametrize("name", sorted(DISCOVERED))
def test_every_error_is_a_gradio_error(name):
    """``spaces`` populates ``ExceptionResult.gradio_error`` only for
    ``isinstance(exc, gr.Error)``; any class that loses this property is
    reduced to its bare name in transit again."""
    assert issubclass(DISCOVERED[name], gr.Error)
    assert isinstance(_construct(name), gr.Error)


def test_the_hierarchy_still_roots_at_ltxstudioerror():
    """``ui/render.py`` branches on ``isinstance(exc, LTXStudioError)`` — the
    gr.Error graft must not loosen the family the UI catches."""
    for name in sorted(DISCOVERED):
        assert issubclass(DISCOVERED[name], errors.LTXStudioError)


def test_the_valueerror_widening_is_known_and_audited():
    """gr.Error derives from gradio's ``AppError``, which derives from
    ``ValueError`` — so this inheritance makes every error here a ValueError.
    That widening was audited (every ``except ValueError`` in the package
    guards stdlib parsing or ``inspect.signature``; ``lora/apply.py``
    re-raises ``LoRAError`` first). This test pins the fact so a future
    gradio bump that moves gr.Error's base is noticed, not discovered."""
    assert issubclass(errors.LTXStudioError, ValueError)


# --------------------------------------------------------------------------
# half two — survival: the transmitted object must cross a pickle boundary
# --------------------------------------------------------------------------


@pytest.mark.parametrize("name", sorted(DISCOVERED))
def test_pickle_round_trip_preserves_class_message_and_every_attribute(name):
    """22 of 27 classes were measured unpicklable before ``__reduce__``:
    keyword-only constructors versus a one-tuple ``args`` that pickle feeds
    back positionally. Class identity, ``args``, and the full instance dict
    (mode, reason, message, print_exception — all of it) must survive."""
    exc = _construct(name)
    back = pickle.loads(pickle.dumps(exc))
    assert type(back) is type(exc)
    assert str(back) == str(exc)
    assert back.args == exc.args
    assert vars(back) == vars(exc)


@pytest.mark.parametrize("name", sorted(GROUPING_BASES))
def test_bases_round_trip_bare_too(name):
    """The five grouping bases were the only picklable classes before; raised
    bare they must stay picklable — with ``args == ()`` intact, not swapped
    for gr.Error's ``"Error raised."`` default."""
    back = pickle.loads(pickle.dumps(DISCOVERED[name]()))
    assert type(back) is DISCOVERED[name]
    assert back.args == ()
    assert str(back) == ""


# --------------------------------------------------------------------------
# what the pair must NOT change: the words
# --------------------------------------------------------------------------

#: str() snapshots taken from the hierarchy BEFORE it inherited gr.Error, at
#: the exact kwargs ERROR_SPECS uses. Byte-identical means byte-identical: a
#: leading quote here is gr.Error's repr-``__str__`` leaking through, and it
#: would repr-quote every rendered error detail across the whole UI.
PRE_CHANGE_STR = {
    "DecodeError": "decoding failed for mode 'text_to_video': pipeline returned no frames.",
    "ConfigError": (
        "LTX25_TRANSFORMER_PRECISION='fp4' is not a recognised value. Legal values are: "
        "bf16, fp8. Refusing to fall back to a default silently -- set the variable "
        "correctly or unset it."
    ),
    "DimensionError": (
        "width=700 is not divisible by 32. LTX-2.5 requires every spatial dimension to be "
        "a multiple of 32. Use 672 or 704 instead."
    ),
    "ModelLoadError": "failed to load pipeline component 'transformer': connection reset by peer.",
    "EmptyOutputError": (
        "generation for mode 'text_to_video' finished without producing "
        "/tmp/run-abc/video.mp4. Reporting this rather than returning an empty result."
    ),
    "LoRANotAttachedError": (
        "LoRA /tmp/adapter.safetensors loaded as adapter 'user_lora' but attached to zero "
        "modules of transformer, connectors. The adapter is almost certainly built for a "
        "different architecture; applying it would silently produce base-model output."
    ),
}


@pytest.mark.parametrize("name", sorted(PRE_CHANGE_STR))
def test_str_is_byte_identical_to_the_pre_gradio_hierarchy(name):
    assert str(_construct(name)) == PRE_CHANGE_STR[name]


@pytest.mark.parametrize("name", sorted(DISCOVERED))
def test_no_error_str_is_repr_quoted(name):
    """The class-wide version of the snapshot check: no message may pick up
    gr.Error's quoting, before or after crossing the boundary."""
    exc = _construct(name)
    for obj in (exc, pickle.loads(pickle.dumps(exc))):
        assert not str(obj).startswith(("'", '"'))


def test_root_arity_semantics_match_plain_exception_exactly():
    """gr.Error would absorb a second positional into ``duration`` and turn
    zero args into ``"Error raised."``; the root suppresses both so str/args
    behave as the plain-Exception hierarchy always did."""
    assert str(errors.LTXStudioError()) == ""
    assert errors.LTXStudioError().args == ()
    assert str(errors.LTXStudioError("boom")) == "boom"
    assert str(errors.LTXStudioError("a", "b")) == "('a', 'b')"
    assert errors.LTXStudioError("a", "b").args == ("a", "b")


def shows(message: str, rendered: str) -> bool:
    """The message reaches the user as written: verbatim, or HTML-escaped (the
    browser shows the same text either way)."""
    return message in rendered or html.escape(message) in rendered


@pytest.mark.parametrize("name", sorted(PRE_CHANGE_STR))
def test_render_failure_carries_the_message_and_survives_the_round_trip(name):
    """``ui/render.py::failure`` must show the error's own message verbatim, and
    -- the point of the whole fix -- must render the *same* card for the object
    the parent process unpickles as for the object the worker raised.

    Format-agnostic on purpose: the card's headline and layout belong to the UI
    and are pinned in its own tests."""
    exc = _construct(name)
    rendered = render.failure(exc)
    assert shows(PRE_CHANGE_STR[name], rendered), rendered
    assert render.failure(pickle.loads(pickle.dumps(exc))) == rendered


def test_a_foreign_gradio_error_renders_its_own_text():
    """The platform's own errors (``'GPU task aborted'`` and friends) are
    gr.Error instances that are NOT LTXStudioError. Whatever card they get, the
    platform's text must reach the user verbatim, before and after a round trip."""
    exc = gr.Error("GPU task aborted")
    rendered = render.failure(exc)
    assert shows("GPU task aborted", rendered), rendered
    assert render.failure(pickle.loads(pickle.dumps(exc))) == rendered


# --------------------------------------------------------------------------
# the spaces contract, simulated end to end — zero GPU
# --------------------------------------------------------------------------


def test_the_spaces_marshalling_contract_end_to_end():
    """Replays ``spaces/zero/wrappers.py`` step by step (read at 0.48.3):
    ``exception_result`` gates on ``isinstance(exc, gr.Error)`` and normalises
    ``print_exception``; the queue crossing is a pickle round trip (a
    ``multiprocessing.Queue`` serialises with pickle — an in-process dumps/
    loads exercises the identical path deterministically); the parent reads
    ``print_exception``, prints the worker traceback while it is truthy, sets
    it False and raises the object. Every step must hold or the user is back
    to ``'DecodeError'`` — or worse, ``'PicklingError'``."""
    exc = errors.DecodeError(
        mode="video_to_video",
        reason="the pipeline raised ValueError('bad frames')",
    )

    # worker: exception_result() — the eligibility gate
    assert isinstance(exc, gr.Error), "gradio_error would be None; only the class name travels"
    gradio_error = exc
    gradio_error.print_exception = getattr(gradio_error, "print_exception", True)

    # worker → parent: res_queue.put(res) — the survival gate
    reconstructed = pickle.loads(pickle.dumps(gradio_error))

    # parent: print the traceback (print_exception is True — deliberately, so
    # the Space log keeps the channel §3's diagnosis actually came through),
    # then flip it and raise the real object
    assert isinstance(reconstructed, gr.Error)
    assert reconstructed.print_exception is True
    reconstructed.print_exception = False  # the parent's own mutation must work

    # what arrives is our object, saying our words
    assert type(reconstructed) is errors.DecodeError
    assert isinstance(reconstructed, errors.LTXStudioError)
    assert reconstructed.mode == "video_to_video"
    assert reconstructed.reason == "the pipeline raised ValueError('bad frames')"
    assert str(reconstructed) == (
        "decoding failed for mode 'video_to_video': the pipeline raised ValueError('bad frames')."
    )


def test_print_exception_defaults_true_for_the_whole_hierarchy():
    """Deliberate, not inherited by accident: the spaces parent prints the
    worker's full traceback to the Space log only while the transmitted
    error's ``print_exception`` is truthy. That log line is how the one real
    generation defect this Space has had was diagnosed; suppressing it would
    re-close the channel."""
    for name in sorted(DISCOVERED):
        assert _construct(name).print_exception is True, name
