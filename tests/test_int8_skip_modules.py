"""``TRANSFORMER_INT8_SKIP_MODULES`` against the REAL LTX-2.5 module tree.

Every frame the deployed Space ever produced was correctly-shaped,
high-variance garbage — colour static and flat fields, at every resolution, in
every mode, deterministically — because the int8 config named no
``llm_int8_skip_modules`` and bitsandbytes replaced **every** ``nn.Linear`` in
the transformer, the adaLN/modulation heads and the input/output projections
included.

**Why this file exists rather than one more literal-list assertion.** The
obvious fix is to copy the skip list from another PixArt-lineage DiT:
``norm_out.linear``, ``adaln_single``, ``caption_projection``, ``proj_out``. On
LTX-2.5 the first three match **nothing at all** — ``norm_out`` is an
``elementwise_affine=False`` LayerNorm with zero parameters, there is no
``adaln_single`` module, and ``caption_projection`` is gated off in this
checkpoint's config. Three no-ops and one real name would have looked exactly
like a fix, passed any test that asserted the literal list it was given,
deployed, and produced the same static.

So this asserts nothing about the list's *text*. It re-derives, from the real
checkpoint's 4091-entry weight-map index, **which concrete Linears the list
actually spares** — by re-implementing bitsandbytes' own matcher — and pins the
resulting partition. A wrong name changes the split; a no-op name changes
nothing and is caught as a dead entry.

**Skipped cleanly when the weights are not cached**, so CI without the HF cache
still passes. It reads one JSON index and no tensor data: no torch, no
diffusers, no bitsandbytes, no GPU, no network, and not one byte of the 44 GB
of shards.
"""

from __future__ import annotations

import json
import re
from collections import Counter

import pytest

from ltx25_studio import model_files, pipelines, preflight

# --------------------------------------------------------------------------
# the checkpoint, or a clean skip
# --------------------------------------------------------------------------

INDEX_NAME = "diffusion_pytorch_model.safetensors.index.json"

#: Ground truth for this revision, measured. `weight_map` entries, then the
#: partition of the transformer's `nn.Linear` modules under the skip list.
#: These are the numbers the fix is worth: 316 Linears stay bf16 and 1344 are
#: still quantised, so the memory case for int8 survives essentially intact
#: while the conditioning path stops being rounded.
EXPECTED_WEIGHT_MAP_ENTRIES = 4091
EXPECTED_LINEARS = 1660
EXPECTED_SKIPPED = 316
EXPECTED_QUANTIZED = 1344


@pytest.fixture(scope="module")
def weight_keys():
    """Every tensor name in the transformer's shard index, or skip."""
    index = preflight.snapshot_dir() / "transformer" / INDEX_NAME
    if not index.is_file():
        pytest.skip(
            f"no cached transformer weight index at {index}. These assertions are "
            f"about the real {model_files.REPO_ID} module tree at revision "
            f"{model_files.REVISION}; fetch the weights to run them."
        )
    return sorted(json.loads(index.read_text())["weight_map"])


# --------------------------------------------------------------------------
# bitsandbytes' matcher, re-implemented from the pinned source
# --------------------------------------------------------------------------


def bnb_is_skipped(path: str, skip) -> bool:
    """Whether bnb leaves the Linear at ``path`` alone, given ``skip``.

    Transcribed from ``diffusers/quantizers/bitsandbytes/utils.py`` at the
    pinned SHA (``requirements.txt``), inside ``_replace_with_bnb_linear``::

        for name, module in model.named_children():
            ...
            current_key_name.append(name)
            if isinstance(module, nn.Linear) and name not in modules_to_not_convert:
                current_key_name_str = ".".join(current_key_name)
                if not any(
                    (key + "." in current_key_name_str) or (key == current_key_name_str)
                    for key in modules_to_not_convert
                ):
                    ...replace with Linear8bitLt...

    Two gates, and a name is spared if EITHER fires:

    1. ``name not in modules_to_not_convert`` — ``name`` is the module's
       **local child name** from ``named_children()``, not its dotted path. This
       is the only gate a leaf entry like ``to_gate_logits`` can use.
    2. ``(key + ".") in current_key_name_str`` — plain **substring
       containment** on the full dotted path, not a path-prefix test. This is
       the only gate a parent entry like ``time_embed`` can use, and the reason
       short names spread further than they look: ``time_embed`` also matches
       ``audio_time_embed.linear`` because ``"time_embed."`` occurs inside it.
    """
    local_name = path.rsplit(".", 1)[-1]
    if local_name in skip:  # gate 1
        return True
    return any((key + "." in path) or (key == path) for key in skip)  # gate 2


# --------------------------------------------------------------------------
# which of the checkpoint's modules are nn.Linear
# --------------------------------------------------------------------------

#: Leaf names that are `nn.Linear` in `transformer_ltx2.py` at the pinned SHA.
_LINEAR_LEAF = re.compile(
    r"(^|\.)(proj_in|proj_out|audio_proj_in|audio_proj_out"
    r"|linear|linear_1|linear_2|to_q|to_k|to_v|to_gate_logits)$"
)
#: `Attention.to_out` is a `ModuleList` whose [0] is the Linear.
_TO_OUT = re.compile(r"\.to_out\.0$")
#: `FeedForward.net` is [GELU(proj=Linear), Dropout, Linear].
_FF = re.compile(r"(^|\.)(ff|audio_ff)\.net\.(0\.proj|2)$")


def _is_linear(module_path: str) -> bool:
    return bool(
        _LINEAR_LEAF.search(module_path) or _TO_OUT.search(module_path) or _FF.search(module_path)
    )


def _partition(weight_keys):
    """``(linears, non_linears, bare_parameters)`` for the checkpoint."""
    modules, bare = set(), []
    for key in weight_keys:
        path, _, leaf = key.rpartition(".")
        # A module owns `.weight`/`.bias`; anything else is a bare
        # `nn.Parameter` hanging directly off a module.
        if leaf in ("weight", "bias"):
            modules.add(path)
        else:
            bare.append(key)
    return (
        sorted(p for p in modules if _is_linear(p)),
        sorted(p for p in modules if not _is_linear(p)),
        sorted(bare),
    )


def _collapse(paths):
    """Collapse block indices so a failure prints shapes, not 1660 lines."""
    return sorted(Counter(re.sub(r"\.\d+\.", ".{N}.", p) for p in paths).items())


# --------------------------------------------------------------------------
# the derivation itself is sound
# --------------------------------------------------------------------------


def test_the_index_is_the_revision_this_list_was_read_off(weight_keys):
    """A different checkpoint invalidates every count below."""
    assert len(weight_keys) == EXPECTED_WEIGHT_MAP_ENTRIES, (
        f"the cached transformer index holds {len(weight_keys)} tensors, not "
        f"{EXPECTED_WEIGHT_MAP_ENTRIES}. TRANSFORMER_INT8_SKIP_MODULES was read off "
        f"revision {model_files.REVISION}; on a different tree its counts mean nothing."
    )


def test_no_linear_is_missed_by_the_derivation(weight_keys):
    """The residue must contain no Linears, or the split is undercounted.

    This is the load-bearing check under every count in this file. If
    ``_is_linear`` missed a Linear leaf name, that module would be silently
    absent from both sides of the partition and the totals would still look
    tidy. The transformer's only non-Linear *parameterised* submodules are the
    per-attention ``norm_q``/``norm_k`` RMSNorms
    (``transformer_ltx2.py:371-372``), so the residue is fully enumerable — and
    anything appearing here that is not one of those is a leaf this file does
    not know how to classify.
    """
    _linears, non_linears, _bare = _partition(weight_keys)

    unexplained = [p for p in non_linears if not re.search(r"\.(norm_q|norm_k)$", p)]
    assert not unexplained, (
        f"{len(unexplained)} parameterised submodules are neither a known Linear nor an "
        f"attention RMSNorm: {_collapse(unexplained)}. If any is an nn.Linear, the "
        f"skip/quantise counts below are undercounting and this file is not measuring "
        f"what it claims."
    )


def test_every_scale_shift_table_is_a_bare_parameter(weight_keys):
    """Why the modulation tables need no skip entry — bnb cannot reach them.

    bnb only ever replaces ``nn.Linear`` (``isinstance(module, nn.Linear)``),
    and it walks ``named_children()``, which yields modules. A bare
    ``nn.Parameter`` is neither. LTX-2.5's per-block modulation is exactly that
    — a ``scale_shift_table`` Parameter rather than an adaLN Linear — so every
    one of them is already safe at any precision, and naming them would be a
    harmless no-op that looked like protection.
    """
    _linears, _non_linears, bare = _partition(weight_keys)

    assert bare, "no bare parameters found at all; the checkpoint layout has changed"
    stray = [k for k in bare if not re.search(r"(scale_shift_table|pos_embedding)$", k)]
    assert not stray, (
        f"bare parameters that are not modulation tables or positional embeddings: "
        f"{_collapse(stray)}"
    )


# --------------------------------------------------------------------------
# the partition the skip list actually produces
# --------------------------------------------------------------------------


def test_the_skip_list_produces_the_measured_split(weight_keys):
    """316 spared, 1344 still quantised — the numbers the fix is worth.

    ⚠️ Both halves matter. Too few skips and the conditioning path is still
    rounded and the Space still renders static; too many and int8 stops paying
    for itself on a memory-capped platform, which is the only reason it is on.
    """
    linears, _non_linears, _bare = _partition(weight_keys)
    skip = pipelines.TRANSFORMER_INT8_SKIP_MODULES

    assert len(linears) == EXPECTED_LINEARS, (
        f"{len(linears)} nn.Linear modules, expected {EXPECTED_LINEARS}: {_collapse(linears)}"
    )

    skipped = [p for p in linears if bnb_is_skipped(p, skip)]
    quantized = [p for p in linears if not bnb_is_skipped(p, skip)]

    assert len(skipped) == EXPECTED_SKIPPED, (
        f"the skip list spares {len(skipped)} Linears, expected {EXPECTED_SKIPPED}:\n"
        f"{_collapse(skipped)}"
    )
    assert len(quantized) == EXPECTED_QUANTIZED, (
        f"{len(quantized)} Linears would be quantised, expected {EXPECTED_QUANTIZED}"
    )
    assert len(skipped) + len(quantized) == len(linears)


def test_no_skip_entry_is_a_no_op(weight_keys):
    """The `norm_out.linear` trap: a name that protects nothing.

    This is the test that would have caught the plausible wrong answer. A skip
    list borrowed from another DiT is not rejected by anything — bnb ignores
    names it cannot match, `BitsAndBytesConfig` validates only that the entries
    are strings, and the model loads and runs. It just renders static.

    Each entry must be load-bearing on THIS tree.
    """
    linears, _non_linears, _bare = _partition(weight_keys)

    dead = [
        name
        for name in pipelines.TRANSFORMER_INT8_SKIP_MODULES
        if not any(bnb_is_skipped(p, [name]) for p in linears)
    ]
    assert not dead, (
        f"{dead} match no nn.Linear in the real checkpoint. Names borrowed from "
        f"another DiT fail exactly this way: `norm_out.linear`, `adaln_single` and "
        f"`caption_projection` are all no-ops on LTX-2.5, and a list of them would "
        f"look like a fix, deploy, and render static."
    )


#: Entries whose removal would change the partition, and entries already
#: covered by another entry. Measured, not asserted from intent.
#:
#: ``audio_time_embed`` and ``audio_prompt_adaln`` are **subsumed**: bnb's gate 2
#: is substring containment on the dotted path, so ``"time_embed."`` occurs
#: inside ``"audio_time_embed.emb.timestep_embedder.linear_1"`` and the video
#: entry already spares the audio tower's Linears.
#:
#: They are kept anyway, deliberately — see the constant's own comment. The
#: point of pinning the relationship here is that it is *load-bearing in one
#: direction only*: drop ``prompt_adaln`` and the video prompt adaLN stops being
#: spared even though ``audio_prompt_adaln`` remains, which is not what the pair
#: looks like at a glance.
LOAD_BEARING = (
    "time_embed",
    "prompt_adaln",
    "av_cross_attn_video_scale_shift",
    "av_cross_attn_audio_scale_shift",
    "av_cross_attn_video_a2v_gate",
    "av_cross_attn_audio_v2a_gate",
    "proj_in",
    "proj_out",
    "audio_proj_in",
    "audio_proj_out",
    "to_gate_logits",
)
SUBSUMED = ("audio_time_embed", "audio_prompt_adaln")


def test_which_entries_are_load_bearing_is_pinned(weight_keys):
    """Not every entry is independently necessary, and that is worth knowing.

    Mutation-testing this list found one behaviour-preserving edit: removing
    ``audio_prompt_adaln`` changes nothing, because ``prompt_adaln`` already
    matches ``audio_prompt_adaln.*`` through bnb's substring-containment gate.
    Same for ``audio_time_embed`` under ``time_embed``.

    ⚠️ **The subsumption is asymmetric and that is the trap.** The audio entry
    does not cover the video one — dropping ``prompt_adaln`` while keeping
    ``audio_prompt_adaln`` silently leaves the *video* prompt adaLN quantised,
    which is the original bug confined to one tower and just as invisible.

    The redundant pair is kept on purpose: it states the audio tower's coverage
    explicitly instead of making a reader derive it from a matcher's substring
    semantics, and if upstream ever tightens gate 2 into a real path-prefix test
    — which would be the *correct* fix for that matcher — those entries stop
    being redundant and start being the only thing covering the audio tower.
    """
    linears, _non_linears, _bare = _partition(weight_keys)
    skip = pipelines.TRANSFORMER_INT8_SKIP_MODULES
    baseline = {p for p in linears if bnb_is_skipped(p, skip)}

    def lost_without(name):
        rest = [n for n in skip if n != name]
        return len(baseline) - len({p for p in linears if bnb_is_skipped(p, rest)})

    assert sorted(skip) == sorted(LOAD_BEARING + SUBSUMED), (
        f"the skip list changed to {sorted(skip)}; re-measure which entries are "
        f"load-bearing before updating this pin"
    )
    for name in LOAD_BEARING:
        assert lost_without(name) > 0, (
            f"{name!r} is pinned as load-bearing but removing it now spares the same "
            f"Linears — another entry has grown to cover it, so the list is no longer "
            f"saying what it appears to say"
        )
    for name in SUBSUMED:
        assert lost_without(name) == 0, (
            f"{name!r} is pinned as subsumed by a shorter entry, but removing it now "
            f"loses {lost_without(name)} Linears. The coverage it was relying on is "
            f"gone; this entry is now the only thing sparing them."
        )


@pytest.mark.parametrize("borrowed", ["norm_out.linear", "adaln_single", "caption_projection"])
def test_the_borrowed_pixart_names_match_nothing(weight_keys, borrowed):
    """Pinned as a fact about the checkpoint, not just as advice in a comment.

    * ``norm_out`` is ``nn.LayerNorm(inner_dim, eps=1e-6, elementwise_affine=False)``
      — zero parameters, no ``.linear`` child.
    * ``adaln_single`` does not exist; the modulation Linears are eight
      *top-level* ``LTX2AdaLayerNormSingle`` modules named individually.
    * ``caption_projection`` is gated on ``use_prompt_embeddings``, which is
      ``false`` in this checkpoint's ``transformer/config.json``.

    If a future revision reintroduces any of these, this goes red and the skip
    list needs revisiting — which is the correct outcome, not a nuisance.
    """
    linears, _non_linears, _bare = _partition(weight_keys)

    assert not [p for p in linears if bnb_is_skipped(p, [borrowed])], (
        f"{borrowed!r} now matches real Linears; LTX-2.5 previously had none and the "
        f"skip list was written on that basis"
    )


def test_shortening_a_skip_name_is_caught(weight_keys):
    """Why the entries are spelled out: gate 2 is substring containment.

    ``"proj"`` looks like a tidier way to say ``proj_in``/``proj_out``. It is
    not: every ``FeedForward``'s inner ``net.0.proj`` has ``proj`` as its local
    child name, so gate 1 fires on all 96 of them across the 48 blocks' video
    and audio towers. That un-quantises a large share of the model's parameters
    and destroys the memory case int8 exists to make — silently, since bnb
    reports nothing about what it skipped.
    """
    linears, _non_linears, _bare = _partition(weight_keys)

    over_broad = [p for p in linears if bnb_is_skipped(p, ["proj"])]
    assert len(over_broad) == 96, (
        f"bare 'proj' now matches {len(over_broad)} Linears rather than 96; the "
        f"argument for spelling out proj_in/proj_out/audio_proj_in/audio_proj_out "
        f"has changed shape and the comment on the constant needs rechecking"
    )
    assert len(over_broad) > sum(
        bnb_is_skipped(p, ["proj_in", "proj_out", "audio_proj_in", "audio_proj_out"])
        for p in linears
    )


# --------------------------------------------------------------------------
# what is spared, and what is still paid for
# --------------------------------------------------------------------------


def test_the_whole_conditioning_path_is_spared(weight_keys):
    """No modulation Linear may be left quantised.

    The eight top-level ``LTX2AdaLayerNormSingle`` modules
    (``transformer_ltx2.py:1219-1257``) each contribute
    ``.emb.timestep_embedder.linear_1``, ``.linear_2`` and ``.linear``. They
    compute the shift/scale/gate every one of the 48 blocks is conditioned on,
    so error here is not attenuated — it is multiplied across the whole depth
    and both towers. One of these left behind reproduces the bug in miniature,
    and there is no output short of the final frames that would show it.
    """
    linears, _non_linears, _bare = _partition(weight_keys)
    skip = pipelines.TRANSFORMER_INT8_SKIP_MODULES

    modulation = [
        p
        for p in linears
        if re.search(r"(time_embed|adaln|scale_shift|_gate)\.", p) or p.endswith("to_gate_logits")
    ]
    assert modulation, "found no modulation Linears at all; the derivation is broken"

    still_quantized = [p for p in modulation if not bnb_is_skipped(p, skip)]
    assert not still_quantized, (
        f"{len(still_quantized)} conditioning Linears would still be quantised: "
        f"{_collapse(still_quantized)}. Quantised conditioning is what produced "
        f"correctly-shaped, high-variance garbage at every setting."
    )


def test_the_bulk_of_the_model_is_still_quantised(weight_keys):
    """The other half of the bargain: attention and feed-forward still pay.

    ``LLM.int8()`` on projection and feed-forward weights is the case
    bitsandbytes was designed for and is not implicated in this bug. A "fix"
    that spared them too would render correctly and evict on RAM — a failure
    this suite has already paid for once.
    """
    linears, _non_linears, _bare = _partition(weight_keys)
    skip = pipelines.TRANSFORMER_INT8_SKIP_MODULES

    bulk = [p for p in linears if re.search(r"\.(to_q|to_k|to_v|to_out\.0)$", p) or _FF.search(p)]
    wrongly_spared = [p for p in bulk if bnb_is_skipped(p, skip)]
    assert not wrongly_spared, (
        f"{len(wrongly_spared)} attention/feed-forward Linears would be left at bf16: "
        f"{_collapse(wrongly_spared)}. That is the memory budget the quantisation "
        f"exists to meet."
    )
    assert len(bulk) == EXPECTED_QUANTIZED, (
        f"{len(bulk)} attention/feed-forward Linears, but {EXPECTED_QUANTIZED} are "
        f"quantised overall; the two sets are meant to coincide exactly, so something "
        f"outside them is now being quantised (or one of them is not)"
    )
