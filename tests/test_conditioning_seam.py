"""Spec for the shot-list seam: UI dataframe -> conditioning.py -> pipeline.

PLAN §3.1a, §5.7 rows 1 and 2, §7.2.

**The same shape as the exports blocker, a third time.** ``conditioning.py`` is
complete, documented and thoroughly tested in isolation. The shot-list UI is
complete and thoroughly tested in isolation. And **nothing connects them**: the
Gradio dataframe's rows travel from ``GenerationRequest.conditions`` through
``modes._conditions`` — which does ``specs.extend(request.conditions)`` — and
arrive at the pipeline as *raw list rows*, next to the properly built
``ConditionSpec`` for the uploaded image::

    conditions reaching the pipeline:
        (ConditionSpec(source='img.png', kind='image', index=0, strength=1.0),
         ['a second shot', 3])

``conditioning.build_conditions`` and ``conditioning.normalise_condition`` are
called by nothing in the package. Both sides' tests pass. Neither side is wrong.
There is simply no test across the boundary — which is the exact failure the
exports blocker was.

What that costs, in PLAN's own words:

* §5.7 row 1, "the severe one": ``LTX2VideoCondition.index`` is a **latent**
  index. The conversion lives in ``conditioning.py`` and is never applied, so a
  user's anchor lands wherever the raw number happens to point.
* §5.7 row 2: an out-of-range condition is **silently skipped upstream** with
  nothing but a ``logger.warning``. The user supplies a shot, the shot does
  nothing, nobody is told, and the metered run is spent anyway.

**The rows themselves are now refused.** A dataframe row carries a prompt
segment where a condition needs frames, so it became
``LTX2VideoCondition(frames="<text>")`` and died inside the GPU worker.
``generate.scope_request_to_mode`` refuses such rows before the dispatch, in
plain words. What reaches ``conditioning.py`` is a real keyframe -- an image or a
video at an anchor -- and that is what the seam specs below drive.

**Deliberately not asserted here:** whether the anchor column is a latent index
or a pixel frame. ``fields.SHOT_COLUMNS`` labels it "Anchor (LATENT index)" and
``handlers.on_shots_change`` converts latent→pixel for the readout, while
``conditioning.normalise_condition`` treats the same field as a pixel frame and
converts pixel→latent. The two halves disagree, and picking the winner is a
design call, not a test's to make. These specs hold under either answer.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from conftest import code_tokens, image_input, inputs_for, package_python_files, video_file
from test_exports_contract import factory_returning, one_muxed_file_encoder

from ltx25_studio import audio_conditioning, conditioning, modes
from ltx25_studio import generate as generate_module
from ltx25_studio.errors import LTXStudioError
from ltx25_studio.ui.fields import SHOT_ANCHOR_COLUMN, SHOT_COLUMNS

#: Modes that accept conditioning input, so a shot list is meaningful for them.
CONDITIONABLE_MODES = tuple(
    mode_id
    for mode_id, spec in modes.MODES.items()
    if spec.pipeline == "condition" and {"image", "video"} & set(spec.inputs)
)

#: Modes that make a variation of the user's clip (mode 04): the clip travels as
#: ``video_variation`` and is never a condition. A shot (a real keyframe) still is.
VARIATION_MODES = tuple(
    mode_id
    for mode_id in CONDITIONABLE_MODES
    if modes.get_mode(mode_id).video_conditioning == "variation"
)

NUM_FRAMES = 121
#: The clip a variation mode is given here; its length is the run's length.
VARIATION_CLIP_FRAMES = 17


class CapturingPipe:
    """Records the kwargs it was called with, so the condition list can be read."""

    def __init__(self):
        self.kwargs: dict = {}

    def __call__(self, **kwargs):
        self.kwargs = kwargs
        return type("Output", (), {"frames": ("frame",), "audio": ("sample",)})()


def shot_row(anchor, prompt="a second shot", strength=1.0):
    """One row exactly as ``gr.Dataframe(type="array")`` hands it to the handler."""
    row = [None] * len(SHOT_COLUMNS)
    row[0] = prompt
    row[SHOT_ANCHOR_COLUMN] = anchor
    row[2] = strength
    return row


def keyframe(mode_id, anchor, strength=1.0):
    """A real condition at ``anchor``: an image where the mode takes one, else a video."""
    if modes.accepts(modes.get_mode(mode_id), "image"):
        return conditioning.ConditionSpec(
            source=image_input(), kind="image", index=anchor, strength=strength
        )
    return conditioning.ConditionSpec(
        source=video_file(), kind="video", index=anchor, strength=strength
    )


def _inputs(mode_id):
    fields = inputs_for(mode_id)
    if mode_id in VARIATION_MODES:
        # A variation runs at its clip's length, so the clip must reach past the anchor.
        fields["video"] = video_file(frames=VARIATION_CLIP_FRAMES)
    return fields


def run_with_shots(mode_id, rows, tmp_path, *, num_frames=NUM_FRAMES, sent=None):
    """The conditions the pipeline received; ``sent`` (a dict) also gets every kwarg."""
    pipe = CapturingPipe()
    generate_module.generate(
        mode=mode_id,
        seed=7,
        **_inputs(mode_id),
        num_frames=num_frames,
        conditions=rows,
        pipeline_factory=factory_returning(pipe),
        encoder=one_muxed_file_encoder,
        output_root=tmp_path,
        retention_bytes=10**9,
    )
    if sent is not None:
        sent.update(pipe.kwargs)
    return pipe.kwargs.get("conditions", ())


def assert_the_clip_is_the_variation(mode_id, sent):
    """Mode 04: the clip arrives as decoded frames to vary, never as a condition or a path.

    Every frame of it, as ``uint8``, at the mode's default level (no level was asked for).
    """
    clip = sent.get("video_variation")
    assert isinstance(clip, modes.VariationClip), (
        f"{mode_id}: the clip did not arrive as the variation: {sorted(sent)}"
    )
    assert not isinstance(clip.source, (str, Path)), f"{mode_id}: the clip arrived as a path"
    assert getattr(clip.source, "ndim", None) == 4, f"{mode_id}: {type(clip.source)!r}"
    assert clip.source.dtype == np.uint8, f"{mode_id}: {clip.source.dtype}"
    assert len(clip.source) == VARIATION_CLIP_FRAMES, (
        f"{mode_id}: {len(clip.source)} of the clip's {VARIATION_CLIP_FRAMES} frames arrived"
    )
    assert clip.level == audio_conditioning.DEFAULT_VIDEO_LEVEL, f"{mode_id}: level {clip.level}"


# --------------------------------------------------------------------------
# a raw dataframe row is not a condition
# --------------------------------------------------------------------------


@pytest.mark.parametrize("mode_id", CONDITIONABLE_MODES)
def test_no_raw_dataframe_row_reaches_the_pipeline(mode_id, tmp_path):
    """The observable form of "conditioning.py is not wired in".

    A bare ``list`` from ``gr.Dataframe`` is not a condition under any
    convention. It has no frames -- its first cell is prompt text -- so it is
    refused, in plain words, before the dispatch; it never reaches the pipeline.
    """
    pipe = CapturingPipe()
    with pytest.raises(LTXStudioError) as caught:
        generate_module.generate(
            mode=mode_id,
            seed=7,
            **inputs_for(mode_id),
            conditions=[shot_row(3)],
            pipeline_factory=factory_returning(pipe),
            encoder=one_muxed_file_encoder,
            output_root=tmp_path,
            retention_bytes=10**9,
        )
    assert pipe.kwargs == {}, "a raw dataframe row reached the pipeline"
    assert "shot" in str(caught.value) and "prompt" in str(caught.value), caught.value


@pytest.mark.parametrize("mode_id", CONDITIONABLE_MODES)
def test_every_condition_reaching_the_pipeline_carries_a_source_and_an_index(mode_id, tmp_path):
    """Whatever the convention, a built condition has these three fields.

    For a variation mode the one condition is the shot: the clip is not one.
    """
    sent: dict = {}
    conditions = run_with_shots(mode_id, [keyframe(mode_id, 3)], tmp_path, sent=sent)

    if mode_id in VARIATION_MODES:
        assert_the_clip_is_the_variation(mode_id, sent)
        assert len(conditions) == 1, f"{mode_id}: {conditions!r}"

    for condition in conditions:
        for attribute in ("source", "index", "strength"):
            value = (
                condition.get(attribute)
                if isinstance(condition, dict)
                else getattr(condition, attribute, None)
            )
            assert value is not None, (
                f"mode {mode_id!r} handed the pipeline a condition with no {attribute!r}: "
                f"{condition!r}"
            )


@pytest.mark.parametrize("mode_id", CONDITIONABLE_MODES)
def test_the_users_shot_actually_arrives_as_a_condition(mode_id, tmp_path):
    """A shot the user typed must not be dropped between the form and the model.

    The uploaded image already becomes one condition. A row in the shot list must
    become another — a list that silently comes back the same length as if the
    row were never there is the quietest possible failure.
    """
    sent: dict = {}
    without = run_with_shots(mode_id, [], tmp_path, sent=sent)
    with_shot = run_with_shots(mode_id, [keyframe(mode_id, 3)], tmp_path)

    if mode_id in VARIATION_MODES:
        # Without a shot, a variation reaches the pipeline with no condition at all.
        assert without == (), f"{mode_id} built a condition from its clip: {without!r}"
        assert_the_clip_is_the_variation(mode_id, sent)

    assert len(with_shot) == len(without) + 1, (
        f"mode {mode_id!r}: adding a shot row changed the condition count from "
        f"{len(without)} to {len(with_shot)}; it must add exactly one."
    )


@pytest.mark.parametrize("mode_id", VARIATION_MODES)
def test_a_shot_beside_a_varied_clip_is_placed_by_the_real_pipeline(mode_id, tmp_path, monkeypatch):
    """Through ``generate`` and the runner into the real condition pipeline (tiny, CPU).

    The runner starts the denoise from the clip's latents; the shot must still be
    placed, since upstream drops a condition it cannot place with only a log line.
    A placed shot adds its own tokens to the sequence the transformer denoises.
    """
    pytest.importorskip("torch")
    pytest.importorskip("diffusers")
    from diffusers import LTX2ConditionPipeline
    from test_real_pipeline_tiny import _tiny_snapshot_cached, load_smoke, run_tiny

    if not _tiny_snapshot_cached():
        pytest.skip("the tiny checkpoint is not in the HF cache")
    tiny = load_smoke().tiny_stack(device="cpu")
    calls, lengths = [], []
    original = LTX2ConditionPipeline.__call__

    def spy(self, *args, **kwargs):
        calls.append(kwargs)
        return original(self, *args, **kwargs)

    def length(_module, _args, kwargs):
        lengths.append(kwargs["hidden_states"].shape[1])

    monkeypatch.setattr(LTX2ConditionPipeline, "__call__", spy)
    hook = tiny["base"].transformer.register_forward_pre_hook(length, with_kwargs=True)
    try:
        run_tiny(tiny, mode_id, tmp_path, video=video_file(25), conditions=[keyframe(mode_id, 3)])
    finally:
        hook.remove()

    (sent,) = calls
    conditions = sent.get("conditions") or []
    assert len(conditions) == 1, f"diffusers was handed {conditions!r}, not the one shot"
    assert not isinstance(conditions[0].frames, (str, Path)), conditions[0].frames
    base = tiny["base"]
    _batch, _channels, frames, height, width = sent["latents"].shape  # the clip's
    per_frame = (height // base.transformer_spatial_patch_size) * (
        width // base.transformer_spatial_patch_size
    )
    clip_tokens = frames // base.transformer_temporal_patch_size * per_frame
    assert lengths[0] == clip_tokens + per_frame, (
        f"the transformer denoised {lengths[0]} tokens: the clip is {clip_tokens} and a placed "
        f"one-frame shot adds {per_frame}"
    )


# --------------------------------------------------------------------------
# an anchor that cannot work must say so, not be skipped upstream
# --------------------------------------------------------------------------


@pytest.mark.parametrize("mode_id", CONDITIONABLE_MODES)
def test_an_out_of_range_anchor_is_refused_rather_than_silently_skipped(mode_id, tmp_path):
    """PLAN §5.7 row 2, and ``conditioning.validate_condition_index`` exists for it.

    Upstream skips an out-of-range condition with only a log line and generates
    anyway: the shot has no effect, nobody is told, and the metered run is spent.
    9999 is out of range for a 121-frame clip under *either* reading of the
    anchor column, so this holds whichever convention wins.
    """
    with pytest.raises(LTXStudioError):
        run_with_shots(mode_id, [keyframe(mode_id, 9999)], tmp_path, num_frames=NUM_FRAMES)


@pytest.mark.parametrize("mode_id", CONDITIONABLE_MODES)
def test_a_negative_anchor_is_refused(mode_id, tmp_path):
    """The other end of the same range check."""
    with pytest.raises(LTXStudioError):
        run_with_shots(mode_id, [keyframe(mode_id, -5)], tmp_path, num_frames=NUM_FRAMES)


# --------------------------------------------------------------------------
# the module that owns the severe conversion has to be on the path at all
# --------------------------------------------------------------------------


#: The modules a generation actually runs through. ``ui/`` is excluded on
#: purpose: ``handlers.on_shots_change`` already imports ``conditioning`` for the
#: live readout, and that import is exactly what makes this defect invisible —
#: the module *is* used, just nowhere on the path that generates anything.
GENERATION_PATH = ("generate.py", "modes.py", "pipelines.py", "outputs.py")


def imported_names(path):
    """Every module name imported by a file, from its AST. Prose cannot fake it."""
    import ast

    names = set()
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Import):
            names.update(alias.name.split(".")[-1] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                names.add(node.module.split(".")[-1])
            names.update(alias.name for alias in node.names)
    return names


def test_the_conditioning_module_is_reachable_from_the_generation_path():
    """``conditioning.py`` is imported by no module that runs a generation.

    Stated structurally because each behavioural spec above can be closed by a
    local patch, while the real problem is that the module holding the §5.7
    conversion is not wired into anything. A module nothing calls is not a
    safeguard; it is documentation with a test suite.

    Read from the AST, not the text: ``modes.py``'s own docstring says
    "``conditioning.py`` owns the actual ``LTX2VideoCondition`` construction",
    which is true, aspirational, and invisible to the interpreter.
    """
    callers = [
        path.name
        for path in package_python_files()
        if path.name in GENERATION_PATH and "conditioning" in imported_names(path)
    ]

    assert callers, (
        f"no module on the generation path ({', '.join(GENERATION_PATH)}) imports "
        f"conditioning.py, so build_conditions / normalise_condition / "
        f"validate_condition_index are never called by a generation. The pixel<->latent "
        f"conversion PLAN §5.7 calls 'the severe one' is implemented, tested, and unreachable."
    )


def test_the_conditioning_helpers_are_called_by_something():
    """Names the two functions specifically, so the fix is unambiguous.

    ``code_tokens`` drops comments and string literals, so a docstring promising
    the call cannot satisfy it.
    """
    required = ("normalise_condition", "build_conditions")
    found = {name: [] for name in required}
    for path in package_python_files(exclude=("conditioning.py",)):
        tokens = set(code_tokens(path))
        for name in required:
            if name in tokens:
                found[name].append(path.name)

    unused = sorted(name for name, callers in found.items() if not callers)
    assert not unused, (
        f"{', '.join(unused)} in conditioning.py is called by nothing else in the package. "
        f"The UI collects shot anchors and the pipeline receives them unconverted."
    )


def test_the_conversion_is_not_reimplemented_outside_conditioning():
    """One implementation of the latent<->pixel arithmetic, or the two drift.

    Wiring the seam by copying ``(index - 1) * ratio + 1`` into ``generate.py``
    would close every behavioural spec above and reopen the bug the module's own
    docstring warns about: a readout that disagrees with the generator is worse
    than no readout at all.

    Detected as arithmetic on the AST rather than as text, because ``errors.py``
    quotes the very same formula in a docstring — correctly, and it is not a
    second implementation.
    """
    import ast

    offenders = []
    for path in package_python_files(exclude=("conditioning.py", "constraints.py")):
        for node in ast.walk(ast.parse(path.read_text())):
            if not isinstance(node, ast.BinOp):
                continue
            operands = [node.left, node.right]
            if any(
                isinstance(operand, ast.Name) and operand.id == "temporal_compression_ratio"
                for operand in operands
            ):
                offenders.append(f"{path.name}:{node.lineno}")

    assert not offenders, (
        f"{', '.join(offenders)} does latent<->pixel arithmetic outside conditioning.py. "
        f"That conversion belongs there, once — call pixel_index_for / latent_index_for."
    )


def test_the_conditioning_module_still_exposes_what_the_seam_needs():
    """Guards the names the specs above are written against."""
    for name in ("normalise_condition", "build_conditions", "validate_condition_index"):
        assert hasattr(conditioning, name), f"conditioning.{name} has gone missing"
