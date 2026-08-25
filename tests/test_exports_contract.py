"""Spec for the ``exports`` contract — PLAN §3.2, §3.3, §5.3, §7.2.

**This file exists because 1102 tests were structurally blind to a blocker.**

``modes.py`` declares, per mode, an ``exports`` tuple. ``generate.py`` maps what
the encoder wrote onto ``GenerationResult.video_path`` / ``.audio_path``.
``ui/handlers.py`` reads those two attributes back and treats a declared track
that came back falsy as an internal error — correctly, because PLAN §5.3 says a
success-shaped result carrying an empty path is a bug.

Three layers, one shared assumption, and **nothing tested the seam**. The
original defect was two lines:

    video_path = artifact if "video" in spec.exports else None
    audio_path = artifact if "audio" in spec.exports and video_path is None else None

For the three modes exporting *both* tracks, ``video_path`` was set, so
``video_path is None`` was False, so ``audio_path`` came back ``None`` — and the
UI reported every successful generation in the three flagship modes as
"🛑 Internal error — empty output". ``generate()``'s own tests passed (the
artifact existed) and the UI's own tests passed (the guard fired as designed).
The disagreement between them was invisible to both.

So the contract is specified here, once, for **every** mode, as an invariant
rather than an instance:

* every track named in ``exports`` comes back as a real, non-empty file;
* no track outside ``exports`` comes back populated;
* a result that violates its own mode's ``exports`` never leaves ``generate()`` —
  it raises, at the source, rather than travelling onward as ``None``.

No GPU, no weights, no network: ``pipeline_factory`` and ``encoder`` are the
injected seams PLAN §7.3 put there for exactly this.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from conftest import FakeAudioVae

from ltx25_studio import generate as generate_module
from ltx25_studio import modes, outputs
from ltx25_studio.errors import EmptyOutputError

#: Every mode, in registry order. Parametrising over the registry rather than a
#: hand-copied list means a twelfth mode is covered the day it is added.
MODE_IDS: tuple[str, ...] = tuple(modes.MODES)

#: The only two tracks ``GenerationResult`` can express.
TRACKS = ("video", "audio")

#: Filenames taken from the writer, so the fakes cannot drift from production.
FILENAME_FOR = {"video": outputs.VIDEO_FILENAME, "audio": outputs.AUDIO_FILENAME}

#: ``(mode_id, declared_track)`` for every track every mode promises — the unit
#: the contract is actually defined on.
MODE_TRACK_PAIRS: tuple[tuple[str, str], ...] = tuple(
    (mode_id, track) for mode_id in MODE_IDS for track in modes.MODES[mode_id].exports
)


def _pair_ids(pairs, verb):
    return [f"{mode_id}-{verb}-{track}" for mode_id, track in pairs]


# --------------------------------------------------------------------------
# fakes
# --------------------------------------------------------------------------


class FakeOutput:
    """Duck-types ``LTX2PipelineOutput(frames=..., audio=...)``.

    Both tracks are always present because PLAN §3.2(b) says they always are:
    the model is jointly audio-visual and has no video-off switch. A fake that
    withheld the audio for an audio-less mode would be modelling a pipeline that
    does not exist, and would hide exactly the mapping bug this file is about.
    """

    def __init__(self):
        self.frames = ("frame",)
        self.audio = ("sample",)


class FakePipe:
    def __init__(self):
        self.calls: list[dict] = []
        # The real pipeline has one, and the audio-input modes cannot be
        # encoded without it. A fake without it made those modes look fine.
        self.audio_vae = FakeAudioVae()

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        return FakeOutput()


def factory_returning(pipe):
    def _factory(*args, **kwargs):
        return pipe

    return _factory


def _write(run_dir, name: str) -> Path:
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    target = run_dir / name
    target.write_bytes(b"\x00" * 2048)
    return target


def one_muxed_file_encoder(output, *, run_dir, exports, **kwargs) -> Path:
    """The common case: one container carries every track the mode exports.

    For a joint audio+video mode that single muxed mp4 legitimately *is* both
    tracks, which is precisely why returning it must populate both attributes.
    """
    return _write(run_dir, FILENAME_FOR["video" if "video" in exports else "audio"])


def one_file_per_track_encoder(output, *, run_dir, exports, **kwargs) -> dict:
    """The other legal shape: a separate file per track, returned as a mapping."""
    return {track: _write(run_dir, FILENAME_FOR[track]) for track in exports}


ENCODERS = {
    "one-muxed-file": one_muxed_file_encoder,
    "one-file-per-track": one_file_per_track_encoder,
}


def run(mode_id, tmp_path, *, encoder=one_muxed_file_encoder, **overrides):
    return generate_module.generate(
        mode=mode_id,
        seed=7,
        pipeline_factory=factory_returning(FakePipe()),
        encoder=encoder,
        output_root=tmp_path,
        retention_bytes=10**9,
        **overrides,
    )


# --------------------------------------------------------------------------
# the contract itself
# --------------------------------------------------------------------------


@pytest.mark.parametrize("encoder_name", sorted(ENCODERS))
@pytest.mark.parametrize("mode_id", MODE_IDS)
def test_every_track_the_mode_exports_comes_back_as_a_real_file(mode_id, encoder_name, tmp_path):
    """The core of it: ``exports`` is a promise about the result, per track.

    Failing here for a mode means ``ui/handlers.on_generate`` will render a
    perfectly good generation as "🛑 Internal error — empty output".
    """
    spec = modes.get_mode(mode_id)
    result = run(mode_id, tmp_path, encoder=ENCODERS[encoder_name])

    for track in spec.exports:
        path = getattr(result, f"{track}_path")
        assert path is not None, (
            f"mode {mode_id!r} declares exports={list(spec.exports)} but returned "
            f"{track}_path=None. ui/handlers.py treats a declared track that is falsy on a "
            f"success-shaped result as an internal error, so this reports a good run to the "
            f"user as a bug (PLAN §5.3)."
        )
        path = Path(path)
        assert path.is_file(), (
            f"mode {mode_id!r} returned {track}_path={path} but no such file exists. "
            f"PLAN §5.3: every path on a GenerationResult is guaranteed to exist."
        )
        assert path.stat().st_size > 0, (
            f"mode {mode_id!r} returned a 0-byte {track} artifact at {path}."
        )


@pytest.mark.parametrize("encoder_name", sorted(ENCODERS))
@pytest.mark.parametrize("mode_id", MODE_IDS)
def test_no_track_the_mode_does_not_export_is_populated(mode_id, encoder_name, tmp_path):
    """The other half: ``exports`` is exhaustive, not merely a minimum.

    A populated track the mode never promised is a lie the UI acts on — it will
    hand the user a player for a track this mode does not produce, and the most
    likely thing in it is the *previous* run's output.
    """
    spec = modes.get_mode(mode_id)
    result = run(mode_id, tmp_path, encoder=ENCODERS[encoder_name])

    for track in TRACKS:
        if track in spec.exports:
            continue
        assert getattr(result, f"{track}_path") is None, (
            f"mode {mode_id!r} declares exports={list(spec.exports)}, which does not include "
            f"{track!r}, but came back with {track}_path set."
        )


@pytest.mark.parametrize("mode_id", MODE_IDS)
def test_the_populated_tracks_are_exactly_the_declared_ones(mode_id, tmp_path):
    """Stated once as a set equality, so a failure reads as the drift it is."""
    spec = modes.get_mode(mode_id)
    result = run(mode_id, tmp_path)

    populated = {track for track in TRACKS if getattr(result, f"{track}_path") is not None}
    assert populated == set(spec.exports), (
        f"mode {mode_id!r} declares exports={sorted(spec.exports)} but populated "
        f"{sorted(populated)}. generate.py and modes.py disagree about what this mode returns."
    )


# --------------------------------------------------------------------------
# the live blocker, named
# --------------------------------------------------------------------------

JOINT_MODES: tuple[str, ...] = tuple(
    mode_id for mode_id in MODE_IDS if set(modes.MODES[mode_id].exports) == {"audio", "video"}
)


def test_the_registry_still_declares_four_joint_audio_video_modes():
    """PLAN §3.3 rows 5-7, plus mode 11. If this changes, the regression below is aimed wrong.

    ``audio_to_video`` joined this set when the no-audio-stream defect was fixed:
    it conditions on audio and used to return an mp4 with zero audio streams. The
    ruling and its six reasons are in ``modes.py`` at the mode-11 spec. The count
    is pinned by NAME rather than by length so that a mode arriving here silently
    -- which is how a video-only mode would start muxing sound -- fails loudly.
    """
    assert JOINT_MODES == (
        "text_to_audio_video",
        "image_to_audio_video",
        "image_text_to_audio_video",
        "audio_to_video",
    )


@pytest.mark.parametrize("encoder_name", sorted(ENCODERS))
@pytest.mark.parametrize("mode_id", JOINT_MODES)
def test_a_mode_exporting_both_tracks_returns_both_paths(mode_id, encoder_name, tmp_path):
    """The exact shape of the blocker: ``exports=("audio", "video")`` → BOTH set.

    The original mapping computed ``audio_path`` as "the artifact, but only if
    ``video_path`` is None", so the three flagship modes — the ones the model is
    actually for — always returned ``audio_path=None`` on success.
    """
    result = run(mode_id, tmp_path, encoder=ENCODERS[encoder_name])

    assert result.video_path is not None, f"{mode_id!r} exports video but returned no video path"
    assert result.audio_path is not None, (
        f"{mode_id!r} exports BOTH audio and video, and returned a video path but "
        f"audio_path=None. This is the mapping accident that reported every successful "
        f"generation in the three flagship modes as an internal error."
    )


# --------------------------------------------------------------------------
# generalisation: generate() refuses to violate its own contract
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("mode_id", "dropped"), MODE_TRACK_PAIRS, ids=_pair_ids(MODE_TRACK_PAIRS, "without")
)
def test_generate_refuses_a_result_missing_a_declared_track(mode_id, dropped, tmp_path):
    """The invariant, not the instance: no declared track may travel on as ``None``.

    An encoder that quietly omits a track the mode promised is the same class of
    accident as the mapping bug — and PLAN §5.3 requires it to raise at the
    source rather than reach a layer that can only guess whether it was real.
    """

    def encoder(output, *, run_dir, exports, **kwargs):
        return {
            track: _write(run_dir, FILENAME_FOR[track]) for track in exports if track != dropped
        }

    with pytest.raises(EmptyOutputError) as caught:
        run(mode_id, tmp_path, encoder=encoder)

    assert dropped in str(caught.value), (
        f"the error must name the {dropped!r} track that is missing; got {str(caught.value)!r}"
    )


@pytest.mark.parametrize(
    ("mode_id", "dropped"), MODE_TRACK_PAIRS, ids=_pair_ids(MODE_TRACK_PAIRS, "nulls")
)
def test_generate_refuses_a_declared_track_returned_as_none(mode_id, dropped, tmp_path):
    """The key is present and the value is ``None`` — the falsy-artifact shape itself."""

    def encoder(output, *, run_dir, exports, **kwargs):
        return {
            track: (None if track == dropped else _write(run_dir, FILENAME_FOR[track]))
            for track in exports
        }

    with pytest.raises(EmptyOutputError):
        run(mode_id, tmp_path, encoder=encoder)


@pytest.mark.parametrize(
    ("mode_id", "phantom"), MODE_TRACK_PAIRS, ids=_pair_ids(MODE_TRACK_PAIRS, "unwritten")
)
def test_generate_refuses_a_declared_track_whose_file_was_never_written(mode_id, phantom, tmp_path):
    """A path is not an artifact. PLAN §5.3 forbids trusting one we did not see."""

    def encoder(output, *, run_dir, exports, **kwargs):
        Path(run_dir).mkdir(parents=True, exist_ok=True)
        return {
            track: (
                Path(run_dir) / FILENAME_FOR[track]
                if track == phantom
                else _write(run_dir, FILENAME_FOR[track])
            )
            for track in exports
        }

    with pytest.raises(EmptyOutputError):
        run(mode_id, tmp_path, encoder=encoder)


@pytest.mark.parametrize(
    ("mode_id", "empty"), MODE_TRACK_PAIRS, ids=_pair_ids(MODE_TRACK_PAIRS, "zerobyte")
)
def test_generate_refuses_a_declared_track_that_is_zero_bytes(mode_id, empty, tmp_path):
    """A 0-byte file opens, plays nothing, and reports no error. It is not success."""

    def encoder(output, *, run_dir, exports, **kwargs):
        run_dir = Path(run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        resolved = {}
        for track in exports:
            target = run_dir / FILENAME_FOR[track]
            target.write_bytes(b"" if track == empty else b"\x00" * 2048)
            resolved[track] = target
        return resolved

    with pytest.raises(EmptyOutputError):
        run(mode_id, tmp_path, encoder=encoder)


@pytest.mark.parametrize("mode_id", MODE_IDS)
def test_a_track_the_mode_never_declared_does_not_reach_the_result(mode_id, tmp_path):
    """An over-generous encoder must not widen the mode's contract behind its back.

    The audio-only modes still render video internally (PLAN §3.2b) and the
    writer could hand back both. Only what the registry declares is exported.
    """
    spec = modes.get_mode(mode_id)

    def encoder(output, *, run_dir, exports, **kwargs):
        return {track: _write(run_dir, FILENAME_FOR[track]) for track in TRACKS}

    result = run(mode_id, tmp_path, encoder=encoder)

    for track in TRACKS:
        if track in spec.exports:
            continue
        assert getattr(result, f"{track}_path") is None, (
            f"the encoder offered a {track} artifact and mode {mode_id!r} does not export one, "
            f"but it reached the result anyway."
        )


# --------------------------------------------------------------------------
# the result stays inside its own run
# --------------------------------------------------------------------------


@pytest.mark.parametrize("encoder_name", sorted(ENCODERS))
@pytest.mark.parametrize("mode_id", MODE_IDS)
def test_every_returned_artifact_lives_in_this_runs_directory(mode_id, encoder_name, tmp_path):
    """PLAN §5.3's second rule: no run may hand back another run's file."""
    spec = modes.get_mode(mode_id)
    result = run(mode_id, tmp_path, encoder=ENCODERS[encoder_name])

    for track in spec.exports:
        path = Path(getattr(result, f"{track}_path")).resolve()
        assert path.parent == Path(result.run_dir).resolve(), (
            f"mode {mode_id!r} returned a {track} artifact at {path}, outside its own run "
            f"directory {result.run_dir}."
        )


@pytest.mark.parametrize("mode_id", MODE_IDS)
def test_output_path_is_one_of_the_declared_tracks(mode_id, tmp_path):
    """``output_path`` is "the artifact the user receives" — it must be one of them."""
    spec = modes.get_mode(mode_id)
    result = run(mode_id, tmp_path)

    declared = {getattr(result, f"{track}_path") for track in spec.exports}
    assert result.output_path in declared, (
        f"mode {mode_id!r} returned output_path={result.output_path}, which is none of its "
        f"declared tracks {sorted(str(p) for p in declared)}."
    )


@pytest.mark.parametrize("mode_id", [m for m in MODE_IDS if "video" in modes.MODES[m].exports])
def test_a_video_exporting_mode_hands_back_the_video_as_its_primary_artifact(mode_id, tmp_path):
    """When a mode returns video, the video is the thing the user came for."""
    result = run(mode_id, tmp_path, encoder=one_file_per_track_encoder)
    assert result.output_path == result.video_path


# --------------------------------------------------------------------------
# the contract has to be expressible at all
# --------------------------------------------------------------------------


@pytest.mark.parametrize("mode_id", MODE_IDS)
def test_every_declared_export_has_somewhere_to_live_on_the_result(mode_id):
    """``GenerationResult`` can only carry ``video_path`` and ``audio_path``.

    A mode declaring a third track would be undeliverable *and* invisible: the
    handler's guard only inspects those two, so the missing track would never be
    reported. Catch it in the registry rather than in production.
    """
    spec = modes.get_mode(mode_id)
    unrepresentable = set(spec.exports) - set(TRACKS)
    assert not unrepresentable, (
        f"mode {mode_id!r} declares exports={list(spec.exports)}, but GenerationResult has no "
        f"attribute for {sorted(unrepresentable)} and ui/handlers.py only guards "
        f"{list(TRACKS)}. Such a track would be silently dropped."
    )
    assert spec.exports, f"mode {mode_id!r} declares no exports at all"


# --------------------------------------------------------------------------
# stage 2 returns a different size than was asked for
#
# Requesting 704x480 with stage 2 gives a 1408x960 file. That is correct --
# doubling both axes is what stage 2 IS -- but nothing said so, and
# verify_streams does not assert resolution, so the size a user asked for was
# not the size they got and no part of the system mentioned it.
# --------------------------------------------------------------------------


def test_a_stage_2_run_says_the_output_is_twice_the_requested_size(tmp_path):
    result = run("text_to_video", tmp_path, enable_stage_2=True, width=704, height=480)

    notes = " ".join(result.warnings)
    assert "1408" in notes and "960" in notes, (
        f"a stage-2 run did not tell the user its output is 1408x960 rather than the "
        f"704x480 requested: {result.warnings}"
    )
    assert "704" in notes and "480" in notes, (
        f"the note does not name the size that was actually asked for: {result.warnings}"
    )


def test_a_single_stage_run_says_nothing_about_size(tmp_path):
    """A note on every run is a note nobody reads."""
    result = run("text_to_video", tmp_path, width=704, height=480)

    assert not any("upsamples" in note for note in result.warnings), (
        f"a single-stage run warned about a doubling that did not happen: {result.warnings}"
    )


def test_a_mode_that_skips_stage_2_does_not_warn(tmp_path):
    """`text_to_audio` discards its video, so the registry overrides the toggle."""
    result = run("text_to_audio", tmp_path, enable_stage_2=True, width=704, height=480)

    assert not any("upsamples" in note for note in result.warnings), (
        f"a mode with skip_stage_2 warned about an upsample it never runs: {result.warnings}"
    )
