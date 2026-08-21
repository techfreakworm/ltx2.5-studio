#!/usr/bin/env python
"""Exercise the live Space one mode at a time, and account for what each run costs.

    # the DEFAULT: validates every planned config offline and spends nothing
    .venv/bin/python scripts/space_matrix.py

    # the same plan, actually executed against the Space
    HF_TOKEN=... .venv/bin/python scripts/space_matrix.py --live

    # a named subset, in the given order, chaining fixtures through a directory
    HF_TOKEN=... .venv/bin/python scripts/space_matrix.py --live \
        --modes text_to_video,image_to_video --fixture-dir outputs/space-fixtures

Nine of eleven modes now generate on the Space, exercised by this tool; the two
video-INPUT modes fail and their cause is known (``docs/SPACE_MODE_MATRIX.md``).
It was written when only ``text_to_video`` had ever run, around the three ways
that exercise can lie -- and all three earned their place on the run that
followed.

**1. A cost figure that reads as billing.** There is no readable ZeroGPU quota
endpoint -- all three candidate URLs 404 -- so billed seconds cannot be measured
here, only bounded. Every number this tool prints is wall-clock times the size
multiplier and is labelled ``<=`` and ``UPPER BOUND`` at every single site,
because a bound that loses its label becomes a measurement the moment someone
quotes it into a document. This build has already spent a day undoing exactly
that with the inherited RAM ceiling.

**2. A run that produces nothing and reads as success.** ``ui/handlers.py``
catches every generation failure and returns ``(None, None, <markdown>)`` -- a
success-shaped 3-tuple. A client that only unpacks the tuple sees two Nones and
a string and can carry on perfectly happily. So a missing artifact for a track
the registry says the mode EXPORTS raises here, naming the mode and quoting the
Space's own status text verbatim. That is an acceptance criterion for the
project, not polish.

**3. An artifact that is present, well-formed and empty.** H9 came back with the
correct codec, the correct rate, two channels, ``verify_streams`` exit 0 -- and
50 dB below every sibling. A structural check cannot see silence, and it cannot
see a frozen video either. So every artifact is DECODED: video must have the
frame count it asked for, distinct frames, and measurable motion; audio must
have a level, a sample rate and a channel count. The inverse error is just as
expensive -- a synthetic two-tone fixture once produced a *false* defect report
-- so fixtures here are real generated artifacts, never synthesised, and the
content thresholds sit far below the one real measurement we have (pixel std
72.657, mean frame delta 18.894) rather than near it.

**Dry run is the default; ``--live`` is the flag.** Reversing that -- spend by
default, opt out -- puts a metered resource one forgotten flag away from being
consumed, and the default-safe path is the one that gets typed at 3am.

**The ceiling spans invocations, because the budget does.** It did not, once:
the counter lived in one process's memory, so five separate invocations each
started at zero and each independently announced that they fit. Five runs of
1900 GPU-s would each have printed FITS while spending 9500 of a 2400/day
budget. That is the same defect class as an advertised lever attached to
nothing, except worse -- an inert *lever* merely does nothing, while an inert
*report* tells you that you are safe. :class:`PersistentQuota` is the fix, keyed
to the UTC day and kept at a fixed path a routine flag cannot move.

Assumptions stated rather than hidden, because they are genuinely ambiguous:

* The projected cost of a not-yet-started run comes from
  ``zerogpu.estimate_duration``, which reads the LOCAL config. The Space has its
  own ``LTX25_GPU_DURATION_SECONDS`` (60 at the last observed run) and that is
  the value the platform actually reserves against. ``--reservation-seconds``
  states the Space's value when the two differ.
* ``--sampling-rate`` defaults to the pinned revision's vocoder rate. See
  :data:`PINNED_VOCODER_SAMPLING_RATE` for why that number is here at all when
  ``tests/verify_streams.py`` deliberately refuses to hold it.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
# `tests/` as well as the repo root: `verify_streams` is the registry-driven
# stream checker and lives there. Importing it rather than copying its
# expectations is the whole point -- a second place the truth lives is a second
# place it can drift from, and the drift would be invisible because both sides
# would agree with themselves.
for _path in (REPO, REPO / "tests"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import verify_streams  # noqa: E402

from ltx25_studio import constraints, modes, zerogpu  # noqa: E402
from ltx25_studio.ui import fields  # noqa: E402

__all__ = [
    "API_FIELD_KEYS",
    "BILLING_SIZE_MULTIPLIER",
    "DAILY_QUOTA_SECONDS",
    "DEFAULT_MODE_ORDER",
    "DEFAULT_QUOTA_CEILING_SECONDS",
    "ENDPOINT",
    "ArtifactContentError",
    "AudioContent",
    "FixtureStore",
    "Ledger",
    "MatrixOutcome",
    "MissingFixtureError",
    "PersistentQuota",
    "QuotaCeilingReached",
    "QuotaLedgerUnreadable",
    "RunConfig",
    "RunRecord",
    "SpaceRunError",
    "VideoContent",
    "build_plan",
    "main",
    "predict_args",
    "projected_gpu_seconds",
    "run_matrix",
    "run_one",
    "validate_config",
    "verify_audio_content",
    "verify_video_content",
    "utc_day",
]

# ---------------------------------------------------------------------------
# the Space, and the endpoint's positional signature
# ---------------------------------------------------------------------------

#: The private ZeroGPU Space this exercises. Overridable with ``--space`` so the
#: tool survives a rename or a fork without an edit.
DEFAULT_SPACE_ID = "techfreakworm/LTX2.5-Studio"

#: Gradio names an endpoint after the handler function, and ``blocks.py`` wires
#: ``handlers.on_generate`` with no explicit ``api_name``.
ENDPOINT = "/on_generate"

#: The endpoint's positional signature after ``mode_id``, DERIVED from the one
#: declarative input spec rather than restated.
#:
#: ``fields.FIELD_KEYS`` orders the Blocks ``inputs=`` list and orders the
#: handler's ``zip(..., strict=True)`` unpacking -- one tuple, both jobs. The
#: client signature is that same tuple **minus the** ``gr.State``: Gradio gives a
#: State no ``api_info`` and therefore does not expose it to a client at all, so
#: the endpoint takes 16 parameters after ``mode_id``, not 17.
#:
#: Hand-writing this list is how a field added to the UI silently shifts every
#: value one slot to the left -- prompt into negative_prompt, width into height
#: -- which generates something plausible and wrong with no error anywhere. That
#: is the precise defect ``fields.py`` exists to make impossible, and a client
#: that restates the order re-opens it from outside.
API_FIELD_KEYS: tuple[str, ...] = tuple(f.key for f in fields.FIELDS if f.kind != "state")

# ---------------------------------------------------------------------------
# billing -- every number below is a BOUND
# ---------------------------------------------------------------------------

#: PRO daily allowance. 📄 platform, confirmed by ``whoami-v2`` -> ``isPro: true``.
DAILY_QUOTA_SECONDS = 2400.0

#: ZeroGPU bills effective duration times a size multiplier, and promotes us to
#: ``xlarge`` regardless of what we ask for: promotion is on packed size past a
#: 42.00 GiB threshold and ours is 67.23 GiB. ``xlarge`` bills at 2x 📄.
#:
#: This is why every figure here is an upper bound and not a measurement. See
#: ``zerogpu.estimate_duration``'s docstring, which is authoritative on the
#: reservation/billing split: the pre-check compares the REQUESTED duration
#: against remaining quota, billing is on the EFFECTIVE duration. Wall clock is
#: an over-estimate of effective duration -- it includes queueing, upload,
#: download and our own overhead -- so ``2 x wall`` bounds the charge from above
#: and never from below.
BILLING_SIZE_MULTIPLIER = 2.0

#: 80% of the daily budget. Deliberately not 100%: a ceiling at the budget leaves
#: nothing for the operator to debug with afterwards, and the run that discovers
#: the quota is gone is the run you most wanted to inspect.
DEFAULT_QUOTA_CEILING_SECONDS = 0.8 * DAILY_QUOTA_SECONDS

#: Where the cross-invocation quota ledger lives, relative to the repo root.
#:
#: FIXED, and deliberately not under ``--fixture-dir`` or ``--download-dir``.
#: The whole defect :class:`PersistentQuota` exists to close is a guard that only
#: sees its own invocation; a ledger path that a routine flag can move would
#: reintroduce it, because two invocations with different directories would each
#: keep their own total and each announce that they fit.
#:
#: Inside ``outputs/``, which ``.gitignore`` already excludes, so accounting
#: state never becomes an untracked file in the working tree. Dotted so it does
#: not look like a run directory to anything sweeping that tree.
DEFAULT_QUOTA_LEDGER = "outputs/.space-quota-ledger.json"

# ---------------------------------------------------------------------------
# the cheapest configuration that still proves a code path
# ---------------------------------------------------------------------------

#: 256 is ``modes.MIN_RESOLUTION`` -- read from the registry so the number the
#: audio-only modes are FORCED to and the number we choose cannot drift apart.
CHEAPEST_SIZE = modes.MIN_RESOLUTION

#: ``k*8 + 1`` at k=1. The smallest frame count above the single-frame degenerate
#: case; ``constraints.validate_num_frames`` is what actually checks it.
CHEAPEST_FRAMES = 9

#: Pinned, and it matters: ``seed=-1`` draws a fresh value per call, so any
#: comparison run at the default differs for that reason alone and passes
#: vacuously. Same value as ``scripts/host_matrix.py`` so a Space run and a host
#: run of the same mode are comparable.
SEED = 1234

PROMPT = (
    "A red sports car driving fast along a winding coastal road at sunset, "
    "waves breaking on rocks below, cinematic camera following alongside"
)

#: Read off the pinned revision's ``vocoder.config.output_sampling_rate`` 📄.
#:
#: ``tests/verify_streams.py`` deliberately refuses to hold this number, because
#: on the GPU host the vocoder is loaded and can be asked. Here it cannot: this
#: is a laptop-side client with no weights, and "omit the check" is the option
#: that lets a container written at the wrong rate pass silently. So the number
#: lives here, with the same status as ``constraints.DEFAULT_TEMPORAL_COMPRESSION_RATIO``
#: -- a pre-load default, NOT a constant of the model -- and ``--sampling-rate``
#: overrides it for anyone on a different revision.
PINNED_VOCODER_SAMPLING_RATE = 48000

# ---------------------------------------------------------------------------
# content thresholds
#
# Set far BELOW the one real measurement we have (§9.3 of the Space finding:
# pixel std 72.657, mean frame-to-frame 18.894, 9 of 9 distinct) rather than
# near it. A threshold near the observed value fails the first legitimately
# calmer shot and produces a false defect -- which has already cost this build
# a day. These separate "degenerate" from "anything at all", nothing finer.
# ---------------------------------------------------------------------------

#: Below this the frame is flat -- a blank or single-colour render.
MIN_PIXEL_STD = 1.0

#: Mean absolute inter-frame difference, on the 0..255 scale. A frozen video is
#: exactly 0.0.
MIN_MEAN_FRAME_DELTA = 0.5

#: Two distinct frames is "not frozen". The full ``distinct X of N`` ratio is
#: always REPORTED -- a 7-of-9 is worth a human's attention -- but it is not
#: asserted on, because a legitimately static shot can encode a repeated frame
#: and failing that would be a false defect of exactly the kind this tool exists
#: to avoid. The motion check above is the load-bearing one.
MIN_DISTINCT_FRAMES = 2

#: Below this there is no signal at all. NOT the same as the -40 dBFS
#: "very quiet" flag ``scripts/host_audio_levels.py`` uses: -55 dBFS is faint and
#: audible, and calling it silence is what turned H9 into a false defect report.
#: Silence raises; quiet warns.
SILENT_DBFS = -70.0
VERY_QUIET_DBFS = -40.0

# ---------------------------------------------------------------------------
# the plan
# ---------------------------------------------------------------------------

#: Default order, and it is a DEPENDENCY order, not the registry's table order.
#:
#: Later modes consume real generated artifacts, so every producer must run
#: before its consumers:
#:
#: 1. ``text_to_video``  needs nothing and produces the **video** fixture, and
#:    the **image** fixture by frame extraction.
#: 2. ``text_to_audio``  needs nothing and produces the **audio** fixture. It is
#:    the cheapest audio producer in the registry -- audio-only modes are forced
#:    to ``MIN_RESOLUTION`` with stage 2 off -- and its artifact is a real
#:    ``output.wav`` rather than an AAC track inside an mp4.
#:
#: Everything after that is ordered cheapest-input-first so that a run stopped by
#: the quota ceiling has covered the widest set of code paths, not the deepest.
DEFAULT_MODE_ORDER: tuple[str, ...] = (
    "text_to_video",
    "text_to_audio",
    "text_to_audio_video",
    "image_to_video",
    "image_text_to_video",
    "image_to_audio_video",
    "image_text_to_audio_video",
    "video_to_video",
    "video_to_audio",
    "audio_to_audio",
    "audio_to_video",
)


# ---------------------------------------------------------------------------
# failures -- all three of them loud
# ---------------------------------------------------------------------------


class SpaceRunError(RuntimeError):
    """A run finished and did not produce a track the registry says it exports.

    This is the project's headline acceptance criterion. ``ui/handlers.py``
    converts EVERY generation failure into ``(None, None, <markdown>)``, which
    over the wire is indistinguishable in shape from a success. Nothing further
    down the client would notice; the only thing that can is a check against what
    the mode DECLARED, which is what this carries.
    """

    def __init__(self, *, mode_id: str, missing, status, wall_seconds=None):
        self.mode_id = mode_id
        self.missing = tuple(missing)
        self.status = status
        self.wall_seconds = wall_seconds
        tracks = " and ".join(self.missing)
        timing = "" if wall_seconds is None else f" after {wall_seconds:.2f}s of wall clock"
        quoted = _quote(status) or (
            "    (the Space returned NO status text at all -- which is itself a defect:\n"
            "     `ui/render.py` guarantees every failure path produces a non-empty string,\n"
            "     so an empty one means the response never reached that layer.)"
        )
        # Guarded: this exception is also raised for a response whose SHAPE is
        # wrong, where there is no mode to look up. An error class that raises a
        # different error while formatting its own message replaces the finding
        # with a red herring, which is the one thing it must never do.
        try:
            declared = str(modes.get_mode(mode_id).exports)
        except Exception:  # noqa: BLE001
            declared = "(mode not resolvable from this response)"
        super().__init__(
            f"{mode_id}: the Space produced no {tracks} file{timing}, and "
            f"`ltx25_studio.modes` declares this mode exports {declared}.\n\n"
            f"The Space's own status text, verbatim:\n\n{quoted}\n\n"
            f"This is a FAILED run, not an empty one. `ui/handlers.on_generate` returns "
            f"(None, None, <error markdown>) for every failure, so the response is "
            f"success-SHAPED and a client that only unpacks the tuple sees nothing wrong. "
            f"Read the status above first; if it names 'GPU task aborted' the forked worker "
            f"was killed and there will be NO traceback in the Space log -- the absence is "
            f"the finding, not a gap in it."
        )


class ArtifactContentError(RuntimeError):
    """A file exists, parses, and contains nothing anyone wanted.

    Separate from :class:`SpaceRunError` on purpose: that one means the run
    failed, this one means the run *succeeded* and the output is empty. They need
    different next actions, so they are different exceptions.
    """


class MissingFixtureError(RuntimeError):
    """A mode needs an input kind no earlier run produced."""


class QuotaCeilingReached(RuntimeError):
    """Starting the next run would project past the ceiling. Nothing was spent."""

    def __init__(self, *, config, projected: float, ledger):
        self.config = config
        self.projected = projected
        self.ledger = ledger
        super().__init__(
            f"refusing to start {config.mode_id}: {ledger.spent_upper_bound:.1f} GPU-s already "
            f"bounded plus a projected <= {projected:.1f} GPU-s would reach "
            f"{ledger.spent_upper_bound + projected:.1f}, past the "
            f"{ledger.ceiling_seconds:.1f} GPU-s ceiling."
        )


def _quote(text) -> str:
    """Indent a block so the Space's words are visibly not ours."""
    body = ("" if text is None else str(text)).strip()
    if not body:
        return ""
    return "\n".join(f"    {line}" for line in body.splitlines())


# ---------------------------------------------------------------------------
# one planned run
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RunConfig:
    """One planned generation: the mode plus every value the endpoint takes.

    Frozen because a config is written into the ledger and the log, and a config
    that can be mutated afterwards is a config the record no longer describes.
    """

    mode_id: str
    width: int = CHEAPEST_SIZE
    height: int = CHEAPEST_SIZE
    num_frames: int = CHEAPEST_FRAMES
    seed: int = SEED
    prompt: str = PROMPT
    negative_prompt: str = ""
    enable_stage_2: bool = False
    auto_duration: bool = False
    min_seconds: float = 2.0
    max_seconds: float = 5.0
    strength: float = 0.8
    noise_scale: float = 0.5

    @property
    def spec(self):
        """The registry entry. Resolved every access, never snapshotted."""
        return modes.get_mode(self.mode_id)

    @property
    def effective_width(self) -> int:
        """What will actually render, which is not always what we asked for.

        ``min_resolution`` modes are pinned to ``modes.MIN_RESOLUTION`` by the
        registry no matter what the request says. Reporting the requested value
        for those would put a number in the cost log that no run used.
        """
        return modes.MIN_RESOLUTION if self.spec.min_resolution else self.width

    @property
    def effective_height(self) -> int:
        return modes.MIN_RESOLUTION if self.spec.min_resolution else self.height

    @property
    def effective_stage_2(self) -> bool:
        """``generate.stage_2_enabled()``: the registry wins over the request."""
        return bool(self.enable_stage_2) and not self.spec.skip_stage_2

    @property
    def needs(self) -> tuple[str, ...]:
        """Input kinds that are FILES, from the registry's ``inputs``.

        ``"text"`` is an input too, and it is not a fixture -- the intersection
        with ``fields.MEDIA_KINDS`` is what separates the two, and it is the same
        set ``components.py`` uses to decide which widgets carry an asset.
        """
        return tuple(k for k in self.spec.inputs if k in fields.MEDIA_KINDS)

    @property
    def geometry(self) -> str:
        if self.auto_duration:
            return f"{self.effective_width}x{self.effective_height}x(duration head)"
        return f"{self.effective_width}x{self.effective_height}x{self.num_frames}"


def validate_config(config: RunConfig) -> RunConfig:
    """Check a config against the real validators, or raise. Costs nothing.

    ``constraints`` is the module that already knows these rules, and it knows
    them for a reason worth restating: upstream does NOT raise on a bad
    ``num_frames``, it floors silently, so 120 frames becomes 113 and nobody is
    told. Re-deriving the modulus here would be a second copy of a rule whose
    whole point is that the first copy exists.
    """
    constraints.validate_dimensions(config.effective_width, config.effective_height)
    if not config.auto_duration:
        # auto_duration drops num_frames entirely -- `generate()` pops it and
        # passes the seconds bounds instead -- so validating it would reject a
        # legal plan on a value that never reaches the model.
        constraints.validate_num_frames(config.num_frames)
    return config


def build_plan(mode_ids=None, **overrides) -> tuple[RunConfig, ...]:
    """Ordered configs for the named modes, defaulting to :data:`DEFAULT_MODE_ORDER`.

    Raises on an unknown mode via ``modes.get_mode``, which names what IS
    registered rather than just what was not.
    """
    chosen = tuple(DEFAULT_MODE_ORDER if mode_ids is None else mode_ids)
    for mode_id in chosen:
        modes.get_mode(mode_id)
    return tuple(RunConfig(mode_id=mode_id, **overrides) for mode_id in chosen)


# ---------------------------------------------------------------------------
# the ledger -- bounds, never measurements
# ---------------------------------------------------------------------------


def projected_gpu_seconds(config: RunConfig, *, reservation_seconds=None) -> float:
    """Upper bound on ONE worker attempt, before the run starts.

    ``zerogpu.estimate_duration`` is the app's own estimator and the platform
    kills a task at its requested duration, so effective duration <= requested
    duration and ``requested x multiplier`` bounds the charge for one attempt.

    **For one attempt.** ``spaces`` re-forks a killed worker -- the failed run in
    the Space finding burned 100.35s of wall clock against a 60s reservation
    across two attempts -- so a run CAN exceed this. That is exactly why the
    ledger charges observed wall clock rather than this figure, and why this one
    is used only to decide whether to start.

    ``reservation_seconds`` exists because the estimator reads the LOCAL config
    and the Space reserves against its own ``LTX25_GPU_DURATION_SECONDS``.
    """
    params: dict = {"enable_stage_2": config.effective_stage_2}
    if reservation_seconds is not None:
        params["gpu_duration_seconds"] = int(reservation_seconds)
    return BILLING_SIZE_MULTIPLIER * float(zerogpu.estimate_duration(**params))


def utc_day(now=None) -> str:
    """Today's date in UTC, as ``YYYY-MM-DD``.

    UTC and not local time, because the budget is the platform's and the
    platform does not know where the operator is. A ledger keyed on local
    midnight resets at a different instant than the thing it is tracking.
    """
    moment = datetime.now(UTC) if now is None else now
    return moment.strftime("%Y-%m-%d")


class QuotaLedgerUnreadable(RuntimeError):
    """The ledger file exists and could not be read. Refuses rather than zeroing.

    A corrupt accounting file is not evidence that nothing was spent, and
    treating it as zero is how a guard silently hands back the whole budget at
    the worst possible moment. ``--reset-quota-ledger`` is the deliberate way to
    say "yes, start today's count from zero".
    """


class PersistentQuota:
    """Cumulative upper-bound spend for one UTC day, shared across invocations.

    **The defect this closes.** The ceiling used to live entirely in one
    process's memory, so every invocation started at zero and every invocation
    independently announced that it fit. Five runs of 1900 GPU-s each would each
    print FITS while spending 9500 of a 2400/day budget. That is an advertised
    guard attached to nothing -- and an inert *report* is worse than an inert
    lever, because the lever merely does nothing while the report actively tells
    you that you are safe.

    **Why a JSON file, and why at a FIXED path.** The state has to outlive the
    process and be found by the next one whatever flags it was given. Putting it
    under ``--fixture-dir`` or ``--download-dir`` would reintroduce the very
    defect: two invocations pointed at different directories would each keep
    their own total and each announce FITS. So the location is fixed, and
    ``--quota-ledger`` exists to move it deliberately rather than by accident.
    JSON rather than a database because the file's whole job is to be readable
    by a human at 3am who wants to know how much of today is gone.

    **Not keyed on the Space.** The 2400 s/day allowance is per ACCOUNT, not per
    Space, so two Spaces on one token share one budget and must share one
    ledger.

    **Every number in it is an upper bound.** The file says so in its own
    ``unit`` field, so the label survives being read by something other than
    this program.

    ⏸️ **Whether the platform's own reset is a UTC-day boundary or a rolling
    24-hour window is NOT established here.** The daily figure is documented; the
    reset mechanism was never probed, and there is no readable quota endpoint to
    probe it with. If it is a rolling window, this ledger's midnight reset
    under-counts for up to a day -- the unsafe direction -- so a rollover prints
    the previous day's total rather than silently discarding it.
    """

    SCHEMA = 1
    UNIT = "gpu-seconds, wall-clock-derived UPPER BOUND — NOT billed seconds"

    def __init__(self, path, *, day=None, reset: bool = False):
        self.path = Path(path)
        self.day = day or utc_day()
        self.spent_upper_bound = 0.0
        self.invocations = 0
        #: ``(day, spent)`` discarded by a rollover, so the report can say what
        #: was dropped instead of quietly starting from zero.
        self.rolled_over_from: tuple[str, float] | None = None
        if not reset:
            self._read()

    def _read(self) -> None:
        if not self.path.exists():
            return
        try:
            data = json.loads(self.path.read_text())
            stored_day = str(data["utc_day"])
            stored_spent = float(data["spent_upper_bound"])
            stored_invocations = int(data.get("invocations", 0))
        except Exception as exc:
            raise QuotaLedgerUnreadable(
                f"{self.path} exists and could not be read as a quota ledger "
                f"({type(exc).__name__}: {exc}).\n\n"
                f"Refusing to treat that as zero: a corrupt accounting file is not evidence "
                f"that nothing was spent, and starting from zero here is how the guard hands "
                f"back a whole day's budget at the worst possible moment. Inspect the file, or "
                f"pass --reset-quota-ledger to start today's count from zero on purpose."
            ) from exc
        if stored_day != self.day:
            self.rolled_over_from = (stored_day, stored_spent)
            return
        self.spent_upper_bound = stored_spent
        self.invocations = stored_invocations

    def begin_invocation(self) -> None:
        """Count this process against the day, before it spends anything."""
        self.invocations += 1
        self.save()

    def add(self, gpu_seconds_upper_bound: float) -> None:
        self.spent_upper_bound += float(gpu_seconds_upper_bound)
        self.save()

    def save(self) -> None:
        """Write atomically: a crash mid-write must not produce a file we refuse."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema": self.SCHEMA,
            "unit": self.UNIT,
            "utc_day": self.day,
            "spent_upper_bound": round(self.spent_upper_bound, 3),
            "invocations": self.invocations,
            "daily_quota_seconds": DAILY_QUOTA_SECONDS,
            "note": (
                "Derived from wall clock x the xlarge size multiplier. There is no readable "
                "ZeroGPU quota endpoint, so this bounds the spend from above and is never a "
                "measurement of billed seconds."
            ),
        }
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        temporary.write_text(json.dumps(payload, indent=2) + "\n")
        temporary.replace(self.path)

    def render(self) -> str:
        lines = [
            f"quota ledger {self.path}",
            f"  carried      <= {self.spent_upper_bound:.1f} GPU-s already bounded on UTC "
            f"{self.day} across {self.invocations} earlier invocation(s)  "
            f"*** UPPER BOUND, NOT MEASURED BILLING ***",
        ]
        if self.rolled_over_from is not None:
            previous_day, previous_spent = self.rolled_over_from
            lines.append(
                f"  NEW UTC DAY  reset to 0; UTC {previous_day} ended at <= "
                f"{previous_spent:.1f} GPU-s upper bound."
            )
            lines.append(
                "               ⏸️ If the platform's reset is a rolling 24h window rather than "
                "a UTC-day boundary, some of that may still count against you. This ledger "
                "cannot tell, and no readable quota endpoint exists to ask."
            )
        return "\n".join(lines)


@dataclass
class Ledger:
    """Cumulative upper-bound spend, and the refusal that keeps it under a ceiling.

    Two numbers, kept apart on purpose. ``projected`` decides whether to START a
    run; ``spent_upper_bound`` accumulates what was OBSERVED. They are different
    meters and blurring them is how a plan that fit becomes a budget that did
    not: the projection can under-shoot when the platform retries a killed
    worker, and the observation cannot.

    ``store`` is what makes the ceiling mean anything across invocations. Without
    it this counter starts at zero every time the process does, and five runs of
    1900 each would each announce that they fit while spending 9500 of a 2400
    budget -- an advertised guard attached to nothing. See :class:`PersistentQuota`.
    """

    ceiling_seconds: float = DEFAULT_QUOTA_CEILING_SECONDS
    daily_quota_seconds: float = DAILY_QUOTA_SECONDS
    reservation_seconds: int | None = None
    spent_upper_bound: float = 0.0
    records: list = field(default_factory=list)
    #: Cross-invocation accounting. ``None`` means this ledger sees only itself,
    #: which is correct for a unit test and wrong for a real run.
    store: PersistentQuota | None = None
    #: What was already on the books when this process started, so the report can
    #: separate "this invocation" from "today".
    carried_upper_bound: float = 0.0

    @classmethod
    def from_store(cls, store: PersistentQuota, **kwargs) -> Ledger:
        """A ledger that starts where today's accounting already stands."""
        return cls(
            store=store,
            spent_upper_bound=store.spent_upper_bound,
            carried_upper_bound=store.spent_upper_bound,
            **kwargs,
        )

    @property
    def spent_this_invocation(self) -> float:
        return self.spent_upper_bound - self.carried_upper_bound

    def projected_for(self, config: RunConfig) -> float:
        return projected_gpu_seconds(config, reservation_seconds=self.reservation_seconds)

    def would_cross(self, config: RunConfig) -> bool:
        """True if starting this run could reach the ceiling. Never borrows ahead."""
        return (self.spent_upper_bound + self.projected_for(config)) > self.ceiling_seconds

    def check_before(self, config: RunConfig) -> float:
        """Raise :class:`QuotaCeilingReached` rather than start. Returns the projection."""
        projected = self.projected_for(config)
        if (self.spent_upper_bound + projected) > self.ceiling_seconds:
            raise QuotaCeilingReached(config=config, projected=projected, ledger=self)
        return projected

    def charge(self, record: RunRecord) -> None:
        self.spent_upper_bound += record.gpu_seconds_upper_bound
        self.records.append(record)
        if self.store is not None:
            # Write THROUGH, per run, not once at the end. A matrix that dies on
            # run 7 has still spent the first six, and an accounting file that
            # only lands on a clean exit loses exactly the runs worth knowing
            # about -- which is the same shape as the failure this whole class
            # exists to close, one level out.
            self.store.add(record.gpu_seconds_upper_bound)

    @property
    def headroom(self) -> float:
        return self.ceiling_seconds - self.spent_upper_bound

    def render(self) -> str:
        percent = 100.0 * self.spent_upper_bound / self.daily_quota_seconds
        line = (
            f"cumulative <= {self.spent_upper_bound:.1f} GPU-s of a "
            f"{self.ceiling_seconds:.1f} ceiling "
            f"({self.headroom:.1f} left) · <= {percent:.1f}% of the "
            f"{self.daily_quota_seconds:.0f}/day budget  "
            f"*** UPPER BOUND, NOT MEASURED BILLING ***"
        )
        if self.carried_upper_bound:
            line += (
                f"\n             of which <= {self.carried_upper_bound:.1f} carried from "
                f"{self.store.invocations - 1 if self.store else '?'} earlier invocation(s) "
                f"on UTC {self.store.day if self.store else '?'}"
            )
        return line


# ---------------------------------------------------------------------------
# what one run produced
# ---------------------------------------------------------------------------


@dataclass
class VideoContent:
    """Decoded video, as numbers. A stream table cannot see any of this."""

    frames: int
    distinct_frames: int
    pixel_std: float
    mean_abs_delta: float
    min_abs_delta: float
    max_abs_delta: float

    def render(self) -> str:
        return (
            f"{self.frames} frames · {self.distinct_frames} distinct · "
            f"pixel std {self.pixel_std:.3f} · mean |Δ| {self.mean_abs_delta:.3f} "
            f"(per-pair {self.min_abs_delta:.2f}..{self.max_abs_delta:.2f})"
        )


@dataclass
class AudioContent:
    """Decoded audio, as numbers -- including the level, always."""

    sample_rate: int
    channels: int
    samples: int
    peak: float
    rms: float
    dbfs: float

    def render(self) -> str:
        flag = "  *** VERY QUIET (<-40 dBFS) ***" if self.dbfs < VERY_QUIET_DBFS else ""
        return (
            f"{self.sample_rate} Hz · {self.channels}ch · {self.samples} samples · "
            f"peak {self.peak:.4f} · rms {self.rms:.6f} = {self.dbfs:.1f} dBFS{flag}"
        )


@dataclass
class RunRecord:
    """One completed run: what it was, what it cost, what it produced."""

    config: RunConfig
    wall_seconds: float
    artifacts: dict  # track -> (path, byte size)
    status: str = ""
    projected_gpu_seconds: float = 0.0
    stream_verdicts: dict = field(default_factory=dict)
    video: VideoContent | None = None
    audio: AudioContent | None = None
    fixtures_saved: tuple[str, ...] = ()

    @property
    def mode_id(self) -> str:
        return self.config.mode_id

    @property
    def gpu_seconds_upper_bound(self) -> float:
        """Wall clock times the size multiplier. Never a measurement.

        Wall clock over-estimates effective duration -- it contains queueing,
        upload, download and our own overhead -- so this bounds the charge from
        above. There is no readable quota endpoint to check it against; if one
        ever appears, this is the number to compare it to.
        """
        return BILLING_SIZE_MULTIPLIER * self.wall_seconds

    def render_cost(self) -> str:
        return (
            f"<= {self.gpu_seconds_upper_bound:.1f} GPU-s  "
            f"({self.wall_seconds:.2f}s wall clock x {BILLING_SIZE_MULTIPLIER:g} size "
            f"multiplier)  *** UPPER BOUND, NOT MEASURED BILLING ***"
        )


@dataclass
class MatrixOutcome:
    """The whole session. ``stopped_at`` is the run that was REFUSED, not one that failed."""

    records: list
    ledger: Ledger
    stopped_at: RunConfig | None = None
    stop_reason: str = ""

    @property
    def stopped(self) -> bool:
        return self.stopped_at is not None


# ---------------------------------------------------------------------------
# content verification -- decode it, do not merely open it
# ---------------------------------------------------------------------------


def _decode_video_frames(path, *, limit=512):
    """Every video frame as an rgb24 array, or an empty list."""
    import av

    out = []
    with av.open(str(path)) as container:
        if not container.streams.video:
            return out
        for frame in container.decode(video=0):
            out.append(frame.to_ndarray(format="rgb24"))
            if len(out) >= limit:
                break
    return out


def verify_video_content(path, *, expected_frames=None) -> VideoContent:
    """Decode the video and refuse a blank or frozen one. Raises, never returns False.

    ``verify_streams`` already proved the container declares a video stream with
    the right geometry and rate. That is the half a green table can be wrong
    about: a file of nine identical grey frames satisfies all of it. So this
    measures what is actually IN the frames.

    Three separate degeneracies, because they fail differently:

    * **no frames at all** -- a header with no packets behind it;
    * **flat frames** (pixel std ~0) -- a blank or single-colour render;
    * **no motion** (frame-to-frame delta 0) -- the same frame N times, which is
      what a stuck decode or a discarded temporal axis looks like.
    """
    import numpy as np

    frames = _decode_video_frames(path)
    if not frames:
        raise ArtifactContentError(
            f"{path} yielded ZERO video frames. Either it carries no video stream at all, or "
            f"the container header has no packets behind it. A header is not evidence that "
            f"anything is behind it, and 'the file opened' is not a pass."
        )

    stack = np.stack(frames).astype("float32")
    distinct = len({frame.tobytes() for frame in frames})
    pixel_std = float(stack.std())
    if len(frames) > 1:
        deltas = [float(np.abs(stack[i + 1] - stack[i]).mean()) for i in range(len(frames) - 1)]
    else:
        deltas = [0.0]
    content = VideoContent(
        frames=len(frames),
        distinct_frames=distinct,
        pixel_std=pixel_std,
        mean_abs_delta=float(sum(deltas) / len(deltas)),
        min_abs_delta=min(deltas),
        max_abs_delta=max(deltas),
    )

    problems = []
    if expected_frames is not None and content.frames != expected_frames:
        problems.append(
            f"decoded {content.frames} frames, asked for {expected_frames} -- upstream floors "
            f"an illegal num_frames silently, so a short video is a real signal, not rounding"
        )
    if content.pixel_std < MIN_PIXEL_STD:
        problems.append(
            f"global pixel std {content.pixel_std:.4f} < {MIN_PIXEL_STD} -- the frames are flat. "
            f"A blank render passes every structural check there is"
        )
    if content.distinct_frames < MIN_DISTINCT_FRAMES:
        problems.append(
            f"{content.distinct_frames} distinct frame(s) of {content.frames} -- every frame is "
            f"byte-identical, so nothing moved"
        )
    if content.mean_abs_delta < MIN_MEAN_FRAME_DELTA:
        problems.append(
            f"mean frame-to-frame |Δ| {content.mean_abs_delta:.4f} < {MIN_MEAN_FRAME_DELTA} -- "
            f"the video is frozen. The one healthy run measured 18.894 here"
        )
    if problems:
        raise ArtifactContentError(
            f"{path} parses as video and contains nothing:\n  "
            + "\n  ".join(f"✗ {p}" for p in problems)
            + f"\n  measured: {content.render()}"
        )
    return content


def _decode_audio_planes(path):
    """(planes, sample_rate, channels) for the file's first audio stream."""
    import av

    with av.open(str(path)) as container:
        if not container.streams.audio:
            return [], None, None
        stream = container.streams.audio[0]
        rate = int(stream.rate) if stream.rate else None
        layout = getattr(stream, "layout", None)
        channels = int(getattr(layout, "nb_channels", 0) or 0) or None
        planes = [frame.to_ndarray() for frame in container.decode(stream)]
    return planes, rate, channels


def verify_audio_content(
    path,
    *,
    expected_sample_rate=None,
    expected_channels=verify_streams.EXPECTED_AUDIO_CHANNELS,
) -> AudioContent:
    """Decode the audio and refuse a silent one. Reports dBFS whatever happens.

    H9 is the reason this exists as a hard gate rather than a log line: correct
    codec, correct rate, two channels, ``verify_streams`` exit 0 -- and 50 dB
    below every sibling. Nothing structural could see it.

    The threshold separates **silence** from **quiet**, and the distinction is
    load-bearing. -55 dBFS is faint and audible; calling it silence is what
    turned H9 into a false defect report that cost a day. So
    :data:`SILENT_DBFS` raises and :data:`VERY_QUIET_DBFS` only annotates.
    """
    import numpy as np

    planes, rate, channels = _decode_audio_planes(path)
    if not planes:
        raise ArtifactContentError(
            f"{path} declares an audio stream and decoded to ZERO samples. The stream "
            f"header is not evidence that anything is behind it."
        )

    raw = np.concatenate(planes, axis=-1)
    if np.issubdtype(raw.dtype, np.integer):
        # av hands back int16 for pcm_s16le; the level arithmetic wants full scale
        # at 1.0 so the dBFS figure is comparable with host_audio_levels.py.
        raw = raw.astype("float32") / 32768.0
    else:
        raw = raw.astype("float32")

    rms = float(np.sqrt((raw**2).mean()))
    content = AudioContent(
        sample_rate=int(rate or 0),
        channels=int(channels or (raw.shape[0] if raw.ndim > 1 else 1)),
        samples=int(raw.shape[-1]),
        peak=float(np.abs(raw).max()),
        rms=rms,
        # The epsilon keeps digital silence finite so it prints as a number
        # instead of -inf; it lands around -240 dBFS, far under any threshold.
        dbfs=float(20 * np.log10(rms + 1e-12)),
    )

    problems = []
    if content.dbfs < SILENT_DBFS:
        problems.append(
            f"level {content.dbfs:.1f} dBFS < {SILENT_DBFS} -- this track is SILENT. "
            f"Every structural check passes on silence"
        )
    if expected_sample_rate is not None and content.sample_rate != int(expected_sample_rate):
        problems.append(
            f"sample rate {content.sample_rate} Hz, expected {int(expected_sample_rate)} -- the "
            f"container disagrees with the vocoder and the track plays at the wrong pitch"
        )
    if expected_channels is not None and content.channels != int(expected_channels):
        problems.append(
            f"{content.channels} channel(s), expected {int(expected_channels)} -- something "
            f"downmixed the vocoder's stereo output"
        )
    if problems:
        raise ArtifactContentError(
            f"{path} parses as audio and is not what the mode promised:\n  "
            + "\n  ".join(f"✗ {p}" for p in problems)
            + f"\n  measured: {content.render()}"
        )
    return content


# ---------------------------------------------------------------------------
# fixtures -- real generated artifacts, never synthesised
# ---------------------------------------------------------------------------


class FixtureStore:
    """Real artifacts from earlier runs, keyed by the input KIND that consumes them.

    **Generated, not synthesised, and that is a correctness property rather than
    a preference.** ``host_matrix.py`` builds a two-tone wav when no generated
    one exists, and that fixture is part of how a false defect got reported: a
    synthetic input exercises a different distribution from anything the model
    produces, so a failure against it says nothing about the real path. Here the
    only inputs are files the Space itself made.

    The image fixture is extracted from a produced video rather than being a
    separate render, because there is no mode that produces a still and paying
    for one would be a whole extra generation for a frame we already have.
    """

    FILENAMES = {"image": "image.png", "video": "video.mp4", "audio": "audio.wav"}

    def __init__(self, directory):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)

    def path_for(self, kind: str) -> Path:
        try:
            return self.directory / self.FILENAMES[kind]
        except KeyError:
            raise KeyError(
                f"no fixture kind {kind!r}; known kinds: {', '.join(sorted(self.FILENAMES))}"
            ) from None

    def has(self, kind: str) -> bool:
        path = self.path_for(kind)
        return path.exists() and path.stat().st_size > 0

    def require(self, kind: str, *, mode_id: str) -> Path:
        """The fixture, or a refusal that says which run would have produced it."""
        if self.has(kind):
            return self.path_for(kind)
        # Named from the registry rather than listed here: the image fixture is a
        # frame out of a produced video, so image and video have the same
        # producers, and only a mode needing no file input can be run first.
        wanted_track = "audio" if kind == "audio" else "video"
        producers = [
            spec.id
            for spec in modes.MODES.values()
            if wanted_track in spec.exports and not (set(spec.inputs) & fields.MEDIA_KINDS)
        ]
        raise MissingFixtureError(
            f"{mode_id} needs a real {kind} input and {self.path_for(kind)} does not exist. "
            f"Run one of {producers[:4]} first -- with --fixture-dir {self.directory} -- so the "
            f"input is something the model actually produced. Do NOT substitute a synthesised "
            f"file: a two-tone wav has already produced one false defect report on this build."
        )

    def adopt(self, config: RunConfig, artifacts: dict, log=print) -> tuple[str, ...]:
        """Harvest whatever this run's artifacts can serve as a later input.

        Driven by the mode's declared ``exports`` rather than by sniffing the
        file, so a mode that stops exporting a track stops seeding that fixture.
        """
        saved: list[str] = []
        exports = config.spec.exports

        if "video" in exports:
            source = artifacts.get("video", (None, 0))[0]
            if source:
                shutil.copyfile(source, self.path_for("video"))
                saved.append("video")
                log(f"  fixture      video  <- {source} -> {self.path_for('video')}")
                try:
                    frame_index = extract_image_fixture(source, self.path_for("image"))
                except Exception as exc:  # noqa: BLE001 -- a failed extract is not a failed run
                    log(f"  fixture      image  EXTRACT FAILED: {type(exc).__name__}: {exc}")
                else:
                    saved.append("image")
                    log(
                        f"  fixture      image  <- frame {frame_index} of that video "
                        f"-> {self.path_for('image')}"
                    )

        if "audio" in exports:
            source = artifacts.get("audio", (None, 0))[0]
            if source:
                try:
                    extract_audio_fixture(source, self.path_for("audio"))
                except Exception as exc:  # noqa: BLE001
                    log(f"  fixture      audio  EXTRACT FAILED: {type(exc).__name__}: {exc}")
                else:
                    saved.append("audio")
                    log(f"  fixture      audio  <- {source} -> {self.path_for('audio')}")

        return tuple(saved)


def extract_image_fixture(video_path, dest) -> int:
    """Write the MIDDLE frame of a video as a PNG. Returns the frame index.

    The middle, not frame 0, and the reason is the same one behind
    ``host_matrix.make_image``'s refusal to use a flat fill: for a conditioned
    run frame 0 is the input echoed back, so an image fixture taken from there is
    a copy of an earlier fixture and conditions the next run on itself. That is a
    degenerate signal the model can produce something plausible from while the
    anchor conversion is wrong.
    """
    frames = _decode_video_frames(video_path)
    if not frames:
        raise ArtifactContentError(
            f"{video_path} decoded to no frames, so no image fixture can come from it."
        )
    index = len(frames) // 2
    from PIL import Image

    Path(dest).parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(frames[index]).save(str(dest))
    return index


def extract_audio_fixture(source, dest) -> Path:
    """Write a real ``.wav`` carrying the source's audio.

    Audio arrives two ways: audio-only modes write ``output.wav`` (pcm_s16le, the
    vocoder's own samples) and audio+video modes mux AAC into ``output.mp4``.
    Copying an mp4 to ``audio.wav`` would work -- ``av`` sniffs content, not
    extension -- and would be a small lie of exactly the kind that becomes a
    large one later, so the mp4 case is transcoded rather than renamed.
    """
    import av

    source, dest = Path(source), Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)

    with av.open(str(source)) as probe:
        if not probe.streams.audio:
            raise ArtifactContentError(
                f"{source} carries no audio stream, so there is nothing to extract."
            )
        stream = probe.streams.audio[0]
        already_wav = probe.format.name in ("wav", "wave") and stream.codec_context.name.startswith(
            "pcm"
        )
    if already_wav:
        shutil.copyfile(source, dest)
        return dest

    with av.open(str(source)) as container, av.open(str(dest), "w", format="wav") as out:
        stream = container.streams.audio[0]
        layout = getattr(stream, "layout", None)
        out_stream = out.add_stream(
            "pcm_s16le",
            rate=int(stream.rate),
            layout=getattr(layout, "name", "stereo") or "stereo",
        )
        for frame in container.decode(stream):
            # The muxer sets its own timestamps; keeping the source's would make
            # a wav whose duration disagrees with its sample count.
            frame.pts = None
            for packet in out_stream.encode(frame):
                out.mux(packet)
        for packet in out_stream.encode():
            out.mux(packet)
    return dest


# ---------------------------------------------------------------------------
# the call
# ---------------------------------------------------------------------------


def empty_shot_list() -> dict:
    """The Dataframe payload meaning "no shot list".

    Headers come from ``fields.SHOT_COLUMNS`` -- the same tuple the Dataframe is
    built from -- so a reordered or renamed column cannot leave this sending a
    table the component refuses to parse.
    """
    return {"headers": list(fields.SHOT_HEADERS), "data": [], "metadata": None}


def _handle_file(path):
    """Wrap a local path for upload.

    Imported lazily: this module is worth being able to import, read and unit
    test on a machine with no ``gradio_client`` at all, and a local stand-in for
    ``handle_file`` would be a payload shape that is wrong in a way nothing here
    could detect.
    """
    from gradio_client import handle_file

    return handle_file(str(path))


def predict_args(config: RunConfig, *, fixtures: FixtureStore | None = None) -> list:
    """The endpoint's positional arguments, in :data:`API_FIELD_KEYS` order.

    Built as a dict keyed by field name and then READ OUT through
    ``API_FIELD_KEYS``, never appended positionally. Appending is how a value
    lands one slot left of where it belongs, and the resulting run generates
    something plausible with no error anywhere -- see the ``fields.py`` docstring
    for the week that cost.
    """
    payload = {
        "prompt": config.prompt,
        "negative_prompt": config.negative_prompt,
        "image": None,
        "video": None,
        "audio": None,
        "conditions": empty_shot_list(),
        # The EFFECTIVE geometry, so the request and the render agree: the
        # registry pins min_resolution modes regardless of what we send, and a
        # log showing 704 next to a run that rendered 256 is worse than useless.
        "width": config.effective_width,
        "height": config.effective_height,
        "auto_duration": config.auto_duration,
        "num_frames": config.num_frames,
        "min_seconds": config.min_seconds,
        "max_seconds": config.max_seconds,
        # The REQUEST value, not the effective one: `generate.stage_2_enabled()`
        # applies the registry override server-side, and sending the already
        # overridden value would hide a registry that stopped applying it.
        "enable_stage_2": config.enable_stage_2,
        "seed": config.seed,
        "strength": config.strength,
        "noise_scale": config.noise_scale,
    }
    for kind in config.needs:
        if fixtures is None:
            raise MissingFixtureError(
                f"{config.mode_id} needs a real {kind} input and no fixture directory was given."
            )
        payload[kind] = _handle_file(fixtures.require(kind, mode_id=config.mode_id))

    missing_keys = [key for key in API_FIELD_KEYS if key not in payload]
    if missing_keys:
        # A field added to `fields.FIELDS` that nothing here fills would otherwise
        # KeyError several lines later with no hint at the cause.
        raise RuntimeError(
            f"the UI declares field(s) {missing_keys} that scripts/space_matrix.py does not "
            f"send. Add them to `predict_args`; the endpoint's argument order comes from "
            f"`fields.FIELDS` and cannot be patched around here."
        )
    return [config.mode_id, *(payload[key] for key in API_FIELD_KEYS)]


def _as_path(value):
    """A local path out of whatever a Gradio file output deserialised into.

    All three returns are declared ``filepath``, so a plain string is the normal
    case. The Mapping and sequence branches are there because a Video component
    has historically round-tripped as ``{"video": ..., "subtitles": ...}`` and a
    client that only handled the string case would read that as "no artifact" --
    which this tool would then raise on. A false SpaceRunError is as expensive as
    a missed one.
    """
    if value is None:
        return None
    if isinstance(value, str):
        return value or None
    if isinstance(value, Mapping):
        for key in ("video", "path", "name", "url"):
            found = _as_path(value.get(key))
            if found:
                return found
        return None
    if isinstance(value, (list, tuple)) and value:
        return _as_path(value[0])
    return None


def unpack_result(result) -> tuple:
    """``(video_path, audio_path, status)`` out of the endpoint's 3-tuple."""
    if isinstance(result, (list, tuple)) and len(result) >= 3:
        video, audio, status = result[0], result[1], result[2]
        return _as_path(video), _as_path(audio), "" if status is None else str(status)
    raise SpaceRunError(
        mode_id="<unknown>",
        missing=("video", "audio"),
        status=f"the endpoint returned {type(result).__name__} {result!r}, not the "
        f"(video, audio, status) triple `ui/blocks.py` wires as its outputs",
    )


def run_one(
    client,
    config: RunConfig,
    *,
    fixtures: FixtureStore | None = None,
    ledger: Ledger | None = None,
    sampling_rate=PINNED_VOCODER_SAMPLING_RATE,
    endpoint: str = ENDPOINT,
    log=print,
) -> RunRecord:
    """One generation: reserve, call, time, verify loudly, harvest fixtures.

    Raises rather than returning a failed record. A caller that wanted to survey
    "which modes work" would be tempted to collect failures and carry on, and
    that is precisely the shape that lets an empty result read as a data point.
    """
    spec = config.spec
    projected = ledger.check_before(config) if ledger is not None else 0.0
    log(f">>> {config.mode_id}  pipeline={spec.pipeline}  exports={spec.exports}")
    log(
        f"  geometry     {config.geometry}  seed={config.seed}  "
        f"stage2={'on' if config.effective_stage_2 else 'off'}"
    )
    if spec.min_resolution:
        log(f"  (registry FORCES {modes.MIN_RESOLUTION}px and stage 2 off for this mode)")
    log(f"  projected    <= {projected:.1f} GPU-s for one worker attempt (upper bound)")

    args = predict_args(config, fixtures=fixtures)

    started = time.time()
    try:
        result = client.predict(*args, api_name=endpoint)
    except Exception as exc:
        wall = time.time() - started
        # The wall clock still has to be charged: a call that raised on our side
        # may well have reserved and burned a GPU slot on theirs, and quietly
        # dropping it is how the ledger starts under-counting.
        if ledger is not None:
            ledger.charge(RunRecord(config=config, wall_seconds=wall, artifacts={}))
        raise SpaceRunError(
            mode_id=config.mode_id,
            missing=spec.exports,
            status=f"the client raised {type(exc).__name__}: {exc}",
            wall_seconds=wall,
        ) from exc
    wall_seconds = time.time() - started

    video_path, audio_path, status = unpack_result(result)
    record = RunRecord(
        config=config,
        wall_seconds=wall_seconds,
        artifacts={},
        status=status,
        projected_gpu_seconds=projected,
    )
    if ledger is not None:
        ledger.charge(record)
    log(f"  PREDICT      {wall_seconds:.2f}s wall clock")
    log(f"  COST         {record.render_cost()}")
    if ledger is not None:
        log(f"  {ledger.render()}")
        if record.gpu_seconds_upper_bound > projected * 1.05:
            log(
                f"  NOTE         observed bound exceeds the projection "
                f"({record.gpu_seconds_upper_bound:.1f} > {projected:.1f}) -- the platform "
                f"re-forks a killed worker, so a run can cost more than one attempt"
            )

    produced = {"video": video_path, "audio": audio_path}
    missing = [track for track in spec.exports if not produced.get(track)]
    if missing:
        raise SpaceRunError(
            mode_id=config.mode_id,
            missing=missing,
            status=status,
            wall_seconds=wall_seconds,
        )

    for track in spec.exports:
        path = Path(produced[track])
        size = path.stat().st_size if path.exists() else 0
        record.artifacts[track] = (str(path), size)
        log(f"  artifact     {track:<5} {path}  ({size} bytes)")
        if size == 0:
            raise ArtifactContentError(
                f"{config.mode_id}: the Space returned a path for {track} and the file at "
                f"{path} is ZERO BYTES. A path is not an artifact.\n\nThe Space's status "
                f"text:\n\n{_quote(status)}"
            )

    # Structure first, from the registry, on every call -- then content, because
    # a green stream table and a silent file are indistinguishable.
    #
    # Deduplicated by PATH: `outputs.write_output` returns ONE file and maps both
    # written tracks onto it, so an audio+video mode has video_path == audio_path
    # == output.mp4. Checking it twice would print the same verdict twice and
    # read as two independent confirmations of one fact.
    for container in dict.fromkeys(written for written, _size in record.artifacts.values()):
        verdict = verify_streams.verify(
            config.mode_id,
            container,
            frame_rate=None,
            sampling_rate=sampling_rate if "audio" in spec.exports else None,
        )
        record.stream_verdicts[container] = verdict
        log(f"  streams      {verdict.render().splitlines()[0]}")
        if not verdict.ok:
            raise ArtifactContentError(
                f"{config.mode_id}: {container} does not match what the registry says this mode "
                f"exports.\n  " + "\n  ".join(f"✗ {p}" for p in verdict.problems)
            )

    if "video" in spec.exports:
        record.video = verify_video_content(
            record.artifacts["video"][0],
            expected_frames=None if config.auto_duration else config.num_frames,
        )
        log(f"  video        {record.video.render()}")
    if "audio" in spec.exports:
        record.audio = verify_audio_content(
            record.artifacts["audio"][0],
            expected_sample_rate=sampling_rate,
        )
        log(f"  audio        {record.audio.render()}")

    if fixtures is not None:
        record.fixtures_saved = fixtures.adopt(config, record.artifacts, log=log)
    return record


def run_matrix(
    client,
    configs,
    *,
    fixtures: FixtureStore | None = None,
    ledger: Ledger | None = None,
    sampling_rate=PINNED_VOCODER_SAMPLING_RATE,
    endpoint: str = ENDPOINT,
    log=print,
) -> MatrixOutcome:
    """Run the plan in order, stopping cleanly at the ceiling.

    A failed run RAISES out of here. Only the quota ceiling stops it gracefully,
    because only the quota ceiling is a decision rather than a defect.
    """
    ledger = Ledger() if ledger is None else ledger
    records: list[RunRecord] = []
    for index, config in enumerate(configs, start=1):
        log(f"\n--- {index}/{len(configs)} ---")
        try:
            record = run_one(
                client,
                config,
                fixtures=fixtures,
                ledger=ledger,
                sampling_rate=sampling_rate,
                endpoint=endpoint,
                log=log,
            )
        except QuotaCeilingReached as stop:
            log(f"\n*** STOPPED AT THE QUOTA CEILING before {config.mode_id} ***")
            log(f"  {stop}")
            log("  Nothing was spent on this run. The remaining plan was not started.")
            return MatrixOutcome(
                records=records, ledger=ledger, stopped_at=config, stop_reason=str(stop)
            )
        records.append(record)
    return MatrixOutcome(records=records, ledger=ledger)


# ---------------------------------------------------------------------------
# reporting
# ---------------------------------------------------------------------------


def render_plan(configs, ledger: Ledger, *, live: bool) -> str:
    """The ordered plan with per-run and cumulative bounds. Spends nothing."""
    banner = (
        "=== PLAN — LIVE, this WILL spend GPU-seconds ==="
        if live
        else "=== PLAN — DRY RUN, ZERO GPU-seconds will be spent ==="
    )
    lines = [
        banner,
        f"  {'#':>2}  {'mode':<26} {'geometry':<26} {'stage2':<7} {'needs':<12} "
        f"{'<= GPU-s':>9} {'<= cumul':>9}",
    ]
    running = ledger.spent_upper_bound
    crossed = None
    for index, config in enumerate(configs, start=1):
        projected = ledger.projected_for(config)
        running += projected
        if crossed is None and running > ledger.ceiling_seconds:
            crossed = index
        lines.append(
            f"  {index:>2}  {config.mode_id:<26} {config.geometry:<26} "
            f"{'on' if config.effective_stage_2 else 'off':<7} "
            f"{','.join(config.needs) or '-':<12} {projected:>9.1f} {running:>9.1f}"
        )
    lines.append("")
    # "carried" and "this plan" stay SEPARATE numbers on purpose. Folding them
    # into one "plan total" is how a figure that includes four earlier
    # invocations gets quoted as the cost of this one.
    if ledger.spent_upper_bound:
        lines.append(
            f"  carried      <= {ledger.spent_upper_bound:.1f} GPU-s already bounded earlier "
            f"today (UTC {ledger.store.day if ledger.store else '?'})"
        )
    lines.append(
        f"  ceiling      {ledger.ceiling_seconds:.1f} GPU-s "
        f"({100 * ledger.ceiling_seconds / ledger.daily_quota_seconds:.0f}% of the "
        f"{ledger.daily_quota_seconds:.0f}/day PRO budget)"
    )
    lines.append(f"  this plan    <= {running - ledger.spent_upper_bound:.1f} GPU-s")
    lines.append(
        f"  projected    <= {running:.1f} GPU-s for UTC today once this plan finishes  "
        f"*** UPPER BOUND from zerogpu.estimate_duration x {BILLING_SIZE_MULTIPLIER:g}, "
        f"NOT measured billing ***"
    )
    if crossed is None:
        lines.append(
            f"  verdict      FITS, with {ledger.ceiling_seconds - running:.1f} GPU-s spare"
        )
    else:
        lines.append(
            f"  verdict      DOES NOT FIT — run #{crossed} ({configs[crossed - 1].mode_id}) is "
            f"where the ceiling is projected to bind. It will be refused, not started."
        )
    return "\n".join(lines)


def render_summary(outcome: MatrixOutcome) -> str:
    lines = ["", "=== SUMMARY ==="]
    for record in outcome.records:
        artifacts = " ".join(
            f"{track}={size}B" for track, (_path, size) in sorted(record.artifacts.items())
        )
        lines.append(
            f"  {record.mode_id:<26} {record.wall_seconds:>7.2f}s wall  "
            f"<= {record.gpu_seconds_upper_bound:>7.1f} GPU-s  {artifacts}"
        )
        if record.video:
            lines.append(f"  {'':<26} video: {record.video.render()}")
        if record.audio:
            lines.append(f"  {'':<26} audio: {record.audio.render()}")
    if outcome.stopped:
        lines.append(f"  STOPPED before {outcome.stopped_at.mode_id}: {outcome.stop_reason}")
    lines.append("")
    if outcome.ledger.carried_upper_bound:
        lines.append(
            f"  this invocation <= {outcome.ledger.spent_this_invocation:.1f} GPU-s upper bound"
        )
    lines.append(f"  {outcome.ledger.render()}")
    if outcome.ledger.store is not None:
        lines.append(f"  persisted to {outcome.ledger.store.path} (UTC {outcome.ledger.store.day})")
    lines.append(
        "  There is no readable ZeroGPU quota endpoint, so the figure above is a BOUND "
        "derived from wall clock. Do not quote it as billed seconds."
    )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def connect(space_id: str, *, token=None, download_dir=None, log=print):
    """A ``gradio_client.Client`` for the Space. The token is never written down.

    It is read from the environment by the caller and handed straight to the
    client -- it does not reach a file, a log line or an argv entry, and this
    function does not echo it either.
    """
    from gradio_client import Client

    log(f"connecting to {space_id} …")
    started = time.time()
    kwargs = {"token": token} if token else {}
    if download_dir is not None:
        kwargs["download_files"] = str(download_dir)
    client = Client(space_id, **kwargs)
    log(f"connected in {time.time() - started:.2f}s (costs no GPU quota)")
    return client


def _parse_args(argv):
    parser = argparse.ArgumentParser(
        prog="space_matrix.py",
        description=(
            "Exercise the live LTX-2.5 Studio Space mode by mode and bound what each run "
            "costs. Dry run by default; --live is what spends."
        ),
    )
    parser.add_argument(
        "--modes",
        default=None,
        help=(
            "comma-separated mode ids, run in the given order. Default is the documented "
            f"dependency order: {','.join(DEFAULT_MODE_ORDER)}"
        ),
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="actually call the Space. Without this nothing is sent and nothing is spent.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="force the dry run even alongside --live (the safe direction wins).",
    )
    parser.add_argument("--space", default=DEFAULT_SPACE_ID)
    parser.add_argument(
        "--quota-ceiling",
        type=float,
        default=DEFAULT_QUOTA_CEILING_SECONDS,
        help=f"cumulative upper-bound GPU-seconds to stop at (default {DEFAULT_QUOTA_CEILING_SECONDS:.0f}).",
    )
    parser.add_argument(
        "--reservation-seconds",
        type=int,
        default=None,
        help=(
            "the Space's own LTX25_GPU_DURATION_SECONDS. The local estimator cannot read it, "
            "and it is what the platform reserves against."
        ),
    )
    parser.add_argument(
        "--quota-ledger",
        default=None,
        help=(
            "cross-invocation spend ledger, keyed to the UTC day "
            f"(default {DEFAULT_QUOTA_LEDGER}). Deliberately NOT derived from --fixture-dir: "
            "a ledger a routine flag can move is a ceiling that only sees one invocation."
        ),
    )
    parser.add_argument(
        "--reset-quota-ledger",
        action="store_true",
        help=(
            "start today's count from zero on purpose. The only way past an unreadable "
            "ledger, which is otherwise refused rather than silently treated as zero."
        ),
    )
    parser.add_argument("--fixture-dir", default=None)
    parser.add_argument("--download-dir", default=None)
    parser.add_argument(
        "--sampling-rate",
        type=int,
        default=PINNED_VOCODER_SAMPLING_RATE,
        help=f"expected audio sample rate (default {PINNED_VOCODER_SAMPLING_RATE}, the pinned vocoder's).",
    )
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--prompt", default=PROMPT)
    parser.add_argument("--width", type=int, default=CHEAPEST_SIZE)
    parser.add_argument("--height", type=int, default=CHEAPEST_SIZE)
    parser.add_argument("--num-frames", type=int, default=CHEAPEST_FRAMES)
    parser.add_argument("--enable-stage-2", action="store_true")
    return parser.parse_args(argv)


def main(argv=None, *, log=print) -> int:
    """Exit codes: 0 = plan complete · 2 = the plan is illegal · 3 = stopped at the ceiling.

    A run that FAILS does not get an exit code -- it raises, and the traceback is
    the loudest thing available. Converting it to a return value here would put
    it one level away from being ignored.
    """
    import os

    args = _parse_args(argv)
    mode_ids = None
    if args.modes:
        mode_ids = tuple(m.strip() for m in args.modes.split(",") if m.strip())

    try:
        configs = build_plan(
            mode_ids,
            seed=args.seed,
            prompt=args.prompt,
            width=args.width,
            height=args.height,
            num_frames=args.num_frames,
            enable_stage_2=args.enable_stage_2,
        )
        for config in configs:
            validate_config(config)
    except Exception as exc:  # noqa: BLE001 -- a bad plan is a usage error, not a crash
        log(f"the plan is not runnable: {type(exc).__name__}: {exc}")
        return 2

    # Read BEFORE the plan is rendered, so the plan is judged against today's
    # real standing rather than against an empty counter. A dry run reads it and
    # never writes it -- that is what makes --dry-run free in the ledger as well
    # as on the GPU.
    ledger_path = args.quota_ledger or (REPO / DEFAULT_QUOTA_LEDGER)
    try:
        store = PersistentQuota(ledger_path, reset=args.reset_quota_ledger)
    except QuotaLedgerUnreadable as exc:
        log(f"{exc}")
        return 2

    ledger = Ledger.from_store(
        store,
        ceiling_seconds=args.quota_ceiling,
        reservation_seconds=args.reservation_seconds,
    )
    live = args.live and not args.dry_run
    log(store.render())
    log("")
    log(render_plan(configs, ledger, live=live))

    if not live:
        log("")
        log(
            "DRY RUN: every config above was validated offline with "
            "`constraints.validate_dimensions` / `validate_num_frames`."
        )
        log(
            "ZERO GPU-seconds were spent and the quota ledger was READ, not written. "
            "Re-run with --live to execute this plan."
        )
        return 0

    # Defaults INSIDE `outputs/`, which `.gitignore` already excludes. A fixture
    # directory anywhere else leaves generated media as untracked files in the
    # working tree, and the first person to `git add -A` commits a video.
    fixture_dir = args.fixture_dir or (REPO / "outputs" / "space-fixtures")
    fixtures = FixtureStore(fixture_dir)
    log(f"\nfixture directory: {fixtures.directory}")

    # Counted against the day BEFORE anything is spent. An invocation that dies
    # on its first run must still show up in the ledger, or "how many times did
    # we hit the Space today" is answerable only for the runs that succeeded.
    store.begin_invocation()

    client = connect(
        args.space,
        token=os.environ.get("HF_TOKEN"),
        download_dir=args.download_dir,
        log=log,
    )
    try:
        outcome = run_matrix(
            client,
            configs,
            fixtures=fixtures,
            ledger=ledger,
            sampling_rate=args.sampling_rate,
            log=log,
        )
    except Exception:
        # The ledger is the one thing worth printing on the way out: a run that
        # blew up still spent quota, and losing that number means the next
        # session starts blind to how much of the day's budget is gone.
        log(render_summary(MatrixOutcome(records=ledger.records, ledger=ledger)))
        raise

    log(render_summary(outcome))
    return 3 if outcome.stopped else 0


if __name__ == "__main__":
    raise SystemExit(main())
