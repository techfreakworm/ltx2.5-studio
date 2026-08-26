"""Typed exception hierarchy. Every error names the thing that went wrong.

PLAN §5.2, §5.3, §4.3, §7.2.

"Fail loudly" is a contract here, not a slogan. The predecessor's week-long
outage happened because four separate layers each turned a failure into a
falsy value: a missing file became ``logger.warning + continue``, a node
exception became a FAILURE tuple, an empty path was pushed unconditionally,
and the UI rendered the empty path as a silent form reset. So:

* every error descends from one root, :class:`LTXStudioError`, which the UI can
  catch as a family;
* four grouping bases -- :class:`ConstraintError`, :class:`ModelFileError`,
  :class:`LoRAError`, :class:`GenerationError` -- let callers catch by area;
* every concrete error takes **keyword arguments naming the offending thing**
  and renders a sentence containing them. An error that says "generation
  failed" is not an error, it is a shrug.

Concrete errors also keep their inputs as attributes, so the UI can render a
structured message without re-parsing prose.

The root inherits ``gradio.exceptions.Error`` and defines ``__reduce__``, as a
pair, so these errors survive the ZeroGPU fork boundary with their messages
intact -- docs/SPACE_MODE_MATRIX.md §4 measured what happens otherwise: every
``reason=`` this module so carefully attaches was reduced to the bare class
name in transit, and "fail loudly" quietly degraded to "fail by name only"
inside every ``@spaces.GPU`` call. See :class:`LTXStudioError` for why both
halves are needed and why neither works alone.

Imports nothing else from this package (PLAN §7.2).
"""

from __future__ import annotations

# The ONE gradio import allowed outside ui/ (test_architecture.py pins it to
# exactly this name). It is unconditional and at module level on purpose:
# gradio is a pinned runtime dependency (requirements.txt, ==6.24.0), and the
# fork-boundary property below has to hold wherever an error is *constructed*
# -- a fallback base for gradio-less environments would make that property
# silently vary by environment, which is the inert-guard shape this build
# keeps finding (SPACE_MODE_MATRIX §8). If gradio is absent, this package
# should fail to import, loudly, right here.
from gradio.exceptions import Error as _GradioError

__all__ = [
    "AudioVideoDesyncError",
    "ConfigError",
    "ConstraintError",
    "DecodeError",
    "DimensionError",
    "EmptyOutputError",
    "EncodeError",
    "FrameCountError",
    "GenerationError",
    "LTXStudioError",
    "LoRAConventionError",
    "LoRAError",
    "LoRADimensionError",
    "LoRAFetchError",
    "LoRAFormatError",
    "LoRAHeaderError",
    "LoRAInContextError",
    "LoRAModuleNotFoundError",
    "LoRANotAttachedError",
    "LoRAPairingError",
    "LoRATooLargeError",
    "LoRATruncatedError",
    "LoRAVersionError",
    "MissingModelFileError",
    "ModelFileError",
    "ModelLoadError",
    "PromptEnhancementError",
    "UnknownComponentError",
    "UnknownModeError",
]


def _join(values) -> str:
    """Render a sequence as a readable comma-separated list."""
    return ", ".join(str(v) for v in values)


def _gb(num_bytes: int) -> str:
    """Bytes as decimal GB, matching how the Hub reports file sizes."""
    return f"{num_bytes / 1e9:.1f} GB"


def _rebuild(cls, args, state):
    """Reconstruct an error for pickle without calling its ``__init__``.

    Every concrete class in this module takes **keyword-only** arguments while
    ``self.args`` holds the one formatted sentence, so pickle's default path --
    ``cls(*self.args)`` -- feeds the sentence back positionally and the
    constructor refuses it. That is not hypothetical: 22 of the 27 classes
    here were measured unpicklable through exactly that mismatch
    (SPACE_MODE_MATRIX §4). Rebuilding via ``__new__`` restores what an
    instance actually *is* -- its class, its ``args``, its attributes --
    without re-running construction logic that already ran once.
    """
    exc = cls.__new__(cls)
    exc.args = args
    exc.__dict__.update(state)
    return exc


# ---------------------------------------------------------------------------
# roots
# ---------------------------------------------------------------------------


class LTXStudioError(_GradioError):
    """Root of every error this package raises.

    The UI catches this to decide what to show a human; nothing else in the
    package is allowed to signal failure by returning a falsy value instead
    (PLAN §5.3).

    **Why the base is ``gradio.exceptions.Error``** (SPACE_MODE_MATRIX §4): on
    ZeroGPU every ``@spaces.GPU`` call runs in a forked worker, and ``spaces``
    transmits the raised exception *object* back to the parent only when
    ``isinstance(exc, gr.Error)`` -- anything else is reduced to its class
    name, which is how ``DecodeError(reason=...)`` reached a live user as the
    eleven characters ``'DecodeError'``. The inheritance must be real and must
    sit here at the root: the worker *constructs* the exception, so the
    property has to hold at creation time, in library code -- it cannot be
    grafted on from ``ui/`` afterwards.

    Inheriting alone would be **worse than nothing**: an eligible-but-
    unpicklable error dies crossing the result queue, the spaces worker
    catches the ``PicklingError`` and ships *that*, and the user sees
    ``'PicklingError'`` -- losing even the class name that survives today.
    ``__reduce__`` below is the other half of the pair; neither works alone.

    Eligibility to travel must not change what these errors *say*, so two
    inherited behaviours are deliberately reversed:

    * ``__str__`` -- gr.Error renders ``repr(self.message)``; the single
      quotes around ``'DecodeError'`` on the Space were exactly that.
      ``ui/render.py`` quotes ``str(exc)`` verbatim into markdown, so we
      restore ``BaseException.__str__``.
    * ``__init__`` -- gr.Error would absorb a second positional argument into
      ``duration`` and turn zero arguments into its ``"Error raised."``
      default. We keep ``BaseException`` arity semantics exactly.

    ``print_exception`` is left at gr.Error's default (``True``) on purpose:
    the spaces parent prints the worker's full traceback to the Space log only
    when the transmitted error's ``print_exception`` is truthy, and that log
    line is how the one real generation defect this Space has had was actually
    diagnosed. Suppressing it would re-hide the channel §4 exists to keep open.

    Inheriting gr.Error also makes every error here a ``ValueError`` (via
    gradio's ``AppError``). Audited: no ``except ValueError`` in this package
    sits on a path our errors travel -- the five catch sites guard stdlib
    parsing and ``inspect.signature`` only, and ``lora/apply.py`` re-raises
    ``LoRAError`` before its ``ValueError`` arm.
    """

    def __init__(self, *args):
        # Concrete subclasses pass exactly one formatted sentence; tests may
        # construct bases bare. Hand gr.Error the message (it stores it and
        # its modal defaults -- title, duration, visible, print_exception),
        # then restore ``args`` so str/repr semantics match plain Exception
        # for every arity, including () and (a, b).
        super().__init__(str(args[0]) if args else "")
        self.args = args

    def __str__(self):
        # Not gr.Error's repr(self.message) -- see the class docstring.
        return BaseException.__str__(self)

    def __reduce__(self):
        # ``args`` is a C-level slot, absent from __dict__, so it is carried
        # explicitly; the dict carries everything else -- mode, reason,
        # message, print_exception, all of it. ``_rebuild`` skips __init__.
        return (_rebuild, (self.__class__, self.args, dict(vars(self))))


class ConstraintError(LTXStudioError):
    """An input violates a model constraint (PLAN §5.5)."""


class ModelFileError(LTXStudioError):
    """A model file or component could not be resolved (PLAN §5.2)."""


class LoRAError(LTXStudioError):
    """A user-supplied LoRA failed validation or failed to attach (PLAN §4.3)."""


class GenerationError(LTXStudioError):
    """Generation started and did not produce a usable artifact (PLAN §5.3)."""


# ---------------------------------------------------------------------------
# configuration -- PLAN §5.1
# ---------------------------------------------------------------------------


class ConfigError(LTXStudioError):
    """An environment variable holds a value we do not recognise.

    Deliberately outside :class:`GenerationError` and :class:`ConstraintError`:
    ``ui/`` has to tell an operator who set a Space variable wrong from a user
    whose run failed, and those are different messages to different people.

    Raised rather than defaulted, because a silent fallback is how a Space runs
    for a week in a configuration nobody chose.
    """

    def __init__(self, *, variable, value, legal):
        self.variable = variable
        self.value = value
        self.legal = tuple(legal)
        super().__init__(
            f"{variable}={value!r} is not a recognised value. Legal values are: "
            f"{_join(self.legal)}. Refusing to fall back to a default silently -- set the "
            f"variable correctly or unset it."
        )


# ---------------------------------------------------------------------------
# constraints -- PLAN §5.5
# ---------------------------------------------------------------------------


class FrameCountError(ConstraintError):
    """``num_frames`` is not ``k * temporal_compression_ratio + 1``.

    Upstream does not raise for this: it computes
    ``(num_frames - 1) // temporal_compression_ratio + 1`` and **silently
    floors** a non-conforming value. We raise, and we name the two legal
    counts on either side so the user can pick one.
    """

    def __init__(self, *, num_frames, temporal_compression_ratio, nearest):
        self.num_frames = num_frames
        self.temporal_compression_ratio = temporal_compression_ratio
        self.nearest = tuple(nearest)
        if len(set(self.nearest)) == 1:
            advice = f"Use {self.nearest[0]} instead."
        else:
            advice = f"Use {self.nearest[0]} or {self.nearest[-1]} instead."
        super().__init__(
            f"num_frames={num_frames} is not valid: it must be a multiple of the VAE temporal "
            f"compression ratio ({temporal_compression_ratio}) plus 1. Upstream would silently "
            f"round this down rather than fail. {advice}"
        )


class DimensionError(ConstraintError):
    """A spatial dimension is not divisible by the required multiple."""

    def __init__(self, *, name, value, multiple):
        self.name = name
        self.value = value
        self.multiple = multiple
        lower = (value // multiple) * multiple
        upper = lower + multiple
        super().__init__(
            f"{name}={value} is not divisible by {multiple}. LTX-2.5 requires every spatial "
            f"dimension to be a multiple of {multiple}. Use {lower} or {upper} instead."
        )


class AudioVideoDesyncError(ConstraintError):
    """The audio and video tracks would not be the same length.

    Reachable whenever pre-encoded ``audio_latents`` are supplied: upstream
    derives the audio length from the latent shape and **overrides** the
    duration-derived value, while ``num_frames`` still governs the video.
    """

    def __init__(self, *, video_seconds, audio_seconds, largest_fitting_num_frames=None):
        self.video_seconds = video_seconds
        self.audio_seconds = audio_seconds
        self.largest_fitting_num_frames = largest_fitting_num_frames
        message = (
            f"video and audio lengths disagree: video is {video_seconds} s but audio is "
            f"{audio_seconds} s. Supplying audio_latents overrides the audio length while "
            f"num_frames still governs the video; adjust one so they match."
        )
        if largest_fitting_num_frames is not None:
            # Optional because only some raise sites can compute it: the
            # file-upload path knows the frame rate and can name the exact
            # change that would fit, and an error that names the fix beats one
            # that only names the mismatch.
            message += (
                f" The longest video this audio covers is num_frames={largest_fitting_num_frames}."
            )
        super().__init__(message)


# ---------------------------------------------------------------------------
# model files -- PLAN §5.2
# ---------------------------------------------------------------------------


class MissingModelFileError(ModelFileError):
    """A required weight file is absent from the Hub or the local cache.

    Names the URL because this has happened for real: Lightricks deleted a 2.3
    checkpoint with no warning and broke a live Space. The operator needs to
    know which page to open.
    """

    def __init__(self, *, filename, url):
        self.filename = filename
        self.url = url
        super().__init__(
            f"required model file {filename!r} could not be resolved. Check whether it still "
            f"exists upstream: {url}"
        )


class UnknownComponentError(ModelFileError):
    """A component was requested that the model-file table does not declare."""

    def __init__(self, *, name, known):
        self.name = name
        self.known = tuple(known)
        super().__init__(
            f"unknown model component {name!r}. Declared components are: {_join(self.known)}."
        )


# ---------------------------------------------------------------------------
# LoRA -- PLAN §4.3
# ---------------------------------------------------------------------------


class LoRAFetchError(LoRAError):
    """Stage 0 -- a Hub reference could not be resolved to a downloadable file.

    Everything that can go wrong BEFORE any bytes exist locally: a repo that
    does not exist (or is private, which the Hub deliberately reports the same
    way), a gated repo, a repo with no ``.safetensors`` file, an ambiguous
    repo where no convention picks one file, or the network dying mid-fetch.
    Distinct from :class:`LoRAFormatError` on purpose: "could not be read.
    Only safetensors adapters are accepted" is actionable advice about a
    *file*, and every one of these failures happens while there is no file.
    """

    def __init__(self, *, source, reason):
        self.source = source
        self.reason = reason
        super().__init__(f"LoRA {source} could not be fetched from the Hub: {reason}")


class LoRATooLargeError(LoRAError):
    """Stage 1 -- the upload exceeds the configured size bound."""

    def __init__(self, *, source, size_bytes, max_bytes):
        self.source = source
        self.size_bytes = size_bytes
        self.max_bytes = max_bytes
        super().__init__(
            f"LoRA {source} is {size_bytes} bytes ({_gb(size_bytes)}), which exceeds the limit of "
            f"{max_bytes} bytes ({_gb(max_bytes)}). Refusing before download."
        )


class LoRAFormatError(LoRAError):
    """Stage 3 -- the file is not a safetensors container at all."""

    def __init__(self, *, source, reason):
        self.source = source
        self.reason = reason
        super().__init__(
            f"LoRA {source} could not be read: {reason}. Only safetensors adapters are accepted."
        )


class LoRAHeaderError(LoRAError):
    """Stage 3 -- the safetensors header is unparseable or implausible.

    A wildly large declared header length usually means the response was an
    HTML error page (a gated repo, say) rather than a safetensors file.
    """

    def __init__(self, *, source, declared_header_bytes):
        self.source = source
        self.declared_header_bytes = declared_header_bytes
        super().__init__(
            f"LoRA {source} has an unreadable safetensors header: it declares a header length of "
            f"{declared_header_bytes} bytes. This usually means the file is not safetensors at "
            f"all -- an HTML error page from a gated repository, for example."
        )


class LoRATruncatedError(LoRAError):
    """Stage 3 -- the file is shorter than its own header claims."""

    def __init__(self, *, source, declared_bytes, available_bytes):
        self.source = source
        self.declared_bytes = declared_bytes
        self.available_bytes = available_bytes
        super().__init__(
            f"LoRA {source} is truncated: its header declares {declared_bytes} bytes but only "
            f"{available_bytes} bytes are available. Re-download the adapter."
        )


class LoRAVersionError(LoRAError):
    """Stage 4 -- an explicit version signal contradicts this model.

    Only raised on an *explicit* mismatch. Absent version metadata warns and
    proceeds: measured against real public LTX LoRAs, most carry no version
    signal at all, and rejecting on absence would reject the most-downloaded
    LTX adapter on the Hub.
    """

    def __init__(self, *, source, found, expected):
        self.source = source
        self.found = found
        self.expected = expected
        super().__init__(
            f"LoRA {source} declares version {found!r} but this pipeline is {expected}. "
            f"Loading it would attach to the wrong architecture."
        )


class LoRAConventionError(LoRAError):
    """Stage 5 -- the key naming convention is not one we recognise.

    The message names the prefixes actually found, because "unrecognised
    format" tells the user nothing they can act on.
    """

    def __init__(self, *, source, found_prefixes, known_prefixes):
        self.source = source
        self.found_prefixes = tuple(found_prefixes)
        self.known_prefixes = tuple(known_prefixes)
        super().__init__(
            f"LoRA {source} uses an unrecognised key convention. Found prefixes: "
            f"{_join(self.found_prefixes)}. Expected one of: {_join(self.known_prefixes)}."
        )


class LoRAPairingError(LoRAError):
    """Stage 6 -- some low-rank factors have no counterpart."""

    def __init__(self, *, source, unpaired_keys):
        self.source = source
        self.unpaired_keys = tuple(unpaired_keys)
        super().__init__(
            f"LoRA {source} has unpaired low-rank factors: every lora_A/lora_down needs a "
            f"matching lora_B/lora_up. Unpaired keys: {_join(self.unpaired_keys)}."
        )


class LoRAInContextError(LoRAError):
    """Stage 5.5 -- an in-context (IC) LoRA, which the plain pipeline cannot use.

    Marked by ``reference_downscale_factor`` in ``__metadata__``, which maps to
    a kwarg on the in-context pipeline that must match the factor the adapter
    was trained with. Applied to the plain pipeline an IC-LoRA **neither errors
    nor works** -- the silent-garbage shape again, arriving through a door stage
    7 does not watch.
    """

    def __init__(self, *, source, marker, value):
        self.source = source
        self.marker = marker
        self.value = value
        super().__init__(
            f"LoRA {source} is an in-context (IC) LoRA: its metadata declares "
            f"{marker}={value!r}. IC-LoRAs condition on a reference image and only work through "
            f"the in-context pipeline, which must be given the same {marker}. Applied to the "
            f"standard pipeline this adapter would load without error and have no effect."
        )


class LoRADimensionError(LoRAError):
    """Stage 6.5 -- a low-rank factor does not match its target module's shape.

    The gate that catches an adapter trained for a different LTX model. Note the
    comparison is **per module**: this transformer carries a video stream at 4096
    and an audio stream at 2048, each with a 4x feed-forward expansion, so
    "contains 2048" is not a signal -- "2048 at a module that expects 4096" is.
    """

    def __init__(self, *, source, module, expected, actual, key=""):
        self.source = source
        self.module = module
        self.expected = expected
        self.actual = actual
        self.key = key
        super().__init__(
            f"LoRA {source} does not fit this model: {actual} at {module}, expected {expected}"
            f"{f' (key {key})' if key else ''}. The adapter was trained for a different LTX "
            f"model; applying it would corrupt that module."
        )


class LoRAModuleNotFoundError(LoRAError):
    """Stage 6.5 -- the adapter targets modules this model does not have.

    Distinct from :class:`LoRADimensionError` on purpose: a wrong *dimension*
    and a wrong *module* are different diagnoses, and the user needs to know
    which. Missing modules are what a genuinely foreign adapter looks like.
    """

    def __init__(self, *, source, missing_modules, module_count):
        self.source = source
        self.missing_modules = tuple(missing_modules)
        self.module_count = module_count
        super().__init__(
            f"LoRA {source} targets modules this model does not have: "
            f"{_join(self.missing_modules[:5])}"
            f"{' and others' if len(self.missing_modules) > 5 else ''}. "
            f"{len(self.missing_modules)} of {module_count} targeted modules are absent, so the "
            f"adapter is built for a different architecture."
        )


class LoRANotAttachedError(LoRAError):
    """Stage 7 -- the adapter loaded but attached to zero modules.

    The anti-silent-garbage check. Upstream's only validation is a ``'lora'``
    substring test, which a LoRA for a completely different architecture
    passes -- it then attaches to nothing, generation proceeds, and the output
    silently looks like the base model. This is the error that stops that.
    """

    def __init__(self, *, source, adapter_name, components):
        self.source = source
        self.adapter_name = adapter_name
        self.components = tuple(components)
        super().__init__(
            f"LoRA {source} loaded as adapter {adapter_name!r} but attached to zero modules of "
            f"{_join(self.components)}. The adapter is almost certainly built for a different "
            f"architecture; applying it would silently produce base-model output."
        )


# ---------------------------------------------------------------------------
# generation -- PLAN §5.3
# ---------------------------------------------------------------------------


class ModelLoadError(GenerationError):
    """A pipeline component failed to load."""

    def __init__(self, *, component, reason):
        self.component = component
        self.reason = reason
        super().__init__(f"failed to load pipeline component {component!r}: {reason}.")


class PromptEnhancementError(GenerationError):
    """The prompt enhancer could not be fetched, loaded or run.

    A :class:`GenerationError` rather than a :class:`ModelFileError` because of
    what it means for the run: enhancement happens on the way into a generation
    the user asked for, and when it fails **nothing is generated**. The
    alternative -- carrying on with the prompt the user typed while the UI says
    enhancement is on -- is precisely the silent substitution PLAN §5.3 exists
    to forbid, so this raises instead.

    Names the repository, because the enhancer is the one model this app loads
    that is NOT part of the pinned pack: it is resolved at runtime from
    ``LTX25_PROMPT_ENHANCER_REPO`` (default
    ``google/gemma-4-E2B-it-qat-mobile-transformers``), so "which model?" is a
    real question with a configurable answer, and an error that omits it sends
    the operator to the wrong Hub page.
    """

    def __init__(self, *, repo_id, reason):
        self.repo_id = repo_id
        self.reason = reason
        super().__init__(
            f"prompt enhancement failed and nothing was generated. The enhancer {repo_id!r} "
            f"could not be used: {reason}. Turn the enhance-prompt toggle off to run with the "
            f"prompt you typed."
        )


class DecodeError(GenerationError):
    """Denoising or decoding produced nothing usable."""

    def __init__(self, *, mode, reason):
        self.mode = mode
        self.reason = reason
        super().__init__(f"decoding failed for mode {mode!r}: {reason}.")


class EncodeError(GenerationError):
    """Muxing or writing the output file failed."""

    def __init__(self, *, path, reason):
        self.path = path
        self.reason = reason
        super().__init__(f"failed to write output {path}: {reason}.")


class EmptyOutputError(GenerationError):
    """Generation reported success but produced no file.

    The invariant that the predecessor lacked. A run that finishes without an
    artifact raises here, at the source, rather than returning an empty path
    for some later layer to render as a silent no-op.
    """

    def __init__(self, *, mode, expected_path):
        self.mode = mode
        self.expected_path = expected_path
        super().__init__(
            f"generation for mode {mode!r} finished without producing {expected_path}. "
            f"Reporting this rather than returning an empty result."
        )


# ---------------------------------------------------------------------------
# modes -- PLAN §3.3
# ---------------------------------------------------------------------------


class UnknownModeError(LTXStudioError):
    """A mode was requested that the registry does not define."""

    def __init__(self, *, mode, known):
        self.mode = mode
        self.known = tuple(known)
        super().__init__(f"unknown mode {mode!r}. Registered modes are: {_join(self.known)}.")
