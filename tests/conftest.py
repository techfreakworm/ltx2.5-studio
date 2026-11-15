"""Shared test infrastructure for the LTX-2.5 Studio spec suite.

Every fixture here obeys the hard constraint from PLAN §9.1: **no GPU, no network,
no model weights**. Nothing in this file imports torch, diffusers, safetensors or
gradio.

Two deliberate choices worth knowing about:

1. **safetensors files are synthesised byte-for-byte in pure Python.**
   The format is trivial and documented: an 8-byte little-endian ``u64`` header
   length, that many bytes of UTF-8 JSON, then the tensor payload. Writing it
   ourselves (rather than via ``safetensors.torch.save_file``) means the LoRA
   suite runs on a bare venv, and — more importantly — it lets us craft the
   *malformed* files that PLAN §4.3 stage 3 requires: a truncated header, a
   corrupt header length, a payload that is never read. A torch-based writer
   cannot produce those. ``test_lora_inspect.py`` cross-checks the synthesised
   bytes against the real ``safetensors`` library when it happens to be
   installed, so the fixtures cannot quietly drift from the real format.

2. **Fakes mirror the *failure modes* of the real objects, not just their happy
   path.** ``fake_pipeline`` sets ``peft_config`` on a zero-attach load, exactly
   as peft does, because an implementation that checks ``peft_config`` instead of
   counting attached modules is the silent-garbage bug PLAN §4.3 stage 7 exists
   to catch. The fake must be able to express the bug or the test is theatre.
"""

from __future__ import annotations

import builtins
import io
import json
import struct
from pathlib import Path
from types import SimpleNamespace

import pytest

# --------------------------------------------------------------------------
# process-lifetime caches — reset between specs
# --------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _drop_resident_enhancer():
    """Empty ``enhance._ENHANCER`` around every test in the suite.

    The enhancer is deliberately **resident for the life of the worker process**
    (``enhance.py``, and the +30.78 s per-call reload that bought). A pytest
    session is one process, so without this the first spec to call ``enhance()``
    would serve its fake model to every spec that followed — cross-test
    contamination that reads as a mysterious unrelated failure.

    Autouse and global rather than local to ``test_enhance.py``: the cache is
    reachable from anything that calls ``generate()`` without injecting an
    ``enhancer=`` seam, so scoping the reset to the file that happens to use it
    today is a trap for whoever adds the next caller.

    Imported inside the fixture so ``conftest``'s own import surface stays as
    light as its docstring claims.
    """
    from ltx25_studio import enhance

    _drop(enhance)
    yield
    _drop(enhance)


def _drop(enhance):
    """``clear_enhancer_cache()``, but it can never be the thing that fails a test.

    The real free imports torch when the cached model reports a CUDA device, and
    a spec that installed a *fake* cuda-reporting model has usually had its fake
    ``sys.modules['torch']`` undone by ``monkeypatch`` before this teardown runs —
    autouse fixtures tear down last. On a machine with no torch that import then
    raises during cleanup and reports as an error in an unrelated test.

    So: attempt the real free, because that is what exercises the production path,
    and fall back to emptying the dict, because isolation is this fixture's only
    job. ``test_dropping_the_resident_enhancer_gives_the_vram_back`` is what
    actually holds the free honest.
    """
    try:
        enhance.clear_enhancer_cache()
    except Exception:  # noqa: BLE001 - cleanup must not fail somebody else's test
        enhance._ENHANCER.clear()


# --------------------------------------------------------------------------
# safetensors synthesis (pure Python — no torch, no safetensors)
# --------------------------------------------------------------------------

DTYPE_SIZES = {
    "F64": 8,
    "F32": 4,
    "F16": 2,
    "BF16": 2,
    "I64": 8,
    "I32": 4,
    "I16": 2,
    "U8": 1,
    "I8": 1,
    "BOOL": 1,
}

#: Key prefixes verified against real LTX-2.5 adapters (PLAN §4.1, §4.2).
NATIVE_PREFIX = "diffusion_model."
DIFFUSERS_PREFIX = "transformer."
NATIVE_CONNECTOR_PREFIX = "text_embedding_projection."
DIFFUSERS_CONNECTOR_PREFIX = "connectors."

ALL_PREFIXES = (
    NATIVE_PREFIX,
    DIFFUSERS_PREFIX,
    NATIVE_CONNECTOR_PREFIX,
    DIFFUSERS_CONNECTOR_PREFIX,
)

#: Module paths the fake transformer exposes, with ``(in_features, out_features)``.
#:
#: The dimensions are the measured ones from PLAN §4.5: the transformer carries
#: **two streams** — video at 4096 and audio at 2048, each with a 4x feed-forward
#: expansion. That is why stage 6.5 compares per module: "contains 2048" is not a
#: signal, "2048 at a module that expects 4096" is.
TRANSFORMER_MODULE_DIMS = {
    "transformer_blocks.0.attn1.to_q": (4096, 4096),
    "transformer_blocks.0.attn1.to_k": (4096, 4096),
    "transformer_blocks.0.attn1.to_v": (4096, 4096),
    "transformer_blocks.1.attn1.to_q": (4096, 4096),
    "transformer_blocks.0.ff.net.2": (16384, 4096),
    "transformer_blocks.0.audio_ff.net.0.proj": (2048, 8192),
    "transformer_blocks.0.audio_ff.net.2": (8192, 2048),
    "adaln_single.linear": (256, 4096),
    "audio_adaln_single.linear": (256, 2048),
}

#: Adapter keys that resolve to one of these are what "attached" means.
TRANSFORMER_MODULE_PATHS = tuple(TRANSFORMER_MODULE_DIMS)

CONNECTOR_MODULE_DIMS = {
    "layers.0.proj": (4096, 4096),
    "layers.1.proj": (4096, 4096),
}

CONNECTOR_MODULE_PATHS = tuple(CONNECTOR_MODULE_DIMS)

#: Every module dimension the fake pipeline knows about, in one lookup.
MODULE_DIMS = {**TRANSFORMER_MODULE_DIMS, **CONNECTOR_MODULE_DIMS}

#: What an unmapped module is assumed to be — the video stream's hidden dim.
DEFAULT_MODULE_DIMS = (4096, 4096)

#: Rank of the synthetic adapters. The real ones range from 32 to 450.
DEFAULT_RANK = 32

#: The official adapter's metadata, read from its real header (PLAN §4.2).
OFFICIAL_METADATA = {
    "lora_rank": "450",
    "lora_alpha": "450",
    "model_version": "2.5.0",
    "license": "LTX-2.5 Community License",
}


def build_safetensors_bytes(
    tensors,
    metadata=None,
    *,
    header_length_override=None,
    payload_bytes_override=None,
    align_header=True,
):
    """Return the bytes of a safetensors file.

    ``tensors`` maps tensor name -> ``(dtype, shape)``. The payload is zero-filled;
    nothing here ever needs to be a real tensor, which is the whole point of
    PLAN §4.3 stage 2 (header-only parsing).

    ``header_length_override`` writes a deliberately wrong 8-byte length prefix
    (the "corrupt header length" case). ``payload_bytes_override`` writes fewer
    payload bytes than the header declares.
    """
    header: dict[str, object] = {}
    offset = 0
    for name, (dtype, shape) in tensors.items():
        count = 1
        for dim in shape:
            count *= dim
        nbytes = count * DTYPE_SIZES[dtype]
        header[name] = {
            "dtype": dtype,
            "shape": list(shape),
            "data_offsets": [offset, offset + nbytes],
        }
        offset += nbytes
    if metadata is not None:
        header["__metadata__"] = {str(k): str(v) for k, v in metadata.items()}

    blob = json.dumps(header).encode("utf-8")
    if align_header:
        pad = (8 - (len(blob) % 8)) % 8
        blob += b" " * pad

    declared = len(blob) if header_length_override is None else header_length_override
    payload_len = offset if payload_bytes_override is None else payload_bytes_override
    return struct.pack("<Q", declared) + blob + (b"\x00" * payload_len)


def lora_pair(prefix, module_path, *, style="peft", rank=DEFAULT_RANK, dims=None):
    """Return the ``{key: (dtype, shape)}`` entries for one LoRA A/B pair.

    Shapes follow the real convention measured in PLAN §4.5:
    ``lora_A = [rank, in_features]`` and ``lora_B = [out_features, rank]``.
    Dimensions default to the fake transformer's own, so a fixture built from the
    real module paths is dimensionally *correct* and passes stage 6.5 — leaving
    the dimension tests free to break exactly one thing on purpose.
    """
    if style == "peft":
        down, up = "lora_A.weight", "lora_B.weight"
    elif style == "kohya":
        down, up = "lora_down.weight", "lora_up.weight"
    else:  # pragma: no cover - guard against a typo in a test
        raise ValueError(f"unknown style {style!r}")
    in_features, out_features = dims or MODULE_DIMS.get(module_path, DEFAULT_MODULE_DIMS)
    base = f"{prefix}{module_path}"
    return {
        f"{base}.{down}": ("F32", [rank, in_features]),
        f"{base}.{up}": ("F32", [out_features, rank]),
    }


def lora_tensors(prefix, module_paths, *, style="peft", rank=DEFAULT_RANK, dims=None):
    """``dims`` maps a module path to ``(in_features, out_features)`` overrides."""
    tensors: dict[str, tuple[str, list[int]]] = {}
    overrides = dims or {}
    for path in module_paths:
        tensors.update(lora_pair(prefix, path, style=style, rank=rank, dims=overrides.get(path)))
    return tensors


def read_safetensors_header(path):
    """Independent reference reader — used by the fakes, never by the code under test."""
    data = Path(path).read_bytes()
    (declared,) = struct.unpack("<Q", data[:8])
    return json.loads(data[8 : 8 + declared].decode("utf-8"))


# --------------------------------------------------------------------------
# safetensors fixtures
# --------------------------------------------------------------------------


@pytest.fixture
def write_safetensors(tmp_path):
    """Factory: write a synthetic safetensors file and return its path."""

    def _write(name="adapter.safetensors", tensors=None, metadata=None, **kwargs):
        if tensors is None:
            tensors = lora_tensors(NATIVE_PREFIX, TRANSFORMER_MODULE_PATHS)
        target = tmp_path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(build_safetensors_bytes(tensors, metadata, **kwargs))
        return target

    return _write


@pytest.fixture
def good_lora(write_safetensors):
    """A well-formed native-convention adapter that targets the fake transformer."""
    return write_safetensors(
        name="good-lora.safetensors",
        tensors=lora_tensors(NATIVE_PREFIX, TRANSFORMER_MODULE_PATHS),
        metadata=OFFICIAL_METADATA,
    )


@pytest.fixture
def wrong_architecture_lora(write_safetensors):
    """PLAN §9.3.1: passes stages 1-6, attaches to ZERO modules.

    Every key contains ``lora`` (so diffusers' only built-in check passes), the
    prefix is a recognised one, pairs are complete and ``model_version`` says
    2.5.0. The module paths are simply from a different architecture — which is
    exactly the checkpoint that produces silent garbage today.
    """
    return write_safetensors(
        name="wrong-arch-lora.safetensors",
        tensors=lora_tensors(
            NATIVE_PREFIX,
            (
                "down_blocks.0.resnets.0.conv1",
                "mid_block.attentions.0.to_out.0",
                "up_blocks.3.attentions.2.proj_out",
            ),
        ),
        metadata=OFFICIAL_METADATA,
    )


@pytest.fixture
def wrong_dimension_lora(write_safetensors):
    """The ``charlied98/ltxv_2b_lora_soft_draw_v1`` shape — PLAN §4.3 stage 6.5.

    An LTX-**Video-2B** adapter. Its module names are the ones LTX-2.5 uses, so
    it passes stages 4, 5 and 6, passes diffusers' built-in check, and **passes
    stage 7** — it really does attach. The only thing wrong with it is that
    ``to_k.lora_A`` is ``[32, 2048]`` where this model is 4096.

    Stage 6.5 is the only gate that catches this file.
    """
    video_stream = (
        "transformer_blocks.0.attn1.to_q",
        "transformer_blocks.0.attn1.to_k",
        "transformer_blocks.0.attn1.to_v",
    )
    return write_safetensors(
        name="ltxv-2b-lora.safetensors",
        tensors=lora_tensors(
            NATIVE_PREFIX,
            video_stream,
            dims={path: (2048, 2048) for path in video_stream},
        ),
        metadata={},
    )


@pytest.fixture
def two_stream_lora(write_safetensors):
    """The ``joyfox/LTX-2.3-Transition-LORA`` shape — dimensionally correct.

    It contains 2048-wide tensors, at the audio-stream modules where 2048 is
    right. A stage 6.5 that tests "does this file contain 2048" instead of
    "2048 at a module that expects 4096" rejects the most-downloaded LTX LoRA on
    the Hub.
    """
    return write_safetensors(
        name="two-stream-lora.safetensors",
        tensors=lora_tensors(NATIVE_PREFIX, TRANSFORMER_MODULE_PATHS),
        metadata={},
    )


@pytest.fixture
def ic_lora(write_safetensors):
    """The ``DoctorDiffusion/LTX-2.3-IC-LoRA-Colorizer`` shape — PLAN §4.3 stage 5.5.

    An in-context LoRA. On the plain pipeline it will not error and will not
    work — silent garbage arriving through a door stage 7 does not watch.
    """
    return write_safetensors(
        name="ic-lora-colorizer.safetensors",
        tensors=lora_tensors(NATIVE_PREFIX, TRANSFORMER_MODULE_PATHS),
        metadata={"reference_downscale_factor": "1"},
    )


@pytest.fixture
def sd_scripts_lora(write_safetensors):
    """The ``vrgamedevgirl84/LTX_2.3_Crisp_Enhance_Style_LoRa`` shape — PLAN §4.5.

    No ``model_version`` anywhere, but a rich sd-scripts/musubi metadata block
    whose ``modelspec.architecture`` is a *better* architecture signal.
    """
    return write_safetensors(
        name="sd-scripts-lora.safetensors",
        tensors=lora_tensors(NATIVE_PREFIX, TRANSFORMER_MODULE_PATHS),
        metadata={
            "ss_ltx_version": "2.3",
            "ss_base_model_version": "ltx2_v1",
            "modelspec.architecture": "LTX2/lora",
            "ss_network_module": "networks.lora_ltx2",
            "ss_sd_model_name": "ltx-2.3-22b-dev.safetensors",
            "ss_network_dim": "64",
            "ss_network_alpha": "32",
            "ss_lora_target_preset": "full",
        },
    )


# --------------------------------------------------------------------------
# fake pipeline (duck-types the diffusers surface we touch)
# --------------------------------------------------------------------------


class FakeModule:
    """Minimal ``torch.nn.Module`` stand-in: a named module tree."""

    def __init__(self):
        self._children: dict[str, FakeModule] = {}

    @classmethod
    def from_paths(cls, paths):
        root = cls()
        for path in paths:
            node = root
            for part in path.split("."):
                node = node._children.setdefault(part, cls())
        return root

    def named_modules(self, prefix=""):
        if prefix:
            yield prefix, self
        for name, child in self._children.items():
            child_prefix = f"{prefix}.{name}" if prefix else name
            yield from child.named_modules(child_prefix)

    def get_module(self, path):
        node = self
        for part in path.split("."):
            node = node._children.get(part)
            if node is None:
                return None
        return node


class FakeLinear(FakeModule):
    """A leaf that looks like ``torch.nn.Linear`` to a dimension check.

    Stage 6.5 builds its expectation map by walking the loaded transformer's
    ``nn.Linear`` modules — never a hardcoded table — so the fake has to carry
    real ``in_features`` / ``out_features`` for that walk to find anything.
    """

    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features


class FakeComponent(FakeModule):
    """A pipeline component (``transformer`` / ``connectors``) that peft can decorate."""

    def __init__(self, paths, dims=None):
        super().__init__()
        dims = {**MODULE_DIMS, **(dims or {})}
        for path in paths:
            parts = path.split(".")
            node = self
            for part in parts[:-1]:
                node = node._children.setdefault(part, FakeModule())
            in_features, out_features = dims.get(path, DEFAULT_MODULE_DIMS)
            node._children.setdefault(parts[-1], FakeLinear(in_features, out_features))
        self.peft_config: dict[str, object] = {}


class ShapedTensor:
    """Enough tensor to follow a rearrangement, with no torch installed.

    The offline gate has none, so a test of a real transform can only skip --
    and a skipped test of an axis order is how two half-fixes went unnoticed.
    This models exactly the operations the layout steps use.
    """

    def __init__(self, shape, dtype="float32"):
        self.shape = tuple(shape)
        self.dtype = dtype

    def transpose(self, a, b):
        axes = list(self.shape)
        axes[a], axes[b] = axes[b], axes[a]
        return ShapedTensor(axes, self.dtype)

    def unsqueeze(self, axis):
        axes = list(self.shape)
        axes.insert(axis, 1)
        return ShapedTensor(axes, self.dtype)

    def to(self, dtype):
        return ShapedTensor(self.shape, dtype)

    def __getitem__(self, key):
        # Models `waveform[None]`, which is how the batch axis is added --
        # `[None]` rather than `.unsqueeze(0)` because it means the same in
        # torch and numpy and the offline gate has no torch.
        if key is None:
            return self.unsqueeze(0)
        raise TypeError(f"ShapedTensor only models [None] indexing, got {key!r}")


class FakeMel:
    """A mel carrying every attribute the hand-off to the VAE can disagree on.

    4D on purpose, and **on the CPU on purpose**: the encoder wants
    [batch, channels, time, mel], and the real mel is computed on the waveform's
    device while the VAE's weights are on the accelerator. A fake that already
    matched the VAE would pass a hand-off that moves neither.
    """

    def __init__(self, dtype, shape=(1, 2, 401, 64), device="cpu"):
        self.dtype = dtype
        self.shape = shape
        self.device = device

    def to(self, **kwargs):
        return FakeMel(
            kwargs.get("dtype", self.dtype),
            self.shape,
            kwargs.get("device", self.device),
        )


class FakeLatentDist:
    """Duck-types diffusers' ``DiagonalGaussianDistribution`` -- **with its signature.**

    The real one is ``sample(self, generator: torch.Generator | None = None)``,
    and ``audio._sampled`` hands it the run's generator positionally. This fake
    used to be ``sample=lambda: latents``, which takes nothing: in a gate with
    no torch ``_seeded_generator`` returns None, the draw is bare, and the
    lambda agreed with it. Install requirements.txt and the generator is real,
    so every audio-input test raised ``TypeError`` inside the fake -- 34
    failures and 20 errors measuring the fixture, not the code.

    It **records** what each draw was handed (None for a bare draw) in the list
    it is given, which is the VAE's ``sample_generators``. Accepting the
    generator alone would make the gate pass while proving nothing about
    seeding; recording it lets a test check that the encode draws with the
    SAME generator the pipeline gets (``generate._prepared_audio``).
    """

    def __init__(self, latents, draws: list):
        self._latents = latents
        self._draws = draws

    def sample(self, generator=None):
        self._draws.append(generator)
        return self._latents


class FakeAudioVae:
    """Duck-types an audio VAE closely enough to really encode a waveform.

    Rich rather than minimal on purpose. ``encode_waveform_to_latents`` reads all
    six PLAN 5.4 config parameters, *finds* a mel front-end rather than building
    one, and asserts the latent rank before returning -- so a thinner fake would
    let a broken encode pass. That is the fixture failure that hid the bfloat16
    bug, and this path is the one that was connected to nothing.

    Its sampling rate matches the stub loader's, so ``_resampled`` short-circuits
    and the whole encode runs with no torch present.
    """

    #: ⚠️ **bf16, because that is what the checkpoint is.** A float32 fake would
    #: pass a broken cast -- the mel is computed in float32 and float32 is the
    #: dtype that works, which is why three dtype-boundary defects reached the
    #: card before one reached the gate.
    dtype = "bfloat16"

    #: ⚠️ The weights live on the accelerator; the mel is computed wherever the
    #: waveform is, which is the CPU. A fake VAE on the CPU would pass a
    #: hand-off that never moves the tensor.
    device = "cuda"

    #: Downsampling applied to BOTH axes by `encode`, from the real config.
    mel_compression_ratio = 4

    def __init__(self, sampling_rate=16000):
        # ⚠️ These are the REAL key names, read off audio_vae/config.json at
        # revision 871165de…, not our semantic ones. The previous fake used our
        # names, so it agreed with the code and with nothing else -- the reader
        # raised on the actual checkpoint while every offline test passed.
        self.config = SimpleNamespace(
            sample_rate=sampling_rate,
            mel_hop_length=160,
            mel_bins=64,
            latent_channels=8,
            in_channels=2,
        )
        # Instance attributes on the model, as upstream sets them in __init__.
        # They are absent from config.json entirely.
        self.temporal_compression_ratio = 4
        self.mels: list = []
        self.encoded: list = []
        #: One entry per ``latent_dist.sample`` call: the generator it was handed.
        self.sample_generators: list = []

    def mel_spectrogram(self, waveform):
        """A mel whose TIME axis is derived from the waveform, in encoder layout.

        float32, as the real transform computes it -- the cast to the VAE's
        dtype happens at the hand-off, which is what this lets us check.

        ⚠️ The time axis is computed, not fixed. A constant-shaped mel makes
        every duration look the same, which is exactly the symptom a doubled
        transpose produced on the card: reconstructions pinned to one length
        regardless of input. A fixture that cannot express duration cannot
        catch that.
        """
        self.mels.append(waveform)
        batch, channels, samples = waveform.shape
        frames = samples // self.config.mel_hop_length + 1
        return FakeMel("float32", (batch, channels, frames, self.config.mel_bins))

    def encode(self, mel):
        """Model what the real encoder does: **preserve axis order, downsample both.**

        ⚠️ The output shape is DERIVED from the input's, not fixed. That is the
        whole point: the encoder does not reorder anything, so the mel's layout
        decides the latents' layout. A fake returning a constant shape agrees
        with whatever the code does to it -- it would pass a missing transpose
        and a doubled one alike, and a doubled one shipped.

        Matches the card exactly: a (1, 2, 401, 64) mel gives (1, 8, 101, 16).
        """
        self.encoded.append(mel)
        batch, _channels, first, second = mel.shape
        ratio = self.mel_compression_ratio
        latents = ShapedTensor(
            (
                batch,
                self.config.latent_channels,
                -(-first // ratio),  # ceil, as the strided convolutions give
                second // ratio,
            )
        )
        return SimpleNamespace(latent_dist=FakeLatentDist(latents, self.sample_generators))


def waveform_of(channels: int, samples: int = 4096):
    """A waveform shaped the way ``torchaudio.load`` returns one: [channels, samples].

    A real array rather than a sentinel, because ``match_channel_count`` reads
    ``.shape[0]`` and widens by fancy indexing. A shapeless stub would sail past
    the channel check and leave the mono path -- the one most users will hit --
    untested.
    """
    import numpy as np

    return np.zeros((channels, samples), dtype=np.float32)


class FakeVocoder:
    """The vocoder, with the REAL key names from vocoder/config.json.

    It carries the analysis window the audio VAE's config does not declare, so
    the forward mel transform needs it as well as the VAE. Keys are the
    checkpoint's own, not ours -- a fake using our names would pass a wrong
    lookup, which is exactly how the audio-VAE parameters went unnoticed.
    """

    def __init__(self):
        self.config = SimpleNamespace(
            filter_length=512,
            window_length=512,
            hop_length=80,  # NOT the mel hop; see audio.MEL_PARAM_SOURCES
            num_mel_channels=64,
            input_sampling_rate=16000,
            output_sampling_rate=48000,
            upsample_factors=[5, 2, 2, 2, 2, 2],
        )


#: What the stub loader hands back. STEREO, matching the fake VAE's in_channels,
#: so the default seam runs need no conversion and the mono case is explicit.
#: LONG ENOUGH to cover the request the seam's field defaults produce — 121
#: frames at 24 fps is 5.04 s — because `generate._prepared_audio` now
#: reconciles upload length with the video (trim long, REFUSE short): the old
#: 0.26 s stub would be refused against every default request, dragging every
#: encode-seam test into the length policy's territory instead of the seam
#: they are about. Derived from the same numbers, not a magic constant.
STUB_WAVEFORM = waveform_of(2, samples=round(121 / 24.0 * 16000))
#: Matches the audio VAE's own rate, so `_resampled` short-circuits and the
#: offline gate needs no torchaudio. The VAE takes 16000; the VOCODER emits
#: 48000. They are not symmetric and assuming they were is a real trap.
STUB_SAMPLING_RATE = 16000


@pytest.fixture
def audio_input_ready(monkeypatch):
    """Make the audio-input path runnable in the offline gate.

    ``load_waveform`` reads a real file with torchaudio, and the gate has
    neither the file nor torchaudio. Stubbing only the **loader** leaves the rest
    of the path -- resample, mel front-end, encode, shape assertion -- genuinely
    exercised, which is the part that was dead.
    """
    from ltx25_studio import audio as audio_module

    monkeypatch.setattr(
        audio_module, "load_waveform", lambda source: (STUB_WAVEFORM, STUB_SAMPLING_RATE)
    )


class FakePipeline:
    """Duck-types the bits of an ``LTX2*Pipeline`` that ``lora/apply.py`` touches.

    ``load_lora_weights`` reproduces upstream behaviour *including its hole*:

    * the only validation is the ``'lora'`` substring test (PLAN §4.2);
    * ``peft_config`` is populated whether or not anything attached — so code
      that checks ``peft_config`` instead of counting modules still sees a
      "successful" load. That is the silent-garbage path stage 7 must close.
    """

    _lora_loadable_modules = ["transformer", "connectors"]

    def __init__(self, transformer_paths=None, connector_paths=None, dims=None):
        # The real pipeline carries one, and `generate._prepared_audio` unwraps
        # to `.pipeline` to find it. A fake without it made the audio-input
        # modes look like they had nothing to encode with.
        self.audio_vae = FakeAudioVae()
        self.transformer = FakeComponent(
            TRANSFORMER_MODULE_PATHS if transformer_paths is None else transformer_paths,
            dims=dims,
        )
        self.connectors = FakeComponent(
            CONNECTOR_MODULE_PATHS if connector_paths is None else connector_paths,
            dims=dims,
        )
        self.calls: list[dict] = []
        self.fuse_calls: list[dict] = []
        self.unfuse_calls: list[dict] = []

    # -- upstream conversion table (PLAN §4.1) ----------------------------
    _ROUTES = {
        NATIVE_PREFIX: "transformer",
        DIFFUSERS_PREFIX: "transformer",
        NATIVE_CONNECTOR_PREFIX: "connectors",
        DIFFUSERS_CONNECTOR_PREFIX: "connectors",
    }

    def load_lora_weights(self, source, *args, adapter_name="default", **kwargs):
        self.calls.append({"source": source, "adapter_name": adapter_name, "kwargs": kwargs})
        header = read_safetensors_header(source)
        keys = [k for k in header if k != "__metadata__"]

        # The one and only upstream check, quoted verbatim from diffusers.
        if not all("lora" in k for k in keys):
            raise ValueError(
                "Invalid LoRA checkpoint. Make sure all LoRA param names contain "
                "`'lora'` substring."
            )

        attached = 0
        for key in keys:
            component_name = None
            stripped = key
            for prefix, route in self._ROUTES.items():
                if key.startswith(prefix):
                    component_name = route
                    stripped = key[len(prefix) :]
                    break
            if component_name is None:
                continue
            module_path = stripped
            for suffix in (
                ".lora_A.weight",
                ".lora_B.weight",
                ".lora_down.weight",
                ".lora_up.weight",
            ):
                if module_path.endswith(suffix):
                    module_path = module_path[: -len(suffix)]
                    break
            component = getattr(self, component_name)
            module = component.get_module(module_path)
            if module is None:
                continue
            lora_a = getattr(module, "lora_A", None)
            if lora_a is None:
                lora_a = {}
                module.lora_A = lora_a
                module.lora_B = {}
            if adapter_name not in lora_a:
                lora_a[adapter_name] = object()
                module.lora_B[adapter_name] = object()
                attached += 1

        # THE TRAP: peft records the adapter config even when nothing attached.
        for component_name in self._lora_loadable_modules:
            component = getattr(self, component_name, None)
            if component is not None:
                component.peft_config[adapter_name] = {"r": 450, "lora_alpha": 450}
        return attached

    def get_list_adapters(self):
        """Also optimistic: reports the adapter whether or not anything attached."""
        return {
            name: list(getattr(self, name).peft_config)
            for name in self._lora_loadable_modules
            if getattr(self, name, None) is not None
        }

    def set_adapters(self, *args, **kwargs):
        self.calls.append({"set_adapters": (args, kwargs)})

    def delete_adapters(self, adapter_names):
        """Mirrors ``LoraBaseMixin.delete_adapters``: drop the named adapters
        from every loadable component. Added when ``generate()`` grew its
        unload step -- the real pipeline has this, and a fake without it turns
        the lifecycle's cannot-detach refusal into a false alarm."""
        names = [adapter_names] if isinstance(adapter_names, str) else list(adapter_names)
        self.calls.append({"delete_adapters": names})
        for name in names:
            for component_name in self._lora_loadable_modules:
                component = getattr(self, component_name, None)
                if component is None:
                    continue
                component.peft_config.pop(name, None)
                for _, module in component.named_modules():
                    lora_a = getattr(module, "lora_A", None)
                    if lora_a is not None:
                        lora_a.pop(name, None)
                        module.lora_B.pop(name, None)

    def fuse_lora(self, *args, **kwargs):
        self.fuse_calls.append({"args": args, "kwargs": kwargs})

    def unfuse_lora(self, *args, **kwargs):
        self.unfuse_calls.append({"args": args, "kwargs": kwargs})


@pytest.fixture
def fake_pipeline():
    return FakePipeline()


@pytest.fixture
def make_pipeline():
    def _make(transformer_paths=None, connector_paths=None, dims=None):
        return FakePipeline(transformer_paths, connector_paths, dims=dims)

    return _make


@pytest.fixture
def fake_transformer():
    """Just the transformer, for the stage 6.5 dimension checks."""
    return FakeComponent(TRANSFORMER_MODULE_PATHS)


# --------------------------------------------------------------------------
# byte-counting read spy — proof of PLAN §4.3 stage 2
# --------------------------------------------------------------------------


class _CountingFile:
    def __init__(self, wrapped, spy):
        self._wrapped = wrapped
        self._spy = spy

    def read(self, *args, **kwargs):
        chunk = self._wrapped.read(*args, **kwargs)
        self._spy.bytes_read += len(chunk)
        return chunk

    def readinto(self, buf):
        n = self._wrapped.readinto(buf)
        self._spy.bytes_read += n or 0
        return n

    def readline(self, *args, **kwargs):
        chunk = self._wrapped.readline(*args, **kwargs)
        self._spy.bytes_read += len(chunk)
        return chunk

    def __enter__(self):
        self._wrapped.__enter__()
        return self

    def __exit__(self, *exc):
        return self._wrapped.__exit__(*exc)

    def __iter__(self):
        return iter(self._wrapped)

    def __getattr__(self, name):
        return getattr(self._wrapped, name)


class ReadSpy:
    def __init__(self):
        self.bytes_read = 0
        self.opens = 0
        self.watched: set[str] = set()

    def watch(self, path):
        self.watched.add(str(Path(path).resolve()))


@pytest.fixture
def read_spy(monkeypatch):
    """Counts bytes read from watched files via ``open()`` / ``io.open()``.

    Both names are patched: ``pathlib.Path.open`` and ``Path.read_bytes`` route
    through ``io.open``, not ``builtins.open``, so patching only one leaves a
    hole a header parser could slip through.
    """
    spy = ReadSpy()
    real_open = builtins.open

    def fake_open(file, mode="r", *args, **kwargs):
        handle = real_open(file, mode, *args, **kwargs)
        try:
            resolved = str(Path(file).resolve())
        except (TypeError, ValueError, OSError):
            return handle
        if resolved in spy.watched and "b" in mode and "w" not in mode and "a" not in mode:
            spy.opens += 1
            return _CountingFile(handle, spy)
        return handle

    monkeypatch.setattr(builtins, "open", fake_open)
    monkeypatch.setattr(io, "open", fake_open)
    return spy


# --------------------------------------------------------------------------
# generation request factory
# --------------------------------------------------------------------------


def request_fields(**overrides):
    """Every field a mode kwarg-builder could plausibly read, with legal defaults.

    Deliberately a plain namespace: PLAN §7.2 says the mode registry is data and
    the builders are pure, so they must not require a concrete class from another
    module. If a builder only works with the real ``GenerationRequest``, that is
    a coupling the registry should not have.
    """
    fields = {
        "mode": "text_to_video",
        "prompt": "a cat playing a grand piano, cinematic",
        "negative_prompt": "",
        "width": 704,
        "height": 480,
        "num_frames": 121,
        "seed": 1234,
        "image": None,
        "video": None,
        "audio": None,
        "audio_latents": None,
        "noise_scale": 0.5,
        "conditions": (),
        "strength": 0.8,
        "guidance_scale": 1.0,
        "num_inference_steps": 8,
        "loras": (),
    }
    fields.update(overrides)
    return SimpleNamespace(**fields)


@pytest.fixture
def make_request():
    return request_fields


# --------------------------------------------------------------------------
# source-file helpers for the architecture specs
# --------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent
PACKAGE_ROOT = REPO_ROOT / "ltx25_studio"


def code_string_constants(path):
    """Yield ``(lineno, value)`` for every string literal that is not a docstring.

    Comments are invisible to ``ast`` by construction, and module/class/function
    docstrings are filtered explicitly — so this sees prose nowhere and code
    everywhere, which is what the architecture rules want to police.
    """
    import ast

    tree = ast.parse(Path(path).read_text())
    docstrings = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            body = getattr(node, "body", None)
            if (
                body
                and isinstance(body[0], ast.Expr)
                and isinstance(body[0].value, ast.Constant)
                and isinstance(body[0].value.value, str)
            ):
                docstrings.add(id(body[0].value))
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            if id(node) in docstrings:
                continue
            yield node.lineno, node.value


def code_tokens(path):
    """Every code token in a file, with comments and string literals dropped."""
    import io as _io
    import tokenize as _tokenize

    source = Path(path).read_text()
    for token in _tokenize.generate_tokens(_io.StringIO(source).readline):
        if token.type in (_tokenize.COMMENT, _tokenize.STRING, _tokenize.NL, _tokenize.NEWLINE):
            continue
        if token.string.strip():
            yield token.string


def package_python_files(exclude=()):
    excluded = {str(PACKAGE_ROOT / name) for name in exclude}
    return sorted(
        p
        for p in PACKAGE_ROOT.rglob("*.py")
        if "__pycache__" not in p.parts and str(p) not in excluded
    )
