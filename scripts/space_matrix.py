#!/usr/bin/env python
"""Run a plan of calls against the private LTX-2.5 Studio Space under a hard daily GPU budget.

    # DRY RUN (the default): resolve, validate and price every call; contacts nothing
    python scripts/space_matrix.py --plan scripts/space_plan.example.json \\
        --assets-dir ~/assets --quota-ledger ~/space-ledger.jsonl

    # the same plan, executed
    python scripts/space_matrix.py --plan plan.json --assets-dir ~/assets \\
        --quota-ledger ~/space-ledger.jsonl --out ~/space-runs/day1 --live

    # one call from flags; the historical 11-mode loop is --plan default
    python scripts/space_matrix.py --mode text_to_video --width 704 --height 480 --num-frames 49

    # read-only, one GET each, no GPU time
    python scripts/space_matrix.py status
    python scripts/space_matrix.py info
    python scripts/space_matrix.py logs --out run.log

Exit codes: 0 plan complete · 2 illegal plan or contract mismatch (nothing spent) · 3 stopped at
the ceiling · 4 the platform refused on quota · 5 a call failed (with --keep-going the rest of the
plan still runs, and the exit is still 5).

Cost figures are BOUNDS. There is no readable ZeroGPU quota endpoint, so billed seconds cannot be
measured, only bounded: the conservative figure is the wall clock from the moment the Space starts
the event to its completion, times the size multiplier. The ceiling check before each call adds
that call's full reservation (``zerogpu.estimate_duration``, times the multiplier) to today's
conservative total.

The token is read when a command needs it (``HF_TOKEN``, else the Hub's cached token file), is
sent only as the ``Authorization`` header to the Space and the Hub API, and is redacted from
every line, file and error this script writes.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import os
import re
import shutil
import sys
import time
import traceback
import urllib.error
import urllib.parse
import urllib.request
import uuid
from dataclasses import dataclass, field, replace
from dataclasses import fields as dataclass_fields
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parent.parent
# `tests/` for `verify_streams`, the registry-driven stream checker.
for _path in (REPO, REPO / "tests"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import verify_streams  # noqa: E402

from ltx25_studio import constraints, modes, outputs, zerogpu  # noqa: E402
from ltx25_studio.ui import fields  # noqa: E402

__all__ = [
    "API_FIELD_KEYS",
    "DEFAULT_MODE_ORDER",
    "DEFAULT_QUOTA_CEILING_SECONDS",
    "DEFAULT_RESERVATION_SECONDS",
    "ENDPOINT",
    "EXIT_CALL_FAILED",
    "EXIT_CEILING",
    "EXIT_ILLEGAL",
    "EXIT_OK",
    "EXIT_QUOTA",
    "LORA_ENDPOINT",
    "Call",
    "Ledger",
    "PlanError",
    "QuotaLedgerUnreadable",
    "Secret",
    "default_plan",
    "generate_contract_problems",
    "load_plan",
    "main",
    "measure_audio",
    "measure_video",
    "quota_multiplier",
    "quota_refusal",
    "read_token",
    "redact",
    "reservation_seconds",
    "utc_day",
]

# ---------------------------------------------------------------------------
# the Space and its published signature
# ---------------------------------------------------------------------------

DEFAULT_SPACE_ID = "techfreakworm/LTX2.5-Studio"
DEFAULT_HUB_URL = "https://huggingface.co"
ENDPOINT = "/on_generate"
LORA_ENDPOINT = "/on_validate_lora"
LORA_WEIGHT_ENDPOINT = "/on_lora_weight"
API_PREFIX = "/gradio_api"

#: ``/on_generate``'s parameters after ``mode_id``: the field spec minus ``gr.State``, which
#: Gradio does not publish. Read again from ``fields.FIELDS`` at call time (see :func:`api_fields`)
#: so a field appended after import is still sent.
API_FIELD_KEYS: tuple[str, ...] = tuple(f.key for f in fields.FIELDS if f.kind != "state")

#: Field kind -> the component name Gradio publishes. Only ``text`` differs from its kind.
_COMPONENT_FOR_KIND = {"text": "textbox"}

# ---------------------------------------------------------------------------
# budget
# ---------------------------------------------------------------------------

DAILY_QUOTA_SECONDS = 2400.0
DEFAULT_QUOTA_CEILING_SECONDS = 1920.0
#: The Space's ``LTX25_GPU_DURATION_SECONDS``; the local config default is not what it runs.
DEFAULT_RESERVATION_SECONDS = 180
#: The Space's ``LTX25_ZEROGPU_SIZE``; ``zerogpu.quota_multiplier`` turns it into the quota factor.
DEFAULT_ZEROGPU_SIZE = "xlarge"
DEFAULT_CALL_TIMEOUT_SECONDS = 1800.0
DEFAULT_READ_TIMEOUT_SECONDS = 120.0

EXIT_OK = 0
EXIT_ILLEGAL = 2
EXIT_CEILING = 3
EXIT_QUOTA = 4
EXIT_CALL_FAILED = 5

#: Every wording of a ZeroGPU quota refusal in ``spaces`` (``zero/client.py``): the error title,
#: the PRO/free/IP message, the runs limit, the Space-wide limit and the pending-credits refusal.
_QUOTA_REFUSAL = re.compile(
    r"ZeroGPU quota exceeded"
    r"|exceeded your [\w ]*?ZeroGPU quota"
    r"|exceeded your ZeroGPU runs limit"
    r"|exceeded [\w ]*GPU quota"
    r"|reached its GPU limit"
    r"|ZeroGPU pending credits exceeded"
    r"|too many ZeroGPU credits allocated",
    re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# content thresholds (measurements, reported with their thresholds; never stop a plan)
# ---------------------------------------------------------------------------

MIN_PIXEL_STD = 1.0
MIN_DISTINCT_FRAMES = 2
#: mean |Δ| over the frames' own spatial std; frozen probes measured <= 0.000523, the stillest
#: real post-fix render 0.004948.
MIN_RELATIVE_FRAME_DELTA = 0.001
LOW_MEAN_FRAME_DELTA = 0.5

#: The pinned vocoder's rate; ``expect.sample_rate`` overrides it per call.
PINNED_VOCODER_SAMPLING_RATE = 48000

# ---------------------------------------------------------------------------
# the historical default plan (--plan default)
# ---------------------------------------------------------------------------

CHEAPEST_SIZE = modes.MIN_RESOLUTION
CHEAPEST_FRAMES = 9
SEED = 1234
PROMPT = (
    "A red sports car driving fast along a winding coastal road at sunset, "
    "waves breaking on rocks below, cinematic camera following alongside"
)

#: Producers first: text_to_video makes the video (and, by frame, the image) the conditioned
#: modes chain from; text_to_audio makes the audio. Each chain is explicit in the plan.
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

MEDIA_KINDS = ("image", "video", "audio")
_CALL_KEYS = frozenset({"label", "mode", "loras", "expect", "note"})
_EXPECT_KEYS = {
    "width": int,
    "height": int,
    "frames": int,
    "has_audio": bool,
    "has_video": bool,
    "sample_rate": int,
    "frame_rate": float,
}
_LABEL = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
_HUB_ID = re.compile(r"^[A-Za-z0-9][\w.-]*/[\w.-]+(/[\w./-]+)?$")


# ---------------------------------------------------------------------------
# clocks (module-level so tests can drive them)
# ---------------------------------------------------------------------------


def _utcnow() -> datetime:
    return datetime.now(UTC)


def _monotonic() -> float:
    return time.monotonic()


def utc_day(now: datetime | None = None) -> str:
    return (now or _utcnow()).strftime("%Y-%m-%d")


def next_utc_midnight(now: datetime | None = None) -> datetime:
    moment = now or _utcnow()
    return datetime(moment.year, moment.month, moment.day, tzinfo=UTC) + timedelta(days=1)


# ---------------------------------------------------------------------------
# the token
# ---------------------------------------------------------------------------

_SECRETS: list[str] = []
REDACTED = "<redacted>"


class Secret:
    """The token. Never printed, pickled or formatted; :meth:`reveal` is read by the transport."""

    __slots__ = ("_value",)

    def __init__(self, value: str):
        self._value = value
        if value and value not in _SECRETS:
            _SECRETS.append(value)

    def reveal(self) -> str:
        return self._value

    def __repr__(self) -> str:
        return f"Secret({REDACTED})"

    __str__ = __repr__

    def __reduce__(self):
        raise TypeError("a Secret is not picklable")


def token_path() -> Path:
    if os.environ.get("HF_TOKEN_PATH"):
        return Path(os.environ["HF_TOKEN_PATH"]).expanduser()
    home = os.environ.get("HF_HOME") or "~/.cache/huggingface"
    return Path(home).expanduser() / "token"


def read_token() -> Secret | None:
    """``HF_TOKEN``, else the cached token file. Read at call time, never at import."""
    value = (os.environ.get("HF_TOKEN") or "").strip()
    if not value:
        try:
            value = token_path().read_text().strip()
        except OSError:
            value = ""
    return Secret(value) if value else None


def redact(text: Any) -> str:
    out = "" if text is None else str(text)
    for secret in _SECRETS:
        out = out.replace(secret, REDACTED)
    return out


def _redacted_json(payload: Any) -> str:
    return redact(json.dumps(payload, indent=2, sort_keys=False, default=str)) + "\n"


# ---------------------------------------------------------------------------
# errors
# ---------------------------------------------------------------------------


class PlanError(ValueError):
    """The plan cannot run as written. Nothing was sent."""


class ContractMismatch(RuntimeError):
    """The live Space's published signature is not what the local field spec builds."""


class QuotaLedgerUnreadable(RuntimeError):
    """The ledger exists and is not a ledger. Refused rather than read as zero."""


class HttpError(RuntimeError):
    def __init__(self, code: int, url: str, body: str):
        self.code = code
        super().__init__(redact(f"HTTP {code} from {_strip_query(url)}: {body[:500]}"))


class TransportError(RuntimeError):
    pass


class CallTimeout(RuntimeError):
    pass


def _strip_query(url: str) -> str:
    return url.split("?", 1)[0]


# ---------------------------------------------------------------------------
# HTTP
# ---------------------------------------------------------------------------


def _origin(url: str) -> str:
    parts = urllib.parse.urlsplit(url)
    return f"{parts.scheme}://{parts.netloc}".lower()


class Http:
    """urllib with the token attached only to trusted origins and never carried across a redirect."""

    def __init__(self, secret: Secret | None, trusted: tuple[str, ...] = ()):
        self._secret = secret
        self._trusted = {_origin(u) for u in trusted}
        self._opener = urllib.request.build_opener()

    def trust(self, url: str) -> None:
        self._trusted.add(_origin(url))

    def open(self, method: str, url: str, *, data: bytes | None = None, headers=None, timeout: float = 60.0):
        request = urllib.request.Request(url, data=data, method=method, headers=dict(headers or {}))
        if self._secret is not None and _origin(url) in self._trusted:
            # Unredirected: urllib drops it on any redirect, to any host.
            request.add_unredirected_header("Authorization", f"Bearer {self._secret.reveal()}")
        try:
            return self._opener.open(request, timeout=timeout)
        except urllib.error.HTTPError as exc:
            try:
                body = exc.read(2000).decode("utf-8", "replace")
            except Exception:  # noqa: BLE001
                body = ""
            raise HttpError(exc.code, url, body) from None
        except (urllib.error.URLError, OSError) as exc:
            raise TransportError(redact(f"{type(exc).__name__} reaching {_strip_query(url)}: {exc}")) from None

    def get_json(self, url: str, *, timeout: float = 60.0) -> Any:
        with self.open("GET", url, headers={"Accept": "application/json"}, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))

    def post_json(self, url: str, payload: Any, *, timeout: float = 60.0) -> Any:
        body = json.dumps(payload).encode()
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        with self.open("POST", url, data=body, headers=headers, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))


def iter_sse(response):
    """``data:`` payloads of an event stream, one parsed JSON value per event."""
    buffer: list[str] = []
    while True:
        raw = response.readline()
        if not raw:
            if buffer:
                yield json.loads("\n".join(buffer))
            return
        line = raw.decode("utf-8", "replace").rstrip("\r\n")
        if not line:
            if buffer:
                text = "\n".join(buffer)
                buffer = []
                yield json.loads(text)
            continue
        if line.startswith("data:"):
            buffer.append(line[5:].lstrip())


def space_host(space_id: str) -> str:
    """``owner/Name.x`` -> ``https://owner-name-x.hf.space``, the Space's own subdomain."""
    slug = re.sub(r"[^a-z0-9]+", "-", space_id.lower()).strip("-")
    return f"https://{slug}.hf.space"


class SpaceApp:
    """The Gradio app's HTTP surface: config, info, upload, queue, files."""

    def __init__(self, base_url: str, http: Http, *, api_prefix: str = API_PREFIX):
        self.base = base_url.rstrip("/")
        self.http = http
        self.prefix = "/" + api_prefix.strip("/") if api_prefix else ""
        http.trust(self.base)

    def url(self, path: str) -> str:
        return f"{self.base}{self.prefix}{path}"

    def config(self) -> dict:
        return self.http.get_json(f"{self.base}/config")

    def info(self) -> dict:
        return self.http.get_json(self.url("/info"))

    def upload(self, local: Path) -> dict:
        boundary = uuid.uuid4().hex
        content = Path(local).read_bytes()
        name = Path(local).name
        body = (
            f"--{boundary}\r\nContent-Disposition: form-data; name=\"files\"; filename=\"{name}\"\r\n"
            f"Content-Type: application/octet-stream\r\n\r\n"
        ).encode() + content + f"\r\n--{boundary}--\r\n".encode()
        headers = {"Content-Type": f"multipart/form-data; boundary={boundary}", "Accept": "application/json"}
        with self.http.open("POST", self.url("/upload"), data=body, headers=headers, timeout=600) as response:
            paths = json.loads(response.read().decode("utf-8"))
        return {"path": paths[0], "orig_name": name, "meta": {"_type": "gradio.FileData"}}

    def join(self, fn_index: int, data: list, session_hash: str) -> str:
        payload = {
            "data": data,
            "event_data": None,
            "fn_index": fn_index,
            "trigger_id": None,
            "session_hash": session_hash,
        }
        return str(self.http.post_json(self.url("/queue/join"), payload)["event_id"])

    def stream(self, session_hash: str, *, read_timeout: float):
        query = urllib.parse.urlencode({"session_hash": session_hash})
        return self.http.open(
            "GET", self.url(f"/queue/data?{query}"), headers={"Accept": "text/event-stream"}, timeout=read_timeout
        )

    def download(self, server_path: str, dest: Path) -> Path:
        url = self.url("/file=" + urllib.parse.quote(server_path, safe="/:"))
        dest.parent.mkdir(parents=True, exist_ok=True)
        with self.http.open("GET", url, timeout=600) as response, dest.open("wb") as handle:
            shutil.copyfileobj(response, handle)
        return dest


class Hub:
    """The Hub API's read-only Space endpoints."""

    def __init__(self, hub_url: str, http: Http):
        self.base = hub_url.rstrip("/")
        self.http = http
        http.trust(self.base)

    def space(self, space_id: str) -> dict:
        return self.http.get_json(f"{self.base}/api/spaces/{space_id}")

    def logs(self, space_id: str, *, kind: str, max_seconds: float) -> list[str]:
        lines: list[str] = []
        deadline = _monotonic() + max_seconds
        url = f"{self.base}/api/spaces/{space_id}/logs/{kind}"
        try:
            with self.http.open("GET", url, headers={"Accept": "text/event-stream"}, timeout=min(max_seconds, 10.0)) as response:
                for event in iter_sse(response):
                    if isinstance(event, dict):
                        lines.append(f"{event.get('timestamp', '')} {event.get('data', '')}".strip())
                    else:
                        lines.append(str(event))
                    if _monotonic() >= deadline:
                        break
        except TimeoutError:
            pass
        except TransportError as exc:
            if "timed out" not in str(exc):
                raise
        return lines


# ---------------------------------------------------------------------------
# the plan
# ---------------------------------------------------------------------------


def api_fields() -> tuple:
    """Non-state fields, from the spec as it stands now."""
    return tuple(f for f in fields.FIELDS if f.kind != "state")


def _field(key: str):
    return next((f for f in fields.FIELDS if f.key == key), None)


def _accepts(spec, kind: str) -> bool:
    predicate = getattr(modes, "accepts", None)
    return bool(predicate(spec, kind)) if predicate else kind in spec.inputs


def _requires(spec, kind: str) -> bool:
    predicate = getattr(modes, "requires", None)
    return bool(predicate(spec, kind)) if predicate else kind in spec.inputs


def _skip_stage_2(spec) -> bool:
    return bool(getattr(spec, "skip_stage_2", False))


@dataclass(frozen=True)
class MediaInput:
    kind: str
    path: Path | None = None
    from_call: str | None = None
    #: image chains only: a frame index, ``"middle"`` or ``"last"``.
    frame: int | str | None = None

    @property
    def chained(self) -> bool:
        return self.from_call is not None

    def describe(self) -> dict:
        if self.chained:
            out = {"chained": True, "from_call": self.from_call}
            if self.kind == "image":
                out["frame"] = self.frame
            return out
        size = self.path.stat().st_size if self.path and self.path.exists() else None
        return {"chained": False, "path": str(self.path), "bytes": size}


@dataclass(frozen=True)
class LoraInput:
    source: str
    weight: float = 1.0
    local: Path | None = None

    @property
    def label(self) -> str:
        return self.local.name if self.local else self.source

    def describe(self) -> dict:
        return {
            "source": self.source,
            "route": "upload" if self.local else "hub",
            "local": str(self.local) if self.local else None,
            "weight": self.weight,
        }


@dataclass(frozen=True)
class Call:
    label: str
    mode: str
    values: dict
    media: dict
    loras: tuple = ()
    expect: dict = field(default_factory=dict)
    note: str = ""
    #: From ``generate.prepare_request``: the frame count the app will run (None: the model
    #: chooses) and the notes it will carry ("length set from your audio ...").
    app_frames: int | None = None
    app_notes: tuple = ()

    @property
    def spec(self):
        return modes.get_mode(self.mode)

    def value(self, key: str) -> Any:
        if key in self.values:
            return self.values[key]
        spec = _field(key)
        return None if spec is None else spec.default

    @property
    def min_resolution(self) -> bool:
        return bool(getattr(self.spec, "min_resolution", False))

    @property
    def render_size(self) -> tuple[int, int]:
        if self.min_resolution:
            return (modes.MIN_RESOLUTION, modes.MIN_RESOLUTION)
        return (int(self.value("width")), int(self.value("height")))

    @property
    def stage_2(self) -> bool:
        return bool(self.value("enable_stage_2")) and not _skip_stage_2(self.spec)

    @property
    def exports(self) -> tuple[str, ...]:
        return tuple(self.spec.exports)

    def expected_geometry(self) -> tuple[int, int] | None:
        if "video" not in self.exports:
            return None
        width, height = self.render_size
        if self.stage_2:
            width, height = 2 * width, 2 * height
        return (int(self.expect.get("width", width)), int(self.expect.get("height", height)))

    def expected_frames(self) -> int | None:
        if "frames" in self.expect:
            return int(self.expect["frames"])
        if "video" not in self.exports or _accepts(self.spec, "video"):
            # A clip can set the length; only the plan can say what to expect.
            return None
        if self.app_frames is not None:
            return self.app_frames
        if self.value("auto_duration"):
            return None
        return int(self.value("num_frames"))


def _check_value(spec, value, problems: list[str], where: str) -> None:
    kind = spec.kind
    if kind == "text" and not isinstance(value, str):
        problems.append(f"{where}: {spec.key} must be text")
    elif kind == "checkbox" and not isinstance(value, bool):
        problems.append(f"{where}: {spec.key} must be true or false")
    elif kind in ("number", "slider"):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            problems.append(f"{where}: {spec.key} must be a number")
            return
        low, high = spec.options.get("minimum"), spec.options.get("maximum")
        if low is not None and value < low or high is not None and value > high:
            problems.append(f"{where}: {spec.key}={value} is outside {low}..{high}")
    elif kind in ("dropdown", "radio"):
        choices = spec.options.get("choices")
        if choices:
            allowed = [c[1] if isinstance(c, (list, tuple)) else c for c in choices]
            if value not in allowed:
                problems.append(f"{where}: {spec.key}={value!r} is not one of {allowed}")
    elif kind == "dataframe" and not isinstance(value, list):
        problems.append(f"{where}: {spec.key} must be a list of rows")


def _resolve_path(raw: str, base: Path) -> Path:
    path = Path(raw).expanduser()
    return path if path.is_absolute() else (base / path)


def _parse_call(raw: dict, *, index: int, base: Path, earlier: dict, problems: list[str]) -> Call | None:
    before = len(problems)
    where = f"call {index + 1}"
    if not isinstance(raw, dict):
        problems.append(f"{where}: must be an object")
        return None
    label = raw.get("label")
    if not isinstance(label, str) or not _LABEL.match(label):
        problems.append(f"{where}: 'label' must match {_LABEL.pattern} (it names the evidence folder)")
        return None
    where = f"call {label!r}"
    if label in earlier:
        problems.append(f"{where}: the label is used twice")
        return None
    try:
        spec = modes.get_mode(str(raw.get("mode")))
    except Exception as exc:  # noqa: BLE001
        problems.append(f"{where}: {exc}")
        return None

    values: dict = {}
    media: dict = {}
    loras: list[LoraInput] = []
    expect: dict = {}
    for key, value in raw.items():
        if key in ("label", "mode", "note"):
            continue
        if key == "expect":
            if not isinstance(value, dict):
                problems.append(f"{where}: 'expect' must be an object")
                continue
            for name, wanted in value.items():
                kind = _EXPECT_KEYS.get(name)
                if kind is None:
                    problems.append(f"{where}: expect.{name} is not one of {sorted(_EXPECT_KEYS)}")
                elif kind is bool and not isinstance(wanted, bool):
                    problems.append(f"{where}: expect.{name} must be true or false")
                elif kind is not bool and (isinstance(wanted, bool) or not isinstance(wanted, (int, float))):
                    problems.append(f"{where}: expect.{name} must be a number")
                else:
                    expect[name] = wanted
            continue
        if key == "loras":
            if not isinstance(value, list):
                problems.append(f"{where}: 'loras' must be a list of {{source, weight}}")
                continue
            for entry in value:
                lora = _parse_lora(entry, base=base, where=where, problems=problems)
                if lora is not None:
                    loras.append(lora)
            continue
        spec_field = _field(key)
        if spec_field is None:
            known = ", ".join(f.key for f in api_fields())
            problems.append(
                f"{where}: {key!r} is not a field of this app's {ENDPOINT} (fields: {known}; "
                f"call keys: {', '.join(sorted(_CALL_KEYS))})"
            )
            continue
        if spec_field.kind == "state":
            problems.append(f"{where}: {key!r} is staged through {LORA_ENDPOINT}; use the call's 'loras' list")
            continue
        if spec_field.kind in MEDIA_KINDS:
            item = _parse_media(spec_field.kind, value, base=base, where=where, earlier=earlier, problems=problems)
            if item is not None:
                media[spec_field.kind] = item
            continue
        _check_value(spec_field, value, problems, where)
        values[key] = value

    call = Call(label=label, mode=spec.id, values=values, media=media, loras=tuple(loras), expect=expect,
                note=str(raw.get("note", "")))

    for kind in MEDIA_KINDS:
        if kind in media and not _accepts(spec, kind):
            problems.append(f"{where}: {spec.id} takes no {kind}")
        if kind not in media and _requires(spec, kind):
            problems.append(
                f"{where}: {spec.id} needs a {kind}: give a file under --assets-dir, or "
                f'{{"from_call": "<earlier label>"}} to use an earlier call\'s output'
            )
    if str(values.get("prompt", "") or "").strip() and not _accepts(spec, "text"):
        problems.append(f"{where}: {spec.id} takes no prompt; remove 'prompt'")
    try:
        width, height = call.render_size
        constraints.validate_dimensions(width, height)
        constraints.validate_dimension_limit(width, name="width")
        constraints.validate_dimension_limit(height, name="height")
    except Exception as exc:  # noqa: BLE001
        problems.append(f"{where}: {exc}")
    try:
        if not call.value("auto_duration"):
            constraints.validate_num_frames(int(call.value("num_frames")))
            constraints.validate_num_frames_limit(int(call.value("num_frames")))
        else:
            if float(call.value("min_seconds")) >= float(call.value("max_seconds")):
                problems.append(f"{where}: max_seconds must be greater than min_seconds")
            constraints.validate_seconds_limit(float(call.value("max_seconds")))
    except Exception as exc:  # noqa: BLE001
        problems.append(f"{where}: {exc}")
    if len(problems) > before:
        return call
    chained_audio = "audio" in media and media["audio"].chained
    if chained_audio:
        # Its length is only known once the earlier call has produced it; checked again live.
        return call
    try:
        frames, notes = app_gate(call, {k: str(v.path) if not v.chained else f"<from {v.from_call}>"
                                        for k, v in media.items()})
    except Exception as exc:  # noqa: BLE001 -- the app's own words
        problems.append(f"{where}: the app would refuse this call before reserving a GPU: {exc}")
        return call
    return replace(call, app_frames=frames, app_notes=notes)


def _app_request(call: Call, media_paths: dict[str, str]):
    """The ``GenerationRequest`` the Space's handler would build from this call's fields."""
    from ltx25_studio import generate

    known = {f.name for f in dataclass_fields(generate.GenerationRequest)}
    width, height = call.render_size
    kwargs: dict = {"mode": call.mode}
    for spec in api_fields():
        if spec.key not in known:
            continue
        if spec.kind in MEDIA_KINDS:
            kwargs[spec.key] = media_paths.get(spec.kind)
        elif spec.key in ("width", "height"):
            kwargs[spec.key] = width if spec.key == "width" else height
        elif spec.kind == "dataframe":
            kwargs[spec.key] = tuple(call.value(spec.key) or ())
        else:
            kwargs[spec.key] = call.value(spec.key)
    return generate.GenerationRequest(**kwargs)


def app_gate(call: Call, media_paths: dict[str, str]) -> tuple[int | None, tuple[str, ...]]:
    """``generate.prepare_request``, the checks the app runs before it reserves a GPU.

    Raises what the Space would refuse the call with. Returns the frame count the run will use
    (``None`` when the model chooses) and the notes the run will carry.
    """
    from ltx25_studio import generate

    scoped, _spec, notes = generate.prepare_request(_app_request(call, media_paths))
    frames = None if scoped.auto_duration else int(scoped.num_frames)
    return frames, tuple(str(note) for note in notes)


def _parse_media(kind, value, *, base, where, earlier, problems) -> MediaInput | None:
    if isinstance(value, str):
        path = _resolve_path(value, base)
        if not path.is_file():
            problems.append(f"{where}: {kind} file {path} does not exist")
            return None
        return MediaInput(kind=kind, path=path)
    if isinstance(value, dict) and "from_call" in value:
        source = value["from_call"]
        extra = set(value) - {"from_call", "frame"}
        if extra:
            problems.append(f"{where}: {kind}: unknown key(s) {sorted(extra)}")
        producer = earlier.get(source)
        if producer is None:
            problems.append(f"{where}: {kind} from_call {source!r} is not an earlier call in this plan")
            return None
        track = "audio" if kind == "audio" else "video"
        if track not in producer.exports:
            problems.append(f"{where}: {kind} from_call {source!r}: {producer.mode} exports no {track}")
            return None
        frame = value.get("frame", "middle") if kind == "image" else None
        if kind == "image" and not (isinstance(frame, int) and not isinstance(frame, bool) and frame >= 0
                                    or frame in ("middle", "last")):
            problems.append(f"{where}: image frame must be an index >= 0, 'middle' or 'last'")
            return None
        if kind != "image" and "frame" in value:
            problems.append(f"{where}: 'frame' applies to an image chain only")
        return MediaInput(kind=kind, from_call=source, frame=frame)
    problems.append(f"{where}: {kind} must be a file path or {{\"from_call\": \"<label>\"}}")
    return None


def _parse_lora(entry, *, base, where, problems) -> LoraInput | None:
    if isinstance(entry, str):
        entry = {"source": entry}
    if not isinstance(entry, dict) or not isinstance(entry.get("source"), str):
        problems.append(f"{where}: each LoRA needs a 'source' (a Hub repo id or a local .safetensors)")
        return None
    extra = set(entry) - {"source", "weight"}
    if extra:
        problems.append(f"{where}: LoRA: unknown key(s) {sorted(extra)}")
    weight = entry.get("weight", 1.0)
    if isinstance(weight, bool) or not isinstance(weight, (int, float)) or weight <= 0:
        problems.append(f"{where}: LoRA weight must be a positive number")
        return None
    source = entry["source"].strip()
    candidate = _resolve_path(source, base)
    if source.endswith(".safetensors") and candidate.is_file():
        return LoraInput(source=source, weight=float(weight), local=candidate)
    if _HUB_ID.match(source) and not source.startswith((".", "/", "~")):
        return LoraInput(source=source, weight=float(weight))
    problems.append(f"{where}: LoRA {source!r} is neither a file under --assets-dir nor a Hub repo id")
    return None


def parse_plan(document: Any, *, base: Path) -> list[Call]:
    problems: list[str] = []
    defaults: dict = {}
    if isinstance(document, dict):
        unknown = set(document) - {"defaults", "calls", "note"}
        if unknown:
            problems.append(f"plan: unknown top-level key(s) {sorted(unknown)}")
        defaults = document.get("defaults") or {}
        entries = document.get("calls")
        if not isinstance(defaults, dict):
            problems.append("plan: 'defaults' must be an object")
            defaults = {}
    else:
        entries = document
    if not isinstance(entries, list) or not entries:
        raise PlanError("the plan must be a non-empty list of calls (or {'defaults': {...}, 'calls': [...]})")
    calls: dict[str, Call] = {}
    for index, raw in enumerate(entries):
        merged = {**defaults, **raw} if isinstance(raw, dict) else raw
        if isinstance(raw, dict) and isinstance(defaults.get("expect"), dict) and isinstance(raw.get("expect"), dict):
            merged["expect"] = {**defaults["expect"], **raw["expect"]}
        call = _parse_call(merged, index=index, base=base, earlier=calls, problems=problems)
        if call is not None:
            calls[call.label] = call
    if problems:
        raise PlanError("\n".join(problems))
    return list(calls.values())


def load_plan(path: str | Path, *, assets_dir: Path | None = None) -> list[Call]:
    plan_path = Path(path).expanduser()
    try:
        document = json.loads(plan_path.read_text())
    except (OSError, ValueError) as exc:
        raise PlanError(f"{plan_path}: {exc}") from None
    return parse_plan(document, base=assets_dir or plan_path.resolve().parent)


def default_plan(mode_ids=None, overrides: dict | None = None) -> list[Call]:
    """The historical 11-mode loop, with every chain written out."""
    chosen = list(DEFAULT_MODE_ORDER if mode_ids is None else mode_ids)
    entries: list[dict] = []
    labels: dict[str, int] = {}
    producers: dict[str, str] = {}
    problems: list[str] = []
    for mode_id in chosen:
        try:
            spec = modes.get_mode(mode_id)
        except Exception as exc:  # noqa: BLE001
            raise PlanError(str(exc)) from None
        labels[mode_id] = labels.get(mode_id, 0) + 1
        label = mode_id if labels[mode_id] == 1 else f"{mode_id}-{labels[mode_id]}"
        entry: dict = {"label": label, "mode": spec.id, "width": CHEAPEST_SIZE, "height": CHEAPEST_SIZE,
                       "num_frames": CHEAPEST_FRAMES, "seed": SEED}
        if _accepts(spec, "text"):
            entry["prompt"] = PROMPT
        entry.update(overrides or {})
        if not _accepts(spec, "text"):
            entry.pop("prompt", None)
        for kind in MEDIA_KINDS:
            if not _requires(spec, kind):
                continue
            track = "audio" if kind == "audio" else "video"
            source = producers.get(track)
            if source is None:
                problems.append(f"{label} needs a {kind}, and nothing before it in this plan produces {track}")
                continue
            entry[kind] = {"from_call": source, "frame": "middle"} if kind == "image" else {"from_call": source}
        for track in spec.exports:
            producers.setdefault(track, label)
        entries.append(entry)
    if problems:
        raise PlanError("\n".join(problems))
    return parse_plan(entries, base=Path.cwd())


# ---------------------------------------------------------------------------
# what a call costs
# ---------------------------------------------------------------------------


def quota_multiplier(size: str = DEFAULT_ZEROGPU_SIZE) -> float:
    """The app's own quota factor for the Space's GPU size."""
    return float(zerogpu.quota_multiplier(size))


def reservation_seconds(call: Call, *, base_seconds: int) -> int:
    """The app's own estimator, at the Space's base duration."""
    return int(
        zerogpu.estimate_duration(
            gpu_duration_seconds=int(base_seconds),
            enable_stage_2=call.stage_2,
            loras=[lora.describe() for lora in call.loras] or None,
            enhance_prompt=bool(call.value("enhance_prompt")),
        )
    )


def quota_refusal(*texts: Any) -> str | None:
    """The first platform quota-refusal wording in ``texts``, or ``None``."""
    for text in texts:
        if not text:
            continue
        found = _QUOTA_REFUSAL.search(html.unescape(str(text)))
        if found:
            return found.group(0)
    return None


# ---------------------------------------------------------------------------
# the ledger: one JSON record per line, append-only, keyed by UTC day
# ---------------------------------------------------------------------------


class Ledger:
    """Append-only spend records. The day's total is the sum of ``billed_conservative``."""

    SCHEMA = 2
    UNIT = "GPU-seconds: an UPPER BOUND from wall clock x size multiplier, NOT billed seconds"

    def __init__(self, path: str | Path):
        self.path = Path(path).expanduser()

    def records(self) -> list[dict]:
        if not self.path.exists():
            return []
        try:
            text = self.path.read_text()
        except OSError as exc:
            raise QuotaLedgerUnreadable(f"{self.path}: {exc}") from None
        records = []
        for number, line in enumerate(text.splitlines(), start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                if not isinstance(record, dict) or "kind" not in record or "utc_day" not in record:
                    raise ValueError("not a ledger record")
                if record["kind"] == "call":
                    float(record["billed_conservative"])
            except (ValueError, KeyError, TypeError) as exc:
                raise QuotaLedgerUnreadable(
                    f"{self.path} line {number} is not a ledger record ({exc}). Refusing to read it "
                    f"as zero spend: a damaged ledger is not evidence that nothing was spent. This "
                    f"version writes one JSON record per line; fix or move the file by hand."
                ) from None
            records.append(record)
        return records

    def spent(self, day: str) -> float:
        return sum(
            float(r["billed_conservative"]) for r in self.records() if r["kind"] == "call" and r["utc_day"] == day
        )

    def previous_day(self, day: str) -> tuple[str, float] | None:
        days: dict[str, float] = {}
        for record in self.records():
            if record["kind"] == "call" and record["utc_day"] < day:
                days[record["utc_day"]] = days.get(record["utc_day"], 0.0) + float(record["billed_conservative"])
        if not days:
            return None
        last = max(days)
        return last, days[last]

    def append(self, record: dict) -> None:
        line = redact(json.dumps({"schema": self.SCHEMA, **record}, sort_keys=True)) + "\n"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        descriptor = os.open(self.path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600)
        try:
            os.write(descriptor, line.encode())
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


# ---------------------------------------------------------------------------
# the live contract
# ---------------------------------------------------------------------------


@dataclass
class Endpoint:
    name: str
    fn_index: int
    input_types: list[str]
    parameters: list[dict]
    returns: list[dict]


def find_endpoint(config: dict, info: dict, name: str) -> Endpoint | None:
    published = (info.get("named_endpoints") or {}).get(name)
    dependency = next(
        (d for d in config.get("dependencies", []) if "/" + str(d.get("api_name") or "") == name), None
    )
    if published is None or dependency is None:
        return None
    types = {c["id"]: str(c.get("type", "")).lower() for c in config.get("components", [])}
    return Endpoint(
        name=name,
        fn_index=int(dependency["id"]),
        input_types=[types.get(i, "?") for i in dependency.get("inputs", [])],
        parameters=list(published.get("parameters") or []),
        returns=list(published.get("returns") or []),
    )


def _expected_component(kind: str) -> str:
    return _COMPONENT_FOR_KIND.get(kind, kind)


def generate_contract_problems(parameters: list[dict], *, input_types: list[str] | None = None,
                               plan_modes=()) -> tuple[list[str], list[str]]:
    """``(problems, table)`` comparing ``/on_generate``'s published signature with the field spec."""
    expected = [("mode", None, None)] + [(f.key, f.kind, f.label) for f in api_fields()]
    problems: list[str] = []
    table = [f"  {'#':>2}  {'expected':<44} live"]
    for slot in range(max(len(expected), len(parameters))):
        want = expected[slot] if slot < len(expected) else None
        have = parameters[slot] if slot < len(parameters) else None
        live = "-" if have is None else f"{have.get('component')} {have.get('label')!r}"
        if want is None:
            problems.append(f"slot {slot}: the Space publishes {live} that the field spec does not have")
            table.append(f"  {slot:>2}  {'-':<44} {live}  ✗")
            continue
        key, kind, label = want
        shown = "mode (a choice)" if kind is None else f"{key} ({_expected_component(kind)} {label!r})"
        mark = "✓"
        if have is None:
            problems.append(f"slot {slot}: the field spec sends {key!r} and the Space has no parameter there")
            mark = "✗"
        elif kind is None:
            if str(have.get("component", "")).lower() not in ("radio", "dropdown"):
                problems.append(f"slot 0 should be the mode selector, the Space has {live}")
                mark = "✗"
        else:
            component = str(have.get("component", "")).lower()
            if component != _expected_component(kind):
                problems.append(f"slot {slot}: {key!r} is a {kind}; the Space has {live}")
                mark = "✗"
            elif label and have.get("label") and not str(have.get("label")).strip().startswith(str(label).strip()):
                # A prefix, because the UI may add detail to a label (stage 2 says what size it returns).
                # Labels are the only thing that tells two same-kind neighbours (width, height) apart.
                problems.append(f"slot {slot}: {key!r} is labelled {label!r} here; the Space has {live}")
                mark = "✗"
        table.append(f"  {slot:>2}  {shown:<44} {live}  {mark}")
    if parameters:
        choices = (parameters[0].get("type") or {}).get("enum")
        if choices:
            for mode_id in plan_modes:
                if mode_id not in choices:
                    problems.append(f"the Space does not offer mode {mode_id!r} (it offers {choices})")
    if input_types is not None:
        want_types = ["mode", *(f.kind for f in fields.FIELDS)]
        if len(input_types) != len(want_types):
            problems.append(
                f"{ENDPOINT} takes {len(input_types)} inputs on the Space; the field spec builds {len(want_types)}"
            )
        else:
            want_state = [i for i, kind in enumerate(want_types) if kind == "state"]
            have_state = [i for i, kind in enumerate(input_types) if kind == "state"]
            if want_state != have_state:
                problems.append(f"{ENDPOINT}'s State inputs sit at {have_state} on the Space, {want_state} here")
    return problems, table


@dataclass
class LoraRoute:
    endpoint: Endpoint
    source_slot: int | None
    upload_slot: int | None
    weight_slot: int | None
    weight_endpoint: Endpoint | None = None


_SOURCE_NAMES = {"repo_id", "repo", "source", "hub_id", "repo_or_url"}


def _lora_route(config: dict, info: dict) -> LoraRoute | None:
    endpoint = find_endpoint(config, info, LORA_ENDPOINT)
    if endpoint is None:
        return None
    source = upload = weight = None
    for slot, parameter in enumerate(endpoint.parameters):
        name = str(parameter.get("parameter_name") or "").lower()
        component = str(parameter.get("component", "")).lower()
        label = str(parameter.get("label") or "").lower()
        if weight is None and component in ("slider", "number") and ("weight" in name or "strength" in name
                                                                      or "strength" in label):
            weight = slot
        elif upload is None and (component in ("file", "uploadbutton") or "upload" in name):
            upload = slot
        elif source is None and component == "textbox" and (name in _SOURCE_NAMES or "repo" in label):
            source = slot
    weight_endpoint = find_endpoint(config, info, LORA_WEIGHT_ENDPOINT)
    if weight_endpoint is not None:
        names = [str(p.get("parameter_name") or "").lower() for p in weight_endpoint.parameters]
        if "index" not in names or "weight" not in names:
            weight_endpoint = None
    return LoraRoute(endpoint, source, upload, weight, weight_endpoint)


def lora_contract_problems(route: LoraRoute | None, calls: list[Call]) -> list[str]:
    problems: list[str] = []
    with_loras = [c for c in calls if c.loras]
    if not with_loras:
        return problems
    if route is None:
        return [f"{', '.join(c.label for c in with_loras)} stage LoRAs, and the Space publishes no {LORA_ENDPOINT}"]
    for call in with_loras:
        for lora in call.loras:
            if lora.local is not None and route.upload_slot is None:
                problems.append(f"{call.label}: {LORA_ENDPOINT} has no upload parameter for {lora.source}")
            if lora.local is None and route.source_slot is None:
                problems.append(f"{call.label}: {LORA_ENDPOINT} has no Hub repo id parameter for {lora.source}")
            if lora.weight != 1.0 and route.weight_slot is None and route.weight_endpoint is None:
                names = [p.get("parameter_name") for p in route.endpoint.parameters]
                problems.append(
                    f"{call.label}: {lora.source} asks for strength {lora.weight:g}, and the Space exposes no "
                    f"public way to set an adapter's strength ({LORA_ENDPOINT} takes {names}; there is no "
                    f"{LORA_WEIGHT_ENDPOINT} with 'index' and 'weight'). Drop the weight (the adapter runs at "
                    f"its staged default) or publish that route."
                )
    return problems


def build_data(input_types: list[str], values: list) -> list:
    """Values for the non-State inputs in order, ``None`` in every State slot."""
    remaining = list(values)
    data = []
    for kind in input_types:
        if kind == "state":
            data.append(None)
        else:
            if not remaining:
                raise ContractMismatch(f"{len(values)} values for {input_types}")
            data.append(remaining.pop(0))
    if remaining:
        raise ContractMismatch(f"{len(values)} values for {input_types}")
    return data


def resolved_request(call: Call) -> dict:
    """Every non-state field by name, as it will be sent (media described, not uploaded)."""
    out: dict = {}
    width, height = call.render_size
    for spec in api_fields():
        if spec.kind in MEDIA_KINDS:
            item = call.media.get(spec.kind)
            out[spec.key] = None if item is None else item.describe()
        elif spec.key == "width":
            out[spec.key] = width
        elif spec.key == "height":
            out[spec.key] = height
        else:
            out[spec.key] = call.value(spec.key)
    return out


def generate_values(call: Call, uploaded: dict) -> list:
    values: list = [call.mode]
    request = resolved_request(call)
    for spec in api_fields():
        if spec.kind in MEDIA_KINDS:
            values.append(uploaded.get(spec.kind))
        elif spec.kind == "dataframe":
            values.append({"headers": list(spec.options.get("headers") or []), "data": request[spec.key] or [],
                           "metadata": None})
        else:
            values.append(request[spec.key])
    return values


# ---------------------------------------------------------------------------
# measurements (reported with thresholds; none of them stops a plan)
# ---------------------------------------------------------------------------


def _decode_video_frames(path, *, limit=512):
    import av

    frames = []
    with av.open(str(path)) as container:
        if not container.streams.video:
            return frames
        for frame in container.decode(video=0):
            frames.append(frame.to_ndarray(format="rgb24"))
            if len(frames) >= limit:
                break
    return frames


def _check(name, value, threshold, verdict, detail=""):
    return {"name": name, "value": value, "threshold": threshold, "verdict": verdict, "detail": detail}


def _verbatim(value: Any) -> Any:
    """Any value as JSON-able data, field for field, without interpreting it."""
    import dataclasses
    import enum

    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {f.name: _verbatim(getattr(value, f.name)) for f in dataclasses.fields(value)}
    if isinstance(value, enum.Enum):
        return _verbatim(value.value)
    if isinstance(value, dict):
        return {str(k): _verbatim(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_verbatim(v) for v in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:  # noqa: BLE001
            pass
    if hasattr(value, "__dict__"):
        return {k: _verbatim(v) for k, v in vars(value).items() if not k.startswith("_")}
    return str(value)


def _structure_check_today(frames) -> dict:
    structure = outputs.measure_structure(frames)
    failure = outputs.structure_failure(structure)
    return {
        "check": "outputs.structure_failure",
        "measurements": structure,
        "conditions": [failure] if failure else [],
        "notes": outputs.structure_warnings(structure),
    }


def picture_health(frames) -> Any:
    """The app's own picture-health check, recorded verbatim. Nothing here judges it.

    The single call site: when ``outputs``' recalibrated check lands, this return line is the
    only one to change (to ``_verbatim(outputs.<its check>(frames))``).
    """
    return _verbatim(_structure_check_today(frames))


def fired_conditions(health: Any) -> list[str]:
    conditions = health.get("conditions") if isinstance(health, dict) else None
    return [str(c) for c in conditions or [] if c]


def measure_video(path) -> dict:
    """Frame statistics (this harness's own thresholds) and the app's picture-health record."""
    import numpy as np

    try:
        frames = _decode_video_frames(path)
    except Exception as exc:  # noqa: BLE001
        return {"decoded_frames": 0, "checks": [_check("decodes", 0, "> 0 frames", "fail", f"{type(exc).__name__}: {exc}")]}
    if not frames:
        return {"decoded_frames": 0, "checks": [_check("decodes", 0, "> 0 frames", "fail", "no video frames")]}
    stack = np.stack(frames).astype("float32")
    deltas = [float(np.abs(stack[i + 1] - stack[i]).mean()) for i in range(len(frames) - 1)] or [0.0]
    spatial_std = float(np.median([frame.std() for frame in stack]))
    mean_delta = float(sum(deltas) / len(deltas))
    relative = None if spatial_std < MIN_PIXEL_STD else mean_delta / spatial_std
    distinct = len({frame.tobytes() for frame in frames})
    checks = [
        _check("flatness", round(spatial_std, 4), f">= {MIN_PIXEL_STD} median per-frame std",
               "pass" if spatial_std >= MIN_PIXEL_STD else "fail"),
        _check("distinct_frames", distinct, f">= {MIN_DISTINCT_FRAMES}",
               "pass" if distinct >= MIN_DISTINCT_FRAMES or len(frames) == 1 else "fail"),
        _check("freeze_floor", None if relative is None else round(relative, 6),
               f">= {MIN_RELATIVE_FRAME_DELTA} mean |delta| / spatial std",
               "n/a" if relative is None or len(frames) < 2 else
               ("pass" if relative >= MIN_RELATIVE_FRAME_DELTA else "fail")),
        _check("motion", round(mean_delta, 4), f">= {LOW_MEAN_FRAME_DELTA} mean |delta| (warn only)",
               "pass" if mean_delta >= LOW_MEAN_FRAME_DELTA else "warn",
               "a still-conditioned mode is expected to land low"),
    ]
    return {
        "decoded_frames": len(frames),
        "distinct_frames": distinct,
        "spatial_std": spatial_std,
        "stack_std": float(stack.std()),
        "mean_abs_delta": mean_delta,
        "min_abs_delta": min(deltas),
        "max_abs_delta": max(deltas),
        "relative_delta": relative,
        "checks": checks,
        "picture_health": picture_health(frames),
    }


#: Audio measurements. Reported, never judged: only digital silence fails a call.
SILENT_DBFS = -70.0
SILENCE_WINDOW_SECONDS = 0.02
SILENCE_WINDOW_DBFS = -60.0
CLIP_LEVEL = 0.999
SPECTRUM_FFT = 2048
SPECTRUM_MAX_FRAMES = 4096
ENERGY_BANDS_HZ = ((0, 250), (250, 2000), (2000, 6000), (6000, None))


def _dbfs(rms: float) -> float:
    import math

    return 20.0 * math.log10(rms + 1e-12)


def _decode_audio_track(path, index: int):
    """One audio stream as float32 ``[channels, samples]`` and its rate."""
    import av
    import numpy as np

    with av.open(str(path)) as container:
        stream = container.streams.audio[index]
        layout = getattr(getattr(stream, "layout", None), "name", None) or "stereo"
        resampler = av.AudioResampler(format="fltp", layout=layout, rate=int(stream.rate))
        blocks = []
        for frame in container.decode(stream):
            for out in resampler.resample(frame):
                blocks.append(out.to_ndarray())
        for out in resampler.resample(None):
            blocks.append(out.to_ndarray())
        rate = int(stream.rate)
    if not blocks:
        return np.zeros((1, 0), dtype="float32"), rate
    return np.concatenate(blocks, axis=-1).astype("float32"), rate


def _spectral(mono, rate: int) -> dict:
    import numpy as np

    if mono.size < SPECTRUM_FFT:
        mono = np.pad(mono, (0, SPECTRUM_FFT - mono.size))
    hop = SPECTRUM_FFT // 2
    count = 1 + (mono.size - SPECTRUM_FFT) // hop
    starts = hop * np.unique(np.linspace(0, count - 1, min(count, SPECTRUM_MAX_FRAMES)).astype(int))
    frames = mono[starts[:, None] + np.arange(SPECTRUM_FFT)[None, :]] * np.hanning(SPECTRUM_FFT)
    power = (np.abs(np.fft.rfft(frames, axis=1)) ** 2)[:, 1:]
    freqs = np.fft.rfftfreq(SPECTRUM_FFT, 1.0 / rate)[1:]
    mean = power.mean(axis=0)
    total = float(mean.sum())
    if total <= 0.0:
        return {"spectral_flatness": None, "spectral_flatness_frame_median": None,
                "spectral_centroid_hz": None, "band_energy_share": None}
    tiny = 1e-20

    def flatness(spectrum):
        return float(np.exp(np.mean(np.log(spectrum + tiny))) / (np.mean(spectrum) + tiny))

    live = power[power.sum(axis=1) > total * 1e-6]
    bands = {}
    for low, high in ENERGY_BANDS_HZ:
        mask = (freqs >= low) & (freqs < high) if high else (freqs >= low)
        bands[f"{low}-{high or 'nyquist'}"] = round(float(mean[mask].sum()) / total, 4)
    return {
        "spectral_flatness": round(flatness(mean), 4),
        "spectral_flatness_frame_median": round(float(np.median([flatness(f) for f in live])), 4) if len(live) else None,
        "spectral_centroid_hz": round(float((freqs * mean).sum() / total), 1),
        "band_energy_share": bands,
    }


def measure_audio(path) -> dict:
    """Every audio stream of the file, as measurements. No verdicts."""
    import av
    import numpy as np

    try:
        with av.open(str(path)) as container:
            count = len(container.streams.audio)
    except Exception as exc:  # noqa: BLE001
        return {"tracks": [], "error": f"{type(exc).__name__}: {exc}"}
    tracks = []
    for index in range(count):
        try:
            samples, rate = _decode_audio_track(path, index)
        except Exception as exc:  # noqa: BLE001
            tracks.append({"index": index, "error": f"{type(exc).__name__}: {exc}"})
            continue
        channels, length = samples.shape
        if length == 0:
            tracks.append({"index": index, "sample_rate": rate, "channels": channels, "samples": 0,
                           "level_dbfs": _dbfs(0.0)})
            continue
        rms = float(np.sqrt(np.mean(samples.astype("float64") ** 2)))
        peak = float(np.abs(samples).max())
        window = max(1, int(round(SILENCE_WINDOW_SECONDS * rate)))
        usable = (length // window) * window
        if usable:
            windows = samples[:, :usable].reshape(channels, -1, window)
            window_rms = np.sqrt(np.mean(windows.astype("float64") ** 2, axis=(0, 2)))
            silence = float(np.mean(20.0 * np.log10(window_rms + 1e-12) < SILENCE_WINDOW_DBFS))
        else:
            silence = None
        clipped = int(np.count_nonzero(np.abs(samples) >= CLIP_LEVEL))
        tracks.append({
            "index": index,
            "sample_rate": rate,
            "channels": channels,
            "samples": length,
            "seconds": round(length / rate, 4),
            "level_dbfs": round(_dbfs(rms), 2),
            "rms": rms,
            "peak": peak,
            "peak_dbfs": round(_dbfs(peak), 2),
            "clipped_samples": clipped,
            "clipped_ratio": clipped / float(samples.size),
            "silence_ratio": silence,
            **_spectral(samples.mean(axis=0), rate),
        })
    return {
        "tracks": tracks,
        "definitions": {
            "level_dbfs": "RMS over every sample of every channel",
            "clipped_samples": f"samples with |x| >= {CLIP_LEVEL} of full scale",
            "silence_ratio": f"share of {SILENCE_WINDOW_SECONDS * 1000:g} ms windows below {SILENCE_WINDOW_DBFS:g} dBFS",
            "spectral_flatness": "geometric / arithmetic mean of the mean power spectrum (about 1 for noise, "
                                 "far below 1 for speech or music)",
            "spectral_flatness_frame_median": f"the same per {SPECTRUM_FFT}-sample frame, median over frames",
            "spectral_centroid_hz": "power-weighted mean frequency",
            "band_energy_share": "share of spectral energy per band, Hz",
        },
        "note": "Measurements only: nothing here is a verdict. Only digital silence fails a call.",
    }


def _audio_line(measured: dict) -> str:
    parts = []
    for track in measured.get("tracks", []):
        if "error" in track:
            parts.append(f"track {track['index']}: {track['error']}")
            continue
        flat = track.get("spectral_flatness")
        centroid = track.get("spectral_centroid_hz")
        silence = track.get("silence_ratio")
        parts.append(
            f"{track['level_dbfs']:.1f} dBFS · peak {track.get('peak_dbfs', 0):.1f} · clipped "
            f"{track.get('clipped_samples', 0)} · silence {0 if silence is None else 100 * silence:.0f}% · flatness "
            f"{'n/a' if flat is None else f'{flat:.3f}'} · centroid "
            f"{'n/a' if centroid is None else f'{centroid:.0f} Hz'}"
        )
    return " | ".join(parts)


def _count_frames(path) -> int | None:
    try:
        return len(_decode_video_frames(path, limit=100_000))
    except Exception:  # noqa: BLE001
        return None


def verify_call(call: Call, artifacts: dict[str, Path], *, sampling_rate: int | None) -> dict:
    """Stream layout, geometry and frame count against the call's expectation, plus measurements."""
    problems: list[str] = []
    streams = []
    rate = call.expect.get("sample_rate", sampling_rate)
    geometry = call.expected_geometry()
    for path in dict.fromkeys(artifacts.values()):
        has_video = "video" in call.exports and artifacts.get("video") == path
        verdict = verify_streams.verify(
            call.mode,
            path,
            frame_rate=call.expect.get("frame_rate"),
            sampling_rate=rate if "audio" in call.exports else None,
            geometry=geometry if has_video else None,
        )
        streams.append({"path": str(path), "ok": verdict.ok, "probed": verdict.probed,
                        "expected": verdict.expected, "actual": verdict.actual, "problems": verdict.problems})
        problems += [f"{Path(path).name}: {p}" for p in verdict.problems]
    probe = streams[0]["actual"] if streams else {}
    for track in ("video", "audio"):
        wanted = call.expect.get(f"has_{track}")
        if wanted is not None and bool(probe.get(track)) != bool(wanted):
            problems.append(f"expected has_{track}={wanted}, the artifact has {track}={bool(probe.get(track))}")
    frames = None
    if "video" in artifacts:
        frames = _count_frames(artifacts["video"])
        wanted_frames = call.expected_frames()
        if wanted_frames is not None and frames != wanted_frames:
            problems.append(f"decoded {frames} frames, expected {wanted_frames}")
    content: dict = {}
    if "video" in artifacts:
        content["video"] = measure_video(artifacts["video"])
    if "audio" in artifacts:
        content["audio"] = measure_audio(artifacts["audio"])
        for track in content["audio"].get("tracks", []):
            level = track.get("level_dbfs")
            if level is not None and level < SILENT_DBFS:
                problems.append(f"audio track {track['index']} is digital silence: {level:.1f} dBFS < {SILENT_DBFS:g}")
    return {
        "ok": not problems,
        "problems": problems,
        "expected": {"geometry": list(geometry) if geometry else None, "frames": call.expected_frames(),
                     "sample_rate": rate if "audio" in call.exports else None,
                     **{k: v for k, v in call.expect.items() if k.startswith("has_")}},
        "decoded_frames": frames,
        "streams": streams,
        "content": content,
        "content_note": (
            "Content figures are measurements. The frame checks carry this harness's thresholds, the "
            "picture-health record is the app's own check verbatim, and audio has no verdict. None of "
            "them stops a plan; no artifact is a pass until a frame has been looked at."
        ),
    }


def _flags(verify: dict) -> list[str]:
    flagged = []
    content = verify.get("content") or {}
    for check in (content.get("video") or {}).get("checks", []):
        if check["verdict"] in ("fail", "warn"):
            flagged.append(f"video.{check['name']}={check['verdict']}")
    for condition in fired_conditions((content.get("video") or {}).get("picture_health")):
        flagged.append(f"picture: {condition}")
    return flagged


# ---------------------------------------------------------------------------
# chained inputs
# ---------------------------------------------------------------------------


def extract_image(video: Path, dest: Path, frame) -> int:
    from PIL import Image

    frames = _decode_video_frames(video, limit=100_000)
    if not frames:
        raise RuntimeError(f"{video} has no frames to take an image from")
    index = {"middle": len(frames) // 2, "last": len(frames) - 1}.get(frame, frame)
    if not isinstance(index, int) or not 0 <= index < len(frames):
        raise RuntimeError(f"{video} has {len(frames)} frames; frame {frame!r} does not exist")
    dest.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(frames[index]).save(str(dest))
    return index


def extract_audio(source: Path, dest: Path) -> Path:
    import av

    dest.parent.mkdir(parents=True, exist_ok=True)
    with av.open(str(source)) as probe:
        if not probe.streams.audio:
            raise RuntimeError(f"{source} has no audio stream")
        stream = probe.streams.audio[0]
        already = probe.format.name in ("wav", "wave") and stream.codec_context.name.startswith("pcm")
    if already:
        shutil.copyfile(source, dest)
        return dest
    with av.open(str(source)) as container, av.open(str(dest), "w", format="wav") as out:
        stream = container.streams.audio[0]
        layout = getattr(getattr(stream, "layout", None), "name", None) or "stereo"
        out_stream = out.add_stream("pcm_s16le", rate=int(stream.rate), layout=layout)
        for frame in container.decode(stream):
            frame.pts = None
            for packet in out_stream.encode(frame):
                out.mux(packet)
        for packet in out_stream.encode():
            out.mux(packet)
    return dest


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


# ---------------------------------------------------------------------------
# one live call
# ---------------------------------------------------------------------------


class Evidence:
    """``--out/<label>/``: every file written here is redacted."""

    def __init__(self, directory: Path):
        self.dir = directory
        self.dir.mkdir(parents=True, exist_ok=True)
        self._events = (self.dir / "events.jsonl").open("a", encoding="utf-8")

    def event(self, record: dict) -> None:
        self._events.write(redact(json.dumps(record, sort_keys=True, default=str)) + "\n")
        self._events.flush()

    def write(self, name: str, payload: Any) -> Path:
        path = self.dir / name
        path.write_text(_redacted_json(payload))
        return path

    def close(self) -> None:
        self._events.close()


@dataclass
class Outcome:
    label: str
    status: str  # ok | mismatch | failed | timeout | quota | skipped
    detail: str = ""
    category: str | None = None
    billed_conservative: float = 0.0
    reserved: int = 0
    artifacts: dict = field(default_factory=dict)
    flags: list = field(default_factory=list)
    audio: str = ""

    @property
    def failed(self) -> bool:
        return self.status in ("mismatch", "failed", "timeout")


@dataclass
class Session:
    """Everything one live invocation shares."""

    app: SpaceApp
    generate: Endpoint
    lora: LoraRoute | None
    ledger: Ledger
    out: Path
    log: Any
    invocation: str
    ceiling: float
    base_seconds: int
    multiplier: float
    call_timeout: float
    read_timeout: float
    sampling_rate: int
    space_sha: str | None
    produced: dict = field(default_factory=dict)


class CeilingReached(RuntimeError):
    pass


class QuotaRefused(RuntimeError):
    pass


def _find_record(data: list, status: str) -> dict | None:
    """A structured run record: an extra dict output, or JSON embedded in the status."""
    for extra in data[3:]:
        if isinstance(extra, dict):
            return extra
    found = re.search(r'<script[^>]*type="application/json"[^>]*>(.*?)</script>', status or "", re.S)
    if found is None:
        found = re.search(r"<!--\s*run-record\s*(\{.*?\})\s*-->", status or "", re.S)
    if found is None:
        return None
    try:
        record = json.loads(html.unescape(found.group(1)))
    except ValueError:
        return None
    return record if isinstance(record, dict) else None


def _record_value(record: dict | None, *names) -> Any:
    if not record:
        return None
    for name in names:
        value = record.get(name)
        if value is None and isinstance(record.get("failure"), dict):
            value = record["failure"].get(name)
        if value is not None:
            return value
    return None


#: The success report states the worker's own GPU time this way (``render.run_summary``).
_REPORTED_GPU_TIME = re.compile(r"<dt>Took</dt><dd>(?:<span[^>]*>)?([\d.]+)(?:</span>)? s, GPU time</dd>")


def _reported_gpu_seconds(status: str) -> float | None:
    found = _REPORTED_GPU_TIME.search(status or "")
    return float(found.group(1)) if found else None


def _category(record: dict | None, status: str) -> str | None:
    value = _record_value(record, "category")
    if value:
        return str(value)
    found = re.search(r'data-(?:failure-)?category="([\w-]+)"', status or "")
    return found.group(1) if found else None


def _plain(markdown: str) -> str:
    text = re.sub(r"<script.*?</script>", "", markdown or "", flags=re.S)
    text = re.sub(r"<[^>]+>", " ", text)
    return re.sub(r"[ \t]+", " ", html.unescape(text)).strip()


def _file_path(value) -> str | None:
    if isinstance(value, str):
        return value or None
    if isinstance(value, dict):
        if isinstance(value.get("path"), str):
            return value["path"]
        for key in ("video", "audio", "file"):
            inner = _file_path(value.get(key))
            if inner:
                return inner
    return None


def _run_events(ctx: Session, evidence: Evidence, endpoint: Endpoint, data: list, session_hash: str,
                started: float, deadline: float, timing: dict, key: str) -> dict:
    """Join one event and read its stream to completion. Records every message."""
    evidence.event({"t": round(_monotonic() - started, 4), "endpoint": endpoint.name, "client": "join"})
    event_id = ctx.app.join(endpoint.fn_index, data, session_hash)
    timing[f"{key}_event_id_t"] = round(_monotonic() - started, 4)
    evidence.event({"t": timing[f"{key}_event_id_t"], "endpoint": endpoint.name, "client": "event_id",
                    "event_id": event_id})
    try:
        with ctx.app.stream(session_hash, read_timeout=min(ctx.read_timeout, ctx.call_timeout)) as response:
            for message in iter_sse(response):
                now = _monotonic()
                t = round(now - started, 4)
                evidence.event({"t": t, "utc": _utcnow().isoformat(), "endpoint": endpoint.name, **message})
                kind = message.get("msg") if isinstance(message, dict) else None
                if kind == "process_starts":
                    timing.setdefault(f"{key}_process_starts_t", t)
                elif kind == "progress":
                    timing.setdefault(f"{key}_first_progress_t", t)
                    if key == "generate":
                        for unit in message.get("progress_data") or []:
                            ctx.log(f"    progress  {unit.get('desc') or ''} {unit.get('index')}/{unit.get('length')}"
                                    f" {unit.get('unit') or ''}".rstrip())
                elif kind == "log" and key == "generate":
                    ctx.log(f"    space     {message.get('title') or ''}: {message.get('log')}")
                elif kind in ("process_completed", "unexpected_error") and message.get("event_id") in (event_id, None):
                    timing[f"{key}_completed_t"] = t
                    return message
                elif kind == "close_stream":
                    break
                if now > deadline:
                    raise CallTimeout(f"no completion within {ctx.call_timeout:g} s")
    except TimeoutError:
        raise CallTimeout(f"the event stream went silent for {ctx.read_timeout:g} s") from None
    except TransportError as exc:
        if "timed out" in str(exc):
            raise CallTimeout(f"the event stream went silent for {ctx.read_timeout:g} s") from None
        raise
    raise TransportError("the event stream closed before the call completed")


def _validate_loras(ctx: Session, call: Call, evidence: Evidence, session_hash: str, started: float,
                    timing: dict) -> list[dict]:
    route = ctx.lora
    verdicts = []
    for index, lora in enumerate(call.loras):
        values: list = [None] * len(route.endpoint.parameters)
        if lora.local is not None:
            values[route.upload_slot] = ctx.app.upload(lora.local)
            if route.source_slot is not None:
                values[route.source_slot] = ""
        else:
            values[route.source_slot] = lora.source
        if route.weight_slot is not None:
            values[route.weight_slot] = lora.weight
        data = build_data(route.endpoint.input_types, values)
        deadline = _monotonic() + ctx.call_timeout
        message = _run_events(ctx, evidence, route.endpoint, data, session_hash, started, deadline, timing,
                              f"lora{index}")
        output = message.get("output") or {}
        texts = [str(v) for v in output.get("data") or [] if isinstance(v, str)]
        verdict = texts[0] if texts else str(output.get("error") or message.get("message") or "")
        staged_list = texts[1:]
        name = Path(lora.label).name
        staged = any(name in text for text in staged_list) if staged_list else None
        if message.get("success") is False or message.get("msg") == "unexpected_error":
            staged = False
        entry = {**lora.describe(), "verdict": verdict, "staged_list": staged_list, "staged": staged}
        if staged and lora.weight != 1.0 and route.weight_slot is None and route.weight_endpoint is not None:
            weights = {"index": index, "weight": lora.weight}
            names = [str(p.get("parameter_name") or "").lower() for p in route.weight_endpoint.parameters]
            data = build_data(route.weight_endpoint.input_types, [weights.get(n) for n in names])
            done = _run_events(ctx, evidence, route.weight_endpoint, data, session_hash, started,
                               _monotonic() + ctx.call_timeout, timing, f"lora{index}_weight")
            entry["weight_set"] = done.get("success", True) is not False
        verdicts.append(entry)
        ctx.log(f"    lora      {lora.label}: {'staged' if staged else 'NOT staged' if staged is False else 'unconfirmed'}"
                f" · {_plain(verdict)[:160]}")
        if staged is False:
            break
    return verdicts


def _resolve_media(ctx: Session, call: Call, evidence: Evidence) -> dict[str, dict]:
    resolved: dict[str, dict] = {}
    for kind, item in call.media.items():
        if not item.chained:
            resolved[kind] = {"local": item.path, "chained": False, "sha256": _sha256(item.path)}
            continue
        artifacts = ctx.produced.get(item.from_call)
        track = "audio" if kind == "audio" else "video"
        source = (artifacts or {}).get(track)
        if source is None:
            raise RuntimeError(f"{kind} is chained from {item.from_call!r}, which produced no {track}")
        inputs = evidence.dir / "inputs"
        if kind == "image":
            dest = inputs / "image.png"
            index = extract_image(Path(source), dest, item.frame)
        elif kind == "video":
            dest = inputs / f"video{Path(source).suffix or '.mp4'}"
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, dest)
            index = None
        else:
            dest = extract_audio(Path(source), inputs / "audio.wav")
            index = None
        resolved[kind] = {"local": dest, "chained": True, "from_call": item.from_call, "source_artifact": str(source),
                          "frame_index": index, "sha256": _sha256(dest)}
    return resolved


def run_call(ctx: Session, call: Call) -> Outcome:
    day = utc_day()
    reserved = reservation_seconds(call, base_seconds=ctx.base_seconds)
    worst = reserved * ctx.multiplier
    spent = ctx.ledger.spent(day)
    if spent + worst > ctx.ceiling:
        rollover = next_utc_midnight()
        left = max(0.0, ctx.ceiling - spent)
        raise CeilingReached(
            f"stopped before {call.label}: today (UTC {day}) has <= {spent:.1f} GPU-s on the ledger, and "
            f"this call reserves {reserved} s x {ctx.multiplier:g} = {worst:.1f}, which would pass the "
            f"{ctx.ceiling:.0f} GPU-s ceiling. {left:.1f} GPU-s are left today; the UTC day rolls over at "
            f"{rollover.isoformat().replace('+00:00', 'Z')} (in {_duration(rollover - _utcnow())}). Nothing was sent."
        )

    evidence = Evidence(ctx.out / call.label)
    started = _monotonic()
    timing: dict = {"reserved_seconds": reserved, "multiplier": ctx.multiplier, "worst_case_billed": worst}
    request = {
        "label": call.label,
        "mode": call.mode,
        "note": call.note,
        "fields": resolved_request(call),
        "loras": [lora.describe() for lora in call.loras],
        "expect": call.expect,
        "reservation_seconds": reserved,
        "space_sha": ctx.space_sha,
    }
    evidence.write("request.json", request)
    ctx.log(f"\n>>> {call.label}  {call.mode}  reserve {reserved} s (<= {worst:.0f} GPU-s worst case)")
    session_hash = uuid.uuid4().hex
    outcome = Outcome(label=call.label, status="failed", reserved=reserved)
    message: dict | None = None
    ledger_record = {"kind": "call", "utc_day": day, "invocation": ctx.invocation, "label": call.label,
                     "mode": call.mode, "reserved_seconds": reserved, "multiplier": ctx.multiplier,
                     "billed_conservative": 0.0, "billed_worker": None, "space_sha": ctx.space_sha}
    try:
        try:
            media = _resolve_media(ctx, call, evidence)
        except Exception as exc:  # noqa: BLE001
            request["media"] = {"error": redact(str(exc))}
            evidence.write("request.json", request)
            outcome.detail = f"input could not be prepared: {redact(exc)}"
            outcome.category = "input"
            ctx.ledger.append({**ledger_record, "at": _utcnow().isoformat(), "outcome": "failed",
                               "error": outcome.detail})
            return outcome
        request["media"] = {k: {kk: str(vv) if isinstance(vv, Path) else vv for kk, vv in v.items()}
                            for k, v in media.items()}
        try:
            frames, notes = app_gate(call, {k: str(v["local"]) for k, v in media.items()})
        except Exception as exc:  # noqa: BLE001 -- refused before dispatch: nothing billed
            outcome.detail = f"the app refuses this call before reserving a GPU: {redact(exc)}"
            outcome.category = "input"
            evidence.write("request.json", request)
            ctx.ledger.append({**ledger_record, "at": _utcnow().isoformat(), "outcome": "failed",
                               "error": outcome.detail})
            evidence.write("result.json", {"outcome": "failed", "detail": outcome.detail})
            return outcome
        call = replace(call, app_frames=frames, app_notes=notes)
        request["app_notes"] = list(notes)
        request["app_frames"] = frames
        evidence.write("request.json", request)

        if call.loras:
            verdicts = _validate_loras(ctx, call, evidence, session_hash, started, timing)
            request["loras"] = verdicts
            evidence.write("request.json", request)
            rejected = [v for v in verdicts if v["staged"] is False]
            if rejected:
                outcome.detail = f"LoRA not staged: {_plain(rejected[0]['verdict'])[:400]}"
                outcome.category = "adapter"
                ctx.ledger.append({**ledger_record, "at": _utcnow().isoformat(), "outcome": "failed",
                                   "error": outcome.detail})
                evidence.write("result.json", {"outcome": "failed", "detail": outcome.detail, "loras": verdicts})
                return outcome

        uploaded = {kind: ctx.app.upload(item["local"]) for kind, item in media.items()}
        timing["uploaded_t"] = round(_monotonic() - started, 4)
        data = build_data(ctx.generate.input_types, generate_values(call, uploaded))
        deadline = _monotonic() + ctx.call_timeout
        try:
            message = _run_events(ctx, evidence, ctx.generate, data, session_hash, started, deadline, timing,
                                  "generate")
        except (CallTimeout, TransportError) as exc:
            window = _monotonic() - started - timing.get("generate_process_starts_t", timing.get("generate_event_id_t", 0.0))
            billed = max(window, reserved) * ctx.multiplier
            outcome.status = "timeout" if isinstance(exc, CallTimeout) else "failed"
            outcome.detail = redact(str(exc))
            outcome.category = "timeout" if isinstance(exc, CallTimeout) else "transport"
            outcome.billed_conservative = billed
            timing.update(billed_conservative=billed, billed_rule="max(window, reservation) x multiplier: no completion")
            ctx.ledger.append({**ledger_record, "at": _utcnow().isoformat(), "outcome": outcome.status,
                               "billed_conservative": round(billed, 3), "error": outcome.detail})
            evidence.write("timing.json", timing)
            evidence.write("result.json", {"outcome": outcome.status, "detail": outcome.detail})
            return outcome

        start_t = timing.get("generate_process_starts_t", timing.get("generate_event_id_t", 0.0))
        window = max(0.0, timing["generate_completed_t"] - start_t)
        billed = window * ctx.multiplier
        output = message.get("output") or {}
        data_out = list(output.get("data") or [])
        status = str(data_out[2]) if len(data_out) > 2 and data_out[2] is not None else ""
        record = _find_record(data_out, status)
        worker = _record_value(record, "worker_seconds", "worker_elapsed_seconds", "elapsed_seconds")
        worker_source = "record" if worker is not None else None
        if worker is None:
            worker = _reported_gpu_seconds(status)
            worker_source = "the report's 'Took ... s, GPU time' (whole seconds)" if worker is not None else None
        billed_worker = None if worker is None else float(worker) * ctx.multiplier
        error = None
        if message.get("success") is False or message.get("msg") == "unexpected_error":
            error = output.get("error") or message.get("message") or (
                "the Space raised an error and did not say what (its show_error is off)"
            )
        refusal = quota_refusal(status, error, output.get("title"))
        category = "quota" if refusal else _category(record, status)
        if category == "quota" and refusal is None:
            refusal = "the app classified the failure as quota"
        timing.update(window_seconds=round(window, 4), billed_conservative=round(billed, 3),
                      billed_worker=None if billed_worker is None else round(billed_worker, 3),
                      worker_elapsed_seconds=worker, worker_seconds_source=worker_source,
                      billed_rule="(completed - process_starts) x multiplier")
        outcome.billed_conservative = billed
        outcome.category = category
        ctx.ledger.append({**ledger_record, "at": _utcnow().isoformat(),
                           "outcome": "quota_refusal" if refusal else "completed",
                           "billed_conservative": round(billed, 3),
                           "billed_worker": None if billed_worker is None else round(billed_worker, 3),
                           "error": redact(refusal or error or "") or None, "category": category})
        evidence.write("timing.json", timing)
        result = {"success": message.get("success"), "status_markdown": status, "status_text": _plain(status),
                  "error": error, "title": output.get("title"), "record": record, "category": category,
                  "outputs": data_out}

        if refusal:
            outcome.status = "quota"
            outcome.detail = redact(error or _plain(status) or refusal)
            evidence.write("result.json", {**result, "outcome": "quota"})
            return outcome

        artifacts: dict[str, Path] = {}
        downloaded: dict[str, Path] = {}
        for index, track in enumerate(("video", "audio")):
            server_path = _file_path(data_out[index]) if index < len(data_out) else None
            if not server_path:
                continue
            if server_path not in downloaded:
                downloaded[server_path] = ctx.app.download(server_path, evidence.dir / f"{track}_{Path(server_path).name}")
            artifacts[track] = downloaded[server_path]
        timing["downloaded_t"] = round(_monotonic() - started, 4)
        evidence.write("timing.json", timing)
        result["artifacts"] = {k: {"path": str(v), "bytes": v.stat().st_size, "sha256": _sha256(v)} for k, v in artifacts.items()}
        missing = [track for track in call.exports if track not in artifacts]
        if error or missing:
            outcome.detail = redact(error or f"no {' and '.join(missing)} file; the Space said: {_plain(status)[:600]}")
            outcome.status = "failed"
            evidence.write("result.json", {**result, "outcome": "failed", "detail": outcome.detail})
            return outcome

        verify = verify_call(call, artifacts, sampling_rate=ctx.sampling_rate)
        evidence.write("verify.json", verify)
        outcome.artifacts = artifacts
        outcome.flags = _flags(verify)
        outcome.audio = _audio_line((verify["content"] or {}).get("audio") or {})
        outcome.status = "ok" if verify["ok"] else "mismatch"
        outcome.detail = "; ".join(verify["problems"])
        evidence.write("result.json", {**result, "outcome": outcome.status, "detail": outcome.detail})
        ctx.produced[call.label] = {k: str(v) for k, v in artifacts.items()}
        return outcome
    except Exception as exc:  # noqa: BLE001 -- recorded, then the plan stops (or keeps going)
        outcome.status = "failed"
        outcome.category = outcome.category or "harness"
        outcome.detail = redact(f"{type(exc).__name__}: {exc}")
        if message is None and "generate_event_id_t" in timing:
            billed = max(_monotonic() - started, reserved) * ctx.multiplier
            outcome.billed_conservative = billed
            ctx.ledger.append({**ledger_record, "at": _utcnow().isoformat(), "outcome": "failed",
                               "billed_conservative": round(billed, 3), "error": outcome.detail})
        evidence.write("result.json", {"outcome": "failed", "detail": outcome.detail,
                                       "traceback": redact("".join(traceback.format_exception(exc)))})
        return outcome
    finally:
        evidence.close()


def _duration(delta: timedelta) -> str:
    seconds = max(0, int(delta.total_seconds()))
    return f"{seconds // 3600}h {(seconds % 3600) // 60:02d}m"


# ---------------------------------------------------------------------------
# reporting
# ---------------------------------------------------------------------------


def render_plan(calls: list[Call], *, base_seconds: int, multiplier: float, spent: float, ceiling: float,
                live: bool, ledger_note: str) -> list[str]:
    lines = [
        "=== PLAN — LIVE: this WILL spend GPU-seconds ===" if live
        else "=== PLAN — DRY RUN: nothing is contacted and nothing is spent ===",
        f"  reservation base {base_seconds} s (the Space's LTX25_GPU_DURATION_SECONDS) · "
        f"multiplier x{multiplier:g} · ceiling {ceiling:.0f} GPU-s/UTC day",
        f"  {ledger_note}",
    ]
    running = spent
    binds = None
    for number, call in enumerate(calls, start=1):
        reserved = reservation_seconds(call, base_seconds=base_seconds)
        worst = reserved * multiplier
        running += worst
        if binds is None and running > ceiling:
            binds = number
        parts = [f"base {base_seconds}"]
        if call.stage_2:
            parts.append("x1.5 stage 2")
        if call.loras:
            parts.append(f"+{zerogpu.LORA_DURATION_SECONDS} LoRA")
        if call.value("enhance_prompt"):
            parts.append(f"+{zerogpu.ENHANCE_DURATION_SECONDS} enhancer")
        lines.append("")
        lines.append(f"  {number:>2}. {call.label}  [{call.mode}]")
        request = {k: v for k, v in resolved_request(call).items()}
        lines.append(f"      request   {json.dumps(request, sort_keys=False, default=str)}")
        for kind, item in call.media.items():
            if item.chained:
                where = f"chained from {item.from_call}" + (f" (frame {item.frame})" if kind == "image" else "")
            else:
                where = str(item.path)
            lines.append(f"      {kind:<9} {where}")
        for lora in call.loras:
            lines.append(f"      lora      {lora.source} at {lora.weight:g} via {'upload' if lora.local else 'Hub id'}")
        geometry = call.expected_geometry()
        expect = {"geometry": f"{geometry[0]}x{geometry[1]}" if geometry else None,
                  "frames": call.expected_frames(),
                  **{k: v for k, v in call.expect.items() if k not in ("width", "height", "frames")}}
        lines.append(f"      expect    {json.dumps({k: v for k, v in expect.items() if v is not None})}")
        for note in call.app_notes:
            lines.append(f"      app       {note}")
        lines.append(f"      reserve   {reserved} s ({', '.join(parts)}) -> worst case <= {worst:.0f} GPU-s; "
                     f"day total <= {running:.0f}")
    plan_total = running - spent
    lines += [
        "",
        f"  this plan    <= {plan_total:.0f} GPU-s worst case (every call at its full reservation)",
        f"  today        <= {spent:.1f} GPU-s already on the ledger (UTC {utc_day()})",
        f"  remaining    {max(0.0, ceiling - spent):.1f} GPU-s before the {ceiling:.0f} ceiling",
    ]
    if binds is None:
        lines.append(f"  verdict      FITS, with {ceiling - running:.0f} GPU-s to spare at worst case")
    else:
        lines.append(
            f"  verdict      DOES NOT FIT: call #{binds} ({calls[binds - 1].label}) is where the ceiling binds. "
            f"It and everything after it will be refused, not started. The UTC day rolls over at "
            f"{next_utc_midnight().isoformat().replace('+00:00', 'Z')}."
        )
    lines.append("  Every figure is an UPPER BOUND, not measured billing.")
    return lines


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

_FIELD_FLAGS = (
    ("--prompt", "prompt", str),
    ("--width", "width", int),
    ("--height", "height", int),
    ("--num-frames", "num_frames", int),
    ("--seed", "seed", int),
    ("--min-seconds", "min_seconds", float),
    ("--max-seconds", "max_seconds", float),
    ("--strength", "strength", float),
    ("--noise-scale", "noise_scale", float),
)
_BOOL_FLAGS = (
    ("--enable-stage-2", "enable_stage_2"),
    ("--enhance-prompt", "enhance_prompt"),
    ("--auto-duration", "auto_duration"),
)

SUBCOMMANDS = ("status", "info", "logs")

_EPILOG = f"""
read-only subcommands (one GET each, no GPU time; see `<command> --help`):
  status            the Space's runtime stage, hardware and deployed commit
  info              compare the Space's /gradio_api/info for {ENDPOINT} with the local field spec
  logs --out FILE   save the Space's runtime (or build) log

plan file: a JSON list of calls, or {{"defaults": {{...}}, "calls": [...]}}. A call has a "label",
a "mode", any {ENDPOINT} field by its key, input media as file paths under --assets-dir
("image", "video", "audio") or {{"from_call": "<earlier label>", "frame": 0}}, optional "loras"
([{{"source": "<hub id>" | "<file.safetensors>", "weight": 1.0}}]) and optional "expect"
({", ".join(sorted(_EXPECT_KEYS))}).

exit codes: 0 plan complete · 2 illegal plan or contract mismatch (nothing spent) · 3 stopped at
the ceiling · 4 platform quota refusal · 5 a call failed (still 5 with --keep-going)
"""


def _run_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="space_matrix.py",
        description="Run a plan of calls against the private LTX-2.5 Studio Space inside a hard daily "
        "GPU budget. Dry run by default; --live is what spends.",
        epilog=_EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    plan = parser.add_argument_group("what to run")
    plan.add_argument("--plan", help="a plan JSON file, or 'default' for the historical 11-mode loop "
                      "(the default when no --mode is given)")
    plan.add_argument("--modes", help="with --plan default: comma-separated mode ids, run in that order")
    plan.add_argument("--only", help="comma-separated call labels: run only these calls of the plan")
    plan.add_argument("--assets-dir", help="where relative media and LoRA paths resolve (default: the plan "
                      "file's folder)")
    plan.add_argument("--mode", help="a single call from flags: the mode id")
    plan.add_argument("--label", help="with --mode: the call's label (default: the mode id)")
    for flag, key, kind in _FIELD_FLAGS:
        plan.add_argument(flag, type=kind, default=None, help=f"the {key} field (--mode, or --plan default)")
    for flag, key in _BOOL_FLAGS:
        plan.add_argument(flag, action="store_true", default=None, help=f"set {key} (--mode, or --plan default)")
    plan.add_argument("--image", help="with --mode: the input image file")
    plan.add_argument("--video", help="with --mode: the input video file")
    plan.add_argument("--audio", help="with --mode: the input audio file")
    plan.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                      help="with --mode: any field by key (JSON value, else a string); repeatable")
    plan.add_argument("--lora", action="append", default=[], metavar="SOURCE[@WEIGHT]",
                      help="with --mode: stage a LoRA (Hub id or local .safetensors); repeatable")
    plan.add_argument("--expect", action="append", default=[], metavar="KEY=VALUE",
                      help=f"with --mode: an expectation ({', '.join(sorted(_EXPECT_KEYS))}); repeatable")

    run = parser.add_argument_group("running and budget")
    run.add_argument("--live", action="store_true", help="actually call the Space. Without it nothing is "
                     "contacted and nothing is spent.")
    run.add_argument("--dry-run", action="store_true", help="force the dry run even with --live")
    run.add_argument("--quota-ledger", help="the append-only spend ledger (one JSON record per line, keyed by "
                     "UTC day). Required with --live; read, never written, by a dry run.")
    run.add_argument("--quota-ceiling", type=float, default=DEFAULT_QUOTA_CEILING_SECONDS,
                     help=f"GPU-seconds per UTC day never to pass (default {DEFAULT_QUOTA_CEILING_SECONDS:.0f})")
    run.add_argument("--reservation-seconds", type=int, default=DEFAULT_RESERVATION_SECONDS,
                     help=f"the Space's LTX25_GPU_DURATION_SECONDS, the base the reservation is computed "
                     f"from (default {DEFAULT_RESERVATION_SECONDS})")
    run.add_argument("--zerogpu-size", default=DEFAULT_ZEROGPU_SIZE,
                     help=f"the Space's LTX25_ZEROGPU_SIZE; zerogpu.quota_multiplier turns it into the quota "
                     f"factor (default {DEFAULT_ZEROGPU_SIZE}, which counts double)")
    run.add_argument("--out", help="evidence directory, one folder per call label. Required with --live.")
    run.add_argument("--keep-going", action="store_true", help="after a failed call, run the rest of the plan "
                     "(a quota refusal or the ceiling still stops at once)")
    run.add_argument("--call-timeout", type=float, default=DEFAULT_CALL_TIMEOUT_SECONDS,
                     help=f"seconds to wait for one call to complete (default {DEFAULT_CALL_TIMEOUT_SECONDS:.0f}); "
                     "a timed-out call is recorded, charged at least its reservation, and never retried")
    run.add_argument("--read-timeout", type=float, default=DEFAULT_READ_TIMEOUT_SECONDS,
                     help=f"seconds of silence on the event stream before giving up (default "
                     f"{DEFAULT_READ_TIMEOUT_SECONDS:.0f})")
    run.add_argument("--sampling-rate", type=int, default=PINNED_VOCODER_SAMPLING_RATE,
                     help=f"expected audio sample rate (default {PINNED_VOCODER_SAMPLING_RATE}); "
                     "expect.sample_rate overrides it per call")

    where = parser.add_argument_group("where")
    _where_flags(where)
    return parser


def _where_flags(group) -> None:
    group.add_argument("--space", default=DEFAULT_SPACE_ID, help=f"the Space id (default {DEFAULT_SPACE_ID})")
    group.add_argument("--space-url", help="the Space app's base URL (default: https://<owner>-<name>.hf.space)")
    group.add_argument("--hub-url", default=DEFAULT_HUB_URL, help=f"the Hub API base (default {DEFAULT_HUB_URL})")


def _sub_parser(name: str) -> argparse.ArgumentParser:
    descriptions = {
        "status": "Print the Space's runtime stage, hardware and deployed commit. One GET to the Hub API.",
        "info": f"Compare the Space's /gradio_api/info for {ENDPOINT} with the local field spec. One GET to "
                "the Space. Exits 2 on a mismatch.",
        "logs": "Save the Space's runtime (or build) log to a file. One GET to the Hub API, read for at most "
                "--max-seconds.",
    }
    parser = argparse.ArgumentParser(prog=f"space_matrix.py {name}", description=descriptions[name])
    _where_flags(parser)
    if name == "logs":
        parser.add_argument("--out", required=True, help="the file to write")
        parser.add_argument("--kind", choices=("run", "build"), default="run", help="which log (default run)")
        parser.add_argument("--max-seconds", type=float, default=20.0,
                            help="stop reading the (endless) log stream after this long (default 20)")
    return parser


def _parse_kv(items: list[str], what: str) -> dict:
    out = {}
    for item in items:
        key, sep, raw = item.partition("=")
        if not sep or not key:
            raise PlanError(f"{what} {item!r} must be KEY=VALUE")
        try:
            out[key] = json.loads(raw)
        except ValueError:
            out[key] = raw
    return out


def _flag_values(args) -> dict:
    values = {key: getattr(args, flag[2:].replace("-", "_")) for flag, key, _ in _FIELD_FLAGS}
    values.update({key: getattr(args, flag[2:].replace("-", "_")) for flag, key in _BOOL_FLAGS})
    return {k: v for k, v in values.items() if v is not None}


def build_calls(args) -> list[Call]:
    flags = _flag_values(args)
    single = args.mode is not None
    singles = [args.image, args.video, args.audio, args.label, *args.set, *args.lora, *args.expect]
    if not single and any(singles):
        raise PlanError("--image/--video/--audio/--label/--set/--lora/--expect describe a single call: add --mode")
    if single and (args.plan or args.modes):
        raise PlanError("--mode describes one call; it cannot be combined with --plan or --modes")
    if single:
        entry: dict = {"label": args.label or args.mode, "mode": args.mode, **flags, **_parse_kv(args.set, "--set")}
        for kind in MEDIA_KINDS:
            if getattr(args, kind):
                entry[kind] = getattr(args, kind)
        loras = []
        for item in args.lora:
            source, sep, weight = item.rpartition("@")
            loras.append({"source": source, "weight": float(weight)} if sep and _is_number(weight)
                         else {"source": item})
        if loras:
            entry["loras"] = loras
        if args.expect:
            entry["expect"] = _parse_kv(args.expect, "--expect")
        base = Path(args.assets_dir).expanduser() if args.assets_dir else Path.cwd()
        calls = parse_plan([entry], base=base)
    elif args.plan in (None, "default"):
        mode_ids = [m.strip() for m in args.modes.split(",") if m.strip()] if args.modes else None
        calls = default_plan(mode_ids, flags)
    else:
        if args.modes:
            raise PlanError("--modes selects from --plan default; use --only with a plan file")
        if flags:
            raise PlanError(f"{', '.join('--' + k.replace('_', '-') for k in flags)}: a plan file sets its own "
                            "fields; edit the file")
        calls = load_plan(args.plan, assets_dir=Path(args.assets_dir).expanduser() if args.assets_dir else None)
    if args.only:
        wanted = [label.strip() for label in args.only.split(",") if label.strip()]
        known = {c.label for c in calls}
        unknown = [label for label in wanted if label not in known]
        if unknown:
            raise PlanError(f"--only: no call labelled {unknown} (labels: {sorted(known)})")
        calls = [c for c in calls if c.label in wanted]
        chosen = {c.label for c in calls}
        for call in calls:
            for item in call.media.values():
                if item.chained and item.from_call not in chosen:
                    raise PlanError(f"--only: {call.label} chains from {item.from_call}, which is not selected")
    return calls


def _is_number(text: str) -> bool:
    try:
        float(text)
    except ValueError:
        return False
    return True


class _Out:
    """Every line the CLI prints passes through here and is redacted."""

    def __init__(self, log):
        self._log = log

    def __call__(self, line: Any = "") -> None:
        self._log(redact(line))


def main(argv=None, *, log=print) -> int:
    out = _Out(log)
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] in SUBCOMMANDS:
        return _subcommand(argv[0], argv[1:], out)
    args = _run_parser().parse_args(argv)
    try:
        calls = build_calls(args)
    except PlanError as exc:
        out(f"the plan is not runnable:\n{exc}")
        return EXIT_ILLEGAL
    live = args.live and not args.dry_run
    try:
        multiplier = quota_multiplier(args.zerogpu_size)
    except Exception as exc:  # noqa: BLE001 -- the app's ConfigError names the legal sizes
        out(f"--zerogpu-size: {exc}")
        return EXIT_ILLEGAL

    ledger = Ledger(args.quota_ledger) if args.quota_ledger else None
    if live and ledger is None:
        out("--live needs --quota-ledger PATH: the day's spend must be kept outside any checkout.")
        return EXIT_ILLEGAL
    if live and not args.out:
        out("--live needs --out DIR for the per-call evidence.")
        return EXIT_ILLEGAL
    try:
        spent = ledger.spent(utc_day()) if ledger else 0.0
        previous = ledger.previous_day(utc_day()) if ledger else None
    except QuotaLedgerUnreadable as exc:
        out(str(exc))
        return EXIT_ILLEGAL
    if ledger is None:
        note = "no --quota-ledger given: this dry run assumes nothing has been spent today"
    else:
        note = f"ledger {ledger.path}: <= {spent:.1f} GPU-s on UTC {utc_day()}"
        if previous:
            note += f" (the last earlier day, UTC {previous[0]}, ended at <= {previous[1]:.1f})"
    for line in render_plan(calls, base_seconds=args.reservation_seconds, multiplier=multiplier, spent=spent,
                            ceiling=args.quota_ceiling, live=live, ledger_note=note):
        out(line)
    if not live:
        out("")
        out("DRY RUN: every call above was validated offline. Nothing was contacted, ZERO GPU-seconds were "
            "spent, and the ledger was read, not written. Add --live to execute.")
        return EXIT_OK
    return _live(args, calls, ledger, multiplier, out)


def _live(args, calls: list[Call], ledger: Ledger, multiplier: float, out) -> int:
    out_dir = Path(args.out).expanduser()
    taken = [c.label for c in calls if (out_dir / c.label).exists() and any((out_dir / c.label).iterdir())]
    if taken:
        out(f"{out_dir} already holds evidence for {taken}; use a fresh --out so runs never mix.")
        return EXIT_ILLEGAL
    secret = read_token()
    if secret is None:
        out(f"no token: set HF_TOKEN or log in so {token_path()} exists. The Space is private.")
        return EXIT_ILLEGAL
    http = Http(secret)
    hub = Hub(args.hub_url, http)
    try:
        status = hub.space(args.space)
    except (HttpError, TransportError) as exc:
        out(f"could not read the Space's status: {exc}")
        return EXIT_ILLEGAL
    runtime = status.get("runtime") or {}
    stage = runtime.get("stage")
    if stage != "RUNNING":
        out(f"the Space is {stage}, not RUNNING. Wake it or wait for the build; nothing was sent.")
        return EXIT_ILLEGAL
    base_url = args.space_url or status.get("host") or space_host(args.space)
    if not base_url.startswith(("http://", "https://")):
        base_url = "https://" + base_url
    sha = runtime.get("sha") or status.get("sha")
    try:
        app = SpaceApp(base_url, http)
        config = app.config()
        app.prefix = "/" + str(config.get("api_prefix") or API_PREFIX).strip("/")
        info = app.info()
    except (HttpError, TransportError, ValueError) as exc:
        out(f"could not read the Space's API: {exc}")
        return EXIT_ILLEGAL
    generate = find_endpoint(config, info, ENDPOINT)
    problems: list[str] = []
    table: list[str] = []
    if generate is None:
        problems.append(f"the Space publishes no {ENDPOINT}")
    else:
        problems, table = generate_contract_problems(generate.parameters, input_types=generate.input_types,
                                                     plan_modes=[c.mode for c in calls])
    route = _lora_route(config, info)
    problems += lora_contract_problems(route, calls)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "contract.json").write_text(_redacted_json(
        {"space": args.space, "space_url": base_url, "space_sha": sha, "problems": problems, "table": table,
         "info": info.get("named_endpoints", {}).get(ENDPOINT)}))
    if problems:
        out(f"\nCONTRACT MISMATCH: the Space at {sha or '?'} does not take what this checkout's field spec builds. "
            "Nothing was spent.")
        for line in table:
            out(line)
        for problem in problems:
            out(f"  ✗ {problem}")
        return EXIT_ILLEGAL
    out(f"\ncontract     {ENDPOINT} matches the field spec ({len(generate.parameters)} parameters) · Space "
        f"commit {sha or '?'} · {base_url}")

    invocation = uuid.uuid4().hex[:12]
    ledger.append({"kind": "invocation", "utc_day": utc_day(), "at": _utcnow().isoformat(), "invocation": invocation,
                   "space": args.space, "space_sha": sha, "calls": [c.label for c in calls]})
    ctx = Session(app=app, generate=generate, lora=route, ledger=ledger, out=out_dir, log=out,
                  invocation=invocation, ceiling=args.quota_ceiling, base_seconds=args.reservation_seconds,
                  multiplier=multiplier, call_timeout=args.call_timeout, read_timeout=args.read_timeout,
                  sampling_rate=args.sampling_rate, space_sha=sha)
    outcomes: list[Outcome] = []
    code = EXIT_OK
    stop = ""
    for call in calls:
        try:
            outcome = run_call(ctx, call)
        except CeilingReached as exc:
            stop = str(exc)
            code = EXIT_CEILING
            out(f"\n*** STOPPED AT THE CEILING ***\n  {exc}")
            break
        except QuotaLedgerUnreadable as exc:
            stop = str(exc)
            code = EXIT_ILLEGAL
            out(f"\n*** STOPPED: the ledger became unreadable ***\n  {exc}")
            break
        outcomes.append(outcome)
        out(f"  -> {outcome.status.upper()}  billed <= {outcome.billed_conservative:.1f} GPU-s (conservative)"
            + (f"  [{outcome.category}]" if outcome.category else "")
            + (f"  {outcome.detail}" if outcome.detail else "")
            + (f"  flags: {', '.join(outcome.flags)}" if outcome.flags else ""))
        if outcome.audio:
            out(f"     audio  {outcome.audio}  (measurements, no verdict)")
        if outcome.status == "quota":
            stop = outcome.detail
            code = EXIT_QUOTA
            out("\n*** THE PLATFORM REFUSED ON QUOTA: stopping now. Nothing is retried. ***")
            break
        if outcome.failed:
            code = EXIT_CALL_FAILED
            if not args.keep_going:
                stop = f"{call.label} {outcome.status}"
                out("  stopping: a call failed (pass --keep-going to run the rest)")
                break
    day = utc_day()
    try:
        spent_today = ledger.spent(day)
    except QuotaLedgerUnreadable:
        spent_today = float("nan")
    this_run = sum(o.billed_conservative for o in outcomes)
    summary = {
        "invocation": invocation,
        "space_sha": sha,
        "exit_code": code,
        "stopped": stop or None,
        "calls": [{"label": o.label, "status": o.status, "category": o.category, "detail": o.detail,
                   "reserved_seconds": o.reserved, "billed_conservative": round(o.billed_conservative, 3),
                   "flags": o.flags, "audio": o.audio or None} for o in outcomes],
        "not_run": [c.label for c in calls[len(outcomes):]],
        "this_invocation_billed_conservative": round(this_run, 3),
        "utc_day": day,
        "spent_today_conservative": round(spent_today, 3),
        "ceiling": args.quota_ceiling,
    }
    (out_dir / "summary.json").write_text(_redacted_json(summary))
    out("\n=== SUMMARY ===")
    for o in outcomes:
        out(f"  {o.label:<32} {o.status:<9} <= {o.billed_conservative:7.1f} GPU-s"
            + (f"  [{'; '.join(o.flags)}]" if o.flags else ""))
    for label in summary["not_run"]:
        out(f"  {label:<32} not run")
    out(f"  this invocation <= {this_run:.1f} GPU-s · today (UTC {day}) <= {spent_today:.1f} of {args.quota_ceiling:.0f}"
        " · UPPER BOUNDS from wall clock, not measured billing")
    out(f"  evidence {out_dir} · ledger {ledger.path}")
    return code


def _subcommand(name: str, argv: list[str], out) -> int:
    args = _sub_parser(name).parse_args(argv)
    http = Http(read_token())
    try:
        if name == "status":
            payload = Hub(args.hub_url, http).space(args.space)
            runtime = payload.get("runtime") or {}
            hardware = runtime.get("hardware") or {}
            out(f"space      {payload.get('id', args.space)}{' (private)' if payload.get('private') else ''}")
            out(f"stage      {runtime.get('stage')}")
            out(f"hardware   {hardware.get('current')} (requested {hardware.get('requested')})")
            out(f"commit     {payload.get('sha')} (repo head) · running {runtime.get('sha')}")
            out(f"host       {payload.get('host') or space_host(args.space)}")
            return EXIT_OK
        if name == "logs":
            lines = Hub(args.hub_url, http).logs(args.space, kind=args.kind, max_seconds=args.max_seconds)
            target = Path(args.out).expanduser()
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(redact("\n".join(lines) + ("\n" if lines else "")))
            out(f"wrote {len(lines)} {args.kind}-log line(s) to {target}")
            return EXIT_OK
        app = SpaceApp(args.space_url or space_host(args.space), http)
        info = app.info()
        published = (info.get("named_endpoints") or {}).get(ENDPOINT)
        if published is None:
            out(f"the Space publishes no {ENDPOINT}")
            return EXIT_ILLEGAL
        problems, table = generate_contract_problems(list(published.get("parameters") or []))
        for line in table:
            out(line)
        for problem in problems:
            out(f"  ✗ {problem}")
        out("MATCHES the local field spec" if not problems else "MISMATCH: a --live run would refuse")
        return EXIT_OK if not problems else EXIT_ILLEGAL
    except (HttpError, TransportError, ValueError) as exc:
        out(f"{name}: {exc}")
        return 1


def _entry() -> int:
    try:
        return main()
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        return 130
    except Exception as exc:  # noqa: BLE001 -- the traceback is formatted here, so it is redacted here
        print(redact("".join(traceback.format_exception(exc))), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(_entry())
