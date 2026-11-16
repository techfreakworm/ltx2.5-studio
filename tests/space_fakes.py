"""An in-process fake of the private Space and of the Hub API, for ``tests/test_space_matrix.py``.

Listens on 127.0.0.1 only and serves what ``scripts/space_matrix.py`` talks to:

* Hub API: ``GET /api/spaces/<id>`` (status) and ``GET /api/spaces/<id>/logs/<kind>`` (SSE);
* the Gradio app: ``GET /config``, ``GET /gradio_api/info``, ``POST /gradio_api/upload``,
  ``POST /gradio_api/queue/join``, ``GET /gradio_api/queue/data`` (SSE) and
  ``GET /gradio_api/file=<path>``.

``/config`` and ``/gradio_api/info`` are built from ``ltx25_studio.ui.fields`` with the component
mapping ``ui/components.py`` uses;
``test_the_contract_check_passes_on_the_real_app_and_the_fake_publishes_the_same`` checks them
against the real ``build_ui()`` so the fake cannot drift from what Gradio publishes. The files
a run returns follow the app too (:meth:`FakeSpace.app_files`), pinned per mode against the real
handler, ``generate()`` and writer by ``test_the_fake_returns_the_files_the_real_app_returns``.

Every request is recorded (method, path, query, headers, body size), and every queue join is
decoded BY THE FAKE'S OWN slot map, so a test can assert which value reached which field name
without trusting the client's indexing.
"""

from __future__ import annotations

import hashlib
import json
import queue
import re
import shutil
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, unquote, urlparse

SPACE_ID = "techfreakworm/LTX2.5-Studio"

_COMPONENT = {
    "text": "Textbox",
    "number": "Number",
    "checkbox": "Checkbox",
    "slider": "Slider",
    "image": "Image",
    "video": "Video",
    "audio": "Audio",
    "dataframe": "Dataframe",
    "dropdown": "Dropdown",
    "radio": "Radio",
    "state": "State",
}

QUOTA_MESSAGE = (
    "You have exceeded your Pro ZeroGPU quota (360s requested vs. 118s left). Try again in 5:41:07. "
    "Add credits to keep using ZeroGPU."
)


def component_name(kind: str) -> str:
    return _COMPONENT.get(kind, kind.capitalize())


class _AsTheApp:
    def __repr__(self) -> str:
        return "AS_THE_APP"


#: ``Reply.audio`` by default: the audio file the app returns for the call's mode (``app_files``).
AS_THE_APP: Any = _AsTheApp()


@dataclass
class Reply:
    """What the fake does with one ``/on_generate`` join.

    ``kind``: ``ok`` · ``quota`` (the handler caught the platform's refusal and rendered it) ·
    ``quota_raised`` (it escaped: ``success=False`` with the platform's title) · ``raised`` (any
    error that escaped, with ``title`` and ``message``) · ``app_error`` (a failure card: ``status``
    as given, or one built around ``message``) · ``foreign_error`` (an uncaught exception) ·
    ``unexpected_error`` · ``no_artifact`` (success-shaped, no file) · ``timeout`` (never
    completes; heartbeats only).

    ``video`` and ``audio`` are the files an ``ok`` reply returns. ``audio`` left at
    :data:`AS_THE_APP` is what the app returns beside ``video`` for the call's mode; ``None``
    returns no audio file, and a path returns that file (the ``video`` path itself returns the
    video file in both outputs).
    """

    kind: str = "ok"
    video: Path | None = None
    audio: Any = AS_THE_APP
    status: str = "**Done.** Seed 7."
    record: dict | None = None
    message: str = ""
    title: str = ""
    progress_steps: int = 8
    #: The ``log`` event the Space sends while the call waits (the harness prints it).
    log: str = "Waiting for a GPU to become available"


@dataclass
class _Session:
    messages: queue.Queue = field(default_factory=queue.Queue)
    state: dict = field(default_factory=dict)


class FakeSpace:
    """The Space and the Hub API on one local port. Use as a context manager."""

    def __init__(
        self,
        tmp_dir: Path,
        *,
        token: str | None = None,
        space_id: str = SPACE_ID,
        field_spec=None,
        mode_ids=None,
        lora_weight_route: bool = False,
        stage: str = "RUNNING",
        sha: str = "0123456789abcdef0123456789abcdef01234567",
        log_lines=("===== Application Startup =====", "* Running on local URL"),
        logs_hold: bool = False,
        info_patch=None,
        config_patch=None,
        step_delay: float = 0.002,
        heartbeat_interval: float = 0.02,
        echo_token_on_upload: bool = False,
    ):
        if field_spec is None:
            from ltx25_studio.ui import fields

            field_spec = fields.FIELDS
        if mode_ids is None:
            from ltx25_studio import modes

            mode_ids = tuple(modes.MODES)
        self.tmp = Path(tmp_dir)
        self.tmp.mkdir(parents=True, exist_ok=True)
        self.token = token
        self.space_id = space_id
        self.fields = tuple(field_spec)
        self.mode_ids = tuple(mode_ids)
        self.lora_weight_route = lora_weight_route
        self.stage = stage
        self.sha = sha
        self.log_lines = tuple(log_lines)
        self.logs_hold = logs_hold
        self.step_delay = step_delay
        self.heartbeat_interval = heartbeat_interval
        #: A misbehaving server: uploads fail with the request's Authorization header in the body.
        self.echo_token_on_upload = echo_token_on_upload

        self.requests: list[dict] = []
        self.joins: list[dict] = []
        self.generate_calls: list[dict] = []
        self.lora_calls: list[dict] = []
        self.protocol_errors: list[str] = []
        self.replies: deque[Reply] = deque()
        self.lora_replies: deque[tuple[str, str]] = deque()
        self._files: dict[str, Path] = {}
        self._upload_names: dict[str, str] = {}
        self._sessions: dict[str, _Session] = {}
        self._lock = threading.Lock()
        self._stopping = threading.Event()
        self._server: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None

        self._build_contract()
        if info_patch is not None:
            self.info = info_patch(self.info) or self.info
        if config_patch is not None:
            self.config = config_patch(self.config) or self.config

    # ------------------------------------------------------------------ setup

    def _build_contract(self) -> None:
        ids = iter(range(1, 10_000))
        components: list[dict] = []

        def add(kind: str, label: str) -> int:
            cid = next(ids)
            components.append({"id": cid, "type": component_name(kind).lower(), "props": {"label": label}})
            return cid

        self.mode_component = add("radio", "Mode")
        self.field_components: dict[int, Any] = {}
        state_ids: dict[str, int] = {}
        for spec in self.fields:
            cid = add(spec.kind, spec.label)
            self.field_components[cid] = spec
            if spec.kind == "state":
                state_ids[spec.key] = cid
        self.loras_state = state_ids.get("loras") or add("state", "LoRA adapters")
        video_out, audio_out, status_out = add("video", "Video"), add("audio", "Audio"), add("markdown", "")
        repo_id, upload = add("text", "Hub repo id"), add("file", "Or upload a .safetensors file")
        verdict, summary = add("markdown", ""), add("markdown", "")
        index_in, weight_in = add("number", "Adapter"), add("slider", "Strength")
        self.slot_names = {
            0: ["mode", *[f.key for f in self.fields]],
            1: ["repo_id", "upload", "loras"],
            2: ["loras", "index", "weight"],
        }
        dependencies = [
            {
                "id": 0,
                "api_name": "on_generate",
                "inputs": [self.mode_component, *self.field_components],
                "outputs": [video_out, audio_out, status_out],
                "api_visibility": "public",
                "queue": True,
                "backend_fn": True,
            },
            {
                "id": 1,
                "api_name": "on_validate_lora",
                "inputs": [repo_id, upload, self.loras_state],
                "outputs": [verdict, self.loras_state, summary],
                "api_visibility": "public",
                "queue": True,
                "backend_fn": True,
            },
        ]
        if self.lora_weight_route:
            dependencies.append(
                {
                    "id": 2,
                    "api_name": "on_lora_weight",
                    "inputs": [self.loras_state, index_in, weight_in],
                    "outputs": [self.loras_state],
                    "api_visibility": "public",
                    "queue": True,
                    "backend_fn": True,
                }
            )
        self.config = {
            "version": "6.24.0",
            "protocol": "sse_v3",
            "api_prefix": "/gradio_api",
            "components": components,
            "dependencies": dependencies,
        }

        generate_params = [
            {
                "label": "Mode",
                "parameter_name": "param_0",
                "parameter_has_default": True,
                "parameter_default": self.mode_ids[0],
                "component": "Radio",
                "type": {"enum": list(self.mode_ids), "title": "Radio", "type": "string"},
                "python_type": {"type": "str", "description": ""},
            }
        ]
        for index, spec in enumerate((f for f in self.fields if f.kind != "state"), start=1):
            generate_params.append(
                {
                    "label": spec.label,
                    "parameter_name": f"param_{index}",
                    "parameter_has_default": True,
                    "parameter_default": spec.default,
                    "component": component_name(spec.kind),
                    "type": {},
                    "python_type": {"type": "Any", "description": ""},
                }
            )
        endpoints = {
            "/on_generate": {
                "parameters": generate_params,
                "returns": [
                    {"label": "Video", "component": "Video"},
                    {"label": "Audio", "component": "Audio"},
                    {"label": "", "component": "Markdown"},
                ],
                "api_visibility": "public",
            },
            "/on_validate_lora": {
                "parameters": [
                    {"label": "Hub repo id", "parameter_name": "repo_id", "component": "Textbox"},
                    {"label": "Or upload a .safetensors file", "parameter_name": "upload", "component": "File"},
                ],
                "returns": [{"label": "", "component": "Markdown"}, {"label": "", "component": "Markdown"}],
                "api_visibility": "public",
            },
        }
        if self.lora_weight_route:
            endpoints["/on_lora_weight"] = {
                "parameters": [
                    {"label": "Adapter", "parameter_name": "index", "component": "Number"},
                    {"label": "Strength", "parameter_name": "weight", "component": "Slider"},
                ],
                "returns": [],
                "api_visibility": "public",
            }
        self.info = {"named_endpoints": endpoints, "unnamed_endpoints": {}}

    # -------------------------------------------------------------- lifecycle

    def __enter__(self) -> FakeSpace:
        fake = self

        class Handler(_Handler):
            space = fake

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self._server.daemon_threads = True
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stopping.set()
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()

    @property
    def url(self) -> str:
        assert self._server is not None, "use FakeSpace as a context manager"
        return f"http://127.0.0.1:{self._server.server_address[1]}"

    # ---------------------------------------------------------------- scripting

    def reply(self, *replies: Reply) -> FakeSpace:
        self.replies.extend(replies)
        return self

    def lora(self, *verdicts: tuple[str, str]) -> FakeSpace:
        """``("accept", "")`` or ``("reject", "<reason>")`` per ``/on_validate_lora`` join."""
        self.lora_replies.extend(verdicts)
        return self

    def joins_for(self, endpoint: str) -> list[dict]:
        return [j for j in self.joins if j["endpoint"] == endpoint]

    def paths(self) -> list[str]:
        return [r["path"] for r in self.requests]

    def app_files(self, mode_id: str, reply: Reply) -> list[tuple[Path, str] | None]:
        """What an ``ok`` reply returns in ``/on_generate``'s two file outputs, as
        ``(local file, the name it is served under)``.

        ``reply.video`` is served under the app's name for it. With ``reply.audio`` left at
        :data:`AS_THE_APP`, a mode that exports video and audio also returns the video's sound
        as a wav of its own, as ``outputs.write_output`` writes it; any other mode returns no
        audio file beside a video.
        """
        from ltx25_studio import modes, outputs

        video = None if reply.video is None else (Path(reply.video), outputs.VIDEO_FILENAME)
        if reply.audio is AS_THE_APP:
            exports = modes.get_mode(mode_id).exports
            sound = self._sound_of(reply.video) if video and {"audio", "video"} <= set(exports) else None
            audio = None if sound is None else (sound, outputs.AUDIO_FILENAME)
        elif reply.audio is None:
            audio = None
        else:
            audio = (Path(reply.audio), Path(reply.audio).name)
        return [video, audio]

    def _sound_of(self, video: Path) -> Path | None:
        """The video file's first audio stream, written again as a pcm_s16le wav at its rate and layout."""
        import av

        from ltx25_studio import outputs

        digest = hashlib.sha256(Path(video).read_bytes()).hexdigest()[:20]
        target = self.tmp / "sound" / digest / outputs.AUDIO_FILENAME
        with av.open(str(video)) as source:
            if not source.streams.audio:
                return None
            stream = source.streams.audio[0]
            target.parent.mkdir(parents=True, exist_ok=True)
            with av.open(str(target), "w", format="wav") as out:
                layout = getattr(stream.layout, "name", None) or "stereo"
                encoder = out.add_stream("pcm_s16le", rate=int(stream.rate), layout=layout)
                for frame in source.decode(stream):
                    frame.pts = None
                    for packet in encoder.encode(frame):
                        out.mux(packet)
                for packet in encoder.encode():
                    out.mux(packet)
        return target

    def serve_file(self, local: Path, *, name: str | None = None) -> dict:
        """Publish a local file the way Gradio's cache does and return its FileData."""
        local = Path(local)
        digest = hashlib.sha256(local.read_bytes()).hexdigest()[:20]
        server_path = f"/tmp/gradio/{digest}/{name or local.name}"
        self._files[server_path] = local
        return {
            "path": server_path,
            "url": f"{self.url}/gradio_api/file={server_path}",
            "size": local.stat().st_size,
            "orig_name": name or local.name,
            "mime_type": None,
            "is_stream": False,
            "meta": {"_type": "gradio.FileData"},
        }

    # ------------------------------------------------------------------ events

    def _session(self, session_hash: str) -> _Session:
        with self._lock:
            return self._sessions.setdefault(session_hash, _Session())

    def _join(self, body: dict) -> tuple[int, dict]:
        session_hash = body.get("session_hash")
        if not session_hash:
            return 400, {"detail": "Session hash not found."}
        fn_index = body.get("fn_index")
        dependency = next((d for d in self.config["dependencies"] if d["id"] == fn_index), None)
        if dependency is None:
            return 400, {"detail": f"no dependency {fn_index}"}
        data = body.get("data")
        if not isinstance(data, list) or len(data) != len(dependency["inputs"]):
            return 422, {"detail": f"expected {len(dependency['inputs'])} values, got {len(data or [])}"}
        names = self.slot_names[fn_index]
        types = {c["id"]: c["type"] for c in self.config["components"]}
        for slot, cid in enumerate(dependency["inputs"]):
            if types[cid] == "state" and data[slot] is not None:
                self.protocol_errors.append(f"{dependency['api_name']} slot {slot} is a State and got a value")
        params = dict(zip(names, data, strict=True))
        event_id = uuid.uuid4().hex
        session = self._session(session_hash)
        join = {
            "endpoint": "/" + dependency["api_name"],
            "fn_index": fn_index,
            "session_hash": session_hash,
            "data": data,
            "params": params,
            "event_id": event_id,
        }
        self.joins.append(join)
        if fn_index == 0:
            self._generate(join, session)
        elif fn_index == 1:
            self._validate_lora(join, session)
        else:
            self._lora_weight(join, session)
        return 200, {"event_id": event_id}

    def _put(self, session: _Session, event_id: str, msg: dict) -> None:
        session.messages.put({**msg, "event_id": event_id})

    def _queue_prelude(self, session: _Session, event_id: str) -> None:
        self._put(session, event_id, {"msg": "estimation", "rank": 0, "queue_size": 1, "rank_eta": 1.0})
        self._put(session, event_id, {"msg": "process_starts", "eta": 5.0})

    def _completed(self, session: _Session, event_id: str, data: list, *, success: bool = True) -> None:
        output = {"data": data, "is_generating": False, "duration": 1.0, "average_duration": 1.0}
        self._put(session, event_id, {"msg": "process_completed", "output": output, "success": success})

    def _generate(self, join: dict, session: _Session) -> None:
        params = {k: v for k, v in join["params"].items() if k != "mode"}
        self.generate_calls.append(
            {
                "mode": join["params"]["mode"],
                "params": params,
                "loras": [dict(e) for e in session.state.get("loras", [])],
                "session_hash": join["session_hash"],
            }
        )
        event_id = join["event_id"]
        reply = self.replies.popleft() if self.replies else Reply(kind="foreign_error", message="fake: no reply scripted")
        self._queue_prelude(session, event_id)
        self._put(
            session,
            event_id,
            {"msg": "log", "log": reply.log, "level": "info", "title": "ZeroGPU queue", "duration": 10, "visible": True},
        )
        for step in range(1, reply.progress_steps + 1):
            self._put(
                session,
                event_id,
                {
                    "msg": "progress",
                    "progress_data": [
                        {"index": step, "length": reply.progress_steps, "unit": "steps", "progress": None, "desc": "Denoising · pass 1 of 1"}
                    ],
                },
            )
        self._put(session, event_id, {"msg": "heartbeat"})
        status = reply.status
        if reply.record is not None:
            status += f'\n<script type="application/json" class="run-record">{json.dumps(reply.record)}</script>'
        if reply.kind == "ok":
            video, audio = self.app_files(join["params"]["mode"], reply)
            served_video = None if video is None else self.serve_file(video[0], name=video[1])
            if audio is None:
                served_audio = None
            elif video is not None and audio[0] == video[0]:
                served_audio = served_video
            else:
                served_audio = self.serve_file(audio[0], name=audio[1])
            self._completed(session, event_id, [served_video, served_audio, status])
        elif reply.kind == "quota":
            card = (
                "### The run stopped\n\nAny GPU time this run used is billed. Read the detail below.\n\n"
                f'<details class="mr-detail" open><summary>Detail</summary><p>{reply.message or QUOTA_MESSAGE}</p></details>'
            )
            self._completed(session, event_id, [None, None, card])
        elif reply.kind == "quota_raised":
            self._put(
                session,
                event_id,
                {
                    "msg": "process_completed",
                    "output": {"error": reply.message or QUOTA_MESSAGE, "title": "ZeroGPU quota exceeded", "duration": 10, "visible": True},
                    "success": False,
                },
            )
        elif reply.kind == "raised":
            self._put(
                session,
                event_id,
                {
                    "msg": "process_completed",
                    "output": {"error": reply.message, "title": reply.title, "duration": 10, "visible": True},
                    "success": False,
                },
            )
        elif reply.kind == "app_error":
            card = reply.status or (
                "### The model stopped while generating\n\nAny GPU time this run used is billed.\n\n"
                f'<details class="mr-detail"><summary>Detail</summary><p>{reply.message}</p></details>'
            )
            if reply.record is not None:
                card += f'\n<script type="application/json" class="run-record">{json.dumps(reply.record)}</script>'
            self._completed(session, event_id, [None, None, card])
        elif reply.kind == "foreign_error":
            self._put(
                session,
                event_id,
                {"msg": "process_completed", "output": {"error": reply.message or None}, "success": False},
            )
        elif reply.kind == "unexpected_error":
            self._put(session, event_id, {"msg": "unexpected_error", "message": reply.message, "success": False})
        elif reply.kind == "no_artifact":
            self._completed(session, event_id, [None, None, reply.status])
        elif reply.kind == "timeout":
            self._put(session, event_id, {"msg": "__stall__"})
        else:
            raise AssertionError(f"unknown reply kind {reply.kind!r}")

    def _validate_lora(self, join: dict, session: _Session) -> None:
        params = join["params"]
        upload = params.get("upload")
        name = None
        if isinstance(upload, dict):
            name = self._upload_names.get(upload.get("path"), Path(str(upload.get("path"))).name)
        source = name or (params.get("repo_id") or "").strip()
        self.lora_calls.append({"source": source, "params": params, "session_hash": join["session_hash"]})
        verdict, reason = self.lora_replies.popleft() if self.lora_replies else ("accept", "")
        staged = session.state.setdefault("loras", [])
        if verdict == "accept":
            staged.append({"source": source, "weight": 1.0})
            text = f"#### Accepted: {source}"
        else:
            text = f"### Rejected: the adapter does not fit\n\n{source} was not staged. Nothing was spent.\n\n> {reason}"
        summary = (
            "Staged for the next run: " + "; ".join(f"{e['source']} at {e['weight']:g}" for e in staged) + "."
            if staged
            else "No adapter staged."
        )
        event_id = join["event_id"]
        self._queue_prelude(session, event_id)
        self._completed(session, event_id, [text, None, summary])

    def _lora_weight(self, join: dict, session: _Session) -> None:
        params = join["params"]
        staged = session.state.setdefault("loras", [])
        index = int(params["index"])
        if 0 <= index < len(staged):
            staged[index]["weight"] = float(params["weight"])
        event_id = join["event_id"]
        self._queue_prelude(session, event_id)
        self._completed(session, event_id, [None])

    def _upload(self, content_type: str, body: bytes) -> tuple[int, Any]:
        match = re.search(r"boundary=([^;]+)", content_type or "")
        if not match:
            return 400, {"detail": "no boundary"}
        boundary = match.group(1).strip('"').encode()
        paths = []
        for part in body.split(b"--" + boundary):
            if b"\r\n\r\n" not in part:
                continue
            head, _, content = part.partition(b"\r\n\r\n")
            name = re.search(rb'filename="([^"]*)"', head)
            if not name or b'name="files"' not in head:
                continue
            if content.endswith(b"\r\n"):
                content = content[:-2]
            filename = name.group(1).decode()
            digest = hashlib.sha256(content).hexdigest()[:20]
            local = self.tmp / "uploads" / digest / filename
            local.parent.mkdir(parents=True, exist_ok=True)
            local.write_bytes(content)
            server_path = f"/tmp/gradio/{digest}/{filename}"
            self._files[server_path] = local
            self._upload_names[server_path] = filename
            paths.append(server_path)
        return 200, paths

    def uploaded(self, server_path: str) -> Path:
        return self._files[server_path]

    def status_payload(self) -> dict:
        return {
            "id": self.space_id,
            "private": True,
            "sha": self.sha,
            "host": self.url,
            "runtime": {
                "stage": self.stage,
                "sha": self.sha,
                "hardware": {"current": "zero-a10g", "requested": "zero-a10g"},
            },
        }


class _Handler(BaseHTTPRequestHandler):
    space: FakeSpace

    def log_message(self, *args) -> None:  # quiet
        pass

    def _record(self, body: bytes = b"") -> tuple[str, dict]:
        parsed = urlparse(self.path)
        query = parse_qs(parsed.query)
        self.space.requests.append(
            {
                "method": self.command,
                "path": unquote(parsed.path),
                "raw_path": self.path,
                "query": query,
                "headers": {k.lower(): v for k, v in self.headers.items()},
                "body_len": len(body),
                "body": body if len(body) < 65536 else body[:65536],
            }
        )
        return unquote(parsed.path), query

    def _authorised(self) -> bool:
        if self.space.token is None:
            return True
        if self.headers.get("Authorization") == f"Bearer {self.space.token}":
            return True
        self._json(401, {"error": "Invalid credentials in Authorization header"})
        return False

    def _json(self, code: int, payload: Any) -> None:
        body = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _sse_start(self) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()

    def _sse(self, payload: dict) -> None:
        self.wfile.write(f"data: {json.dumps(payload)}\n\n".encode())
        self.wfile.flush()

    def do_GET(self) -> None:  # noqa: N802
        path, query = self._record()
        if not self._authorised():
            return
        space = self.space
        status_path = f"/api/spaces/{space.space_id}"
        try:
            if path == "/redirect":
                # Somewhere else, the way a CDN or a moved file answers.
                self.send_response(302)
                self.send_header("Location", query.get("to", [""])[0])
                self.send_header("Content-Length", "0")
                self.end_headers()
            elif path == status_path:
                self._json(200, space.status_payload())
            elif path.startswith(status_path + "/logs/"):
                self._sse_start()
                for line in space.log_lines:
                    self._sse({"data": line, "timestamp": "2026-11-16T10:00:00Z"})
                while space.logs_hold and not space._stopping.is_set():
                    time.sleep(0.02)
            elif path == "/config":
                self._json(200, space.config)
            elif path == "/gradio_api/info":
                self._json(200, space.info)
            elif path == "/gradio_api/queue/data":
                self._stream(query.get("session_hash", [""])[0])
            elif path.startswith("/gradio_api/file="):
                local = space._files.get(path[len("/gradio_api/file=") :])
                if local is None:
                    self._json(404, {"detail": "File not found"})
                    return
                body = Path(local).read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "application/octet-stream")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            else:
                self._json(404, {"detail": f"no route {path}"})
        except (BrokenPipeError, ConnectionResetError):
            pass

    def do_POST(self) -> None:  # noqa: N802
        length = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(length) if length else b""
        path, _query = self._record(body)
        if not self._authorised():
            return
        if path == "/gradio_api/upload" and self.space.echo_token_on_upload:
            self._json(500, {"detail": f"upload refused for {self.headers.get('Authorization')}"})
        elif path == "/gradio_api/upload":
            code, payload = self.space._upload(self.headers.get("Content-Type", ""), body)
            self._json(code, payload)
        elif path == "/gradio_api/queue/join":
            code, payload = self.space._join(json.loads(body or b"{}"))
            self._json(code, payload)
        else:
            self._json(404, {"detail": f"no route {path}"})

    def _stream(self, session_hash: str) -> None:
        space = self.space
        session = space._session(session_hash)
        self._sse_start()
        stalled = False
        try:
            while not space._stopping.is_set():
                if stalled:
                    time.sleep(space.heartbeat_interval)
                    self._sse({"msg": "heartbeat"})
                    continue
                try:
                    msg = session.messages.get(timeout=space.heartbeat_interval)
                except queue.Empty:
                    self._sse({"msg": "heartbeat"})
                    continue
                if msg.get("msg") == "__stall__":
                    stalled = True
                    continue
                time.sleep(space.step_delay)
                self._sse(msg)
                if msg.get("msg") in ("process_completed", "unexpected_error") and session.messages.empty():
                    # Gradio (sse_v3) ends the stream once the session has no pending event.
                    self._sse({"msg": "close_stream", "event_id": None})
                    return
        except (BrokenPipeError, ConnectionResetError, OSError):
            return


def copy_into(src: Path, dest_dir: Path, name: str | None = None) -> Path:
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / (name or src.name)
    shutil.copyfile(src, dest)
    return dest
