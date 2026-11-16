#!/usr/bin/env python
"""Verify a produced artifact against the mode's DECLARED exports.

    python tests/verify_streams.py <mode_id> <file> [--frame-rate 24] [--sampling-rate <hz>]
                                   [--geometry WxH]

**Do not hardcode a sampling rate.** The vocoder knows its own
(``vocoder.config.output_sampling_rate``, 48000 at the pinned revision) and
``audio.export_sampling_rate_from()`` is the one place that reads it;
``scripts/local_generate.py`` passes the vocoder's value through. A second copy of the
number here would be a second thing to be wrong. Omitting the flag skips the
rate check rather than assuming one.

``--geometry WxH`` checks the picture's size, e.g. that a stage-2 run returned
twice the requested width and height. Omitting it skips the size check.

Part A's pass criterion is the expected stream layout per mode, not that a file
appeared. Read by eye across sixteen runs, that is sixteen chances for a
wrong-but-plausible pass — and the manual-judgement layer is the one that has
produced every confidently-wrong artefact on this build.

Three properties this is built around, each learned here:

**Expectations come from the REGISTRY, resolved on every call.** Never a snapshot
copied into this file. A second place the truth lives is a second place it can
drift from, and the drift would be invisible precisely because both sides would
agree with themselves.

**It must fail in BOTH directions.** A missing declared stream is the obvious
half. An *undeclared* stream present is the half the exports contract has
historically been weak on, and it is the one that means a "video only" mode is
silently shipping sound — or that `exports` has stopped deciding the container's
contents at all.

**"Could not probe" is never a pass.** It exits 2, distinct from both 0 and 1.
`startup_report` once defaulted to "nothing is missing" and reported 37/37 having
checked nothing; this is that same shape one layer out.

Exit codes:  0 = matches   1 = mismatch   2 = could not probe, or a usage error
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ltx25_studio import modes  # noqa: E402

#: The vocoder is stereo. A mono track is a silent quality regression, not a
#: format difference -- it means something downmixed on the way out.
EXPECTED_AUDIO_CHANNELS = 2


@dataclass
class Verdict:
    mode: str
    path: str
    probed: bool
    ok: bool
    expected: dict = field(default_factory=dict)
    actual: dict = field(default_factory=dict)
    problems: list[str] = field(default_factory=list)

    @property
    def exit_code(self) -> int:
        if not self.probed:
            return 2
        return 0 if self.ok else 1

    def render(self) -> str:
        head = {0: "MATCHES", 1: "MISMATCH", 2: "COULD NOT PROBE"}[self.exit_code]
        lines = [f"{head}  mode={self.mode}  file={self.path}"]
        if self.probed:
            lines.append(f"  expected : {self.expected}")
            lines.append(f"  actual   : {self.actual}")
        lines += [f"  ✗ {p}" for p in self.problems]
        return "\n".join(lines)


def expected_streams(mode_id: str) -> dict:
    """Resolved from the registry, every call. Never cached, never copied."""
    spec = modes.get_mode(mode_id)
    return {
        "video": "video" in spec.exports,
        "audio": "audio" in spec.exports,
        "audio_channels": EXPECTED_AUDIO_CHANNELS if "audio" in spec.exports else None,
    }


def probe_streams(path) -> dict | None:
    """The container's actual layout, or None if it cannot be read."""
    try:
        import av
    except ImportError:
        return None
    try:
        with av.open(str(path)) as container:
            actual: dict = {"video": False, "audio": False}
            for stream in container.streams:
                if stream.type == "video":
                    actual["video"] = True
                    actual["video_codec"] = stream.codec_context.name
                    rate = stream.average_rate
                    actual["frame_rate"] = float(rate) if rate else None
                    actual["width"] = stream.codec_context.width
                    actual["height"] = stream.codec_context.height
                    actual["size"] = f"{actual['width']}x{actual['height']}"
                elif stream.type == "audio":
                    actual["audio"] = True
                    actual["audio_codec"] = stream.codec_context.name
                    actual["audio_channels"] = stream.codec_context.channels
                    actual["sampling_rate"] = stream.codec_context.sample_rate
            return actual
    except Exception:
        return None


def verify(mode_id, path, *, frame_rate=None, sampling_rate=None, geometry=None) -> Verdict:
    """Compare a real container against what the mode declares it exports.

    ``geometry`` is an optional ``(width, height)`` the picture must have.
    """
    expected = expected_streams(mode_id)
    if geometry is not None:
        expected["size"] = "{}x{}".format(*geometry)
    actual = probe_streams(path)
    if actual is None:
        return Verdict(
            mode=mode_id,
            path=str(path),
            probed=False,
            ok=False,
            expected=expected,
            problems=[
                f"{path} could not be opened as a media container. This is NOT a pass: "
                f"nothing about the stream layout has been established."
            ],
        )

    problems: list[str] = []
    for track in ("video", "audio"):
        if expected[track] and not actual[track]:
            problems.append(
                f"mode declares it exports {track}, and the file has NO {track} stream — "
                f"the run produced an artifact missing a track it promised"
            )
        if not expected[track] and actual[track]:
            problems.append(
                f"the file has a {track} stream and the mode does NOT declare one — "
                f"`exports` is not deciding the container's contents"
            )

    if expected["audio"] and actual["audio"]:
        channels = actual.get("audio_channels")
        if channels is not None and channels != EXPECTED_AUDIO_CHANNELS:
            problems.append(
                f"audio has {channels} channel(s), expected {EXPECTED_AUDIO_CHANNELS} — "
                f"something downmixed the vocoder's stereo output"
            )
        if sampling_rate is not None and actual.get("sampling_rate") not in (None, sampling_rate):
            problems.append(
                f"audio sample rate is {actual.get('sampling_rate')}, expected {sampling_rate} "
                f"— the container disagrees with the vocoder"
            )

    if (
        frame_rate is not None
        and actual.get("frame_rate") is not None
        and abs(actual["frame_rate"] - frame_rate) > 0.01
    ):
        problems.append(
            f"frame rate is {actual['frame_rate']}, expected {frame_rate} — a container "
            f"written at the wrong rate plays at the wrong speed and reports no error"
        )

    if geometry is not None:
        if not actual["video"]:
            problems.append(
                f"a size of {expected['size']} was asked for and the file has no video stream "
                f"to measure"
            )
        elif (actual.get("width"), actual.get("height")) != tuple(geometry):
            problems.append(
                f"the picture is {actual.get('size')}, expected {expected['size']} -- the file "
                f"is not the size the run was asked for"
            )

    return Verdict(
        mode=mode_id,
        path=str(path),
        probed=True,
        ok=not problems,
        expected=expected,
        actual=actual,
        problems=problems,
    )


def _geometry(value: str) -> tuple[int, int]:
    width, sep, height = value.lower().partition("x")
    try:
        size = (int(width), int(height))
    except ValueError:
        size = None
    if not sep or size is None or min(size) <= 0:
        raise argparse.ArgumentTypeError(f"{value!r} is not WIDTHxHEIGHT, e.g. 1408x960")
    return size


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="verify_streams.py",
        description="Verify a produced artifact against the mode's declared exports.",
        epilog="Exit codes: 0 = matches, 1 = mismatch, 2 = could not probe or a usage error.",
        allow_abbrev=False,
    )
    parser.add_argument("mode_id")
    parser.add_argument("file")
    parser.add_argument("--frame-rate", type=float, metavar="FPS")
    parser.add_argument("--sampling-rate", type=int, metavar="HZ")
    parser.add_argument("--geometry", type=_geometry, metavar="WxH")
    return parser


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    try:
        args = _parser().parse_args(argv)
    except SystemExit as exc:
        # argparse has already printed the problem; a usage error is never a mismatch.
        return 0 if exc.code == 0 else 2
    try:
        verdict = verify(
            args.mode_id,
            args.file,
            frame_rate=args.frame_rate,
            sampling_rate=args.sampling_rate,
            geometry=args.geometry,
        )
    except Exception as exc:
        print(f"COULD NOT PROBE  mode={args.mode_id}  file={args.file}\n  ✗ {exc}")
        return 2
    print(verdict.render())
    return verdict.exit_code


if __name__ == "__main__":
    raise SystemExit(main())
