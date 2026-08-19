"""Audio level across every produced artifact. Catches the silent-but-valid track.

    ./scripts/hinode_run.sh host_audio_levels.py

`verify_streams` proves a mode declared audio and the container has an audio
stream with the right codec, rate and channel count. It cannot tell how much *sound*
is in it — and a track 50 dB below its siblings passes every structural check.
That is the same failure family as an empty output: valid-shaped and not what
anyone wanted.

So this reports peak, rms and dBFS per artifact, and flags:

* **VERY-QUIET** below -40 dBFS — faint, not silent. -55 dBFS is audible when
  turned up, so the flag means "far below its siblings", not "nothing there"
* **clipping** above 1.0 — the decoder overshooting full scale
* **near-mono** when |L-R| is a negligible fraction of the signal's own level;
  judged relative rather than absolutely, because an absolute threshold calls a
  quiet track "stereo" on a difference that is a rounding error next to its
  amplitude.
"""

import glob
import os

import numpy as np


def levels(path):
    import av

    with av.open(path) as container:
        streams = [s for s in container.streams if s.type == "audio"]
        if not streams:
            return None
        stream = streams[0]
        rate = stream.codec_context.sample_rate
        channels = stream.codec_context.channels
        planes = [f.to_ndarray() for f in container.decode(audio=0)]
    if not planes:
        return None
    raw = np.concatenate(planes, axis=-1)
    integer = np.issubdtype(raw.dtype, np.integer)
    raw = raw.astype("float32")
    if integer:
        raw /= 32768.0
    # s16 is PACKED: a frame is (1, n*channels) interleaved, not (channels, n).
    if raw.shape[0] == 1 and channels > 1:
        raw = raw.reshape(-1, channels).T
    peak = float(np.abs(raw).max())
    rms = float(np.sqrt((raw**2).mean()))
    dbfs = 20 * np.log10(rms + 1e-12)
    level = float(np.abs(raw).mean())
    delta = float(np.abs(raw[0] - raw[1]).mean()) if raw.shape[0] >= 2 else None
    ratio = (delta / level) if (delta is not None and level > 0) else None
    return dict(rate=rate, channels=channels, peak=peak, rms=rms, dbfs=dbfs, ratio=ratio)


def main() -> int:
    root = os.environ.get("LEVELS_ROOT", "/home/wakeuser/ltx25-studio/outputs")
    files = sorted(glob.glob(os.path.join(root, "*", "output.*")))
    header = f"{'artifact':<40} {'ch':>2} {'rate':>6} {'peak':>8} {'rms':>10} {'dBFS':>7} {'L-R%':>7}  verdict"
    print(header, flush=True)
    print("-" * len(header), flush=True)
    for path in files:
        label = os.path.join(os.path.basename(os.path.dirname(path)), os.path.basename(path))
        try:
            info = levels(path)
        except Exception as exc:
            print(f"{label:<40} ERROR {type(exc).__name__}: {exc}", flush=True)
            continue
        if info is None:
            continue
        flags = []
        if info["dbfs"] < -40:
            # "very quiet", not "silent". -55 dBFS is faint but plainly audible
            # when turned up, and calling it silence overstates the measurement --
            # which invited a real disagreement with someone who had listened.
            flags.append("VERY-QUIET (<-40 dBFS)")
        if info["peak"] > 1.0:
            flags.append("clipping")
        if info["ratio"] is not None and info["ratio"] <= 0.01:
            flags.append("near-mono")
        ratio = "-" if info["ratio"] is None else f"{info['ratio'] * 100:.2f}"
        print(
            f"{label:<40} {info['channels']:>2} {info['rate']:>6} {info['peak']:>8.4f} "
            f"{info['rms']:>10.6f} {info['dbfs']:>7.1f} {ratio:>7}  "
            f"{', '.join(flags) or 'ok'}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
