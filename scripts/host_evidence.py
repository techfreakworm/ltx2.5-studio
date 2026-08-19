"""Turn artifacts into something a person can judge, and measure the pairs.

    ./scripts/hinode_run.sh host_evidence.py

Two jobs the stream verifier cannot do.

**Frames a person can look at.** Sampled ACROSS the clip, never all from the
start: a pipeline that degrades after frame 40 looks perfect in frame 1. Every
stream check in this matrix passes on a file whose pixels are garbage, and
"output looks right?" is the column that has produced every confidently-wrong
artifact on this build.

**Spectrograms, because silence and noise have identical stream layout.** A wav
with the correct codec, rate and channel count tells you nothing about whether
there is signal in it. A per-channel spectrogram distinguishes silence, noise and
structure instantly, and it also shows whether the two channels actually differ —
a mono downmix is invisible in a channel count of 2.

**Same-seed pairs are measured, not eyeballed.** H14-vs-H1 and the strength pairs
must DIFFER; identical output means the lever reached nothing. Mean absolute pixel
difference over co-located frames is the check, reported alongside file sizes so a
"differs" claim rests on pixels rather than on byte counts that could differ for
container reasons.
"""

import os
from pathlib import Path

import numpy as np

OUT = Path(os.environ.get("EVIDENCE_DIR", "/home/wakeuser/ltx25-studio/evidence"))

#: run_dir -> label. Set EVIDENCE_PAIRS / EVIDENCE_FRAMES to override.
FRAME_SOURCES = os.environ.get("EVIDENCE_FRAMES", "").split(",")
PAIRS = os.environ.get("EVIDENCE_PAIRS", "").split(";")
SPECTRA = os.environ.get("EVIDENCE_SPECTRA", "").split(",")


def log(*a):
    print(*a, flush=True)


def decode_frames(path, indices=None, limit=None):
    """Decode a video into a list of (index, ndarray) in RGB."""
    import av

    frames = []
    with av.open(str(path)) as container:
        for i, frame in enumerate(container.decode(video=0)):
            if indices is not None and i not in indices:
                continue
            frames.append((i, frame.to_ndarray(format="rgb24")))
            if limit and len(frames) >= limit:
                break
            if indices is not None and len(frames) == len(indices):
                break
    return frames


def frame_count(path):
    import av

    with av.open(str(path)) as container:
        return sum(1 for _ in container.decode(video=0))


def save_frames(path, label, count=6):
    """Save `count` frames spread evenly across the clip, not clustered at the start."""
    from PIL import Image

    total = frame_count(path)
    if total == 0:
        log(f"  {label}: NO FRAMES")
        return []
    # Evenly spaced across the whole clip, first and last included.
    picks = sorted({int(round(i * (total - 1) / max(count - 1, 1))) for i in range(count)})
    got = decode_frames(path, indices=set(picks))
    written = []
    for index, array in got:
        dest = OUT / f"{label}_frame{index:04d}.png"
        Image.fromarray(array).save(dest)
        written.append(dest)
        # A frame that is a flat colour is a red flag the stream check cannot see.
        log(f"  {label} frame {index:4d}/{total - 1}: "
            f"mean {array.mean():6.2f} std {array.std():6.2f} "
            f"min {array.min():3d} max {array.max():3d} -> {dest.name}")
    return written


def compare(label_a, path_a, label_b, path_b):
    """Mean absolute pixel difference over co-located frames. Must be > 0."""
    n_a, n_b = frame_count(path_a), frame_count(path_b)
    size_a, size_b = Path(path_a).stat().st_size, Path(path_b).stat().st_size
    log(f"  {label_a}: {n_a} frames, {size_a} bytes")
    log(f"  {label_b}: {n_b} frames, {size_b} bytes")
    shared = min(n_a, n_b)
    if shared == 0:
        log("  cannot compare: one clip has no frames")
        return None
    picks = sorted({int(round(i * (shared - 1) / 4)) for i in range(5)})
    fa = dict(decode_frames(path_a, indices=set(picks)))
    fb = dict(decode_frames(path_b, indices=set(picks)))
    diffs = []
    for i in picks:
        if i not in fa or i not in fb:
            continue
        a, b = fa[i].astype("float32"), fb[i].astype("float32")
        if a.shape != b.shape:
            log(f"    frame {i}: shapes differ {a.shape} vs {b.shape} -> DIFFERENT by geometry")
            return float("inf")
        d = float(np.abs(a - b).mean())
        diffs.append(d)
        log(f"    frame {i:4d}: mean|diff| = {d:8.4f}")
    if not diffs:
        return None
    overall = sum(diffs) / len(diffs)
    verdict = "DIFFER (good)" if overall > 0.01 else "*** IDENTICAL -- the lever reached nothing ***"
    log(f"  mean|diff| across {len(diffs)} frames = {overall:.4f}  -> {verdict}")
    return overall


def spectrogram(path, label):
    """Per-channel spectrogram PNG, plus the numbers that catch silence and mono."""
    import av
    from PIL import Image

    planes = []
    with av.open(str(path)) as container:
        stream = next(s for s in container.streams if s.type == "audio")
        rate, channels = stream.codec_context.sample_rate, stream.codec_context.channels
        for frame in container.decode(audio=0):
            planes.append(frame.to_ndarray())
    raw = np.concatenate(planes, axis=-1)
    # s16 is PACKED: a frame is (1, n*channels) interleaved, not (channels, n).
    if raw.shape[0] == 1 and channels > 1:
        raw = raw.reshape(-1, channels).T
    raw = raw.astype("float32")
    if np.issubdtype(np.asarray(planes[0]).dtype, np.integer):
        raw /= 32768.0

    log(f"  {label}: {channels} ch @ {rate} Hz, {raw.shape[1]} samples "
        f"({raw.shape[1] / rate:.3f} s)")
    for c in range(raw.shape[0]):
        ch = raw[c]
        peak, rms = float(np.abs(ch).max()), float(np.sqrt((ch**2).mean()))
        log(f"    ch{c}: peak {peak:.4f} rms {rms:.6f}"
            + ("   *** SILENT ***" if peak < 1e-4 else ""))
    if raw.shape[0] >= 2:
        # A channel count of 2 does not mean two different channels.
        delta = float(np.abs(raw[0] - raw[1]).mean())
        log(f"    |L-R| mean = {delta:.6f}  -> "
            + ("channels DIFFER (real stereo)" if delta > 1e-5
               else "*** channels IDENTICAL -- effectively mono ***"))

    # STFT by hand: no scipy in the pinned set, and numpy is enough.
    win = 1024
    hop = 256
    window = np.hanning(win).astype("float32")
    images = []
    for c in range(raw.shape[0]):
        ch = raw[c]
        n = max((len(ch) - win) // hop, 1)
        cols = []
        for i in range(n):
            seg = ch[i * hop : i * hop + win]
            if len(seg) < win:
                break
            cols.append(np.abs(np.fft.rfft(seg * window)))
        if not cols:
            continue
        spec = np.stack(cols, axis=1)
        db = 20 * np.log10(spec + 1e-8)
        db = np.clip((db + 80) / 80, 0, 1)  # -80 dB floor
        img = (db[::-1] * 255).astype("uint8")
        images.append(img)
    if images:
        stacked = np.concatenate(images, axis=0) if len(images) > 1 else images[0]
        dest = OUT / f"{label}_spectrogram.png"
        Image.fromarray(stacked).save(dest)
        log(f"    -> {dest.name} ({len(images)} channel(s) stacked, "
            f"{stacked.shape[1]} frames x {stacked.shape[0]} bins, -80 dB floor)")


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    log(f"evidence dir: {OUT}")

    if any(FRAME_SOURCES):
        log("\n=== FRAMES (sampled across the clip, not from the start) ===")
        for entry in FRAME_SOURCES:
            if not entry.strip():
                continue
            label, _, path = entry.partition("=")
            save_frames(path, label.strip(), count=int(os.environ.get("EVIDENCE_FRAME_COUNT", "6")))

    if any(SPECTRA):
        log("\n=== SPECTROGRAMS (silence and noise have identical stream layout) ===")
        for entry in SPECTRA:
            if not entry.strip():
                continue
            label, _, path = entry.partition("=")
            spectrogram(path, label.strip())

    if any(PAIRS):
        log("\n=== SAME-SEED PAIRS (must differ; identical means the lever is dead) ===")
        for entry in PAIRS:
            if not entry.strip():
                continue
            left, _, right = entry.partition("|")
            la, _, pa = left.partition("=")
            lb, _, pb = right.partition("=")
            log(f"{la.strip()} vs {lb.strip()}:")
            compare(la.strip(), pa.strip(), lb.strip(), pb.strip())

    log(f"\nwrote: {sorted(p.name for p in OUT.iterdir())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
