"""Measure what app.py's module-scope startup actually costs on a real card.

Run on hinode; see docs/HINODE_HOST_SETUP.md.

``app.py`` does this at import when ``will_use_gpu()`` is true::

    pipelines.get_pipeline(pipeline="condition",  device="cuda")
    pipelines.get_pipeline(pipeline="text2video", device="cuda")

The question is whether those two calls **share** one loaded copy of the model.
Two independent ``from_pretrained`` calls would materialise two complete copies --
on ZeroGPU a disk cost against the 150 GB cap, but on this 96 GiB card an outright
OOM at startup. So this asserts identity (``is``), not merely equal shapes: two
distinct modules holding identical weights would pass any size comparison and
still cost twice the VRAM.

Prints per-component parameter+buffer bytes so the arithmetic is visible rather
than inferred from a single total.
"""

import resource

import torch

from ltx25_studio import pipelines

GiB = 1024**3


def rss_gib() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024 / GiB


def smi_used_mib() -> int:
    import subprocess

    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    return int(out.splitlines()[0])


def component_bytes(pipe) -> dict[str, int]:
    rows: dict[str, int] = {}
    for name, mod in (getattr(pipe, "components", None) or {}).items():
        if not isinstance(mod, torch.nn.Module):
            continue
        total = sum(p.numel() * p.element_size() for p in mod.parameters())
        total += sum(b.numel() * b.element_size() for b in mod.buffers())
        rows[name] = total
    return rows


def report(label: str, rows: dict[str, int]) -> int:
    total = sum(rows.values())
    print(f"  {label}: {len(rows)} nn.Module components, {total / GiB:.3f} GiB total")
    for name, n in sorted(rows.items(), key=lambda kv: -kv[1]):
        print(f"    {name:<28} {n / GiB:8.3f} GiB")
    return total


def devices_of(pipe) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for name, mod in (getattr(pipe, "components", None) or {}).items():
        if isinstance(mod, torch.nn.Module):
            dev = next((str(p.device) for p in mod.parameters()), "no-params")
            out.setdefault(dev, []).append(name)
    return {k: sorted(v) for k, v in out.items()}


def main() -> int:
    print(
        f"torch {torch.__version__}  device {torch.cuda.get_device_name()}  "
        f"cap {torch.cuda.get_device_capability()}"
    )
    print(
        f"BASELINE  cuda allocated {torch.cuda.memory_allocated() / GiB:.3f} GiB   "
        f"nvidia-smi used {smi_used_mib()} MiB (includes another user's process)\n",
        flush=True,
    )

    print("=== 1. get_pipeline(pipeline='condition', device='cuda') ===", flush=True)
    r1 = pipelines.get_pipeline(pipeline="condition", device="cuda")
    p1 = r1.pipeline
    a1 = torch.cuda.memory_allocated()
    print(f"  class {type(p1).__name__}   peak RSS {rss_gib():.2f} GiB")
    t1 = report("condition", component_bytes(p1))
    print(f"  devices: {devices_of(p1)}")
    print(f"  torch.cuda.memory_allocated() = {a1} bytes ({a1 / GiB:.3f} GiB)")
    print(f"  torch.cuda.memory_reserved()  = {torch.cuda.memory_reserved() / GiB:.3f} GiB")
    print(f"  nvidia-smi used               = {smi_used_mib()} MiB\n", flush=True)

    print("=== 2. get_pipeline(pipeline='text2video', device='cuda') ===", flush=True)
    r2 = pipelines.get_pipeline(pipeline="text2video", device="cuda")
    p2 = r2.pipeline
    a2 = torch.cuda.memory_allocated()
    print(f"  class {type(p2).__name__}   peak RSS {rss_gib():.2f} GiB")
    t2 = report("text2video", component_bytes(p2))
    print(f"  devices: {devices_of(p2)}")
    print(f"  torch.cuda.memory_allocated() = {a2} bytes ({a2 / GiB:.3f} GiB)")
    print(f"  DELTA from pipeline 1         = {a2 - a1} bytes ({(a2 - a1) / GiB:.3f} GiB)")
    print(f"  nvidia-smi used               = {smi_used_mib()} MiB\n", flush=True)

    print("=== 3. DO THEY SHARE COMPONENTS? identity, not equality ===")
    c1 = getattr(p1, "components", None) or {}
    c2 = getattr(p2, "components", None) or {}
    both = sorted(set(c1) & set(c2))
    print(f"  condition components : {sorted(c1)}")
    print(f"  text2video components: {sorted(c2)}")
    print(f"  only in condition    : {sorted(set(c1) - set(c2))}")
    print(f"  only in text2video   : {sorted(set(c2) - set(c1))}")
    shared = 0
    checked = 0
    for name in both:
        a, b = c1[name], c2[name]
        if a is None and b is None:
            print(f"    {name:<28} both None (declared, no weights)")
            continue
        checked += 1
        same = a is b
        shared += int(same)
        print(f"    {name:<28} shared={same!s:<5} id1={id(a)}  id2={id(b)}")
    print(f"  VERDICT: {shared}/{checked} shared by identity")
    print(
        f"  one copy {t1 / GiB:.3f} GiB; two independent copies would need "
        f"{(t1 + t2) / GiB:.3f} GiB against a {97887 * 1024 * 1024 / GiB:.3f} GiB card"
    )
    print(f"  peak RSS overall {rss_gib():.2f} GiB")
    if shared != checked:
        print("  *** NOT SHARED -- this is the two-copy failure, report it ***")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
