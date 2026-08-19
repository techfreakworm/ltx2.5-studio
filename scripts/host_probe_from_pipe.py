"""Probe: does ``from_pipe`` share modules, and what does omitting dtype cost?

Run on hinode; see docs/HINODE_HOST_SETUP.md. Read-only with respect to the app —
it calls diffusers directly and does not touch ``pipelines.derive_pipeline``.

Measured on the host: ``derive_pipeline``'s bare ``from_pipe(base)`` OOMs a 96 GiB
card. The cause is in diffusers, not in us::

    pipeline_utils.py:2155  dtype = _resolve_dtype(...) or torch.float32
    pipeline_utils.py:2253  new_pipeline.to(dtype=dtype)

With no dtype passed, ``from_pipe`` defaults to **float32** and then casts every
shared module. Two consequences, and the second is worse than the OOM:

1. ``t.to(float32)`` on a bf16 tensor allocates a NEW tensor at twice the bytes,
   so the derived pipeline is a second copy — 130 GiB for a 65 GiB model.
2. ``nn.Module.to()`` mutates in place. The modules ARE shared, so casting the
   derived pipeline casts the base's modules too. Had it fit, the base pipeline
   would have silently become fp32.

This probe checks the remedy: pass the base's own dtype, so every ``t.to()`` is a
no-op that returns ``self``. Expected result — every component shared by identity,
and a cuda-allocated delta of zero.
"""

import torch
from diffusers import LTX2Pipeline

from ltx25_studio import pipelines

GiB = 1024**3


def main() -> int:
    print(f"torch {torch.__version__}  {torch.cuda.get_device_name()}")

    base = pipelines._base_pipeline(device="cuda")
    a1 = torch.cuda.memory_allocated()
    print(f"base {type(base).__name__} on cuda: {a1} bytes ({a1 / GiB:.3f} GiB)")
    print(f"base.dtype = {base.dtype}")

    print("\n=== from_pipe(base, dtype=base.dtype) -- the remedy ===", flush=True)
    derived = LTX2Pipeline.from_pipe(base, dtype=base.dtype)
    a2 = torch.cuda.memory_allocated()
    print(f"derived {type(derived).__name__}")
    print(f"cuda allocated after = {a2} bytes ({a2 / GiB:.3f} GiB)")
    print(f"DELTA                = {a2 - a1} bytes ({(a2 - a1) / GiB:.6f} GiB)")

    print("\n=== identity of every shared component ===")
    c1, c2 = base.components, derived.components
    both = sorted(set(c1) & set(c2))
    shared = checked = 0
    for name in both:
        a, b = c1[name], c2[name]
        if a is None and b is None:
            print(f"  {name:<24} both None")
            continue
        checked += 1
        same = a is b
        shared += int(same)
        dt = getattr(a, "dtype", None)
        print(f"  {name:<24} shared={same!s:<5} dtype={dt} id1={id(a)} id2={id(b)}")
    print(f"  only in base    : {sorted(set(c1) - set(c2))}")
    print(f"  only in derived : {sorted(set(c2) - set(c1))}")
    print(f"VERDICT: {shared}/{checked} shared by identity; delta {(a2 - a1) / GiB:.6f} GiB")

    # The base must still be bf16. A cast would have mutated it through the
    # shared modules, which is the silent half of the bug.
    print(f"base.dtype after derive = {base.dtype}  (must still be torch.bfloat16)")

    # Tolerance, not equality. The measured delta is -1024 bytes: the caching
    # allocator's own bookkeeping moves by a KiB or so across the call, in either
    # direction, and demanding an exact zero fails on that noise while a real
    # second copy would be tens of GiB. 1 MiB is far below anything that matters
    # and far above allocator jitter.
    tolerance = 1 << 20
    no_copy = abs(a2 - a1) < tolerance
    ok = shared == checked and no_copy and base.dtype == torch.bfloat16
    print(f"no second copy (|delta| < 1 MiB): {no_copy}")
    print("PROBE:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
