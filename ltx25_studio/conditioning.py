"""UI inputs -> ``list[LTX2VideoCondition]``, including multishot anchoring.

PLAN §3.1(a): image-conditioning and video-conditioning are the SAME mechanism.
``LTX2VideoCondition.frames`` accepts a PIL image or a video, and carries an
``index`` (the frame at which it takes effect) and a ``strength``.

PLAN §3.3: multishot is built on top of this condition list — a prompt segment
plus an anchor frame index per shot — not as a separate pipeline or mode.
"""
