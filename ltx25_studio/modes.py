"""The 11-mode registry: mode metadata as data, plus per-mode kwarg builders.

PLAN §3.2. This module is data, not branching — adding a mode means adding a
registry entry, not adding an ``if`` to ``generate.py`` (§7.2).

The honest part (§3.2): the model is jointly audio-visual and there is no
video-off switch, so modes 8-11 ("audio" modes) still pay full video compute.
The registry carries that cost note so the UI can show it, and carries the
mitigation flags — minimum resolution, skip stage-2 upscale, export one track.
"""
