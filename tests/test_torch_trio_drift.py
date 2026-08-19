"""The missing leg of the torch-trio drift guard — PLAN §5.1, §8.1.

``test_deploy_drift.py`` checks that ``torch`` and ``torchaudio`` are pinned to
the same version, and ``requirements.txt`` says in as many words that this is not
where the trap actually is:

    NOTE for whoever bumps these: the drift test guarding this trio checks that
    torch and torchaudio agree, but does NOT yet cover torchvision — which is
    where the trap actually lives. [...] Verify the torchvision <-> torch pairing
    by hand until that test covers it.

"Verify by hand" is a guard that works right up until someone is in a hurry.

**Why torchvision is the dangerous one.** Every torchvision release
*hard-declares* the exact torch it was built against — ``torchvision 0.28.0``
requires ``torch==2.13.0``. ZeroGPU supports exactly torch 2.8.0 / 2.9.1 /
2.10.0 / 2.11.0 (CUDA 12.8; sm_120 Blackwell needs cu128+), so bumping
torchvision alone does not produce a resolution error that anyone reads — pip
happily satisfies the declaration by dragging torch straight past the ceiling,
and the Space then fails on hardware nobody can reproduce locally.

So the pairing is asserted here rather than trusted to a comment. The rule is
arithmetic (``torchvision 0.N.x`` is built against ``torch 2.(N-15).x``) and both
data points ``requirements.txt`` cites are used as anchors, so if upstream ever
stops numbering that way the anchor test says so before the pairing test starts
lying.

Nothing here downloads anything; it is file parsing, like the rest of the drift
suite. The parser is local rather than imported from ``test_deploy_drift.py``:
six lines of duplication is cheaper than coupling two spec files together.
"""

from __future__ import annotations

import re

import pytest
from conftest import REPO_ROOT

REQUIREMENTS = REPO_ROOT / "requirements.txt"

#: ``torchvision 0.N.x`` hard-declares ``torch 2.(N - OFFSET).x``.
TORCHVISION_MINOR_OFFSET = 15

#: The two pairings ``requirements.txt`` states as fact. They anchor the rule:
#: a rule with no anchor is just an assumption with arithmetic on it.
VERIFIED_PAIRINGS = {
    "0.26.0": "2.11.0",  # the current pin — self-consistent
    "0.28.0": "2.13.0",  # the documented trap
}

#: PLAN §8.1: ZeroGPU is sm_120 Blackwell on CUDA 12.8. These are the builds that
#: exist for it. 2.11.0 is the ceiling.
ZEROGPU_SUPPORTED_TORCH = ("2.8.0", "2.9.1", "2.10.0", "2.11.0")
ZEROGPU_TORCH_CEILING = "2.11.0"

TRIO = ("torch", "torchvision", "torchaudio")


def requirement_pins() -> dict[str, str]:
    """``{distribution: exact version}`` for every ``name==version`` line."""
    pins = {}
    for raw in REQUIREMENTS.read_text().splitlines():
        line = raw.split("#")[0].strip()
        match = re.fullmatch(r"([A-Za-z0-9._-]+)\s*==\s*([0-9][^\s;]*)", line)
        if match:
            pins[match.group(1).lower()] = match.group(2)
    return pins


def version_tuple(version: str) -> tuple[int, ...]:
    return tuple(int(part) for part in version.split("."))


def torch_declared_by(torchvision_version: str) -> str:
    """The torch that this torchvision release hard-declares."""
    major, minor, patch = version_tuple(torchvision_version)
    assert major == 0, f"torchvision {torchvision_version} is not 0.x; the rule assumed it was"
    return f"2.{minor - TORCHVISION_MINOR_OFFSET}.{patch}"


# --------------------------------------------------------------------------
# the rule, before anything is asserted with it
# --------------------------------------------------------------------------


@pytest.mark.parametrize(("torchvision_version", "torch_version"), VERIFIED_PAIRINGS.items())
def test_the_pairing_rule_reproduces_the_pairings_we_verified_by_hand(
    torchvision_version, torch_version
):
    """Anchors the arithmetic on the two data points ``requirements.txt`` cites.

    If upstream ever renumbers, this fails first and loudly — rather than the
    pairing test below quietly checking the wrong thing.
    """
    assert torch_declared_by(torchvision_version) == torch_version, (
        f"the rule says torchvision {torchvision_version} declares "
        f"torch {torch_declared_by(torchvision_version)}, but requirements.txt records "
        f"{torch_version}. The offset is no longer {TORCHVISION_MINOR_OFFSET}."
    )


# --------------------------------------------------------------------------
# the trio moves together, or it does not move
# --------------------------------------------------------------------------


@pytest.mark.parametrize("name", TRIO)
def test_every_member_of_the_torch_trio_is_pinned_exactly(name):
    """A range anywhere in the trio is a resolver free to break the other two."""
    pins = requirement_pins()
    assert name in pins, (
        f"{name} must be pinned with == in requirements.txt; the torch trio moves together "
        f"or the C++ ABI and the CUDA build stop agreeing."
    )


def test_the_torchvision_pin_declares_the_torch_we_actually_pinned():
    """**The leg the existing drift test does not cover.**

    torchvision hard-declares its torch. If that declaration disagrees with our
    torch pin, pip resolves the conflict in torchvision's favour — silently,
    because satisfying a hard declaration is not an error.
    """
    pins = requirement_pins()
    torch = pins.get("torch")
    torchvision = pins.get("torchvision")
    assert torch and torchvision, "both torch and torchvision must be pinned exactly"

    declared = torch_declared_by(torchvision)
    assert declared == torch, (
        f"torchvision {torchvision} hard-declares torch=={declared}, but requirements.txt "
        f"pins torch=={torch}. pip will satisfy torchvision and move torch to {declared} "
        f"without reporting a conflict."
    )


def test_the_torchvision_pin_cannot_drag_torch_past_the_zerogpu_ceiling():
    """The trap, stated as the consequence rather than the mechanism.

    ``torchvision 0.28.0`` declares ``torch==2.13.0``. ZeroGPU tops out at
    {ceiling}. Nothing in the build reports that as an error; the Space just
    fails on hardware nobody can reproduce locally.
    """
    pins = requirement_pins()
    torchvision = pins.get("torchvision")
    assert torchvision, "torchvision must be pinned exactly"

    declared = torch_declared_by(torchvision)
    assert version_tuple(declared) <= version_tuple(ZEROGPU_TORCH_CEILING), (
        f"torchvision {torchvision} declares torch=={declared}, which is past ZeroGPU's "
        f"{ZEROGPU_TORCH_CEILING} ceiling (PLAN §8.1: sm_120 Blackwell, CUDA 12.8). Letting "
        f"torchvision float is exactly how torch ends up somewhere the platform cannot run it."
    )


def test_the_torch_pin_is_one_zerogpu_actually_ships():
    """A version outside the supported set is a build that cannot run at all."""
    torch = requirement_pins().get("torch")
    assert torch in ZEROGPU_SUPPORTED_TORCH, (
        f"torch=={torch} is not one of ZeroGPU's builds {list(ZEROGPU_SUPPORTED_TORCH)} "
        f"(CUDA 12.8, sm_120)."
    )


def test_the_whole_trio_agrees_at_once():
    """All three legs in one assertion, so a bump is checked as one decision.

    torchaudio matches torch exactly (the C++ ABI), torchvision matches torch by
    declaration, and torch itself has to be a build ZeroGPU ships. Any one of the
    three moving alone is a silent break.
    """
    pins = requirement_pins()
    torch, torchvision, torchaudio = (pins.get(name) for name in TRIO)

    assert (torch, torchaudio, torch_declared_by(torchvision)) == (torch, torch, torch), (
        f"the torch trio disagrees: torch=={torch}, torchaudio=={torchaudio}, "
        f"torchvision=={torchvision} (which declares torch=={torch_declared_by(torchvision)})."
    )
    assert torch in ZEROGPU_SUPPORTED_TORCH


def test_the_requirements_note_about_this_hole_is_retired_when_it_is_filled():
    """The comment tells the next maintainer to verify torchvision by hand.

    Now that a test covers it, that instruction is stale — and a stale "verify by
    hand" note is worse than none, because it implies the guard is still absent.
    """
    text = REQUIREMENTS.read_text()
    assert "Verify the torchvision <-> torch pairing by hand" not in text, (
        "requirements.txt still tells the next maintainer to verify the torchvision pairing "
        "by hand and says the drift test does not cover it. tests/test_torch_trio_drift.py "
        "now does — please update that note so it stops describing a hole that is filled."
    )
