"""Spec: a judgement on the live gauge is made on a settled gauge (F73, items 1 and 2).

After the local runner frees device memory, the kernel un-wires its pages over a
few hundred milliseconds (measured on the operator's Mac, 2026-11-16: 24 GB of
text encoder on MPS, the gauge read every 0.1 s after the release gave 62.3,
59.3, 46.3, 38.3 GB, then flat). The transformer's preflight read it inside that
window and refused a run that fitted (committed 67.3 + need 25.1 = 92.4 GB > 90;
eight seconds later the gauge read 38.8). So, on the live gauge only:

* after a release, the gauge is read every 50 ms until two consecutive readings
  are within 0.25 GB, and that reading is the one judged and logged;
* a FAIL is read again every 0.1 s and passes as soon as a reading fits; only a
  failure that persists is refused, with the last reading;
* both waits share ``LTX25_MEMORY_SETTLE_SECONDS`` (default 2.0; 0 switches them
  off). A projected judgement, no gauge or no limit: one reading or none, as
  before.

Every gauge here is a scripted sequence on a clock that only sleeping moves, so
nothing really sleeps. Runs in both environments (no torch).
"""

from __future__ import annotations

import pickle

import pytest

from ltx25_studio import memory
from ltx25_studio.errors import ConfigError

SETTLE = "LTX25_MEMORY_SETTLE_SECONDS"


@pytest.fixture(autouse=True)
def _defaults(monkeypatch):
    monkeypatch.delenv(SETTLE, raising=False)
    monkeypatch.delenv("LTX25_MEMORY_LIMIT_GB", raising=False)


class Gauge:
    """Scripted readings (the last one repeats) on a clock that only sleeping moves."""

    def __init__(self, *readings):
        self.readings = list(readings)
        self.reads = 0
        self.now = 0.0
        self.slept: list[float] = []

    def read(self):
        value = self.readings[min(self.reads, len(self.readings) - 1)]
        self.reads += 1
        return value

    def clock(self):
        return self.now

    def sleep(self, seconds):
        assert seconds > 0
        self.slept.append(seconds)
        self.now += seconds

    def judge(self, phase="load transformer", *, need_gb=25.1, limit_gb=90.0, **extra):
        return memory.preflight(
            phase,
            need_gb=need_gb,
            limit_gb=limit_gb,
            read=self.read,
            clock=self.clock,
            sleep=self.sleep,
            **extra,
        )


class CoarseClock(Gauge):
    """A clock that moves in whole milliseconds: a shorter sleep does not move it."""

    def sleep(self, seconds):
        assert len(self.slept) < 10_000, "the judgement kept waiting on a clock that did not move"
        self.slept.append(seconds)
        self.now += round(seconds, 3)


class Rising(Gauge):
    """A gauge that never settles: 1 GB more at every reading."""

    def __init__(self, start):
        super().__init__(start)
        self.start = start

    def read(self):
        self.reads += 1
        return self.start + self.reads - 1


# --------------------------------------------------------------------------
# item 1: after a release, the settled reading is the one judged and logged
# --------------------------------------------------------------------------


def test_after_a_release_the_gauge_is_read_until_it_stops_falling_and_that_reading_is_judged():
    gauge = Gauge(62.3, 59.3, 46.3, 38.3, 38.3)

    result = gauge.judge(released=True)

    assert result.committed == 38.3 and result.verdict == "OK"
    assert result.settled.describe() == "memory settled in 0.20 s: 62.3 → 38.3 GB"
    assert result.describe() == (
        "preflight[load transformer]: committed 38.3 + need 25.1 = 63.4 GB <= 90.0 GB OK "
        "(headroom 26.6)"
    )
    assert gauge.reads == 5 and gauge.slept == [memory.SETTLE_STEP_SECONDS] * 4


def test_the_sessions_refused_run_is_judged_on_the_settled_gauge():
    """Run 1 of the 19:02 session: 67.3 GB right after the release, 38.8 once settled."""
    gauge = Gauge(67.3, 64.1, 51.2, 40.1, 38.8, 38.8)

    result = gauge.judge(released=True)

    assert result.verdict == "OK"
    assert result.committed == 38.8
    assert result.settled.first == 67.3 and result.settled.last == 38.8
    assert result.settled.steady
    result.require()


def test_a_release_followed_by_a_reading_that_fits_waits_only_for_the_settle():
    gauge = Gauge(40.0, 40.1)

    result = gauge.judge(released=True)

    assert result.verdict == "OK" and result.committed == 40.1
    assert gauge.slept == [memory.SETTLE_STEP_SECONDS], "one settle step, no reading again"
    assert result.first is None


def test_without_a_release_a_reading_that_fits_is_one_reading_and_no_wait():
    gauge = Gauge(62.3, 38.3)

    result = gauge.judge()

    assert result.committed == 62.3 and result.verdict == "OK"
    assert (gauge.reads, gauge.slept, result.settled) == (1, [], None)


def test_the_settle_is_bounded_and_says_so_when_the_gauge_keeps_moving():
    gauge = Rising(40.0)

    result = gauge.judge(released=True, limit_gb=500.0)

    assert result.verdict == "OK"
    assert not result.settled.steady
    assert sum(gauge.slept) <= memory.DEFAULT_SETTLE_SECONDS + 1e-9
    assert result.settled.seconds == pytest.approx(memory.DEFAULT_SETTLE_SECONDS)
    assert result.committed == result.settled.last == 40.0 + gauge.reads - 1
    assert result.settled.describe() == (
        f"memory still moving after 2.00 s: 40.0 → {result.committed:.1f} GB; "
        "judged on the last reading"
    )


# --------------------------------------------------------------------------
# item 2: a refusal is never based on a falling gauge
# --------------------------------------------------------------------------


def test_a_failing_reading_is_read_again_and_passes_as_soon_as_one_fits():
    gauge = Gauge(67.3, 66.0, 63.0, 44.8)

    result = gauge.judge()

    assert result.verdict == "OK" and result.committed == 63.0
    assert gauge.reads == 3, "it stops at the first reading that fits"
    assert gauge.slept == [memory.REREAD_STEP_SECONDS] * 2
    assert result.describe() == (
        "preflight[load transformer]: committed 67.3 + need 25.1 = 92.4 GB > 90.0 GB FAIL "
        "(short by 2.4); read again for 0.2 s: committed 63.0 + need 25.1 = 88.1 GB "
        "<= 90.0 GB OK (headroom 1.9)"
    )
    result.require()


def test_a_settle_that_stops_before_the_gauge_starts_falling_is_caught_by_reading_again():
    """Two readings taken before the kernel starts un-wiring agree; the FAIL is read again."""
    gauge = Gauge(67.3, 67.3, 59.3, 46.3, 38.3)

    result = gauge.judge(released=True)

    assert result.verdict == "OK" and result.committed == 59.3
    assert result.settled.describe() == "memory settled in 0.05 s: 67.3 → 67.3 GB"
    assert result.first.committed == 67.3 and result.first.verdict == "FAIL"
    assert "; read again for 0.1 s: committed 59.3 + need 25.1" in result.describe()


@pytest.mark.parametrize("released", [False, True])
def test_a_failure_that_persists_is_refused_after_the_bound_with_the_last_reading(released):
    gauge = Gauge(67.3, 67.1, 66.9)

    result = gauge.judge(released=released)

    assert result.verdict == "FAIL"
    assert sum(gauge.slept) <= memory.DEFAULT_SETTLE_SECONDS + 1e-9, "slept past the bound"
    assert sum(gauge.slept) == pytest.approx(memory.DEFAULT_SETTLE_SECONDS)
    assert result.committed == 66.9, "the refusal carries the last reading"
    line = result.describe()
    first = "committed 67.3 + need 25.1 = 92.4 GB > 90.0 GB FAIL (short by 2.4)"
    last = "committed 66.9 + need 25.1 = 92.0 GB > 90.0 GB FAIL (short by 2.0)"
    if released:
        # 67.3 then 67.1 agree within 0.25 GB: the settle stops there, and that fails first.
        assert result.settled.first == 67.3 and result.settled.last == 67.1
        first = "committed 67.1 + need 25.1 = 92.2 GB > 90.0 GB FAIL (short by 2.2)"
    assert line.startswith(f"preflight[load transformer]: {first}; read again for ")
    assert line.endswith(f": {last}")
    with pytest.raises(memory.MemoryPreflightError) as caught:
        result.require()
    message = str(caught.value)
    assert "66.9 GB is already in use" in message and "67.3" not in message
    assert "LTX25_MEMORY_LIMIT_GB" in message


# --------------------------------------------------------------------------
# what does not change
# --------------------------------------------------------------------------


@pytest.mark.parametrize("released", [False, True])
def test_no_gauge_is_report_only_exactly_as_before(monkeypatch, released):
    monkeypatch.setenv(SETTLE, "not even a number")  # never read without a gauge
    gauge = Gauge(None)

    result = gauge.judge("x", need_gb=5.0, released=released)

    assert result.verdict == "REPORT"
    assert (
        result.describe()
        == "preflight[x]: committed n/a + need 5.0 GB (not measured here; report only)"
    )
    assert (gauge.reads, gauge.slept, result.settled, result.first) == (1, [], None, None)


def test_no_limit_is_report_only_exactly_as_before():
    gauge = Gauge(95.0, 40.0)

    result = gauge.judge("x", need_gb=5.0, limit_gb=None, released=True)

    assert result.verdict == "REPORT"
    assert (
        result.describe()
        == "preflight[x]: committed 95.0 + need 5.0 = 100.0 GB (no limit; report only)"
    )
    assert (gauge.reads, gauge.slept) == (1, [])


def test_a_projected_judgement_never_reads_the_gauge_or_waits():
    gauge = Gauge(10.0)

    result = gauge.judge(committed_gb=95.0, released=True)

    assert result.verdict == "FAIL" and result.committed == 95.0
    assert (gauge.reads, gauge.slept, result.first) == (0, [], None)


def test_zero_seconds_is_the_single_reading_of_before(monkeypatch):
    monkeypatch.setenv(SETTLE, "0")
    gauge = Gauge(67.3, 38.3)

    result = gauge.judge(released=True)

    assert result.verdict == "FAIL" and result.committed == 67.3
    assert (gauge.reads, gauge.slept, result.settled, result.first) == (1, [], None, None)
    assert result.describe() == (
        "preflight[load transformer]: committed 67.3 + need 25.1 = 92.4 GB > 90.0 GB FAIL "
        "(short by 2.4)"
    )


def test_the_bound_comes_from_the_environment(monkeypatch):
    monkeypatch.setenv(SETTLE, "0.5")
    gauge = Gauge(67.3)

    result = gauge.judge(released=True)

    assert result.verdict == "FAIL"
    assert sum(gauge.slept) == pytest.approx(0.5)


def test_less_than_a_millisecond_left_of_the_bound_is_no_time_to_wait(monkeypatch):
    """A wait that kept asking for the last 0.4 ms never ended on a clock that did not move."""
    monkeypatch.setenv(SETTLE, "0.2004")
    gauge = CoarseClock(67.3)

    result = gauge.judge(released=True)

    assert result.verdict == "FAIL"
    assert gauge.slept == [0.05, 0.1, pytest.approx(0.0504)]
    assert sum(gauge.slept) <= 0.2004 + 1e-9


@pytest.mark.parametrize("value", ["soon", "-1", "nan", "inf", "61", "1e9"])
def test_a_bad_settle_value_is_refused_like_the_other_memory_variables(monkeypatch, value):
    monkeypatch.setenv(SETTLE, value)
    with pytest.raises(ConfigError, match=SETTLE) as caught:
        memory.settle_seconds()
    assert caught.value.variable == SETTLE and caught.value.value == value
    with pytest.raises(ConfigError, match=SETTLE):
        Gauge(40.0).judge(released=True)


@pytest.mark.parametrize(
    ("value", "seconds"), [("0", 0.0), ("0.25", 0.25), ("2", 2.0), ("60", 60.0)]
)
def test_every_settle_value_from_zero_to_a_minute_is_accepted(monkeypatch, value, seconds):
    monkeypatch.setenv(SETTLE, value)
    assert memory.settle_seconds() == seconds


def test_the_defaults_are_the_measured_ones():
    assert memory.settle_seconds() == memory.DEFAULT_SETTLE_SECONDS == 2.0
    assert memory.SETTLE_STEP_SECONDS == 0.05
    assert memory.SETTLE_TOLERANCE_GB == 0.25
    assert memory.REREAD_STEP_SECONDS == 0.1


def test_a_judgement_read_again_survives_the_fork_boundary():
    result = Gauge(67.3, 67.3, 66.9).judge(released=True)
    assert result.first is not None and result.settled is not None

    back = pickle.loads(pickle.dumps(result))
    error = memory.MemoryPreflightError(result)
    clone = pickle.loads(pickle.dumps(error))

    assert back == result and back.describe() == result.describe()
    assert type(clone) is memory.MemoryPreflightError and str(clone) == str(error)
    assert clone.result == result


def test_the_live_judgement_reads_the_modules_gauge_on_the_modules_clock(monkeypatch):
    """What the runner uses when nothing is injected: ``committed_gb``, ``_clock``, ``_sleep``."""
    gauge = Gauge(62.3, 38.3, 38.3)
    monkeypatch.setattr(memory, "committed_gb", lambda **_: gauge.read())
    monkeypatch.setattr(memory, "_clock", gauge.clock)
    monkeypatch.setattr(memory, "_sleep", gauge.sleep)

    result = memory.preflight("load transformer", need_gb=25.1, limit_gb=90.0, released=True)

    assert result.committed == 38.3 and gauge.reads == 3
    assert gauge.slept == [memory.SETTLE_STEP_SECONDS] * 2
