"""Spec: the memory preflight for local runs (issue F14).

The budget rule on the operator's Mac is "committed memory plus what the next
step adds stays under 90 GB", where committed is ``vm_stat``'s
active + wired + occupied-by-compressor pages. These tests pin the parsing
against recorded ``vm_stat`` output, the activation arithmetic against the
token formulas it is built from, and the refusal message.

Runs in both environments: the module must import without torch.
"""

from __future__ import annotations

import os
import pickle
import subprocess
import sys

import pytest
from conftest import REPO_ROOT

from ltx25_studio import memory
from ltx25_studio.errors import ConfigError, LTXStudioError

#: Recorded on the operator's M5 Max (recon K §8.1): 16 KiB pages,
#: active 940,856 + wired 436,565 + compressor 49,637 = 23.4 GB committed.
VM_STAT_M5 = """\
Mach Virtual Memory Statistics: (page size of 16384 bytes)
Pages free:                                  2755162.
Pages active:                                 940856.
Pages inactive:                              3216783.
Pages speculative:                             78098.
Pages throttled:                                   0.
Pages wired down:                             436565.
Pages purgeable:                               43670.
"Translation faults":                    10336319622.
Pages copy-on-write:                      1087033606.
Pages zero filled:                        8020902729.
Pages reactivated:                          23836198.
Pages purged:                              137371107.
File-backed pages:                           3049321.
Anonymous pages:                             2009206.
Pages stored in compressor:                   125593.
Pages occupied by compressor:                  49637.
Decompressions:                             17123731.
Compressions:                               58590852.
Pageins:                                   393781729.
Pageouts:                                     217520.
Swapins:                                      120395.
Swapouts:                                     460311.
"""

#: Same counters, 4 KiB pages (an Intel Mac): the page size must be read, not assumed.
VM_STAT_4K = VM_STAT_M5.replace("page size of 16384 bytes", "page size of 4096 bytes")

#: Free, inactive, speculative, file-backed and "stored in compressor" all moved;
#: active, wired and occupied-by-compressor did not. Committed must not move.
VM_STAT_NOISE = (
    VM_STAT_M5.replace("2755162.", "10.")
    .replace("3216783.", "99999999.")
    .replace("78098.", "1.")
    .replace("3049321.", "7.")
    .replace("125593.", "88888888.")
)

M5_COMMITTED_BYTES = (940_856 + 436_565 + 49_637) * 16_384


# --------------------------------------------------------------------------
# committed memory, from vm_stat
# --------------------------------------------------------------------------


def test_committed_is_active_plus_wired_plus_compressor_pages_times_page_size():
    assert memory.committed_gb(vm_stat_text=VM_STAT_M5, platform="darwin") == pytest.approx(
        M5_COMMITTED_BYTES / 1e9
    )
    assert round(memory.committed_gb(vm_stat_text=VM_STAT_M5, platform="darwin"), 1) == 23.4


def test_the_page_size_is_read_from_the_header_not_assumed():
    assert memory.committed_gb(vm_stat_text=VM_STAT_4K, platform="darwin") == pytest.approx(
        M5_COMMITTED_BYTES / 4 / 1e9
    )


def test_free_inactive_file_backed_and_stored_pages_do_not_count():
    """Never "available", never RSS: only the three committed counters."""
    assert memory.committed_gb(vm_stat_text=VM_STAT_NOISE, platform="darwin") == pytest.approx(
        M5_COMMITTED_BYTES / 1e9
    )


def test_linux_reports_no_committed_figure():
    assert memory.committed_gb(vm_stat_text=VM_STAT_M5, platform="linux") is None


def test_unreadable_vm_stat_on_darwin_is_an_error_not_a_zero():
    """A gauge that silently reads 0 would pass every preflight."""
    with pytest.raises(memory.MemoryPreflightError, match="vm_stat"):
        memory.committed_gb(vm_stat_text="Pages active: 12.\n", platform="darwin")


@pytest.mark.skipif(sys.platform != "darwin", reason="vm_stat exists only on macOS")
def test_the_live_gauge_reads_this_machine():
    value = memory.committed_gb()
    assert value is not None and 0.5 < value < 1024


# --------------------------------------------------------------------------
# the activation estimate (recon K §8.2-8.3)
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("width", "height", "frames", "tokens"),
    [
        (256, 256, 9, 128),
        (512, 512, 49, 1_792),
        (704, 480, 121, 5_280),
        (960, 512, 121, 7_680),
        (960, 512, 241, 14_880),
    ],
)
def test_video_token_count_matches_the_recon_table(width, height, frames, tokens):
    assert memory.video_tokens(width, height, frames) == tokens


def test_activation_estimate_at_704x480x121_is_the_sum_of_the_named_terms():
    """Every term recomputed here from the formula, independently of the module."""
    tokens = ((121 - 1) // 8 + 1) * (480 // 32) * (704 // 32)
    attention = 3 * 32 * tokens**2 * 2
    hidden = 40 * tokens * 4096 * 2
    vae_tiles = 6 * 256 * 56 * (480 // 4) * (704 // 4) * 2
    vae_output = 121 * 480 * 704 * 3 * 4
    expected = (attention + hidden + vae_tiles + vae_output) / 1e9 + 1.0
    got = memory.activation_estimate_gb(704, 480, 121, calibration=1.0)
    assert got == pytest.approx(expected)
    # recon K §8.3 transformer phase 7.1 GB, VAE 3.6 + 0.5 GB, plus the fixed 1 GB.
    assert 12.0 < got < 12.5


def test_the_vae_term_is_tiled_by_the_app_envelope():
    """Above 56 frames / 480 rows / 704 columns the decode works tile by tile."""
    small = memory.activation_terms(512, 512, 49)
    assert small["vae_decode_tiles"] == pytest.approx(
        6 * 256 * 49 * (480 // 4) * (512 // 4) * 2 / 1e9
    )


def test_calibration_multiplies_only_the_size_dependent_part():
    base = memory.activation_estimate_gb(704, 480, 121, calibration=1.0)
    doubled = memory.activation_estimate_gb(704, 480, 121, calibration=2.0)
    assert doubled - memory.FIXED_ACTIVATION_GB == pytest.approx(
        2 * (base - memory.FIXED_ACTIVATION_GB)
    )


def test_calibration_comes_from_the_environment_when_not_passed(monkeypatch):
    monkeypatch.setenv("LTX25_ACTIVATION_CALIBRATION", "1.5")
    assert memory.activation_estimate_gb(704, 480, 121) == pytest.approx(
        memory.activation_estimate_gb(704, 480, 121, calibration=1.5)
    )


def test_a_garbled_calibration_is_refused(monkeypatch):
    monkeypatch.setenv("LTX25_ACTIVATION_CALIBRATION", "lots")
    with pytest.raises(ConfigError, match="LTX25_ACTIVATION_CALIBRATION"):
        memory.activation_estimate_gb(704, 480, 121)


def test_the_load_spike_factor_defaults_to_two_and_reads_the_environment(monkeypatch):
    monkeypatch.delenv("LTX25_LOAD_SPIKE_FACTOR", raising=False)
    assert memory.load_spike_factor() == 2.0
    monkeypatch.setenv("LTX25_LOAD_SPIKE_FACTOR", "1.25")
    assert memory.load_spike_factor() == 1.25
    monkeypatch.setenv("LTX25_LOAD_SPIKE_FACTOR", "0.5")
    with pytest.raises(ConfigError, match="LTX25_LOAD_SPIKE_FACTOR"):
        memory.load_spike_factor()


# --------------------------------------------------------------------------
# the verdict and the line it prints
# --------------------------------------------------------------------------


def test_the_ok_line_names_every_number():
    result = memory.preflight("denoise 704x480x121", need_gb=41.9, limit_gb=90.0, committed_gb=26.4)
    assert result.verdict == "OK"
    assert result.headroom == pytest.approx(21.7)
    assert result.describe() == (
        "preflight[denoise 704x480x121]: committed 26.4 + need 41.9 = 68.3 GB "
        "<= 90.0 GB OK (headroom 21.7)"
    )


def test_the_fail_line_and_the_refusal_name_every_number():
    result = memory.preflight("load transformer", need_gb=45.6, limit_gb=90.0, committed_gb=52.0)
    assert result.verdict == "FAIL"
    assert result.describe() == (
        "preflight[load transformer]: committed 52.0 + need 45.6 = 97.6 GB "
        "> 90.0 GB FAIL (short by 7.6)"
    )
    with pytest.raises(memory.MemoryPreflightError) as caught:
        result.require()
    message = str(caught.value)
    for number in ("52.0", "45.6", "97.6", "90.0", "7.6"):
        assert number in message
    assert "load transformer" in message
    assert "LTX25_MEMORY_LIMIT_GB" in message


def test_require_returns_the_result_when_it_fits():
    result = memory.preflight("load vae", need_gb=1.0, limit_gb=90.0, committed_gb=20.0)
    assert result.require() is result


def test_the_limit_comes_from_the_environment(monkeypatch):
    monkeypatch.setenv("LTX25_MEMORY_LIMIT_GB", "40")
    result = memory.preflight("x", need_gb=1.0, committed_gb=10.0)
    assert result.limit == 40.0


def test_the_darwin_default_limit_is_90(monkeypatch):
    monkeypatch.delenv("LTX25_MEMORY_LIMIT_GB", raising=False)
    assert memory.default_limit_gb(platform="darwin") == 90.0
    assert memory.default_limit_gb(platform="linux") is None


def test_a_garbled_limit_is_refused(monkeypatch):
    monkeypatch.setenv("LTX25_MEMORY_LIMIT_GB", "ninety")
    with pytest.raises(ConfigError, match="LTX25_MEMORY_LIMIT_GB"):
        memory.preflight("x", need_gb=1.0, committed_gb=10.0)


def test_no_limit_is_report_only():
    result = memory.preflight("x", need_gb=5.0, limit_gb=None, committed_gb=10.0)
    assert result.verdict == "REPORT"
    assert result.require() is result
    assert (
        result.describe()
        == "preflight[x]: committed 10.0 + need 5.0 = 15.0 GB (no limit; report only)"
    )


def test_no_committed_figure_is_report_only():
    result = memory.preflight("x", need_gb=5.0, limit_gb=90.0, committed_gb=None, platform="linux")
    assert result.verdict == "REPORT"
    assert "committed n/a" in result.describe()


def test_the_refusal_is_typed_plain_and_survives_the_fork_boundary():
    error = memory.MemoryPreflightError(
        memory.preflight("load text encoder", need_gb=47.8, limit_gb=90.0, committed_gb=60.0)
    )
    assert isinstance(error, LTXStudioError)
    clone = pickle.loads(pickle.dumps(error))
    assert type(clone) is memory.MemoryPreflightError
    assert str(clone) == str(error)
    assert "PLAN" not in str(error) and ".py" not in str(error)


# --------------------------------------------------------------------------
# MPS counters and the torch-less import
# --------------------------------------------------------------------------


def test_mps_stats_is_none_or_three_numbers():
    stats = memory.mps_stats()
    if stats is None:
        return
    assert set(stats) == {"allocated_gb", "driver_gb", "recommended_max_gb"}
    assert all(isinstance(v, float) and v >= 0 for v in stats.values())


def test_the_module_imports_without_torch():
    """Run in a fresh interpreter with torch made unimportable."""
    code = (
        "import sys; sys.modules['torch'] = None\n"
        "import ltx25_studio.memory as m\n"
        "assert m.mps_stats() is None\n"
        "print('ok')\n"
    )
    done = subprocess.run(
        [sys.executable, "-c", code],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=60,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
    )
    assert done.returncode == 0, done.stderr[-2000:]
    assert done.stdout.strip() == "ok"
