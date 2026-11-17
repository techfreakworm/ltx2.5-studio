"""Spec: one bitsandbytes warning does not flood the Space's log (issue #15, S3).

bitsandbytes warns ``MatMul8bitLt: inputs will be cast from torch.bfloat16 to
float16 during quantization`` on every int8 matrix product. After eleven runs it
was 55,400 of the last 55,534 lines of the Space's log, and the platform keeps
only the tail, so the start-up report and every real warning were pushed out.

The message is let through once per process, so the fact stays on record, and
its repeats are dropped. Nothing else changes: another warning from the same
logger, the logger's level and every other logger are left as they were.

Asserted on the import path the Space runs (``import app``) and without
importing bitsandbytes: the filter sits on the logger by its name, which is the
object bitsandbytes' own ``logging.getLogger(__name__)`` returns. ZeroGPU forks
its GPU worker from that process, so the worker inherits the filter.
"""

from __future__ import annotations

import importlib
import logging
import os
import sys

import pytest

from ltx25_studio import generate as generate_module

BNB_LOGGER = "bitsandbytes.autograd._functions"
#: The format string and argument exactly as bitsandbytes 0.50.2 logs them.
CAST = "MatMul8bitLt: inputs will be cast from %s to float16 during quantization"
CAST_LINE = "MatMul8bitLt: inputs will be cast from torch.bfloat16 to float16 during quantization"


@pytest.fixture
def clean_bnb_logger(monkeypatch):
    """The bitsandbytes logger with no filter left by an earlier import of the app."""
    logger = logging.getLogger(BNB_LOGGER)
    monkeypatch.setattr(logger, "filters", [])
    return logger


@pytest.fixture
def imported_app(clean_bnb_logger):
    """``import app`` afresh, as the Space starts it; the GPU dispatch is put back after."""
    pytest.importorskip("gradio")
    previous = generate_module.gpu_dispatch()
    sys.modules.pop("app", None)
    try:
        yield importlib.import_module("app")
    finally:
        generate_module.set_gpu_dispatch(previous)
        sys.modules.pop("app", None)


@pytest.fixture
def log_filter():
    return importlib.import_module("ltx25_studio.log_filter")


def _lines(caplog, name):
    return [record.getMessage() for record in caplog.records if record.name == name]


def test_importing_the_app_lets_the_cast_warning_through_once(imported_app, caplog):
    logger = logging.getLogger(BNB_LOGGER)
    with caplog.at_level(logging.INFO):
        for _ in range(1000):
            logger.warning(CAST, "torch.bfloat16")

    lines = _lines(caplog, BNB_LOGGER)
    assert lines == [CAST_LINE], f"{len(lines)} lines reached the log for 1,000 int8 products"


def test_importing_the_app_twice_installs_one_filter(imported_app, clean_bnb_logger):
    sys.modules.pop("app", None)
    importlib.import_module("app")

    assert len(clean_bnb_logger.filters) == 1, clean_bnb_logger.filters


def test_the_logger_level_is_not_raised(imported_app, clean_bnb_logger):
    assert clean_bnb_logger.level == logging.NOTSET
    assert clean_bnb_logger.getEffectiveLevel() == logging.getLogger().getEffectiveLevel()


def test_a_different_warning_from_the_same_logger_still_passes(
    log_filter, clean_bnb_logger, caplog
):
    log_filter.install()
    other = "MatMul8bitLt: the input has a different shape"
    with caplog.at_level(logging.INFO):
        clean_bnb_logger.warning(CAST, "torch.bfloat16")
        for _ in range(3):
            clean_bnb_logger.warning(other)
            clean_bnb_logger.warning(CAST, "torch.bfloat16")
            clean_bnb_logger.warning("int8 weights were offloaded to the CPU")

    assert _lines(caplog, BNB_LOGGER) == [
        CAST_LINE,
        *[other, "int8 weights were offloaded to the CPU"] * 3,
    ]


def test_each_distinct_cast_is_recorded_once(log_filter, clean_bnb_logger, caplog):
    log_filter.install()
    with caplog.at_level(logging.INFO):
        for _ in range(5):
            clean_bnb_logger.warning(CAST, "torch.bfloat16")
            clean_bnb_logger.warning(CAST, "torch.float32")

    assert _lines(caplog, BNB_LOGGER) == [
        CAST_LINE,
        "MatMul8bitLt: inputs will be cast from torch.float32 to float16 during quantization",
    ]


def test_other_loggers_are_untouched(log_filter, clean_bnb_logger, caplog):
    before = {
        name: list(logging.getLogger(name).filters)
        for name in ("", "ltx25_studio", "bitsandbytes", "bitsandbytes.autograd")
    }
    log_filter.install()
    after = {name: list(logging.getLogger(name).filters) for name in before}
    assert after == before

    neighbour = logging.getLogger("ltx25_studio.generate")
    with caplog.at_level(logging.INFO):
        for _ in range(4):
            neighbour.warning(CAST, "torch.bfloat16")
    assert len(_lines(caplog, "ltx25_studio.generate")) == 4


def test_a_forked_worker_records_the_fact_once_more(
    log_filter, clean_bnb_logger, caplog, monkeypatch
):
    """Once per PROCESS: the ZeroGPU worker is a fork, so it keeps its own count."""
    log_filter.install()
    with caplog.at_level(logging.INFO):
        clean_bnb_logger.warning(CAST, "torch.bfloat16")
        parent = os.getpid()
        monkeypatch.setattr(os, "getpid", lambda: parent + 1)
        for _ in range(3):
            clean_bnb_logger.warning(CAST, "torch.bfloat16")

    assert _lines(caplog, BNB_LOGGER) == [CAST_LINE, CAST_LINE]
