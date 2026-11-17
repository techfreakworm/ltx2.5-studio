"""Keeps one bitsandbytes warning from flooding the log.

bitsandbytes logs ``MatMul8bitLt: inputs will be cast from torch.bfloat16 to
float16 during quantization`` on every int8 matrix product: tens of thousands of
lines a run on the Space, whose log keeps only its tail. The filter lets that
message through once per process, so the fact is on record, and drops its
repeats. Every other record on that logger, and every other logger, passes
untouched; the logger's level is left alone so bitsandbytes' other warnings
still reach the log.
"""

from __future__ import annotations

import logging
import os

#: The logger bitsandbytes' int8 matmul warns on: ``logging.getLogger(__name__)``
#: in ``bitsandbytes/autograd/_functions.py``.
BNB_MATMUL_LOGGER = "bitsandbytes.autograd._functions"

#: The start of that warning's format string; the dtype follows it.
BNB_CAST_PREFIX = "MatMul8bitLt: inputs will be cast from"


class OncePerProcess(logging.Filter):
    """Passes the first record of each message starting with ``prefix`` in each process."""

    def __init__(self, prefix: str) -> None:
        super().__init__()
        self.prefix = prefix
        self._seen: dict[tuple[int, str], object] = {}

    def filter(self, record: logging.LogRecord) -> bool:
        if not (isinstance(record.msg, str) and record.msg.startswith(self.prefix)):
            return True
        # Keyed by pid: a forked worker records the fact once for itself.
        # setdefault is one atomic step, so two threads cannot both pass.
        first = object()
        return self._seen.setdefault((os.getpid(), record.getMessage()), first) is first


def install() -> OncePerProcess:
    """Attach the filter to the bitsandbytes matmul logger; a second call adds nothing.

    bitsandbytes is not imported: the logger is created by name, and the
    library's own ``getLogger(__name__)`` returns this same object, filter
    included, whenever it loads. ZeroGPU forks its GPU worker from the process
    that called this, so the worker inherits the filter.
    """
    logger = logging.getLogger(BNB_MATMUL_LOGGER)
    for existing in logger.filters:
        if isinstance(existing, OncePerProcess):
            return existing
    added = OncePerProcess(BNB_CAST_PREFIX)
    logger.addFilter(added)
    return added
