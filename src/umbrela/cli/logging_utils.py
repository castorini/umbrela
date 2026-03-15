from __future__ import annotations

import logging


def setup_logging(log_level: int = 1, *, quiet: bool = False) -> None:
    if quiet:
        level = logging.CRITICAL
    else:
        level = {
            0: logging.WARNING,
            1: logging.INFO,
            2: logging.DEBUG,
        }.get(log_level, logging.INFO)
    logging.basicConfig(level=level, format="%(levelname)s %(message)s")
