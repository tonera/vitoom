from __future__ import annotations

import logging
import os


def setup_logging(level: int | None = None) -> None:
    """
    Lightweight logging init for vendored runtime modules.
    """
    if level is None:
        env = os.environ.get("ANIMA_LOG_LEVEL", "").upper().strip()
        level = getattr(logging, env, logging.INFO) if env else logging.INFO

    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

