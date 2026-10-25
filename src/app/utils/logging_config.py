"""Общая настройка логирования для скриптов проекта."""

import logging


def setup_logging() -> None:
    """Настраивает логирование."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
