"""Общие функции для запуска генерации признаков."""

import logging
from pathlib import Path

from app.utils.data_loader import download_home_credit_data_from_kaggle

logger = logging.getLogger(__name__)


def prepare_data_directory() -> Path:
    """Проверяет наличие датасетов или загружает их с Kaggle.

    Returns:
        Path: путь, где лежат обрабатываемые файлы
    """
    force = input("Загрузить датасет с Kaggle? (y/n): ").strip().lower() == "y"

    data_root_input = input(
        "Введите путь к датасету (Enter — папка data в корне проекта): ",
    ).strip()

    data_root = Path(data_root_input).expanduser() if data_root_input else None

    logger.info(
        "Начата загрузка/проверка датасетов. Параметры: "
        "force=%s, data_root=%s",
        force,
        data_root,
    )

    return download_home_credit_data_from_kaggle(
        force=force,
        data_root=data_root,
    )


def confirm_feature_generation() -> bool:
    """Запрашивает подтверждение расчёта признаков.

    Returns:
        bool: подтверждение продолжения расчёта признаков от пользователя
    """
    return (
        input("Начать расчёт признаков? (y/n): ").strip().lower() == "y"
    )


def prepare_output_directory(default_path: Path) -> Path:
    """Запрашивает каталог для сохранения результатов.

    Args:
        default_path: путь для сохранения обработанных файцлов по умолчанию

    Returns:
        Path: путь до обработанных файлов
    """
    output_path_input = input(
        f"Введите путь для сохранения датасета (Enter — {default_path}): ",
    ).strip()

    output_path = (
        Path(output_path_input).expanduser()
        if output_path_input
        else default_path
    )

    output_path.mkdir(parents=True, exist_ok=True)

    return output_path