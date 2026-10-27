"""Общие функции для запуска генерации признаков."""

import logging
from pathlib import Path

from app.utils.data_loader import download_home_credit_data_from_kaggle

logger = logging.getLogger(__name__)


def prepare_data_directory(
    *,
    non_interactive: bool = False,
    data_root: Path | None = None,
    force: bool = False,
) -> Path:
    """Проверяет наличие датасетов или загружает их с Kaggle.

    В интерактивном режиме (``non_interactive=False``) параметры
    запрашиваются у пользователя через ``input()``. В неинтерактивном
    режиме (используется DVC-пайплайном) параметры берутся из
    аргументов функции без дополнительных вопросов.

    Args:
        non_interactive: True — не задавать вопросы пользователю и
            использовать значения force/data_root как есть (режим для
            автоматического запуска, например из DVC-стадии).
        data_root: Корневой каталог для данных. Используется только
            при non_interactive=True; в интерактивном режиме
            запрашивается через input().
        force: Принудительно перекачать датасет с Kaggle, даже если
            он уже есть локально. Используется только при
            non_interactive=True.

    Returns:
        Path: путь, где лежат обрабатываемые файлы.
    """
    if non_interactive:
        logger.info(
            "Неинтерактивный режим. Параметры: force=%s, data_root=%s",
            force,
            data_root,
        )
        return download_home_credit_data_from_kaggle(
            force=force, data_root=data_root
        )

    force = input("Загрузить датасет с Kaggle? (y/n): ").strip().lower() == "y"
    data_root_input = input(
        "Введите путь к датасету (Enter — папка data в корне проекта): ",
    ).strip()
    data_root = Path(data_root_input).expanduser() if data_root_input else None

    logger.info(
        "Начата загрузка/проверка датасетов. Параметры: force=%s, data_root=%s",
        force,
        data_root,
    )
    return download_home_credit_data_from_kaggle(
        force=force, data_root=data_root
    )


def confirm_feature_generation(*, non_interactive: bool = False) -> bool:
    """Запрашивает подтверждение расчёта признаков.

    Args:
        non_interactive: True — не спрашивать пользователя и сразу
            подтвердить расчёт (режим для автоматического запуска,
            например из DVC-стадии).

    Returns:
        bool: подтверждение продолжения расчёта признаков. Всегда
            True при non_interactive=True.
    """
    if non_interactive:
        return True
    return input("Начать расчёт признаков? (y/n): ").strip().lower() == "y"


def prepare_output_directory(
    default_path: Path,
    *,
    non_interactive: bool = False,
    output_path: Path | None = None,
) -> Path:
    """Запрашивает каталог для сохранения результатов.

    Args:
        default_path: путь для сохранения обработанных файлов по
            умолчанию.
        non_interactive: True — не спрашивать пользователя и сразу
            использовать output_path (или default_path, если
            output_path не задан).
        output_path: каталог для сохранения результатов. Используется
            только при non_interactive=True; в интерактивном режиме
            запрашивается через input().

    Returns:
        Path: путь до каталога с обработанными файлами (каталог
            гарантированно создан).
    """
    if non_interactive:
        resolved_path = output_path or default_path
        resolved_path.mkdir(parents=True, exist_ok=True)
        return resolved_path

    output_path_input = input(
        f"Введите путь для сохранения датасета (Enter — {default_path}): ",
    ).strip()
    resolved_path = (
        Path(output_path_input).expanduser()
        if output_path_input
        else default_path
    )
    resolved_path.mkdir(parents=True, exist_ok=True)
    return resolved_path
