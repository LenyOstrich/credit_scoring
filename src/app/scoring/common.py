"""Общие утилиты для пайплайна обучения моделей кредитного скоринга."""

import argparse
import logging
import pickle
from pathlib import Path
from typing import Any, cast

import pandas as pd

logger = logging.getLogger(__name__)


def drop_correlated_columns(
    correlation_matrix: pd.DataFrame,
    columns: list[str],
    correlation_threshold: float = 0.9,
) -> list[str]:
    """Убирает колонки, сильно коррелирующие с уже оставленными (greedy).

    Args:
        correlation_matrix: квадратная матрица |корреляций| между
            columns (например, `df[columns].corr().abs()` или
            `df[columns].isna().astype(int).corr().abs()`).
        columns: колонки-кандидаты в исходном порядке, для каждой
            группы дублей остаётся первая встреченная.
        correlation_threshold: порог, выше которого колонка считается
            дублирующей уже оставленную.

    Returns:
        list[str]: подмножество columns без избыточных колонок.
    """
    if len(columns) < 2:
        return list(columns)

    dropped_columns = set()
    for i, column in enumerate(columns):
        if column in dropped_columns:
            continue
        for other_column in columns[i + 1 :]:
            if other_column in dropped_columns:
                continue
            correlation = cast(
                "float", correlation_matrix.loc[column, other_column]
            )
            if correlation >= correlation_threshold:
                dropped_columns.add(other_column)
                logger.info(
                    "%s: сильно коррелирует с %s (corr=%.3f) — исключена как избыточная",
                    other_column,
                    column,
                    correlation,
                )

    return [column for column in columns if column not in dropped_columns]


def build_arg_parser() -> argparse.ArgumentParser:
    """Создаёт парсер аргументов командной строки для скрипта обучения.

    Returns:
        argparse.ArgumentParser: настроенный парсер.
    """
    parser = argparse.ArgumentParser(
        description="Обучение моделей кредитного скоринга."
    )
    parser.add_argument(
        "--raw-data-dir",
        type=Path,
        default=Path("data"),
        help="Директория с исходными CSV Home Credit.",
    )
    parser.add_argument(
        "--features-dir",
        type=Path,
        default=Path("data/processed"),
        help="Директория со сгенерированными в ДЗ-3 признаками.",
    )
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=Path("data/models"),
        help="Куда сохранять pickle-файлы обученных моделей.",
    )
    parser.add_argument(
        "--reports-dir",
        type=Path,
        default=Path("docs/figures"),
        help="Куда сохранять таблицы метрик и графики.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="random_state для воспроизводимости.",
    )
    parser.add_argument(
        "--cv-folds",
        type=int,
        default=3,
        help="Количество фолдов кросс-валидации.",
    )
    return parser


def ensure_directory(path: Path) -> Path:
    """Создаёт директорию (если её ещё нет) и возвращает путь к ней.

    Args:
        path: путь к директории, которая должна существовать.

    Returns:
        Path: тот же путь, после того как директория гарантированно создана.
    """
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_model(model: Any, path: Path) -> None:
    """Сохраняет обученную модель в pickle-файл.

    Args:
        model: обученный sklearn-совместимый estimator (или Pipeline).
        path: путь к pickle-файлу, в который нужно сохранить модель.
    """
    ensure_directory(path.parent)
    with path.open("wb") as file:
        pickle.dump(model, file)
    logger.info("Модель %s сохранена в %s", type(model).__name__, path)
