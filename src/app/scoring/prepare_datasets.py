"""Подготовка и сохранение train/test датасетов для обучения моделей."""

import argparse
import logging
from pathlib import Path

from app.scoring.common import ensure_directory
from app.scoring.data_preparation import (
    DATASET_KINDS,
    prepare_datasets_for_models,
)
from app.utils.logging_config import setup_logging

logger = logging.getLogger(__name__)


def build_arg_parser() -> argparse.ArgumentParser:
    """Создаёт парсер аргументов командной строки для подготовки датасетов.

    Returns:
        argparse.ArgumentParser: настроенный парсер с флагами
            --raw-data-dir, --features-dir, --output-dir и
            --random-state.
    """
    parser = argparse.ArgumentParser(
        description="Подготовка train/test датасетов из сгенерированных признаков."
    )
    parser.add_argument(
        "--raw-data-dir",
        type=Path,
        default=Path("data/home-credit-default-risk"),
    )
    parser.add_argument(
        "--features-dir", type=Path, default=Path("data/processed")
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("data/model_ready")
    )
    parser.add_argument("--random-state", type=int, default=42)
    return parser


def main() -> None:
    """Готовит train/test датасеты для всех семейств моделей и сохраняет их на диск.

    Для каждого набора из data_preparation.DATASET_KINDS
    (linear, tree_based, catboost) вызывает
    prepare_datasets_for_models и сохраняет полученные train/test
    выборки (с колонкой TARGET) в CSV-файлы в output-dir. Используется
    как отдельная DVC-стадия, выходом которой являются подготовленные
    данные, а не обученные модели.
    """
    setup_logging()
    args = build_arg_parser().parse_args()
    output_dir = ensure_directory(args.output_dir)

    datasets = prepare_datasets_for_models(
        raw_data_dir=args.raw_data_dir,
        features_dir=args.features_dir,
        random_state=args.random_state,
    )

    for kind in DATASET_KINDS:
        x_train, x_test, y_train, y_test = datasets[kind]
        x_train.assign(TARGET=y_train).to_csv(
            output_dir / f"{kind}_train.csv", index=False
        )
        x_test.assign(TARGET=y_test).to_csv(
            output_dir / f"{kind}_test.csv", index=False
        )
        logger.info(
            "Сохранён датасет %s: %d train / %d test строк",
            kind,
            len(x_train),
            len(x_test),
        )

    logger.info("Подготовка датасетов завершена. Результаты в %s", output_dir)


if __name__ == "__main__":
    main()
