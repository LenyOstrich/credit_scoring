"""Совместный запуск генерации всех наборов признаков."""

import argparse
import logging
from pathlib import Path

from app.modelling.features import (
    application_train_test_features,
    bureau_balance_features,
    bureau_features,
)
from app.modelling.features.common import (
    confirm_feature_generation,
    prepare_data_directory,
    prepare_output_directory,
)
from app.utils.logging_config import setup_logging

logger = logging.getLogger(__name__)


def build_arg_parser() -> argparse.ArgumentParser:
    """Создаёт парсер аргументов командной строки для генерации признаков.

    Returns:
        argparse.ArgumentParser: настроенный парсер с флагами
            --non-interactive, --data-root, --force-download и
            --output-dir.
    """
    parser = argparse.ArgumentParser(
        description="Генерация признаков Home Credit."
    )
    parser.add_argument("--non-interactive", action="store_true")
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--force-download", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser


def main() -> None:
    """Запускает генерацию признаков для всех датасетов.

    В интерактивном режиме (по умолчанию) запрашивает у пользователя
    подтверждение и пути через input(). С флагом --non-interactive
    (используется DVC-стадией) все вопросы пропускаются, а пути
    берутся из аргументов командной строки.
    """
    setup_logging()
    args = build_arg_parser().parse_args()

    data_path = prepare_data_directory(
        non_interactive=args.non_interactive,
        data_root=args.data_root,
        force=args.force_download,
    )

    if not args.non_interactive and not confirm_feature_generation():
        logger.info("Расчёт признаков отменён пользователем.")
        return

    output_path = prepare_output_directory(
        data_path.parent / "processed",
        non_interactive=args.non_interactive,
        output_path=args.output_dir,
    )

    application_train_test_features.run(data_path, output_path)
    bureau_features.run(data_path, output_path)
    bureau_balance_features.run(data_path, output_path)

    logger.info("Генерация всех признаков завершена")


if __name__ == "__main__":
    main()
