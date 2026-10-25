"""Совместный запуск генерации всех наборов признаков."""

import logging

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


def main() -> None:
    """Запускает генерацию признаков для всех датасетов."""
    setup_logging()

    data_path = prepare_data_directory()

    if not confirm_feature_generation():
        logger.info("Расчёт признаков отменён пользователем.")
        return

    output_path = prepare_output_directory(data_path.parent / "processed")

    application_train_test_features.run(data_path, output_path)
    bureau_features.run(data_path, output_path)
    bureau_balance_features.run(data_path, output_path)

    logger.info("Генерация всех признаков завершена")


if __name__ == "__main__":
    main()
