"""Проверяет генерацию признаков из файла bureau."""

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

REQUIRED_FEATURES = [
    "max_overdue_amount",
    "min_overdue_amount",
    "overdue_to_open_credit_ratio",
]

FEATURE_PREFIXES = (
    "credit_count_",
    "overdue_credit_count_",
    "closed_credit_count_",
)

FILENAME = "bureau_features.csv"


def main() -> None:
    """Проверяет наличие и заполненность созданных признаков bureau."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    project_root = Path(__file__).resolve().parents[1]
    default_path = project_root / "data" / "processed"

    output_path_input = input(
        f"Путь к каталогу с обработанными датасетами "
        f"(Enter — {default_path}): "
    ).strip()

    output_path = (
        Path(output_path_input).expanduser()
        if output_path_input
        else default_path
    )

    if not output_path.is_dir():
        logger.error("Каталог не найден: %s", output_path)
        return

    input_file = output_path / FILENAME

    if not input_file.is_file():
        logger.error("Файл не найден: %s", input_file)
        return

    logger.info("Начата проверка файла %s", input_file)

    df = pd.read_csv(input_file)

    logger.info(
        "Размер датасета: %s строк × %s столбцов",
        f"{df.shape[0]:,}",
        f"{df.shape[1]:,}",
    )

    if "SK_ID_CURR" not in df.columns:
        logger.error("Отсутствует обязательный столбец SK_ID_CURR")
    else:
        logger.info(
            "Уникальных клиентов: %s",
            f"{df['SK_ID_CURR'].nunique():,}",
        )

        if df["SK_ID_CURR"].duplicated().any():
            logger.error("Обнаружены дубликаты SK_ID_CURR")
        else:
            logger.info("Дубликаты SK_ID_CURR отсутствуют")

    missing_features = [
        feature
        for feature in REQUIRED_FEATURES
        if feature not in df.columns
    ]

    for feature in missing_features:
        logger.error("Обязательный признак %s отсутствует", feature)

    dynamic_features = [
        column
        for column in df.columns
        if column.startswith(FEATURE_PREFIXES)
    ]

    if not dynamic_features:
        logger.error("Не найдены признаки с ожидаемыми префиксами")
    else:
        logger.info(
            "Найдено признаков по типам кредитов: %s",
            len(dynamic_features),
        )

    features = [
        feature
        for feature in REQUIRED_FEATURES
        if feature in df.columns
    ] + dynamic_features

    for feature in features:
        missing = df[feature].isna().sum()
        unique = df[feature].nunique()

        logger.info(
            "%s: пропусков=%s, уникальных значений=%s",
            feature,
            f"{missing:,}",
            f"{unique:,}",
        )

        if missing:
            logger.warning(
                "Признак %s содержит пропуски",
                feature,
            )

    logger.info("Проверка завершена")


if __name__ == "__main__":
    main()
