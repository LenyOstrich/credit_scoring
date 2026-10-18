"""Проверяет генерацию признаков из файлов bureau_balance и bureau."""

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

REQUIRED_FEATURES = [
    "credit_count",
    "open_credit_count",
    "closed_credit_count",
    "overdue_credit_count",
    "closed_credit_ratio",
    "open_credit_ratio",
    "overdue_credit_ratio",
    "days_since_last_closed_credit",
    "days_since_last_active_credit_application",
]

OVERDUE_STATUSES = ("1", "2", "3", "4", "5")

FILENAME = "bureau_balance_features.csv"


def main() -> None:
    """Проверяет наличие и заполненность признаков bureau_balance."""
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
        return

    logger.info(
        "Уникальных клиентов: %s",
        f"{df['SK_ID_CURR'].nunique():,}",
    )

    if df["SK_ID_CURR"].duplicated().any():
        logger.error("Обнаружены дубликаты SK_ID_CURR")
    else:
        logger.info("Дубликаты SK_ID_CURR отсутствуют")

    overdue_count_features = [
        f"overdue_status_{status}_credit_count" for status in OVERDUE_STATUSES
    ]
    overdue_ratio_features = [
        f"overdue_status_{status}_credit_ratio" for status in OVERDUE_STATUSES
    ]

    expected_features = (
        REQUIRED_FEATURES + overdue_count_features + overdue_ratio_features
    )

    missing_features = [
        feature for feature in expected_features if feature not in df.columns
    ]

    for feature in missing_features:
        logger.error("Обязательный признак %s отсутствует", feature)

    if not missing_features:
        logger.info("Все ожидаемые признаки присутствуют")

    features = [
        feature for feature in expected_features if feature in df.columns
    ]

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

    count_features = [
        "credit_count",
        "open_credit_count",
        "closed_credit_count",
        "overdue_credit_count",
        *overdue_count_features,
    ]

    for feature in count_features:
        if feature not in df.columns:
            continue

        if df[feature].lt(0).any():
            logger.error(
                "Признак %s содержит отрицательные значения",
                feature,
            )

    ratio_features = [
        "closed_credit_ratio",
        "open_credit_ratio",
        "overdue_credit_ratio",
        *overdue_ratio_features,
    ]

    for feature in ratio_features:
        if feature not in df.columns:
            continue

        invalid = df[feature].notna() & (df[feature].lt(0) | df[feature].gt(1))

        if invalid.any():
            logger.error(
                "Признак %s содержит значения вне диапазона [0, 1]: %s",
                feature,
                int(invalid.sum()),
            )

    logger.info("Проверка завершена")


if __name__ == "__main__":
    main()
