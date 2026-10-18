"""Проверяет генерацию признаков для application_train и application_test."""

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

FEATURES = [
    "docs_number",
    "has_full_apprt_info",
    "age_in_years",
    "years_since_doc_change",
    "doc_change_age",
    "has_doc_change_delay",
    "loan_income_ratio",
    "children_per_adult",
    "income_per_child",
    "income_per_adult",
    "ext_source_score",
    "income_group_diff",
    "interest_rate",
]

FILENAMES = ("application_train.csv", "application_test.csv")


def main() -> None:
    """Проверяет наличие и заполненность созданных признаков."""

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

    missing_files = [
        filename
        for filename in FILENAMES
        if not (output_path / filename).is_file()
    ]

    if missing_files:
        logger.error("Не найдены необходимые файлы: %s", missing_files)
        return

    for filename in FILENAMES:
        logger.info("Начата проверка файла %s", filename)

        df = pd.read_csv(output_path / filename)

        logger.info(
            "Размер датасета: %s строк × %s столбцов",
            f"{df.shape[0]:,}",
            f"{df.shape[1]:,}",
        )

        for feature in FEATURES:
            if feature not in df.columns:
                logger.error("Признак %s отсутствует", feature)
                continue

            missing = df[feature].isna().sum()
            unique = df[feature].nunique()

            logger.info(
                "%s: пропусков=%s, уникальных значений=%s",
                feature,
                f"{missing:,}",
                f"{unique:,}",
            )

        del df

    logger.info("Проверка завершена")


if __name__ == "__main__":
    main()
