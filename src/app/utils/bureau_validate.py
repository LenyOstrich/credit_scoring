"""Модуль для валидации данных в датасете bureau"""

import logging

import pandas as pd

logger = logging.getLogger(__name__)


def validate_bureau_data(bureau: pd.DataFrame) -> pd.DataFrame:
    """Проверяет качество данных о кредитах и удаляет некорректные дубликаты.

    Строки с пропущенными идентификаторами клиента или кредита
    исключаются. Полные дубликаты строк удаляются. Повторяющиеся
    идентификаторы кредитов подсчитываются и записываются в лог,
    но не удаляются автоматически, поскольку записи могут содержать
    разные сведения об одном кредите.

    Args:
        bureau: Исходные данные о кредитах из файла bureau.csv.

    Returns:
        pd.DataFrame: Проверенные данные о кредитах без строк с
            пропущенными идентификаторами и полных дубликатов.
    """
    missing_ids = bureau[["SK_ID_CURR", "SK_ID_BUREAU"]].isna().any(axis=1)

    logger.info(
        "Строк с пропущенными идентификаторами клиента или кредита: %d",
        missing_ids.sum(),
    )

    bureau = bureau.loc[~missing_ids].copy()

    duplicate_rows = bureau.duplicated().sum()

    logger.info("Полных дубликатов в bureau: %d", duplicate_rows)

    bureau = bureau.drop_duplicates()

    duplicate_credit_ids = bureau["SK_ID_BUREAU"].duplicated(keep=False)

    logger.info(
        "Строк с повторяющимся SK_ID_BUREAU: %d",
        duplicate_credit_ids.sum(),
    )

    return bureau
