"""Создание признаков из файла bureau."""

import logging
from pathlib import Path

import pandas as pd

from app.modelling.features.common import (
    confirm_feature_generation,
    prepare_data_directory,
    prepare_output_directory,
    setup_logging,
)

logger = logging.getLogger(__name__)


def calculate_overdue_to_open_credit_ratio(
    bureau: pd.DataFrame,
) -> pd.Series:
    """Рассчитывает долю просрочки от суммы открытых кредитов клиента.

    Для каждого клиента суммирует максимальные суммы просрочки по всем
    кредитам и делит полученную сумму на общую сумму активных кредитов.

    Args:
        bureau: Данные о кредитах из файла bureau.csv.

    Returns:
        pd.Series: Доля просрочки от суммы открытых кредитов для каждого
            клиента. Если сумма активных кредитов равна нулю или отсутствует,
            результат равен NaN.
    """
    active_credits = bureau["CREDIT_ACTIVE"].eq("Active")

    open_credit_amount = (
        bureau["AMT_CREDIT_SUM"]
        .where(active_credits)
        .groupby(bureau["SK_ID_CURR"])
        .sum(min_count=1)
    )

    overdue_amount = (
        bureau["AMT_CREDIT_MAX_OVERDUE"]
        .groupby(bureau["SK_ID_CURR"])
        .sum(min_count=1)
    )

    return overdue_amount / open_credit_amount.where(open_credit_amount != 0)


def calculate_credit_counts_by_type(
    bureau: pd.DataFrame,
) -> pd.DataFrame:
    """Рассчитывает количество кредитов каждого типа для клиента.

    Для каждого клиента подсчитывает количество кредитов в разрезе
    категорий CREDIT_TYPE. Названия столбцов формируются на основе типов
    кредитов.

    Args:
        bureau: Данные о кредитах из файла bureau.csv.

    Returns:
        pd.DataFrame: Таблица с количеством кредитов по типам.
            Индекс содержит SK_ID_CURR, а названия столбцов имеют формат
            credit_count_<тип кредита>.
    """
    credit_counts = pd.crosstab(
        bureau["SK_ID_CURR"],
        bureau["CREDIT_TYPE"],
    )

    credit_counts.columns = [
        f"credit_count_{credit_type}" for credit_type in credit_counts.columns
    ]

    return credit_counts


def calculate_overdue_credit_counts_by_type(
    bureau: pd.DataFrame,
) -> pd.DataFrame:
    """Рассчитывает количество просроченных кредитов каждого типа.

    Кредит считается просроченным, если значение AMT_CREDIT_MAX_OVERDUE
    строго больше нуля. Для каждого клиента подсчитывается количество
    таких кредитов в разрезе категорий CREDIT_TYPE.

    Args:
        bureau: Данные о кредитах из файла bureau.csv.

    Returns:
        pd.DataFrame: Таблица с количеством просроченных кредитов
            по типам. Индекс содержит SK_ID_CURR, а названия столбцов
            имеют формат overdue_credit_count_<тип кредита>.
    """
    overdue = bureau["AMT_CREDIT_MAX_OVERDUE"].gt(0)

    overdue_counts = pd.crosstab(
        bureau.loc[overdue, "SK_ID_CURR"],
        bureau.loc[overdue, "CREDIT_TYPE"],
    )

    overdue_counts.columns = [
        f"overdue_credit_count_{credit_type}"
        for credit_type in overdue_counts.columns
    ]

    return overdue_counts


def calculate_closed_credit_counts_by_type(
    bureau: pd.DataFrame,
) -> pd.DataFrame:
    """Рассчитывает количество закрытых кредитов каждого типа.

    Для каждого клиента подсчитывает количество кредитов со статусом
    CREDIT_ACTIVE = Closed в разрезе категорий CREDIT_TYPE.

    Args:
        bureau: Данные о кредитах из файла bureau.csv.

    Returns:
        pd.DataFrame: Таблица с количеством закрытых кредитов по типам.
            Индекс содержит SK_ID_CURR, а названия столбцов имеют формат
            closed_credit_count_<тип кредита>.
    """
    closed = bureau["CREDIT_ACTIVE"].eq("Closed")

    closed_counts = pd.crosstab(
        bureau.loc[closed, "SK_ID_CURR"],
        bureau.loc[closed, "CREDIT_TYPE"],
    )

    closed_counts.columns = [
        f"closed_credit_count_{credit_type}"
        for credit_type in closed_counts.columns
    ]

    return closed_counts


def run(data_path: Path, output_path: Path) -> None:
    """Рассчитывает признаки для bureau.

    Args:
        data_path: путь до обрабатываемых файлов
        output_path: путь для сохранения файлов
    """
    input_file = data_path / "bureau.csv"

    logger.info("Начат расчёт признаков для bureau.csv")

    bureau = pd.read_csv(input_file)

    logger.info("Расчёт минимальной и максимальной суммы просрочки")
    bureau_features = bureau.groupby("SK_ID_CURR").agg(
        max_overdue_amount=("AMT_CREDIT_MAX_OVERDUE", "max"),
        min_overdue_amount=("AMT_CREDIT_MAX_OVERDUE", "min"),
    )

    logger.info("Расчёт доли просрочки от суммы открытых кредитов")
    overdue_ratio = calculate_overdue_to_open_credit_ratio(bureau)
    overdue_ratio.name = "overdue_to_open_credit_ratio"

    logger.info("Расчёт количества кредитов по типам")
    credit_counts = calculate_credit_counts_by_type(bureau)

    logger.info("Расчёт количества просроченных кредитов по типам")
    overdue_counts = calculate_overdue_credit_counts_by_type(bureau)

    logger.info("Расчёт количества закрытых кредитов по типам")
    closed_counts = calculate_closed_credit_counts_by_type(bureau)

    bureau_features = bureau_features.join(
        [
            overdue_ratio,
            credit_counts,
            overdue_counts,
            closed_counts,
        ],
        how="outer",
    ).fillna(0)

    bureau_features.index.name = "SK_ID_CURR"

    logger.info("Сохранение признаков bureau")
    output_file = output_path / "bureau_features.csv"
    bureau_features.to_csv(output_file, index=True)

    logger.info("Расчёт признаков для bureau завершён")


def main() -> None:
    """Запускает генерацию признаков для bureau."""
    setup_logging()

    data_path = prepare_data_directory()

    if not confirm_feature_generation():
        logger.info("Расчёт признаков отменён пользователем.")
        return

    output_path = prepare_output_directory(data_path.parent / "processed")

    run(data_path, output_path)


if __name__ == "__main__":
    main()
