"""Создание признаков из файла POS_CASH_balance."""

import logging
from pathlib import Path

import pandas as pd

from app.modelling.features.common import (
    confirm_feature_generation,
    prepare_data_directory,
    prepare_output_directory,
)
from app.utils.logging_config import setup_logging

logger = logging.getLogger(__name__)


def calculate_completed_credit_ratio(
    pos_cash: pd.DataFrame,
) -> pd.Series:
    """Рассчитывает долю завершённых кредитов клиента в Home Credit.

    Статус кредита берётся по последнему (самому свежему по
    MONTHS_BALANCE) ежемесячному снимку каждого SK_ID_PREV.

    Args:
        pos_cash: данные о предыдущих POS/cash кредитах из
            POS_CASH_balance.csv.

    Returns:
        pd.Series: доля кредитов со статусом "Completed" на последнем
            снимке, индекс — SK_ID_CURR.
    """
    latest_status = (
        pos_cash.sort_values("MONTHS_BALANCE")
        .groupby("SK_ID_PREV")
        .tail(1)
        .set_index("SK_ID_PREV")["NAME_CONTRACT_STATUS"]
        .eq("Completed")
    )

    credit_to_client = pos_cash[["SK_ID_PREV", "SK_ID_CURR"]].drop_duplicates()
    completed = (
        latest_status.rename("is_completed")
        .reset_index()
        .merge(credit_to_client, on="SK_ID_PREV")
    )

    return completed.groupby("SK_ID_CURR")["is_completed"].mean()


def run(data_path: Path, output_path: Path) -> None:
    """Рассчитывает признаки для POS_CASH_balance.

    Args:
        data_path: путь до обрабатываемых файлов.
        output_path: путь для сохранения файлов.
    """
    input_file = data_path / "POS_CASH_balance.csv"

    logger.info("Начат расчёт признаков для POS_CASH_balance.csv")

    pos_cash = pd.read_csv(input_file)

    logger.info("Расчёт количества кредитов и глубины истории")
    pos_cash_features = pos_cash.groupby("SK_ID_CURR").agg(
        pos_cash_credit_count=("SK_ID_PREV", "nunique"),
        pos_cash_months_count=("MONTHS_BALANCE", "count"),
        pos_cash_max_dpd=("SK_DPD", "max"),
        pos_cash_max_dpd_def=("SK_DPD_DEF", "max"),
        pos_cash_mean_dpd=("SK_DPD", "mean"),
    )

    logger.info("Расчёт доли месяцев с просрочкой")
    overdue_months_ratio = (
        pos_cash.assign(is_overdue=pos_cash["SK_DPD"].gt(0))
        .groupby("SK_ID_CURR")["is_overdue"]
        .mean()
    )
    pos_cash_features["pos_cash_overdue_months_ratio"] = overdue_months_ratio

    logger.info("Расчёт доли завершённых кредитов")
    pos_cash_features["pos_cash_completed_ratio"] = calculate_completed_credit_ratio(
        pos_cash
    )

    pos_cash_features.index.name = "SK_ID_CURR"

    logger.info("Сохранение признаков POS_CASH_balance")
    output_file = output_path / "pos_cash_balance_features.csv"
    pos_cash_features.to_csv(output_file, index=True)

    logger.info("Расчёт признаков для POS_CASH_balance завершён")


def main() -> None:
    """Запускает генерацию признаков для POS_CASH_balance."""
    setup_logging()

    data_path = prepare_data_directory()

    if not confirm_feature_generation():
        logger.info("Расчёт признаков отменён пользователем.")
        return

    output_path = prepare_output_directory(data_path.parent / "processed")

    run(data_path, output_path)


if __name__ == "__main__":
    main()
