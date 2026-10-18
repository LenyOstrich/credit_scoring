"""Создание признаков из файлов bureau_balance и bureau."""

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

# Статусы просрочки из bureau_balance.STATUS.
OVERDUE_STATUSES = ("1", "2", "3", "4", "5")


def calculate_overdue_status_flags(
    bureau_balance: pd.DataFrame,
) -> pd.DataFrame:
    """Определяет наличие просрочки каждого диапазона по кредиту.

    Args:
        bureau_balance: история статусов кредитов из bureau_balance.csv.

    Returns:
        pd.DataFrame: бинарные признаки наличия статусов просрочки
            по каждому SK_ID_BUREAU.
    """
    status_flags = pd.crosstab(
        bureau_balance["SK_ID_BUREAU"],
        bureau_balance["STATUS"].astype("string"),
    )

    status_flags = status_flags.reindex(
        columns=OVERDUE_STATUSES,
        fill_value=0,
    )

    status_flags = status_flags.gt(0).astype("int8")

    status_flags.columns = [
        f"overdue_status_{status}_flag"
        for status in status_flags.columns
    ]

    return status_flags


def prepare_client_credit_features(
    bureau: pd.DataFrame,
    status_flags: pd.DataFrame,
) -> pd.DataFrame:
    """Добавляет к кредитам признаки истории просрочек и статуса.

    Args:
        bureau: информация о кредитах клиентов из bureau.csv.
        status_flags: признаки наличия статусов просрочки по кредитам.

    Returns:
        pd.DataFrame: таблица кредитов с признаками статуса и просрочек.
    """
    client_credits = bureau[
        ["SK_ID_CURR", "SK_ID_BUREAU", "CREDIT_ACTIVE"]
    ].drop_duplicates(subset=["SK_ID_BUREAU"])

    client_credits = client_credits.merge(
        status_flags,
        left_on="SK_ID_BUREAU",
        right_index=True,
        how="left",
    )

    flag_columns = [
        f"overdue_status_{status}_flag" for status in OVERDUE_STATUSES
    ]

    client_credits[flag_columns] = (
        client_credits[flag_columns].fillna(0).astype("int8")
    )

    client_credits["is_open"] = (
        client_credits["CREDIT_ACTIVE"].eq("Active").astype("int8")
    )
    client_credits["is_closed"] = (
        client_credits["CREDIT_ACTIVE"].eq("Closed").astype("int8")
    )
    client_credits["is_overdue"] = (
        client_credits[flag_columns].max(axis=1).astype("int8")
    )

    return client_credits


def calculate_credit_counts(
    client_credits: pd.DataFrame,
) -> pd.DataFrame:
    """Рассчитывает количество кредитов по клиентам.

    Args:
        client_credits: таблица кредитов с признаками статусов.

    Returns:
        pd.DataFrame: количество кредитов и кредитов с просрочкой
            по каждому SK_ID_CURR.
    """
    aggregation = {
        "credit_count": ("SK_ID_BUREAU", "nunique"),
        "open_credit_count": ("is_open", "sum"),
        "closed_credit_count": ("is_closed", "sum"),
        "overdue_credit_count": ("is_overdue", "sum"),
    }

    for status in OVERDUE_STATUSES:
        aggregation[f"overdue_status_{status}_credit_count"] = (
            f"overdue_status_{status}_flag",
            "sum",
        )

    return client_credits.groupby("SK_ID_CURR").agg(**aggregation)


def calculate_credit_ratios(
    credit_counts: pd.DataFrame,
) -> pd.DataFrame:
    """Рассчитывает доли кредитов по статусам и просрочкам.

    Args:
        credit_counts: количество кредитов по каждому SK_ID_CURR.

    Returns:
        pd.DataFrame: доли открытых, закрытых и просроченных кредитов.
    """
    features = credit_counts.copy()
    denominator = features["credit_count"].replace(0, float("nan"))

    features["closed_credit_ratio"] = (
        features["closed_credit_count"] / denominator
    )
    features["open_credit_ratio"] = features["open_credit_count"] / denominator
    features["overdue_credit_ratio"] = (
        features["overdue_credit_count"] / denominator
    )

    for status in OVERDUE_STATUSES:
        count_column = f"overdue_status_{status}_credit_count"
        ratio_column = f"overdue_status_{status}_credit_ratio"

        features[ratio_column] = features[count_column] / denominator

    return features


def calculate_credit_status_features(
    bureau_balance: pd.DataFrame,
    bureau: pd.DataFrame,
) -> pd.DataFrame:
    """Рассчитывает количество и доли кредитов по статусам.

    Args:
        bureau_balance: история статусов кредитов из bureau_balance.csv.
        bureau: информация о кредитах клиентов из bureau.csv.

    Returns:
        pd.DataFrame: признаки количества и долей кредитов с индексом
            SK_ID_CURR.
    """
    logger.info("Агрегация истории статусов кредитов")
    status_flags = calculate_overdue_status_flags(bureau_balance)

    logger.info("Подготовка признаков по каждому кредиту")
    client_credits = prepare_client_credit_features(bureau, status_flags)

    logger.info("Расчёт количества кредитов")
    credit_counts = calculate_credit_counts(client_credits)

    logger.info("Расчёт долей кредитов")
    return calculate_credit_ratios(credit_counts)


def calculate_credit_time_intervals(
    bureau: pd.DataFrame,
) -> pd.DataFrame:
    """Рассчитывает интервалы между датами кредитов и текущей заявкой.

    Args:
        bureau: информация о кредитах клиентов из bureau.csv.

    Returns:
        pd.DataFrame: интервалы в днях с индексом SK_ID_CURR.
    """
    logger.info("Расчёт интервалов между кредитами и текущей заявкой")

    closed = bureau.loc[
        bureau["CREDIT_ACTIVE"].eq("Closed")
        & bureau["DAYS_ENDDATE_FACT"].notna()
    ]

    last_closed = (
        closed.groupby("SK_ID_CURR")["DAYS_ENDDATE_FACT"]
        .max()
        .mul(-1)
        .rename("days_since_last_closed_credit")
    )

    active = bureau.loc[
        bureau["CREDIT_ACTIVE"].eq("Active") & bureau["DAYS_CREDIT"].notna()
    ]

    last_active = (
        active.groupby("SK_ID_CURR")["DAYS_CREDIT"]
        .max()
        .mul(-1)
        .rename("days_since_last_active_credit_application")
    )

    return pd.concat([last_closed, last_active], axis=1)


def run(data_path: Path, output_path: Path) -> None:
    """Рассчитывает признаки bureau_balance для каждого клиента.

    Args:
        data_path: каталог с bureau.csv и bureau_balance.csv.
        output_path: каталог для сохранения признаков.
    """
    balance_file = data_path / "bureau_balance.csv"
    bureau_file = data_path / "bureau.csv"

    logger.info("Начат расчёт признаков для bureau_balance.csv")

    bureau_balance = pd.read_csv(balance_file)
    bureau = pd.read_csv(bureau_file)

    logger.info("Расчёт количества и долей кредитов")
    status_features = calculate_credit_status_features(
        bureau_balance,
        bureau,
    )

    logger.info("Расчёт временных интервалов")
    time_features = calculate_credit_time_intervals(bureau)

    features = status_features.join(time_features, how="outer")
    features.index.name = "SK_ID_CURR"

    output_path.mkdir(parents=True, exist_ok=True)
    output_file = output_path / "bureau_balance_features.csv"

    logger.info("Сохранение признаков в %s", output_file)
    features.to_csv(output_file, index=True)

    logger.info(
        "Генерация признаков bureau_balance завершена. "
        "Количество клиентов: %s",
        len(features),
    )


def main() -> None:
    """Запускает генерацию признаков bureau_balance отдельно."""
    setup_logging()

    data_path = prepare_data_directory()

    if not confirm_feature_generation():
        logger.info("Расчёт признаков отменён пользователем.")
        return

    output_path = prepare_output_directory(data_path.parent / "processed")

    run(data_path, output_path)


if __name__ == "__main__":
    main()
