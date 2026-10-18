"""Создание признаков из файлов application_train и application_test."""

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


def count_docs_number(
    df: pd.DataFrame,
) -> pd.Series:
    """Подсчитывает количество True флагов документов.

    Args:
        df: Датафрейм, по которому ведётся подсчёт

    Returns:
        pd.Series: серия pandas, содержащая количество документов
            для каждой строки исходного Dataframe
    """
    return df.filter(like="FLAG_DOCUMENT").sum(axis=1)


def get_columns_by_description(
    df: pd.DataFrame,
    search_text: str,
) -> list:
    """Получает список колонок из датасета по описанию.

    Args:
        df: Датафрейм с описанием колонок
        search_text: текст из описания к колонкам,
            который необходимо найти

    Returns:
        list: список искомых колонок
    """
    return df.loc[
        df["Description"].str.contains(search_text, na=False),
        "Row",
    ].tolist()


def has_full_information(
    df: pd.DataFrame,
    list_features: list,
    threshold: int = 30,
) -> pd.Series:
    """Проверяет полноту информации по колонкам.

    Args:
        df: Датафрейм, по которому проводится анализ
            полноты информации
        list_features: список названий исследуемых колонок
        threshold: порог отсечения, ниже которого считаем
            информацию полной

    Returns:
            pd.Series: серия pandas, содержащая бинарный
                признак полноты информации
    """
    return (df[list_features].isna().sum(axis=1) < threshold).astype(int)


def days_into_years_converter(days: pd.Series) -> pd.Series:
    """Переводит количество дней в полные года.

    Args:
        days: столбец, содержащий дни для конвертации

    Returns:
        pd.Series: серия pandas, содержащее количество полных лет
    """
    return (days.abs() / 365.25).astype(int)


def income_per_person(
    income: pd.Series,
    people_count: pd.Series,
) -> pd.Series:
    """Рассчитывает средний доход на одного человека.

    Args:
        income: столбец с годовым доходом
        people_count: столбец, содержащий количество человек

    Returns:
        pd.Series: столбец, содержащий средний доход на одного человека
    """
    return income / people_count.where(people_count != 0)


def income_difference_by_group(df: pd.DataFrame) -> pd.Series:
    """Вычисляет разницу между средним доходом в группе по полу и образованию и доходом заявителя.

    Args:
        df: датафрейм, содержащий информацию о поле, образовании и доходе

    Returns:
        pd.Series: признак разницы между средним доходом в группе и доходом
            заявителя
    """
    group_mean_income = df.groupby(["CODE_GENDER", "NAME_EDUCATION_TYPE"])[
        "AMT_INCOME_TOTAL"
    ].transform("mean")

    return group_mean_income - df["AMT_INCOME_TOTAL"]


def get_interest_rate(
    df: pd.DataFrame,
    previous_application: pd.DataFrame,
) -> pd.Series:
    """Добавляет признаки оценочной процентной ставки на основе предыдущих кредитных заявок клиента.

    Args:
        df: текущий датафрейм, для которого необходимо добавить признак
        previous_application: датафрейм с информацией о предыдущих кредитах клиента

    Returns:
        pd.Series: признак, содержащий примерную оценку процентной ставки,
            рассчитанную по предыдущим кредитам клиента
    """
    previous = previous_application[
        ["SK_ID_CURR", "AMT_CREDIT", "AMT_ANNUITY", "CNT_PAYMENT"]
    ].copy()

    valid = (
        previous["AMT_CREDIT"].gt(0)
        & previous["AMT_ANNUITY"].gt(0)
        & previous["CNT_PAYMENT"].gt(0)
    )
    previous = previous.loc[valid].copy()

    previous["interest_rate"] = (
        24
        * (
            previous["CNT_PAYMENT"] * previous["AMT_ANNUITY"]
            - previous["AMT_CREDIT"]
        )
        / (previous["AMT_CREDIT"] * (previous["CNT_PAYMENT"] + 1))
    )

    previous.loc[previous["interest_rate"] < 0, "interest_rate"] = float("nan")

    mean_rate = previous.groupby("SK_ID_CURR")["interest_rate"].mean()

    return df["SK_ID_CURR"].map(mean_rate).rename("interest_rate")


def run(data_path: Path, output_path: Path) -> None:
    r"""Запускает генерацию признаков для application train\test датасетов.

    Args:
        data_path: путь до обрабатываемых файлов
        output_path: путь для сохранения файлов
    """
    description_df = pd.read_csv(
        data_path / "HomeCredit_columns_description.csv",
        encoding="cp1252",
    )

    apprt_features = get_columns_by_description(
        description_df,
        "Normalized information about building where the client lives",
    )

    doc_change_age = [14, 20, 45]

    ext_features = get_columns_by_description(
        description_df, "Normalized score from external data source"
    )

    previous_application = pd.read_csv(data_path / "previous_application.csv")

    for filename in ("application_train.csv", "application_test.csv"):
        logger.info("Начата обработка %s датасета", filename)
        df = pd.read_csv(data_path / filename)

        new_features = {}

        logger.info("Подсчёт количества документов")
        new_features["docs_number"] = count_docs_number(df)

        logger.info("Проверка информации о доме")
        new_features["has_full_apprt_info"] = has_full_information(
            df,
            apprt_features,
        )

        logger.info("Подсчёт количества полных лет")
        new_features["age_in_years"] = days_into_years_converter(
            df["DAYS_BIRTH"]
        )

        logger.info("Подсчёт года смены документа")
        new_features["years_since_doc_change"] = days_into_years_converter(
            df["DAYS_ID_PUBLISH"]
        )

        logger.info("Подсчёт возраста смены документа")
        doc_change_age_in_days = df["DAYS_BIRTH"] - df["DAYS_ID_PUBLISH"]
        new_features["doc_change_age"] = days_into_years_converter(
            doc_change_age_in_days
        )

        logger.info("Подсчёт задержки смены документа")
        new_features["has_doc_change_delay"] = ~new_features[
            "doc_change_age"
        ].isin(doc_change_age)

        logger.info("Подсчёт доли денег на займ за год")
        new_features["loan_income_ratio"] = df["AMT_ANNUITY"] / df[
            "AMT_INCOME_TOTAL"
        ].where(df["AMT_INCOME_TOTAL"] != 0)

        logger.info("Подсчёт количества детей на одного взрослого")
        adults = df["CNT_FAM_MEMBERS"] - df["CNT_CHILDREN"]
        new_features["children_per_adult"] = df["CNT_CHILDREN"] / adults.where(
            adults != 0
        )

        logger.info("Подсчёт среднего дохода на ребёнка")
        new_features["income_per_child"] = income_per_person(
            df["AMT_INCOME_TOTAL"],
            df["CNT_CHILDREN"],
        )

        logger.info("Подсчёт среднего дохода на взрослого")
        new_features["income_per_adult"] = income_per_person(
            df["AMT_INCOME_TOTAL"],
            adults,
        )

        logger.info("Подсчёт взвешенного скора внешних источников")
        """
            т.к. в описании датасета нет оснований полагать,
            что один источник важнее другого, то будет использоваться
            среднее арифметическое доступных внешних оценок, чтобы избежать
            большого количества пропусков
        """
        new_features["ext_source_score"] = df[ext_features].mean(axis=1)

        logger.info(
            "Подсчёт разницы между средним доходом в группе и доходом заявителя"
        )
        new_features["income_group_diff"] = income_difference_by_group(df)

        logger.info("Подсчёт процентной ставки")
        new_features["interest_rate"] = get_interest_rate(
            df, previous_application
        )

        features_df = pd.DataFrame(new_features, index=df.index)
        result_df = pd.concat(
            [
                df[["SK_ID_CURR"]],
                features_df,
            ],
            axis=1,
        )

        output_filename = filename.replace(".csv", "_features.csv")

        logger.info("Подсчёт завершён. Сохранение файла.")
        result_df.to_csv(output_path / output_filename, index=False)
        logger.info("Файл %s сохранён", filename)
        del df, features_df, new_features


def main() -> None:
    """Запускает интерактивную генерацию признаков."""
    setup_logging()

    data_path = prepare_data_directory()

    if not confirm_feature_generation():
        logger.info("Расчёт признаков отменён пользователем.")
        return

    output_path = prepare_output_directory(data_path.parent / "processed")

    run(data_path, output_path)


if __name__ == "__main__":
    main()
