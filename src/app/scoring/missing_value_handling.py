"""Обработка пропусков в датасете кредитного скоринга"""

import logging
from dataclasses import dataclass
from typing import cast

import pandas as pd
from scipy.stats import chi2_contingency
from statsmodels.stats.multitest import multipletests

from app.scoring.common import drop_correlated_columns

logger = logging.getLogger(__name__)

MISSING_CATEGORY_LABEL = "Missing"


@dataclass
class ColumnGroups:
    """Колонки, сгруппированные по типу для обработки пропусков.

    Attributes:
        numeric: числовые колонки (заполняются медианой).
        categorical: категориальные колонки (заполняются
            MISSING_CATEGORY_LABEL).
    """

    numeric: list[str]
    categorical: list[str]


def add_missing_value_flags(
    df: pd.DataFrame,
    columns: list[str],
) -> pd.DataFrame:
    """Добавляет бинарные индикаторы пропуска для перечисленных колонок.

    Args:
        df: датафрейм, в котором проверяется наличие пропусков.
        columns: список колонок, для которых нужны индикаторы пропуска.

    Returns:
        pd.DataFrame: новый датафрейм с добавленными колонками
            `{column}_was_missing`.
    """
    df = df.copy()
    for col in columns:
        df[f"{col}_was_missing"] = df[col].isna().astype(int)
    return df


def compute_numeric_fill_values(
    train_df: pd.DataFrame,
    numeric_col: list[str],
) -> pd.Series:
    """Вычисляет медианы числовых колонок только на обучающей выборке.

    Args:
        train_df: обучающая выборка.
        numeric_col: список числовых колонок, для которых нужно
            вычислить медиану.

    Returns:
        pd.Series: индекс — название колонки, значение — медиана этой
            колонки на train_df.
    """
    return train_df[numeric_col].median()


def fill_numeric_missing_values(
    df: pd.DataFrame, fill_values: pd.Series
) -> pd.DataFrame:
    """Заполняет пропуски в числовых колонках заранее вычисленными значениями.

    Args:
        df: датафрейм (train или test), в котором нужно заполнить
            пропуски.
        fill_values: результат compute_numeric_fill_values, вычисленный
            на train — применяется одинаково к train и test.

    Returns:
        pd.DataFrame: новый датафрейм без пропусков в перечисленных
            в fill_values колонках.
    """
    df = df.copy()
    missing_counts = df[fill_values.index].isna().sum()

    for column, missing_count, fill_value in zip(
        fill_values.index, missing_counts, fill_values, strict=True
    ):
        filled_count = int(missing_count)
        if filled_count == 0:
            continue
        logger.info(
            "%s: заполнено %d пропусков значением %.4f",
            column,
            filled_count,
            fill_value,
        )
    return df.fillna(fill_values)


def fill_categorical_missing_values(
    df: pd.DataFrame,
    categorical_columns: list[str],
    missing_label: str = MISSING_CATEGORY_LABEL,
) -> pd.DataFrame:
    """Заполняет пропуски в категориальных колонках явной категорией.

    Args:
        df: датафрейм (train или test), в котором нужно заполнить пропуски.
        categorical_columns: список категориальных колонок.
        missing_label: значение, которым заполняются пропуски.

    Returns:
        pd.DataFrame: новый датафрейм без пропусков в categorical_columns.
    """
    missing_counts = df[categorical_columns].isna().sum()

    for column, missing_count in zip(
        categorical_columns,
        missing_counts,
        strict=True,
    ):
        filled_count = int(missing_count)
        if filled_count == 0:
            continue
        logger.info(
            "%s: заполнено %d пропусков значением %r",
            column,
            filled_count,
            missing_label,
        )

    return df.fillna(dict.fromkeys(categorical_columns, missing_label))


def find_informative_missing_columns(
    train_df: pd.DataFrame,
    candidate_columns: list[str],
    target_column: str = "TARGET",
    significance_level: float = 0.05,
) -> list[str]:
    """Находит колонки, пропуск в которых статистически значимо связан с TARGET.

    Args:
        train_df: обучающая выборка (после split_train_test), должна
            содержать target_column.
        candidate_columns: список колонок-кандидатов (обычно —
            numeric_columns + categorical_columns, т.е. все колонки, в
            которых в принципе встречаются пропуски).
        target_column: название колонки с целевой переменной.
        significance_level: порог p-value для статистической
            значимости связи.

    Returns:
        list[str]: подмножество candidate_columns, для которых пропуск
            статистически значимо связан с target_column.
    """
    testable_columns = []
    p_values = []

    for column in candidate_columns:
        is_missing = train_df[column].isna()

        if is_missing.nunique() < 2:
            continue

        contingency_table = pd.crosstab(is_missing, train_df[target_column])
        p_value = cast("float", chi2_contingency(contingency_table)[1])

        testable_columns.append(column)
        p_values.append(p_value)

    if not p_values:
        return []

    is_significant, corrected_p_values, _, _ = multipletests(
        p_values, alpha=significance_level, method="fdr_bh"
    )

    informative_columns = []
    for column, p_value, corrected_p_value, significant in zip(
        testable_columns,
        p_values,
        corrected_p_values,
        is_significant,
        strict=True,
    ):
        if significant:
            informative_columns.append(column)
            logger.info(
                "%s: пропуск значимо связан с %s (p=%.4g, FDR-скорр. p=%.4g)",
                column,
                target_column,
                p_value,
                corrected_p_value,
            )
        else:
            logger.debug(
                "%s: пропуск НЕ значим после FDR-поправки (p=%.4g, скорр. p=%.4g)",
                column,
                target_column,
                p_value,
                corrected_p_value,
            )

    return informative_columns


def drop_correlated_missing_flags(
    train_df: pd.DataFrame,
    columns: list[str],
    correlation_threshold: float = 0.9,
) -> list[str]:
    """Убирает колонки с почти одинаковым паттерном пропуска.

    Args:
        train_df: обучающая выборка, корреляция считается только на ней.
        columns: список колонок-кандидатов (результат find_informative_missing_columns).
        correlation_threshold: порог корреляции индикаторов пропуска,
            выше которого колонки считаются дублирующими друг друга.

    Returns:
        list[str]: подмножество columns без дублирующих друг друга по
            паттерну пропуска колонок — для каждой группы дублей
            оставлена только первая встреченная в порядке columns.
    """
    if len(columns) < 2:
        return list(columns)
    missing_indicators = train_df[columns].isna().astype(int)
    correlation_matrix = missing_indicators.corr().abs()

    return drop_correlated_columns(
        correlation_matrix, columns, correlation_threshold
    )


def handle_missing_values(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    columns: ColumnGroups,
    *,
    target_column: str = "TARGET",
    keep_numeric_nan: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Точка входа для обработки пропусков.

    Args:
        train_df: обучающая выборка после split_train_test.
        test_df: тестовая выборка после split_train_test.
        columns: числовые и категориальные колонки для обработки (см.
            ColumnGroups) — числовые заполняются медианой (для строго
            бинарных колонок это равносильно заполнению модой),
            категориальные — MISSING_CATEGORY_LABEL.
        target_column: название колонки с целевой переменной — нужно
            для find_informative_missing_columns.
        keep_numeric_nan: если True — числовые колонки не заполняются
            (остаются NaN для нативной обработки бустингом).
            Категориальные колонки заполняются в любом случае.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: (train_df, test_df) с
            обработанными пропусками и добавленными флагами
            `{column}_was_missing`.
    """
    all_candidate_columns = columns.numeric + columns.categorical
    missing_flag_columns = find_informative_missing_columns(
        train_df,
        all_candidate_columns,
        target_column,
    )

    missing_flag_columns = drop_correlated_missing_flags(
        train_df,
        missing_flag_columns,
    )

    train_df = add_missing_value_flags(train_df, missing_flag_columns)
    test_df = add_missing_value_flags(test_df, missing_flag_columns)

    train_df = fill_categorical_missing_values(train_df, columns.categorical)
    test_df = fill_categorical_missing_values(test_df, columns.categorical)

    if keep_numeric_nan:
        logger.info(
            "keep_numeric_nan=True: числовые колонки оставлены с NaN "
            "для нативной обработки бустингом"
        )
        return train_df, test_df

    fill_values = compute_numeric_fill_values(train_df, columns.numeric)
    train_df = fill_numeric_missing_values(train_df, fill_values)
    test_df = fill_numeric_missing_values(test_df, fill_values)

    return train_df, test_df
