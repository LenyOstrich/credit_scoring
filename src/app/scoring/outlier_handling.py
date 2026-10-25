"""Обработка аномалий и выбросов для датасета кредитного скоринга"""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Источник: https://www.kaggle.com/c/home-credit-default-risk/discussion/57247
DAYS_EMPLOYED_SENTINEL_VALUE = 365243

NON_FEATURE_NUMERIC_COLUMNS = frozenset({"SK_ID_CURR", "TARGET"})


class UnexpectedSentinelValueError(ValueError):
    """Найден похожий на sentinel кластер значений.

    Необходимо добавить функцию-обработчик.
    """


def fix_days_employed_sentinel(df: pd.DataFrame) -> pd.DataFrame:
    """Заменяет служебное значение-заглушку на NaN.

    Args:
        df: датафрейм с колонкой, содержащей аномальные значения

    Returns:
        pd.Dataframe: датафрейм, где аномальное значение заменено на NaN
    """
    df = df.copy()

    sentinel_mask = df["DAYS_EMPLOYED"] == DAYS_EMPLOYED_SENTINEL_VALUE
    replaced_count = int(sentinel_mask.sum())

    df.loc[sentinel_mask, "DAYS_EMPLOYED"] = np.nan

    logger.info(
        "DAYS_EMPLOYED: заменено %d sentinel-значений (%d) на NaN (%.1f%% строк)",
        replaced_count,
        DAYS_EMPLOYED_SENTINEL_VALUE,
        100 * replaced_count / len(df) if len(df) else 0,
    )

    return df


def detect_unexpected_sentinel_clusters(
    df: pd.DataFrame,
    numeric_columns: list[str],
    cluster_share_threshold: float = 0.01,
    iqr_multiplier: float = 10.0,
) -> None:
    """Проверяет, что в данных не осталось необработанных sentinel-кластеров.

    Args:
        df: датафрейм после применения всех известных fix_*_sentinel
            функций.
        numeric_columns: список числовых колонок для проверки.
        cluster_share_threshold: минимальная доля строк с одинаковым
            значением, чтобы считать его подозрительным кластером.
        iqr_multiplier: во сколько межквартильных промежутков значение
            должно отстоять от медианы, чтобы считаться "далёким" от
            основной массы распределения.

    Raises:
        UnexpectedSentinelValueError: если найдена колонка с
            подозрительным необработанным кластером значений.
    """
    for column in numeric_columns:
        values = df[column].dropna()
        if values.empty:
            continue

        value_shares = values.value_counts(normalize=True)
        most_common_value = value_shares.index[0]
        most_common_share = value_shares.iloc[0]

        if most_common_share < cluster_share_threshold:
            continue

        first_quartile, median, third_quartile = values.quantile(
            [0.25, 0.5, 0.75]
        )
        interquartile_range = third_quartile - first_quartile

        if interquartile_range == 0:
            continue

        distance_in_iqr = abs(most_common_value - median) / interquartile_range

        if distance_in_iqr > iqr_multiplier:
            raise UnexpectedSentinelValueError(
                f"Колонка {column!r}: значение {most_common_value!r} "
                f"встречается в {most_common_share:.1%} строк и находится "
                f"на расстоянии {distance_in_iqr:.1f} IQR от медианы "
                f"({median!r}) — похоже на необработанный sentinel. "
                "Исследуйте колонку и добавьте атомарную fix_*_sentinel "
                "функцию по аналогии с fix_days_employed_sentinel."
            )

    logger.info(
        "Проверка на необработанные sentinel-кластеры пройдена (%d колонок)",
        len(numeric_columns),
    )


def treat_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """Исправляет известные аномалии данных.

    Args:
        df: объединённый датафрейм с признаками

    Returns:
        pd.Dataframe: датафрейм с исправленными аномалиями
    """
    df = fix_days_employed_sentinel(df)

    numeric_columns = [
        column
        for column in df.select_dtypes(include="number").columns
        if column not in NON_FEATURE_NUMERIC_COLUMNS
    ]
    detect_unexpected_sentinel_clusters(df, numeric_columns)

    return df


def clip_outliers_to_percentiles(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    columns: list[str],
    lower_quantile: float = 0.01,
    upper_quantile: float = 0.99,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Ограничивает длинные хвосты числовых признаков (winsorization).

    Args:
        train_df: обучающая выборка, по которой вычисляются границы
            перцентилей, чтобы не допустить утечки данных из теста.
        test_df: тестовая выборка — к ней применяются границы,
            вычисленные на train_df.
        columns: список числовых колонок, которые нужно ограничить.
        lower_quantile: нижний перцентиль границы обрезки.
        upper_quantile: верхний перцентиль границы обрезки.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: новые (train_df, test_df) с 
            обрезанными по границам train_df значениями в columns.
    """
    train_df = train_df.copy()
    test_df = test_df.copy()

    for column in columns:
        lower_bound, upper_bound = train_df[column].quantile(
            [lower_quantile, upper_quantile]
        )

        train_clipped_count = int(
            (~train_df[column].between(lower_bound, upper_bound)).sum()
        )
        test_clipped_count = int(
            (~test_df[column].between(lower_bound, upper_bound)).sum()
        )

        train_df[column] = train_df[column].clip(lower_bound, upper_bound)
        test_df[column] = test_df[column].clip(lower_bound, upper_bound)

        logger.info(
            "%s: обрезано %d значений в train и %d в test (границы [%.2f, %.2f])",
            column,
            train_clipped_count,
            test_clipped_count,
            lower_bound,
            upper_bound,
        )

    return train_df, test_df
