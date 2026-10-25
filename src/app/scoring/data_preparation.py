"""Сборка и подготовка датасета для обучения модели кредитного скоринга"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, mannwhitneyu
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from statsmodels.stats.multitest import multipletests

from app.modelling.features.application_train_test_features import (
    get_columns_by_description,
)
from app.scoring.common import drop_correlated_columns
from app.scoring.missing_value_handling import (
    ColumnGroups,
    handle_missing_values,
)
from app.scoring.outlier_handling import (
    clip_outliers_to_percentiles,
    treat_outliers,
)

logger = logging.getLogger(__name__)

HOUSING_INFO_DESCRIPTION = (
    "Normalized information about building where the client lives"
)


def load_application_with_target(raw_data_dir: Path) -> pd.DataFrame:
    """Загружает application_train.csv с целевой переменной TARGET.

    Args:
        raw_data_dir: директория с исходными CSV Home Credit Default Risk.

    Returns:
        pd.DataFrame: датафрейм с колонками SK_ID_CURR, TARGET и
            исходными признаками application_train.
    """
    return pd.read_csv(raw_data_dir / "application_train.csv")


def load_generated_features(
    features_dir: Path, required_files: list[str]
) -> dict[str, pd.DataFrame]:
    """Загружает все сгенерированные таблицы признаков.

    Args:
        features_dir: директория с результатами генерации признаков
            (обычно data/processed/, см. docs/feature_generation.md).
        required_files: список файлов, необходимых к выгрузке.

    Returns:
        dict[str, pd.DataFrame]: словарь {имя_файла: датафрейм}, где у
            каждого датафрейма есть колонка SK_ID_CURR для последующего
            join.

    Raises:
        FileNotFoundError: если один из файлов из required_files
            отсутствует в features_dir.
    """
    features = {}
    for name in required_files:
        file_path = features_dir / f"{name}.csv"
        if not file_path.exists():
            raise FileNotFoundError(
                f"Не найден обязательный файл признаков: {file_path}"
            )
        features[name] = pd.read_csv(file_path)
    return features


def merge_features_by_client_id(
    base_df: pd.DataFrame, feature_tables: dict[str, pd.DataFrame]
) -> pd.DataFrame:
    """Объединяет таблицу заявок с таблицами признаков по SK_ID_CURR.

    Args:
        base_df: датафрейм application_train (с TARGET), к которому
            присоединяются признаки.
        feature_tables: словарь {имя_файла: датафрейм} с дополнительными
            признаками, каждый содержит колонку SK_ID_CURR.

    Returns:
        pd.DataFrame: единая широкая таблица "клиент -> все признаки
            + TARGET", готовая для дальнейшей очистки и отбора признаков.
    """
    for df in feature_tables.values():
        base_df = base_df.merge(
            df, on="SK_ID_CURR", how="left", validate="one_to_one"
        )
    return base_df


def drop_redundant_housing_columns(
    df: pd.DataFrame, raw_data_dir: Path
) -> pd.DataFrame:
    """Удаляет raw-колонки "жилищного" блока, избыточные относительно has_full_apprt_info.

    Args:
        df: датафрейм с признаками, из которого нужно убрать блок.
        raw_data_dir: директория с HomeCredit_columns_description.csv.

    Returns:
        pd.DataFrame: df без колонок жилищного блока.
    """
    description_df = pd.read_csv(
        raw_data_dir / "HomeCredit_columns_description.csv",
        encoding="cp1252",
    )
    housing_columns = get_columns_by_description(
        description_df, HOUSING_INFO_DESCRIPTION
    )
    return df.drop(columns=housing_columns)


def screen_columns_by_significance(
    df: pd.DataFrame,
    candidate_columns: list[str],
    target_column: str = "TARGET",
    significance_level: float = 0.05,
) -> list[str]:
    """FDR-скрининг связи колонок-кандидатов с целевой переменной.

    Args:
        df: датафрейм с признаками и TARGET.
        candidate_columns: колонки для проверки.
        target_column: название колонки с целевой переменной.
        significance_level: порог для скорректированного p-value (FDR).

    Returns:
        list[str]: подмножество candidate_columns, прошедшее проверку.
    """
    testable_columns = []
    p_values = []

    for column in candidate_columns:
        series = df[column].dropna()
        if series.nunique() < 2:
            continue

        if pd.api.types.is_numeric_dtype(series):
            group_0 = df.loc[df[target_column] == 0, column].dropna()
            group_1 = df.loc[df[target_column] == 1, column].dropna()
            if group_0.empty or group_1.empty:
                continue
            p_value = cast("float", mannwhitneyu(group_0, group_1)[1])
        else:
            contingency_table = pd.crosstab(df[column], df[target_column])
            p_value = cast("float", chi2_contingency(contingency_table)[1])

        testable_columns.append(column)
        p_values.append(p_value)

    if not p_values:
        return []

    is_significant, *_ = multipletests(
        p_values, alpha=significance_level, method="fdr_bh"
    )

    return [
        column
        for column, significant in zip(
            testable_columns, is_significant, strict=True
        )
        if significant
    ]


def drop_correlated_numeric_features(
    df: pd.DataFrame,
    numeric_columns: list[str],
    correlation_threshold: float = 0.9,
) -> list[str]:
    """Убирает числовые признаки, сильно коррелирующие с уже оставленными.

    Args:
        df: датафрейм с признаками.
        numeric_columns: числовые колонки-кандидаты.
        correlation_threshold: порог |корреляции|, выше которого вторая
            колонка в паре считается избыточной.

    Returns:
        list[str]: подмножество numeric_columns без избыточных колонок.
    """
    if len(numeric_columns) < 2:
        return list(numeric_columns)

    correlation_matrix = df[numeric_columns].corr().abs()

    return drop_correlated_columns(
        correlation_matrix, numeric_columns, correlation_threshold
    )


def select_features(df: pd.DataFrame) -> list[str]:
    """Отбирает итоговый список признаков для обучения.

    Args:
        df: датафрейм с признаками и TARGET (housing-блок уже исключён
            вызывающим кодом, см. drop_redundant_housing_columns).

    Returns:
        list[str]: названия колонок-признаков (без SK_ID_CURR и
            TARGET), которые будут использоваться для обучения.
    """
    candidate_columns = [
        column
        for column in df.columns
        if column not in ("SK_ID_CURR", "TARGET")
    ]

    screened_columns = screen_columns_by_significance(df, candidate_columns)

    kept_numeric = [
        column
        for column in screened_columns
        if pd.api.types.is_numeric_dtype(df[column])
    ]
    kept_categorical = [
        column for column in screened_columns if column not in kept_numeric
    ]

    kept_numeric = drop_correlated_numeric_features(df, kept_numeric)

    return kept_numeric + kept_categorical


def encode_categorical_features(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    categorical_columns: list[str],
    *,
    for_linear_model: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Кодирует категориальные признаки.

    Args:
        train_df: обучающая выборка.
        test_df: тестовая выборка.
        categorical_columns: список названий категориальных колонок.
        for_linear_model: True — one-hot encoding (для логрег). False —
            компактное кодирование для деревьев/леса/бустинга.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: (train_df, test_df) с
            закодированными категориальными признаками, с одинаковым
            набором колонок в обеих выборках.
    """
    if for_linear_model:
        encoder = OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False,
            drop="first",
        )
        encoded_train = cast(
            "np.ndarray", encoder.fit_transform(train_df[categorical_columns])
        )
        encoded_test = cast(
            "np.ndarray", encoder.transform(test_df[categorical_columns])
        )
        feature_names = encoder.get_feature_names_out(categorical_columns)

        encoded_train_df = pd.DataFrame(
            encoded_train, columns=feature_names, index=train_df.index
        )
        encoded_test_df = pd.DataFrame(
            encoded_test, columns=feature_names, index=test_df.index
        )
        train_df = pd.concat(
            [train_df.drop(columns=categorical_columns), encoded_train_df],
            axis=1,
        )
        test_df = pd.concat(
            [test_df.drop(columns=categorical_columns), encoded_test_df],
            axis=1,
        )
        return train_df, test_df

    encoder = OrdinalEncoder(
        handle_unknown="use_encoded_value",
        unknown_value=-1,
    )
    train_df[categorical_columns] = encoder.fit_transform(
        train_df[categorical_columns]
    )
    test_df[categorical_columns] = encoder.transform(
        test_df[categorical_columns]
    )
    return train_df, test_df


def scale_numeric_features(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    numeric_columns: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Масштабирует числовые признаки (только для логрег).

    Args:
        train_df: обучающая выборка.
        test_df: тестовая выборка.
        numeric_columns: список названий числовых колонок.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: (train_df, test_df) с
            масштабированными колонками.
    """
    train_df = train_df.copy()
    test_df = test_df.copy()
    scaler = StandardScaler().fit(train_df[numeric_columns])
    train_df[numeric_columns] = scaler.transform(train_df[numeric_columns])
    test_df[numeric_columns] = scaler.transform(test_df[numeric_columns])
    return train_df, test_df


@dataclass
class TrainTestSplit:
    """Результат разбиения на train/test.

    Attributes:
        X_train: признаки обучающей выборки (без TARGET).
        X_test: признаки тестовой выборки (без TARGET).
        y_train: целевая переменная обучающей выборки.
        y_test: целевая переменная тестовой выборки.
    """

    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series


def split_train_test(
    df: pd.DataFrame,
    target_column: str,
    test_size: float,
    random_state: int,
) -> TrainTestSplit:
    """Разбивает датасет на обучающую и тестовую выборки.

    Args:
        df: итоговый датафрейм с признаками и целевой переменной.
        target_column: название колонки с целевой переменной (TARGET).
        test_size: доля тестовой выборки (например, 0.2).
        random_state: random_state для воспроизводимости разбиения.

    Returns:
        TrainTestSplit: X_train, X_test, y_train, y_test.
    """
    X = df.drop(columns=[target_column])
    y = df[target_column]
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )
    return TrainTestSplit(
        X_train=cast("pd.DataFrame", X_train),
        X_test=cast("pd.DataFrame", X_test),
        y_train=cast("pd.Series", y_train),
        y_test=cast("pd.Series", y_test),
    )


DATASET_KINDS = ("linear", "tree_based", "catboost")


def prepare_single_model_dataset(
    split: TrainTestSplit,
    columns: ColumnGroups,
    *,
    dataset_kind: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Готовит train/test для одного семейства моделей.

    Args:
        split: X_train/X_test/y_train/y_test.
        columns: числовые и категориальные колонки (см. ColumnGroups).
        dataset_kind: один из DATASET_KINDS.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: подготовленные (train, test).

    Raises:
        ValueError: если dataset_kind не входит в DATASET_KINDS.
    """
    if dataset_kind not in DATASET_KINDS:
        raise ValueError(f"Неизвестный dataset_kind: {dataset_kind!r}")

    train_with_target = split.X_train.assign(TARGET=split.y_train)
    test_with_target = split.X_test.assign(TARGET=split.y_test)

    train_k, test_k = handle_missing_values(
        train_with_target.copy(),
        test_with_target.copy(),
        columns,
        keep_numeric_nan=(dataset_kind == "catboost"),
    )
    train_k = train_k.drop(columns=["TARGET"])
    test_k = test_k.drop(columns=["TARGET"])

    if dataset_kind == "catboost":
        return train_k, test_k

    train_k, test_k = encode_categorical_features(
        train_k,
        test_k,
        columns.categorical,
        for_linear_model=(dataset_kind == "linear"),
    )

    if dataset_kind == "linear":
        train_k, test_k = clip_outliers_to_percentiles(
            train_k, test_k, columns.numeric
        )
        train_k, test_k = scale_numeric_features(
            train_k, test_k, columns.numeric
        )

    return train_k, test_k


def prepare_datasets_for_models(
    raw_data_dir: Path,
    features_dir: Path,
    random_state: int,
) -> dict[str, tuple]:
    """Собирает весь пайплайн подготовки данных в одну функцию.

    Это точка входа, которую будет вызывать `main()`.

    Args:
        raw_data_dir: директория с исходными CSV Home Credit.
        features_dir: директория со сгенерированными в ДЗ-3 признаками.
        random_state: random_state для воспроизводимости.

    Returns:
        dict[str, tuple]: словарь вида
            {
                "linear": (X_train, X_test, y_train, y_test),
                "tree_based": (X_train, X_test, y_train, y_test),
                "catboost": (X_train, X_test, y_train, y_test),
            }
    """
    base_df = load_application_with_target(raw_data_dir)
    feature_tables = load_generated_features(
        features_dir,
        [
            "application_train_features",
            "bureau_features",
            "bureau_balance_features",
        ],
    )
    df = merge_features_by_client_id(base_df, feature_tables)
    df = treat_outliers(df)
    df = drop_redundant_housing_columns(df, raw_data_dir)

    feature_columns = select_features(df)
    split = split_train_test(
        df[[*feature_columns, "TARGET"]],
        "TARGET",
        test_size=0.2,
        random_state=random_state,
    )

    numeric_columns = split.X_train.select_dtypes("number").columns.tolist()
    categorical_columns = split.X_train.select_dtypes(
        exclude="number"
    ).columns.tolist()
    columns = ColumnGroups(numeric_columns, categorical_columns)

    datasets = {}
    for kind in DATASET_KINDS:
        train_k, test_k = prepare_single_model_dataset(
            split, columns, dataset_kind=kind
        )
        datasets[kind] = (train_k, test_k, split.y_train, split.y_test)

    return datasets
