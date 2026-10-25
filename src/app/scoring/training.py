"""Подбор гиперпараметров на кросс-валидации и обучение моделей.

Модуль берёт конфигурации моделей из `models.py` и подготовленные
данные из `data_preparation.py`, запускает поиск гиперпараметров на
кросс-валидации и возвращает обученные на всей обучающей выборке
лучшие модели.
"""

import logging
from typing import Any

import pandas as pd
from sklearn.model_selection import GridSearchCV, StratifiedKFold

from app.scoring.models import ModelConfig

logger = logging.getLogger(__name__)


def run_cross_validation(
    model_config: ModelConfig,
    x_train: pd.DataFrame,
    y_train: pd.Series,
    cv_folds: int,
    random_state: int,
) -> object:
    """Подбирает гиперпараметры модели на кросс-валидации.

    Args:
        model_config: конфигурация модели (estimator + сетка параметров).
        x_train: матрица признаков обучающей выборки, подходящая по
            подготовке для этой модели.
        y_train: целевая переменная обучающей выборки.
        cv_folds: количество фолдов для StratifiedKFold.
        random_state: random_state для воспроизводимости разбиений
            кросс-валидации.

    Returns:
        Обученный (на всей x_train/y_train) лучший estimator после
        поиска по сетке гиперпараметров.
    """
    cv = StratifiedKFold(
        n_splits=cv_folds,
        shuffle=True,
        random_state=random_state,
    )
    grid_search = GridSearchCV(
        estimator=model_config.estimator,
        param_grid=model_config.param_grid,
        scoring="roc_auc",
        cv=cv,
        refit=True,
        n_jobs=-1,
        verbose=2,
    )
    grid_search.fit(x_train, y_train, **model_config.fit_params)

    logger.info(
        "%s: best_params_=%s, best_score_=%.4f",
        model_config.name,
        grid_search.best_params_,
        grid_search.best_score_,
    )
    return grid_search.best_estimator_


def train_all_models(
    model_configs: list[ModelConfig],
    datasets: dict[str, tuple],
    cv_folds: int,
    random_state: int,
) -> dict[str, Any]:
    """Обучает все модели из model_configs с подбором гиперпараметров.

    Args:
        model_configs: список конфигураций моделей (см. models.py).
        datasets: словарь с подготовленными данными по ключам
            "linear"/"tree_based"/"catboost" (см. data_preparation.
            DATASET_KINDS и prepare_datasets_for_models).
        cv_folds: количество фолдов кросс-валидации.
        random_state: random_state для воспроизводимости.

    Returns:
        dict[str, Any]: словарь {имя_модели: обученный_estimator}
            для всех 4 моделей.
    """
    trained_models = {}
    for model_config in model_configs:
        x_train, _, y_train, _ = datasets[model_config.dataset_kind]

        logger.info("Начато обучение %s", model_config.name)
        trained_models[model_config.name] = run_cross_validation(
            model_config,
            x_train,
            y_train,
            cv_folds,
            random_state,
        )
        logger.info("Обучение %s завершено", model_config.name)

    return trained_models
