"""Описание моделей и сеток гиперпараметров для кредитного скоринга."""

from dataclasses import dataclass, field
from typing import Any

from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier


@dataclass
class ModelConfig:
    """Конфигурация одной модели для кросс-валидации и обучения.

    Attributes:
        name: короткое имя модели, используется в логах, итоговой
            таблице метрик и имени pickle-файла (например,
            "logistic_regression").
        estimator: неинициализированный sklearn-совместимый estimator
            (или Pipeline) с базовыми параметрами.
        param_grid: сетка гиперпараметров для GridSearchCV/
            RandomizedSearchCV — dict или list[dict]. Ключи
            должны соответствовать параметрам estimator (с префиксом
            шага Pipeline, если он используется, например "clf__C").
        dataset_kind: какой вариант подготовленных данных использовать
            — "linear", "tree_based" или "catboost" (см.
            data_preparation.DATASET_KINDS).
        fit_params: дополнительные именованные аргументы для
            estimator.fit(), прокидываемые через GridSearchCV.fit(
            **fit_params) на каждом фолде. Нужны для параметров,
            которые нельзя передавать в конструктор estimator'а.
            Для остальных моделей пустой словарь по умолчанию.
    """

    name: str
    estimator: Any
    param_grid: dict | list[dict]
    dataset_kind: str
    fit_params: dict = field(default_factory=dict)


def get_logistic_regression_config(random_state: int) -> ModelConfig:
    """Возвращает конфигурацию логистической регрессии для CV.

    Args:
        random_state: random_state для воспроизводимости (из CLI).

    Returns:
        ModelConfig: Конфигурация одной модели для кросс-валидации и обучения.
    """
    return ModelConfig(
        name="logistic_regression",
        estimator=LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=random_state,
        ),
        param_grid=[
            {
                "l1_ratio": [0],
                "solver": ["lbfgs"],
                "C": [0.01, 0.1, 1, 10],
            },
            {
                "l1_ratio": [1],
                "solver": ["liblinear"],
                "C": [0.01, 0.1],
            },
            {
                "l1_ratio": [0.3, 0.5, 0.7],
                "solver": ["saga"],
                "C": [0.1],
                "max_iter": [300],
            },
        ],
        dataset_kind="linear",
    )


def get_decision_tree_config(random_state: int) -> ModelConfig:
    """Возвращает конфигурацию решающего дерева для CV.

    Args:
        random_state: random_state для воспроизводимости (из CLI).

    Returns:
        ModelConfig: Конфигурация одной модели для кросс-валидации и обучения.
    """
    return ModelConfig(
        name="decision_tree",
        estimator=DecisionTreeClassifier(
            class_weight="balanced",
            random_state=random_state,
        ),
        param_grid={
            "max_depth": [3, 5, 7, 10],
            "min_samples_leaf": [1, 10, 50, 100],
        },
        dataset_kind="tree_based",
    )


def get_random_forest_config(random_state: int) -> ModelConfig:
    """Возвращает конфигурацию случайного леса для CV.

    Args:
        random_state: random_state для воспроизводимости (из CLI).

    Returns:
        ModelConfig: Конфигурация одной модели для кросс-валидации и обучения.
    """
    return ModelConfig(
        name="random_forest",
        estimator=RandomForestClassifier(
            class_weight="balanced",
            random_state=random_state,
            n_jobs=1,
        ),
        param_grid={
            "n_estimators": [100, 300],
            "max_depth": [5, 10, None],
            "min_samples_leaf": [1, 10, 50],
        },
        dataset_kind="tree_based",
    )


def get_gradient_boosting_config(
    categorical_columns: list[str],
    random_state: int,
) -> ModelConfig:
    """Возвращает конфигурацию CatBoostClassifier для CV.

    Args:
        categorical_columns: названия категориальных колонок.
        random_state: random_state для воспроизводимости (из CLI).

    Returns:
        ModelConfig: Конфигурация одной модели для кросс-валидации и обучения.
    """
    return ModelConfig(
        name="catboost",
        estimator=CatBoostClassifier(
            iterations=150,
            auto_class_weights="Balanced",
            random_state=random_state,
            verbose=50,
            allow_writing_files=False,
        ),
        param_grid={
            "depth": [4, 6, 8],
            "learning_rate": [0.01, 0.05, 0.1],
            "l2_leaf_reg": [1, 3, 5, 10],
        },
        dataset_kind="catboost",
        fit_params={"cat_features": categorical_columns},
    )


def build_model_configs(
    categorical_columns: list[str], random_state: int
) -> list[ModelConfig]:
    """Собирает список конфигураций всех 4 моделей для обучения.

    Args:
        categorical_columns: см. get_gradient_boosting_config.
        random_state: random_state для воспроизводимости (из CLI),
            передаётся во все 4 модели.

    Returns:
        list[ModelConfig]: список из 4 конфигураций — логистическая
            регрессия, дерево, случайный лес, CatBoost.
    """
    return [
        get_logistic_regression_config(random_state),
        get_decision_tree_config(random_state),
        get_random_forest_config(random_state),
        get_gradient_boosting_config(categorical_columns, random_state),
    ]
