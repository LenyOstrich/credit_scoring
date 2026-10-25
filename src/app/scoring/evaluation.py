"""Оценка качества обученных моделей и сравнение результатов.

Модуль считает метрики на отложенной тестовой выборке для каждой из
4 обученных моделей, собирает сводную таблицу для сравнения и строит
графики (ROC-кривые, важность признаков).
"""

import logging
from pathlib import Path
from typing import Protocol, cast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.metrics import (
    RocCurveDisplay,
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

logger = logging.getLogger(__name__)


class FittedClassifier(Protocol):
    """Минимальный интерфейс обученной модели, нужный этому модулю."""

    def predict(self, x: pd.DataFrame) -> np.ndarray:
        """Возвращает предсказанные классы (0/1) для x."""
        ...

    def predict_proba(self, x: pd.DataFrame) -> np.ndarray:
        """Возвращает вероятности классов для x (столбцы — классы)."""
        ...


def compute_classification_metrics(
    model: FittedClassifier,
    x_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict[str, float]:
    """Считает метрики качества модели на тестовой выборке.

    Args:
        model: обученный estimator.
        x_test: матрица признаков тестовой выборки.
        y_test: истинные значения целевой переменной тестовой выборки.

    Returns:
        dict[str, float]: roc_auc, average_precision и accuracy/precision/
            recall/f1 при пороге 0.5.
    """
    y_proba = model.predict_proba(x_test)[:, 1]
    y_pred = model.predict(x_test)

    return {
        "roc_auc": float(roc_auc_score(y_test, y_proba)),
        "average_precision": float(average_precision_score(y_test, y_proba)),
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
    }


def build_comparison_table(
    metrics_per_model: dict[str, dict[str, float]],
) -> pd.DataFrame:
    """Собирает сводную таблицу метрик для сравнения моделей.

    Args:
        metrics_per_model: словарь {имя_модели: метрики}, где метрики —
            результат compute_classification_metrics.

    Returns:
        pd.DataFrame: таблица, где строки — модели, колонки — метрики.
    """
    return pd.DataFrame.from_dict(metrics_per_model, orient="index")


def plot_roc_curves(
    models: dict[str, FittedClassifier],
    x_test_per_model: dict[str, pd.DataFrame],
    y_test: pd.Series,
    output_path: Path,
) -> None:
    """Строит и сохраняет график ROC-кривых всех моделей на одном графике.

    Args:
        models: словарь {имя_модели: обученный estimator}.
        x_test_per_model: словарь {имя_модели: соответствующая матрица
            признаков тестовой выборки}, т.к. у линейных и древесных
            моделей подготовка данных отличается.
        y_test: истинные значения целевой переменной (одинаковые для
            всех моделей, т.к. разбиение train/test делается один раз).
        output_path: путь к файлу (.png/.jpg/.pdf), куда сохранить график.
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    for name, model in models.items():
        RocCurveDisplay.from_estimator(
            cast("BaseEstimator", model),
            x_test_per_model[name],
            y_test,
            ax=ax,
            name=name,
        )

    ax.set_title("ROC-кривые моделей кредитного скоринга")
    ax.plot(
        [0, 1], [0, 1], linestyle="--", color="grey", label="Случайная модель"
    )
    ax.legend()
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)
    logger.info("ROC-кривые сохранены: %s", output_path)


def plot_feature_importances(
    model: FittedClassifier,
    feature_names: list[str],
    output_path: Path,
    top_n: int = 20,
) -> None:
    """Строит и сохраняет график важности признаков для одной модели.

    Args:
        model: обученный estimator (дерево/лес/бустинг — feature_importances_,
            логрег — coef_).
        feature_names: список названий признаков в том же порядке, в
            котором они были переданы модели при обучении.
        output_path: путь к файлу (.png/.jpg/.pdf), куда сохранить график.
        top_n: сколько признаков с наибольшей важностью показать.

    Raises:
        ValueError: если у модели нет ни feature_importances_, ни coef_.
    """
    importances = getattr(model, "feature_importances_", None)
    if importances is None:
        coef = getattr(model, "coef_", None)
        if coef is None:
            raise ValueError(
                f"Модель {type(model).__name__} не поддерживает "
                "feature_importances_/coef_"
            )
        importances = abs(coef[0])

    top_importances = (
        pd.Series(importances, index=feature_names)
        .sort_values(ascending=False)
        .head(top_n)
        .sort_values()
    )

    fig, ax = plt.subplots(figsize=(8, max(4, top_n * 0.3)))
    top_importances.plot.barh(ax=ax)
    ax.set_title(f"Топ-{top_n} признаков по важности ({type(model).__name__})")
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)
    logger.info("График важности признаков сохранён: %s", output_path)


def save_metrics_report(
    comparison_table: pd.DataFrame, output_path: Path
) -> None:
    """Сохраняет сводную таблицу метрик в файл отчёта.

    Args:
        comparison_table: таблица метрик (см. build_comparison_table).
        output_path: путь к файлу отчёта (.csv).
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    comparison_table.to_csv(output_path)
    logger.info("Таблица метрик сохранена: %s", output_path)
