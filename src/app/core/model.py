"""Обёртка над обученной моделью кредитного скоринга."""

import json
import pickle
from pathlib import Path

from app.core.api import (
    Features,
    ScoringDecision,
    ScoringResult,
    build_feature_mappings,
)
from app.core.calculator import Calculator


class SimpleModel:
    """Класс-обёртка для модели: бинарное решение по фиксированному порогу.

    Attributes:
        _threshold: порог вероятности дефолта для решения ACCEPTED/DECLINED.
        _approved_amount: фиксированная одобренная сумма при ACCEPTED.
    """

    _threshold = 0.3
    _approved_amount = 100_000

    def __init__(self, model_path: str, schema_path: str):
        """Создаёт обёртку над pickle-моделью.

        Args:
            model_path: путь к pickle-файлу обученной модели
                (например, "models/catboost.pkl").
            schema_path: путь к JSON-схеме признаков
                (например, "docs/model_core/feature_schema.json"),
                содержащей порядок колонок, в котором обучалась модель.
        """
        with open(model_path, "rb") as pickled_model:
            self._model = pickle.load(pickled_model)

        schema = json.loads(Path(schema_path).read_text(encoding="utf-8"))
        self._feature_order = schema["feature_order"]
        self._field_to_column, self._column_to_field = build_feature_mappings(
            schema
        )

    def get_scoring_result(self, features: Features) -> ScoringResult:
        """Возвращает решение по заявке на основе простого порога.

        Args:
            features: уже рассчитанные признаки заявки.

        Returns:
            ScoringResult: решение, сумма, порог и вероятность дефолта.
        """
        proba = self._predict_proba(features)

        decision = ScoringDecision.DECLINED
        amount = 0
        if proba < self._threshold:
            amount = self._approved_amount
            decision = ScoringDecision.ACCEPTED

        return ScoringResult(
            decision=decision,
            amount=amount,
            threshold=self._threshold,
            proba=proba,
        )

    def _predict_proba(self, features: Features) -> float:
        """Считает вероятность дефолта через обёрнутую модель.

        Важен порядок признаков — он должен совпадать с порядком,
        использованным при обучении (self._feature_order).

        Args:
            features: признаки заявки.

        Returns:
            float: вероятность дефолта (класс 1).
        """
        frame = features.to_frame(self._feature_order, self._field_to_column)
        return float(self._model.predict_proba(frame)[:, 1][0])


class AdvancedModel(SimpleModel):
    """Обёртка над моделью с расчётом одобренной суммы через Calculator."""

    def __init__(self, model_path: str, schema_path: str):
        """См. SimpleModel.__init__; дополнительно создаёт Calculator.

        Args:
            model_path: путь к pickle-файлу обученной модели.
            schema_path: путь к JSON-схеме признаков.
        """
        super().__init__(model_path, schema_path)
        schema = json.loads(Path(schema_path).read_text(encoding="utf-8"))
        self._calculator = Calculator(
            fallback_interest_rate=schema["fallback_interest_rate"],
        )

    def get_scoring_result(self, features: Features) -> ScoringResult:
        """Возвращает решение с суммой, подобранной Calculator'ом.

        Args:
            features: уже рассчитанные признаки заявки.

        Returns:
            ScoringResult: решение, подобранная сумма, порог и proba.
        """
        proba = self._predict_proba(features)

        decision = ScoringDecision.DECLINED
        amount = 0
        if proba < self._threshold:
            decision = ScoringDecision.ACCEPTED
            amount = self._calculator.calc_amount(proba, features)

        return ScoringResult(
            decision=decision,
            amount=amount,
            threshold=self._threshold,
            proba=proba,
        )
