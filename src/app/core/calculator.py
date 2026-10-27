"""Калькулятор одобренной суммы по вероятности дефолта.

Логика: минимизировать невозвращённую сумму и максимизировать ожидаемую
прибыль, используя ожидаемую стоимость (expected value) выдачи.
"""

from app.core.api import Features

_LGD = 1.0


class Calculator:
    """Подбирает одобренную сумму на основе ожидаемой доходности.

    Attributes:
        _fallback_interest_rate: ставка маржи для клиентов без
            кредитной истории (interest_rate == NaN).
    """

    def __init__(self, fallback_interest_rate: float):
        """Создаёт калькулятор.

        Args:
            fallback_interest_rate: ставка по умолчанию, если у клиента
                нет признака interest_rate (нет истории предыдущих
                кредитов). Берётся как медиана interest_rate по train.
        """
        self._fallback_interest_rate = fallback_interest_rate

    def _effective_rate(self, features: Features) -> float:
        """Возвращает ставку маржи для клиента (его личную или fallback).

        Args:
            features: признаки заявки.

        Returns:
            float: эффективная ставка маржи, используемая в расчёте EV.
        """
        rate = features.interest_rate
        if rate is None or rate != rate:
            return self._fallback_interest_rate
        return rate

    def calc_amount(self, proba: float, features: Features) -> int:
        """Рассчитывает одобренную сумму.

        Точка безубыточности по вероятности дефолта:
            p* = r / (1 + r),
        где r — ставка маржи клиента (см. _effective_rate). При p < p*
        каждая единица выданной суммы увеличивает ожидаемый доход,
        поэтому выгодно подходить к запрошенной сумме вплотную; при
        p > p* — наоборот, и сумму нужно урезать или отказать.
        Коэффициенты порогов ниже (x/1.2x/2x от p*).

        Args:
            proba: вероятность дефолта клиента.
            features: признаки заявки.

        Returns:
            int: одобренная сумма (0, если выдавать невыгодно).
        """
        rate = self._effective_rate(features)
        breakeven_proba = rate / (1 + rate)
        requested_amount = features.AMT_CREDIT

        if proba < breakeven_proba:
            return int(requested_amount)
        if proba < 1.2 * breakeven_proba:
            return int(requested_amount * 0.5)
        if proba < 2 * breakeven_proba:
            return int(requested_amount * 0.2)
        return 0
