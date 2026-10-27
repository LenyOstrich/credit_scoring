"""Публичный контракт ядра кредитного скоринга.

Содержит типы, через которые внешний код обменивается данными с ядром:
входные признаки, решение и результат.
"""

from dataclasses import dataclass
from enum import Enum, auto

import pandas as pd


class ScoringDecision(Enum):
    """Возможные решения ядра по заявке."""

    ACCEPTED = auto()
    DECLINED = auto()


@dataclass
class ScoringResult:
    """Результат скоринга одной заявки.

    Attributes:
        decision: решение по заявке (одобрено/отказано).
        amount: одобренная сумма (0, если decision == DECLINED).
        threshold: порог вероятности дефолта, применённый для решения.
        proba: вероятность дефолта, посчитанная моделью.
    """

    decision: ScoringDecision
    amount: int
    threshold: float
    proba: float


def build_feature_mappings(
    schema: dict,
) -> tuple[dict[str, str], dict[str, str]]:
    """Строит сопоставления между именами полей Features и именами колонок модели.

    Args:
        schema: распарсенный docs/model_core/feature_schema.json с ключами
            "feature_order" и "field_names" (синхронизированы по индексу).

    Returns:
        tuple[dict[str, str], dict[str, str]]: (field_to_column, column_to_field).
    """
    field_to_column = dict(
        zip(schema["field_names"], schema["feature_order"], strict=True)
    )
    column_to_field = {
        column: field for field, column in field_to_column.items()
    }
    return field_to_column, column_to_field


@dataclass
class Features:
    """Набор уже рассчитанных признаков для одной заявки.

    Поля, их порядок и типы зафиксированы в docs/model_core/feature_schema.json

    Смысл конкретных признаков:
        - сырые поля заявки (AMT_CREDIT, CODE_GENDER, ...) — см.
          data/home-credit-default-risk/HomeCredit_columns_description.csv.
        - инженерные признаки (docs_number, interest_rate,
          ext_source_score, overdue_credit_ratio и т.д.) — см. докстринги
          в app/modelling/features/application_train_test_features.py
          и app/modelling/features/bureau_features.py.
        - флаги `<признак>_was_missing` — индикаторы пропуска, см.
          app.scoring.missing_value_handling.add_missing_value_flags.
        - поля credit_count_*/overdue_credit_count_*/closed_credit_count_*
          переименованы относительно исходных колонок датафрейма (пробелы
          заменены на "_") — соответствие см. в build_feature_mappings.
    """

    CNT_CHILDREN: float
    AMT_INCOME_TOTAL: float
    AMT_CREDIT: float
    REGION_POPULATION_RELATIVE: float
    DAYS_BIRTH: float
    DAYS_EMPLOYED: float
    DAYS_REGISTRATION: float
    DAYS_ID_PUBLISH: float
    OWN_CAR_AGE: float
    FLAG_EMP_PHONE: float
    FLAG_WORK_PHONE: float
    FLAG_PHONE: float
    CNT_FAM_MEMBERS: float
    REGION_RATING_CLIENT: float
    HOUR_APPR_PROCESS_START: float
    REG_REGION_NOT_LIVE_REGION: float
    REG_REGION_NOT_WORK_REGION: float
    REG_CITY_NOT_LIVE_CITY: float
    REG_CITY_NOT_WORK_CITY: float
    LIVE_CITY_NOT_WORK_CITY: float
    EXT_SOURCE_1: float
    EXT_SOURCE_2: float
    EXT_SOURCE_3: float
    OBS_30_CNT_SOCIAL_CIRCLE: float
    DEF_30_CNT_SOCIAL_CIRCLE: float
    DEF_60_CNT_SOCIAL_CIRCLE: float
    DAYS_LAST_PHONE_CHANGE: float
    FLAG_DOCUMENT_2: float
    FLAG_DOCUMENT_3: float
    FLAG_DOCUMENT_6: float
    FLAG_DOCUMENT_8: float
    FLAG_DOCUMENT_9: float
    FLAG_DOCUMENT_13: float
    FLAG_DOCUMENT_14: float
    FLAG_DOCUMENT_15: float
    FLAG_DOCUMENT_16: float
    FLAG_DOCUMENT_18: float
    FLAG_DOCUMENT_21: float
    AMT_REQ_CREDIT_BUREAU_DAY: float
    AMT_REQ_CREDIT_BUREAU_MON: float
    AMT_REQ_CREDIT_BUREAU_QRT: float
    AMT_REQ_CREDIT_BUREAU_YEAR: float
    docs_number: float
    has_full_apprt_info: float
    has_doc_change_delay: str
    loan_income_ratio: float
    ext_source_score: float
    interest_rate: float
    max_overdue_amount: float
    min_overdue_amount: float
    overdue_to_open_credit_ratio: float
    credit_count_Car_loan: float
    credit_count_Consumer_credit: (
        float
    )
    credit_count_Credit_card: float
    credit_count_Loan_for_working_capital_replenishment: float
    credit_count_Microloan: float
    credit_count_Mortgage: float
    overdue_credit_count_Consumer_credit: (
        float
    )
    overdue_credit_count_Credit_card: (
        float
    )
    overdue_credit_count_Loan_for_working_capital_replenishment: float
    closed_credit_count_Car_loan: (
        float
    )
    closed_credit_count_Credit_card: (
        float
    )
    closed_credit_count_Mortgage: float
    open_credit_count: float
    overdue_credit_count: float
    overdue_status_2_credit_count: float
    overdue_status_3_credit_count: float
    overdue_status_4_credit_count: float
    overdue_status_5_credit_count: float
    closed_credit_ratio: float
    overdue_credit_ratio: float
    overdue_status_2_credit_ratio: float
    overdue_status_3_credit_ratio: float
    overdue_status_4_credit_ratio: float
    overdue_status_5_credit_ratio: float
    days_since_last_closed_credit: float
    days_since_last_active_credit_application: float
    NAME_CONTRACT_TYPE: str
    CODE_GENDER: str
    FLAG_OWN_CAR: str
    FLAG_OWN_REALTY: str
    NAME_TYPE_SUITE: str
    NAME_INCOME_TYPE: str
    NAME_EDUCATION_TYPE: str
    NAME_FAMILY_STATUS: str
    NAME_HOUSING_TYPE: str
    OCCUPATION_TYPE: str
    ORGANIZATION_TYPE: str
    DAYS_EMPLOYED_was_missing: float
    OWN_CAR_AGE_was_missing: float
    EXT_SOURCE_1_was_missing: float
    EXT_SOURCE_3_was_missing: float
    OBS_30_CNT_SOCIAL_CIRCLE_was_missing: float
    AMT_REQ_CREDIT_BUREAU_DAY_was_missing: float
    interest_rate_was_missing: float
    days_since_last_closed_credit_was_missing: float
    days_since_last_active_credit_application_was_missing: float
    NAME_TYPE_SUITE_was_missing: float
    OCCUPATION_TYPE_was_missing: float

    @classmethod
    def from_record(
        cls, record: dict, column_to_field: dict[str, str]
    ) -> "Features":
        """Строит Features из значений одной заявки, уже посчитанных заранее.

        Args:
            record: значения признаков одной заявки в виде {имя_колонки: значение}.
            column_to_field: сопоставление имя_колонки -> имя_поля Features
                (см. build_feature_mappings).

        Returns:
            Features: признаки этой заявки, готовые к передаче в
                SimpleModel.get_scoring_result()/AdvancedModel.get_scoring_result()
                для получения решения по заявке.
        """
        translated = {
            column_to_field.get(key, key): value
            for key, value in record.items()
        }
        field_names = set(cls.__dataclass_fields__)
        return cls(
            **{
                key: value
                for key, value in translated.items()
                if key in field_names
            }
        )

    def to_frame(
        self, feature_order: list[str], field_to_column: dict[str, str]
    ) -> pd.DataFrame:
        """Преобразует Features в DataFrame, пригодный для model.predict_proba().

        Args:
            feature_order: порядок исходных колонок.
            field_to_column: сопоставление имя_поля -> имя_колонки.

        Returns:
            pd.DataFrame: одна строка с признаками этой заявки, с
                исходными именами колонок и в порядке feature_order.
        """
        data = {
            field_to_column[field]: value
            for field, value in self.__dict__.items()
            if field in field_to_column
        }
        return pd.DataFrame([data])[feature_order]
