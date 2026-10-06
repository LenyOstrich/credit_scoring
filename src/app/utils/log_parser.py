import json
import logging
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pandas as pd

BUREAU_COLUMNS = [
    "SK_ID_CURR",
    "SK_ID_BUREAU",
    "CREDIT_ACTIVE",
    "CREDIT_CURRENCY",
    "DAYS_CREDIT",
    "CREDIT_DAY_OVERDUE",
    "DAYS_CREDIT_ENDDATE",
    "DAYS_ENDDATE_FACT",
    "AMT_CREDIT_MAX_OVERDUE",
    "CNT_CREDIT_PROLONG",
    "AMT_CREDIT_SUM",
    "AMT_CREDIT_SUM_DEBT",
    "AMT_CREDIT_SUM_LIMIT",
    "AMT_CREDIT_SUM_OVERDUE",
    "CREDIT_TYPE",
    "DAYS_CREDIT_UPDATE",
    "AMT_ANNUITY",
]

POS_CASH_BALANCE_COLUMNS = [
    "SK_ID_PREV",
    "SK_ID_CURR",
    "MONTHS_BALANCE",
    "CNT_INSTALMENT",
    "CNT_INSTALMENT_FUTURE",
    "NAME_CONTRACT_STATUS",
    "SK_DPD",
    "SK_DPD_DEF",
]


logger = logging.getLogger(__name__)


def handle_bureau(data: dict[str, Any]) -> list[Any]:
    """Обрабатывает строку лога типа bureau.

    Извлекает данные кредитной записи и преобразует вложенное
    строковое представление AmtCredit в словарь.

    Args:
        data: Данные события bureau.

    Returns:
        Список значений одной строки таблицы bureau.
    """
    record = data["record"]
    amt_credit = parse_logs_as_object(record["AmtCredit"])
    return [
        record["SK_ID_CURR"],
        record["SK_ID_BUREAU"],
        record["CREDIT_ACTIVE"],
        amt_credit["CREDIT_CURRENCY"],
        record["DAYS_CREDIT"],
        record["CREDIT_DAY_OVERDUE"],
        record["DAYS_CREDIT_ENDDATE"],
        record["DAYS_ENDDATE_FACT"],
        amt_credit["AMT_CREDIT_MAX_OVERDUE"],
        record["CNT_CREDIT_PROLONG"],
        amt_credit["AMT_CREDIT_SUM"],
        amt_credit["AMT_CREDIT_SUM_DEBT"],
        amt_credit["AMT_CREDIT_SUM_LIMIT"],
        amt_credit["AMT_CREDIT_SUM_OVERDUE"],
        data["CREDIT_TYPE"],
        record["DAYS_CREDIT_UPDATE"],
        amt_credit["AMT_ANNUITY"],
    ]


def handle_pos_cash_balance(data: dict[str, Any]) -> list[list[Any]]:
    """Обрабатывает строку лога типа POS_CASH_balance.

    Одно событие может содержать несколько записей в поле records,
    поэтому функция возвращает список строк.

    Args:
        data: Данные события POS_CASH_balance.

    Returns:
        Список строк для таблицы POS_CASH_balance.
    """
    cnt_instalment = data["CNT_INSTALMENT"]
    records = data["records"]

    res = []
    for record in records:
        pos_ids = parse_logs_as_object(record["PosCashBalanceIDs"])

        res.append(
            [
                pos_ids["SK_ID_PREV"],
                pos_ids["SK_ID_CURR"],
                record["MONTHS_BALANCE"],
                cnt_instalment,
                record["CNT_INSTALMENT_FUTURE"],
                pos_ids["NAME_CONTRACT_STATUS"],
                record["SK_DPD"],
                record["SK_DPD_DEF"],
            ]
        )
    return res


def handle_credit_application(data: dict[str, Any]) -> None:
    """Обрабатывает строку типа credit_application.

    Args:
        data: Данные события credit_application.
    """
    return


def handle_system_heartbeat(data: dict[str, Any]) -> None:
    """Обрабатывает строку типа system_heartbeat.

    Args:
        data: Данные события system_heartbeat.
    """
    return


def handle_fraud_alert(data: dict[str, Any]) -> None:
    """Обрабатывает строку типа fraud_alert.

    Args:
        data: Данные события fraud_alert.
    """
    return


def handle_client_profile_update(data: dict[str, Any]) -> None:
    """Обрабатывает строку типа client_profile_update.

    Args:
        data: Данные события client_profile_update.
    """
    return


def handle_unknown(data: dict[str, Any]) -> None:
    """Обрабатывает неизвестный тип строки.

    Args:
        data: Данные неизвестного события.
    """
    return


type Handler = Callable[
    [dict[str, Any]],
    list[Any] | list[list[Any]] | None,
]

handlers: dict[str, Handler] = {
    "bureau": handle_bureau,
    "POS_CASH_balance": handle_pos_cash_balance,
    "credit_application": handle_credit_application,
    "system_heartbeat": handle_system_heartbeat,
    "fraud_alert": handle_fraud_alert,
    "client_profile_update": handle_client_profile_update,
}


def process_command(
    command: str,
    data: dict[str, Any],
) -> list[Any] | list[list[Any]] | None:
    """Вызывает обработчик в зависимости от типа строки лога.

    Args:
        command: Тип события.
        data: Данные события.

    Returns:
        Результат выполнения соответствующего обработчика.
    """
    handler = handlers.get(command, handle_unknown)
    return handler(data)


def read_log(path: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Читает лог-файл и преобразует его в два DataFrame.

    Обрабатываются события типов bureau и POS_CASH_balance.
    Пустые строки, некорректный JSON, записи не в формате словаря
    и события других типов пропускаются.

    Args:
        path: Путь к исходному log-файлу.

    Returns:
        Кортеж из двух DataFrame:
            - таблица bureau;
            - таблица POS_CASH_balance.
    """
    bureau_rows = []
    pos_cash_balance_rows = []
    processed_lines = 0

    logger.info("Начало обработки файла: %s", path)

    with Path.open(path, encoding="utf-8") as f:
        for line in f:
            processed_lines += 1

            if processed_lines % 100_000 == 0:
                logger.info(
                    "Обработано строк: %d",
                    processed_lines,
                )

            line = line.strip()

            if not line:
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue

            if not isinstance(record, dict):
                continue

            record_type = record.get("type")
            data = record.get("data")

            if record_type is None or data is None:
                continue

            if record_type not in {"bureau", "POS_CASH_balance"}:
                continue

            result = process_command(record_type, data)

            if record_type == "bureau":
                bureau_rows.append(result)
            else:
                pos_cash_balance_rows.extend(result)

    logger.info(
        "Чтение файла завершено. Обработано строк: %d",
        processed_lines,
    )

    logger.info("Создание DataFrame...")

    bureau_df = pd.DataFrame(bureau_rows, columns=BUREAU_COLUMNS)

    pos_cash_balance_df = pd.DataFrame(
        pos_cash_balance_rows, columns=POS_CASH_BALANCE_COLUMNS
    )

    logger.info(
        "DataFrame созданы: bureau=%d строк, POS_CASH_balance=%d строк",
        len(bureau_df),
        len(pos_cash_balance_df),
    )

    return bureau_df, pos_cash_balance_df


def parse_logs_as_object(value: str) -> dict[str, Any]:
    """Преобразует строковое представление объекта из лога в словарь.

    Поддерживаются строковые значения, целые и вещественные числа,
    а также значение None.

    Args:
        value: Строковое представление объекта из лога.

    Returns:
        Словарь с разобранными полями объекта.
    """
    pattern = re.compile(r"(\w+)=('(?:[^']*)'|None|[-+]?\d+(?:\.\d+)?)")

    result = {}

    for key, raw_value in pattern.findall(value):
        if raw_value == "None":
            result[key] = None
        elif raw_value.startswith("'") and raw_value.endswith("'"):
            result[key] = raw_value[1:-1]
        elif "." in raw_value:
            result[key] = float(raw_value)
        else:
            result[key] = int(raw_value)

    return result


def save_dataframe(
    path: str | Path,
    df: pd.DataFrame,
    name: str,
) -> None:
    """Сохраняет DataFrame в CSV-файл.

    Если указанная директория не существует, она создаётся.

    Args:
        path: Путь к директории для сохранения файла.
        df: DataFrame, который необходимо сохранить.
        name: Имя выходного CSV-файла без расширения.
    """
    output_path = Path(path)
    output_path.mkdir(parents=True, exist_ok=True)

    logger.info("Идёт сохранение...")

    df.to_csv(
        output_path / f"{name}.csv",
        index=False,
    )


def main():
    """Запускает обработку log-файла.

    Запрашивает у пользователя путь к исходному log-файлу
    и директорию для сохранения результатов, после чего
    преобразует лог и сохраняет полученные таблицы в CSV.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    log_path = Path(input("Введите путь к log-файлу: ").strip())

    pos_cash_balance_path = Path(
        input("Введите путь для POS_CASH_balance.csv: ").strip()
    )

    bureau_path = Path(input("Введите путь для bureau.csv: ").strip())

    if not log_path.is_file():
        print(f"Файл не найден: {log_path}")
        return

    bureau_df, pos_cash_balance_df = read_log(log_path)

    save_dataframe(
        bureau_path / "bureau",
        bureau_df,
        "bureau",
    )

    save_dataframe(
        pos_cash_balance_path / "POS_CASH_balance",
        pos_cash_balance_df,
        "POS_CASH_balance",
    )

    logger.info("Обработка завершена.")


if __name__ == "__main__":
    main()
