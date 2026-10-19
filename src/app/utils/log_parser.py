import csv
import json
import logging
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pandas as pd

DEFAULT_BUREAU_PATH = Path("logs/output/bureau.csv")
DEFAULT_POS_CASH_BALANCE_PATH = Path("logs/output/POS_CASH_balance.csv")

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
    logger.warning(
        "Неизвестный тип события, данные проигнорированы: %s",
        type(data).__name__,
    )


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


def read_log(
    path: str | Path,
    bureau_path: str | Path,
    pos_cash_balance_path: str | Path,
) -> None:
    """Читает log-файл и потоково записывает данные в два CSV.

    Обрабатываются события типов bureau и POS_CASH_balance.
    Пустые строки, некорректный JSON, записи не в формате словаря
    и события других типов пропускаются.

    Args:
        path: Путь к исходному log-файлу.
        bureau_path: Путь к выходному bureau.csv.
        pos_cash_balance_path: Путь к выходному POS_CASH_balance.csv.
    """
    processed_lines = 0
    bureau_rows_count = 0
    pos_cash_balance_rows_count = 0

    bureau_output = Path(bureau_path)
    pos_cash_balance_output = Path(pos_cash_balance_path)

    bureau_output.parent.mkdir(parents=True, exist_ok=True)
    pos_cash_balance_output.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Начало обработки файла: %s", path)

    with (
        Path(path).open(encoding="utf-8") as log_file,
        bureau_output.open("w", newline="", encoding="utf-8") as bureau_file,
        pos_cash_balance_output.open(
            "w", newline="", encoding="utf-8"
        ) as pos_cash_balance_file,
    ):
        bureau_writer = csv.writer(bureau_file)
        pos_cash_balance_writer = csv.writer(pos_cash_balance_file)

        bureau_writer.writerow(BUREAU_COLUMNS)
        pos_cash_balance_writer.writerow(POS_CASH_BALANCE_COLUMNS)

        for line in log_file:
            processed_lines += 1

            if processed_lines % 100_000 == 0:
                logger.info("Обработано строк: %d", processed_lines)

            line = line.strip()

            if not line:
                logger.warning(
                    "Строка %d: пустая строка, пропущено",
                    processed_lines,
                )
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                logger.warning(
                    "Строка %d: некорректный JSON (%s), пропущено",
                    processed_lines,
                    exc.msg,
                )
                continue

            if not isinstance(record, dict):
                logger.warning(
                    "Строка %d: JSON не является объектом "
                    "(получен %s), пропущено",
                    processed_lines,
                    type(record).__name__,
                )
                continue

            record_type = record.get("type")
            data = record.get("data")

            if record_type is None or data is None:
                logger.warning(
                    "Строка %d: отсутствует поле 'type' или 'data', пропущено",
                    processed_lines,
                )
                continue

            if record_type not in {"bureau", "POS_CASH_balance"}:
                logger.debug(
                    "Строка %d: тип '%s' не обрабатывается, пропущено",
                    processed_lines,
                    record_type,
                )
                continue

            try:
                result = process_command(record_type, data)
            except (KeyError, TypeError, ValueError) as exc:
                logger.warning(
                    "Строка %d (тип '%s'): повреждённая запись "
                    "пропущена (%s: %s)",
                    processed_lines,
                    record_type,
                    type(exc).__name__,
                    exc,
                )
                continue

            if record_type == "bureau":
                bureau_writer.writerow(result)
                bureau_rows_count += 1
            else:
                pos_cash_balance_writer.writerows(result)
                pos_cash_balance_rows_count += len(result)

    logger.info(
        "Чтение файла завершено. Обработано строк: %d",
        processed_lines,
    )
    logger.info(
        "Созданы CSV: bureau=%d строк, POS_CASH_balance=%d строк",
        bureau_rows_count,
        pos_cash_balance_rows_count,
    )


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


def main() -> None:
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

    if not log_path.is_file():
        print(f"Файл не найден: {log_path}")
        return

    read_log(
        path=log_path,
        bureau_path=DEFAULT_BUREAU_PATH,
        pos_cash_balance_path=DEFAULT_POS_CASH_BALANCE_PATH,
    )

    logger.info("Обработка завершена.")


if __name__ == "__main__":
    main()
