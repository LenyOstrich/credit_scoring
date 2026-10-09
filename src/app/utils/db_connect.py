"""Утилиты для подключения к БД и выполнения запросов."""

from collections.abc import Generator
from contextlib import contextmanager

import pandas as pd
import psycopg2
from psycopg2.extensions import connection as PGConnection

from config.db_config import DB_CONFIG


@contextmanager
def get_connection() -> Generator[PGConnection]:
    """Открыть соединение с PostgreSQL и гарантированно закрыть его.

    Yields:
        PGConnection: Активное соединение с базой данных PostgreSQL.
    """
    conn = psycopg2.connect(**DB_CONFIG)
    try:
        yield conn
    finally:
        conn.close()


def _fetch_df(
    sql: str,
    params: tuple | None,
    conn: PGConnection,
) -> pd.DataFrame:
    """Выполнить SELECT на готовом соединении и вернуть DataFrame.

    Args:
        sql (str): SQL-запрос.
        params (tuple | None): Параметры запроса.
        conn (PGConnection): Открытое соединение. Не закрывается.

    Returns:
        pd.DataFrame: DataFrame с результатами.

    Raises:
        ValueError: Если запрос не возвращает строк (не SELECT).
    """
    with conn.cursor() as cur:
        cur.execute(sql, params)
        if cur.description is None:
            msg = "query_to_pd поддерживает только SELECT-запросы"
            raise ValueError(msg)
        columns = [d[0] for d in cur.description]
        rows = cur.fetchall()
    return pd.DataFrame(rows, columns=columns)


def query_to_pd(
    sql: str,
    params: tuple | None = None,
    conn: PGConnection | None = None,
) -> pd.DataFrame:
    """Выполнить SQL-запрос и вернуть pandas DataFrame.

    Args:
        sql (str): SQL-запрос для выполнения.
        params (tuple | None): Параметры для SQL-запроса. По умолчанию None.
        conn (PGConnection | None): Готовое соединение. Если None — открывается
            новое и закрывается после выполнения запроса.

    Returns:
        pd.DataFrame: DataFrame с результатами запроса.
    """
    if conn is not None:
        return _fetch_df(sql, params, conn)

    with get_connection() as own_conn:
        return _fetch_df(sql, params, own_conn)
