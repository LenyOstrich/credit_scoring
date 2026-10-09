"""Конфигурация подключения к базе данных."""

import os

from dotenv import load_dotenv

load_dotenv()

_required = ["DB_HOST", "DB_NAME", "DB_USER", "DB_PASSWORD", "DB_SCHEMA"]
_missing = [name for name in _required if not os.getenv(name)]
if _missing:
    msg = f"Не заданы переменные окружения: {', '.join(_missing)}"
    raise RuntimeError(msg)

DB_CONFIG = {
    "host": os.getenv("DB_HOST"),
    "port": int(os.getenv("DB_PORT", "5432")),
    "dbname": os.getenv("DB_NAME"),
    "user": os.getenv("DB_USER"),
    "password": os.getenv("DB_PASSWORD"),
    "options": f"-c search_path={os.getenv('DB_SCHEMA')}",
}

DB_SCHEMA = os.getenv("DB_SCHEMA")
