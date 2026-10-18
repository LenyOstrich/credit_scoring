"""Загрузка датасета Home Credit Default Risk."""

import logging
from pathlib import Path

import kagglehub

logger = logging.getLogger(__name__)

COMPETITION_NAME = "home-credit-default-risk"

EXPECTED_FILES = {
    "application_train.csv",
    "application_test.csv",
    "bureau.csv",
    "bureau_balance.csv",
    "POS_CASH_balance.csv",
    "credit_card_balance.csv",
    "previous_application.csv",
    "installments_payments.csv",
    "HomeCredit_columns_description.csv",
    "sample_submission.csv",
}


def download_home_credit_data_from_kaggle(
    *,
    force: bool = False,
    data_root: str | Path | None = None,
) -> Path:
    """Загружает датасет Home Credit Default Risk с Kaggle.

    Датасет сохраняется в каталоге
    ``data/home-credit-default-risk`` относительно корня проекта.

    Если датасет уже загружен и содержит все ожидаемые файлы,
    повторная загрузка не выполняется, если ``force=False``.

    Если каталог существует, но датасет неполный, функция сообщает
    об ошибке. Для повторной загрузки необходимо установить
    ``force=True``.

    Args:
        force: Принудительно загружать датасет заново, даже если
            он уже существует. При этом существующие файлы могут
            быть перезаписаны.
        data_root: Корневой каталог для данных. Если не указан,
            используется каталог ``data`` в корне проекта.

    Returns:
        Path: Путь к каталогу с загруженным датасетом.

    Raises:
        FileNotFoundError: Если существующий каталог содержит
            не все ожидаемые файлы, а ``force=False``.
    """
    project_root = Path(__file__).resolve().parents[3]
    if data_root is None:
        data_dir = project_root / "data"
    else:
        data_dir = Path(data_root).expanduser()
        if not data_dir.is_absolute():
            data_dir = project_root / data_dir

    target_dir = data_dir / COMPETITION_NAME

    if target_dir.is_dir() and not force:
        missing_files = EXPECTED_FILES - {
            path.name for path in target_dir.iterdir() if path.is_file()
        }

        if missing_files:
            missing_list = ", ".join(sorted(missing_files))
            raise FileNotFoundError(
                f"Датасет в каталоге {target_dir} неполный. "
                f"Отсутствуют файлы: {missing_list}. "
                "Для повторной загрузки используйте force=True."
            )

        logger.info(
            "Датасет уже загружен в %s. Повторная загрузка пропущена.",
            target_dir,
        )

        return target_dir

    target_dir.parent.mkdir(parents=True, exist_ok=True)

    logger.info(
        "Загрузка датасета %s с Kaggle в %s.",
        COMPETITION_NAME,
        target_dir,
    )

    downloaded_path = kagglehub.competition_download(
        COMPETITION_NAME,
        output_dir=str(target_dir),
        force_download=force,
    )

    result_path = Path(downloaded_path)

    missing_files = EXPECTED_FILES - {
        path.name for path in result_path.iterdir() if path.is_file()
    }

    if missing_files:
        missing_list = ", ".join(sorted(missing_files))
        raise FileNotFoundError(
            f"Загрузка завершилась, но датасет неполный. "
            f"Отсутствуют файлы: {missing_list}."
        )

    logger.info("Датасет успешно загружен в %s.", result_path)

    return result_path
