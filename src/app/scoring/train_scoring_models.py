"""Точка входа для обучения моделей кредитного скоринга."""

import logging

from app.scoring import common, data_preparation, evaluation, models, training
from app.utils.logging_config import setup_logging

logger = logging.getLogger(__name__)


def main() -> None:
    """Запускает полный пайплайн обучения моделей кредитного скоринга.

    1. Предобработка датасета (data_preparation).
    2. Обучение 4 моделей с подбором гиперпараметров на кросс-валидации
       (models + training).
    3. Печать итогового качества моделей на тестовом датасете (evaluation).
    4. Сохранение моделей в pickle-файлы (common.save_model).
    """
    args = common.build_arg_parser().parse_args()
    setup_logging()

    models_dir = common.ensure_directory(args.models_dir)
    reports_dir = common.ensure_directory(args.reports_dir)

    logger.info("Подготовка датасетов...")
    datasets = data_preparation.prepare_datasets_for_models(
        raw_data_dir=args.raw_data_dir,
        features_dir=args.features_dir,
        random_state=args.random_state,
    )

    categorical_columns = (
        datasets["catboost"][0]
        .select_dtypes(exclude="number")
        .columns.tolist()
    )
    model_configs = models.build_model_configs(categorical_columns, args.random_state)

    logger.info("Обучение моделей с подбором гиперпараметров...")
    trained_models = training.train_all_models(
        model_configs,
        datasets,
        args.cv_folds,
        args.random_state,
    )

    logger.info("Оценка качества на тестовой выборке...")
    metrics_per_model: dict[str, dict[str, float]] = {}
    x_test_per_model = {}
    for model_config in model_configs:
        _, x_test, _, y_test = datasets[model_config.dataset_kind]
        model = trained_models[model_config.name]

        metrics_per_model[model_config.name] = (
            evaluation.compute_classification_metrics(
                model,
                x_test,
                y_test,
            )
        )
        x_test_per_model[model_config.name] = x_test

    comparison_table = evaluation.build_comparison_table(metrics_per_model)
    print(comparison_table)

    evaluation.save_metrics_report(
        comparison_table,
        reports_dir / "metrics_comparison.csv",
    )

    _, _, _, y_test = datasets["linear"]
    evaluation.plot_roc_curves(
        trained_models,
        x_test_per_model,
        y_test,
        reports_dir / "roc_curves.png",
    )

    for model_config in model_configs:
        model = trained_models[model_config.name]
        x_test = x_test_per_model[model_config.name]
        evaluation.plot_feature_importances(
            model,
            x_test.columns.tolist(),
            reports_dir / f"feature_importance_{model_config.name}.png",
        )

    for model_config in model_configs:
        common.save_model(
            trained_models[model_config.name],
            models_dir / f"{model_config.name}.pkl",
        )

    logger.info("Пайплайн обучения завершён.")


if __name__ == "__main__":
    main()
