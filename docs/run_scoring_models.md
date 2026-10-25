# Запуск обучения моделей кредитного скоринга

Скрипт `app.scoring.train_scoring_models` обучает 4 модели (логистическая
регрессия, решающее дерево, случайный лес, CatBoost) с подбором
гиперпараметров на кросс-валидации и сохраняет результаты.

## 1. Предварительные условия

- Исходные CSV Home Credit Default Risk лежат в `data/` (см. корневой
  `README.md`).
- Признаки ДЗ-3 сгенерированы в `data/processed/` — см.
  [Запуск генерации и проверка признаков](feature_generation.md). Без
  `application_train.csv`/`application_test.csv`, `bureau_features.csv`,
  `bureau_balance_features.csv` в этой директории скрипт упадёт на этапе
  подготовки данных.
- Зависимости установлены (`poetry install`, либо `pip install` в
  виртуальное окружение — см. `pyproject.toml`).

## 2. Запуск

Из корня проекта:

```powershell
python -m app.scoring.train_scoring_models `
    --raw-data-dir data `
    --features-dir data/processed `
    --models-dir data/models `
    --reports-dir docs/figures
```

Все аргументы необязательны — значения выше совпадают со значениями по
умолчанию, поэтому достаточно:

```powershell
python -m app.scoring.train_scoring_models
```

### Аргументы командной строки

| Аргумент | По умолчанию | Назначение |
|---|---|---|
| `--raw-data-dir` | `data` | исходные CSV Home Credit |
| `--features-dir` | `data/processed` | признаки, сгенерированные в ДЗ-3 |
| `--models-dir` | `data/models` | куда сохранять `*.pkl` обученных моделей |
| `--reports-dir` | `docs/figures` | куда сохранять таблицу метрик и графики |
| `--random-state` | `42` | random_state для воспроизводимости (сплит, модели) |
| `--cv-folds` | `3` | число фолдов `StratifiedKFold` в `GridSearchCV` |

## 3. Что делает скрипт

1. Готовит данные (`data_preparation.prepare_datasets_for_models`):
   чистит `DAYS_EMPLOYED`-sentinel, обрабатывает выбросы и пропуски,
   отбирает и кодирует признаки — отдельно для линейной модели, деревьев
   и CatBoost (нативная работа с категориями и NaN).
2. Обучает все 4 модели через `GridSearchCV` (`scoring="roc_auc"`,
   `n_jobs=-1`) — сетки гиперпараметров см. `src/app/scoring/models.py`.
3. Считает метрики на тестовой выборке и печатает сводную таблицу в stdout.
4. Сохраняет `docs/figures/metrics_comparison.csv`, `roc_curves.png` и
   `feature_importance_<model>.png` для каждой модели.
5. Сохраняет 4 pickle-файла в `data/models/`.

## 4. Время выполнения

Ориентир с этой машины (246 008 строк в обучающей выборке, CPU, `n_jobs=-1`):

| Этап | Время |
|---|---|
| Подготовка данных | ~30 сек |
| LogisticRegression (GridSearchCV) | ~9.5 мин |
| DecisionTree | ~1 мин |
| RandomForest | ~18 мин |
| CatBoost | ~34 мин |
| **Итого** | **~1 ч 03 мин** |


## 5. Результат

```text
data/models/
├── logistic_regression.pkl
├── decision_tree.pkl
├── random_forest.pkl
└── catboost.pkl

docs/figures/
├── metrics_comparison.csv
├── roc_curves.png
├── feature_importance_logistic_regression.png
├── feature_importance_decision_tree.png
├── feature_importance_random_forest.png
└── feature_importance_catboost.png
```
