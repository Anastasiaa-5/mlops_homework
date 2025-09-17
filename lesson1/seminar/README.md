# Семинар: Создание воспроизводимого ML-проекта

## Оглавление
1. [Цель и задачи](#цель-и-задачи)
2. [Подготовка среды](#подготовка-среды)
3. [Структура проекта](#структура-проекта)
4. [Управление зависимостями](#управление-зависимостями)
5. [Конфигурация](#конфигурация)
6. [Реализация обучения](#реализация-обучения)
7. [Тестирование](#тестирование)
8. [MLflow трекинг](#mlflow-трекинг)
9. [Качество кода](#качество-кода)
10. [Контейнеризация](#контейнеризация)
11. [Автоматизация](#автоматизация)
12. [Подводные камни](#подводные-камни)

---

## Цель и задачи

### Что мы создаем?
Минимально воспроизводимый ML-проект с:
- Детерминированными результатами
- Трекингом экспериментов (MLflow)
- Качественным кодом (линтеры, тесты)
- Контейнеризацией
- Автоматизацией процессов

### Почему это важно?
- **Воспроизводимость** - основа научного подхода в ML
- **Отслеживание экспериментов** - понимание что работает
- **Качество кода** - поддерживаемость и надежность
- **Автоматизация** - экономия времени и снижение ошибок

---

## Подготовка среды

### Шаг 1: Git-репозиторий

```bash
git clone https://github.com/tam2511/mlops2025.git
cd mlops2025
git checkout -b lesson1
```

**Объяснение:**
- Создаем отдельную ветку для изоляции работы
- Следуем Git Flow для чистой истории

**Подводные камни:**
- Всегда работайте в отдельных ветках
- Проверяйте, что находитесь в правильной ветке перед началом работы

### Шаг 2: Установка Poetry

```bash
# macOS с Homebrew
brew install poetry

# Альтернативно
curl -sSL https://install.python-poetry.org | python3 -
```

**Объяснение:**
Poetry vs pip:
- **Poetry**: управление зависимостями + виртуальное окружение + packaging
- **pip**: только установка пакетов
- Poetry создает `poetry.lock` для точного воспроизведения среды

**Подводные камни:**
- Poetry может конфликтовать с conda/virtualenv
- В некоторых CI системах лучше использовать pip
- Настройка `poetry config virtualenvs.in-project true` для локальных .venv

---

## Структура проекта

### Шаг 3: Создание структуры

```
lesson1/seminar/
├── configs/               # Конфигурационные файлы
│   └── train.yaml
├── src/                   # Исходный код
│   └── app/
│       ├── __init__.py
│       └── train.py
├── tests/                 # Тесты
│   ├── __init__.py
│   └── test_sanity.py
├── data/                  # Данные (в .gitignore)
│   ├── raw/
│   └── processed/
├── models/                # Сохраненные модели (в .gitignore)
├── pyproject.toml         # Конфигурация проекта
├── poetry.lock           # Зафиксированные версии
├── Makefile              # Автоматизация команд
├── Dockerfile            # Контейнеризация
├── .gitignore            # Исключения Git
├── .dockerignore         # Исключения Docker
├── .pre-commit-config.yaml # Pre-commit хуки
└── README.md             # Документация
```

**Объяснение структуры:**
- **src/app/** - основной код приложения
- **configs/** - все конфигурации в одном месте
- **tests/** - тесты рядом с кодом, но отдельно
- **data/**, **models/** - в .gitignore, но структура известна

**Подводные камни:**
- Не коммитьте большие данные в Git
- Используйте .gitkeep для пустых важных папок
- __init__.py нужен для импортов в Python

---

## Управление зависимостями

### Шаг 4: pyproject.toml

```toml
[tool.poetry]
name = "app"
version = "0.1.0"
description = "Reproducible ML project with MLflow tracking"

[tool.poetry.dependencies]
python = ">=3.11,<3.12"        # Фиксируем версию Python
scikit-learn = "1.5"           # Основная ML библиотека
mlflow = "2.14"                # Трекинг экспериментов
pyyaml = "6.0"                 # Конфигурационные файлы

[tool.poetry.group.dev.dependencies]
pytest = "7.4"                 # Тестирование
ruff = "0.6"                   # Быстрый линтер
black = "24.4"                 # Форматирование кода
mypy = "1.8"                   # Проверка типов
pre-commit = "3.7"             # Git хуки
setuptools = "^80.9.0"         # Для совместимости с MLflow
```

**Объяснение выбора версий:**
- **Точные версии для prod зависимостей** - воспроизводимость
- **Гибкие версии для dev инструментов** - получаем обновления безопасности
- **Python 3.11** - баланс новизны и стабильности

**Подводные камни:**
- MLflow требует setuptools (pkg_resources deprecation)
- Слишком строгие версии → сложности с обновлениями
- Слишком мягкие версии → разные результаты в разных средах

### Шаг 5: poetry.lock

```bash
poetry install  # Создает poetry.lock автоматически
```

**Объяснение:**
- `poetry.lock` - точные версии всех зависимостей включая транзитивные
- Обеспечивает идентичное окружение на всех машинах
- Аналог `package-lock.json` в Node.js

**Подводные камни:**
- Всегда коммитьте poetry.lock
- При конфликтах - удалите lock и пересоздайте
- poetry.lock генерируется для вашей ОС - может не работать на других

---

## Конфигурация

### Шаг 6: train.yaml

```yaml
seed: 42                    # Фиксированный seed для воспроизводимости
test_size: 0.2              # Размер тестовой выборки
model:                      # Параметры модели
  C: 1.0                    # Регуляризация
  max_iter: 200             # Максимум итераций
mlflow_experiment: "lesson1-baseline"  # Название эксперимента
```

**Объяснение:**
- **YAML vs JSON** - удобнее для людей, поддерживает комментарии
- **Внешние конфиги** - можно менять без пересборки
- **Структурированность** - группировка логически связанных параметров

**Подводные камни:**
- YAML чувствителен к отступам
- Не используйте табы, только пробелы
- Осторожно с типами данных (строки vs числа)

### Альтернативы конфигурации

```python
# Hydra (для больших проектов)
from hydra import compose, initialize

# Pydantic Settings (для валидации)
from pydantic import BaseSettings

# Argparse (для CLI)
import argparse
```

---

## Реализация обучения

### Шаг 7: Модульная структура train.py

```python
def set_seed(seed: int) -> None:
    """Фиксируем все источники случайности."""
    random.seed(seed)
    np.random.seed(seed)
    # sklearn использует numpy.random

def load_config(config_path: str = "configs/train.yaml") -> Dict[str, Any]:
    """Загрузка конфигурации."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def prepare_data(test_size: float, random_state: int) -> Tuple[...]:
    """Подготовка данных с фиксированным split."""
    # Важно: stratify=y для сбалансированного разбиения
    return train_test_split(..., stratify=y, random_state=random_state)
```

**Объяснение архитектуры:**
- **Функциональный подход** - легче тестировать
- **Типизация** - mypy поймает ошибки на раннем этапе
- **Модульность** - каждая функция делает одну вещь

**Подводные камни детерминизма:**
- Sklearn не всегда детерминирован (особенно на разных ОС)
- Некоторые операции используют системное время
- Многопоточность может нарушить детерминизм
- GPU вычисления часто недетерминированы

### Шаг 8: Обработка ошибок

```python
def main() -> None:
    try:
        config = load_config()
    except FileNotFoundError:
        print("Config file not found!")
        sys.exit(1)
    except yaml.YAMLError as e:
        print(f"Invalid YAML: {e}")
        sys.exit(1)
```

**Зачем нужно:**
- Понятные сообщения об ошибках
- Корректное завершение программы
- Логирование проблем

---

## Тестирование

### Шаг 9: Тесты на разных уровнях

```python
def test_accuracy_threshold():
    """Smoke test - проверяем адекватность модели."""
    # Iris - простая задача, accuracy должна быть высокой
    assert metrics["accuracy"] >= 0.85

def test_determinism():
    """Ключевой тест - два запуска = одинаковый результат."""
    accuracy1 = run_training_pipeline()
    accuracy2 = run_training_pipeline()
    assert abs(accuracy1 - accuracy2) <= 0.001

def test_data_split_determinism():
    """Проверяем детерминизм на уровне данных."""
    # Один и тот же random_state = одинаковые splits
```

**Объяснение типов тестов:**
- **Smoke tests** - "не падает ли все"
- **Property tests** - проверка математических свойств
- **Integration tests** - работа компонентов вместе
- **Determinism tests** - специфично для ML

**Подводные камни тестирования ML:**
- Тесты могут быть нестабильными из-за случайности
- Не тестируйте точные значения метрик (они могут меняться)
- Тестируйте свойства: accuracy > baseline, модель обучается и т.д.

### Шаг 10: Запуск тестов

```bash
poetry run pytest -v       # Подробный вывод
poetry run pytest -x       # Остановка на первой ошибке
poetry run pytest --tb=short  # Короткий traceback
```

---

## MLflow трекинг

### Шаг 11: Настройка MLflow

```python
# Локальное хранение (не засоряем общие папки)
mlflow.set_tracking_uri("file:./mlruns")

# Создание/использование эксперимента
mlflow.set_experiment("lesson1-baseline")

with mlflow.start_run():
    # Логируем ВСЕ важные параметры
    mlflow.log_param("seed", seed)
    mlflow.log_param("test_size", test_size)

    # Метрики
    mlflow.log_metric("accuracy", accuracy)

    # Модель для последующего использования
    mlflow.sklearn.log_model(model, "model")

    # Конфигурация как артефакт
    mlflow.log_artifact("configs/train.yaml")
```

**Что логировать в MLflow:**
- **Параметры**: всё что влияет на результат
- **Метрики**: всё что измеряем
- **Артефакты**: модели, конфиги, важные файлы
- **Теги**: для группировки и поиска

**Подводные камни MLflow:**
- Большие артефакты съедают место на диске
- MLflow UI может быть медленным с большим количеством экспериментов
- Не логируйте секреты и чувствительные данные
- file:// URI не подходит для продакшена

### Альтернативы MLflow

- **Weights & Biases (wandb)** - лучшая визуализация
- **TensorBoard** - для TensorFlow/PyTorch
- **Neptune** - корпоративное решение
- **ClearML** - полный MLOps stack

---

## Качество кода

### Шаг 12: Линтеры и форматеры

```bash
# Ruff - быстрая замена flake8, isort, и др.
poetry run ruff check .
poetry run ruff check . --fix  # Автоисправление

# Black - бескомпромиссный форматер
poetry run black .
poetry run black --check .     # Только проверка

# MyPy - статическая проверка типов
poetry run mypy src/
```

**Конфигурация в pyproject.toml:**
```toml
[tool.ruff.lint]
select = ["E", "F", "W", "C90"]  # Какие ошибки искать

[tool.black]
line-length = 88                # Стандарт черного

[tool.mypy]
disallow_untyped_defs = true    # Требовать типизацию
```

**Подводные камни:**
- Не настраивайте слишком строго сразу
- Black может конфликтовать с другими форматерами
- MyPy требует типизации - добавляйте постепенно

### Шаг 13: Pre-commit хуки

```yaml
repos:
  - repo: https://github.com/psf/black
    hooks:
      - id: black

  - repo: https://github.com/astral-sh/ruff-pre-commit
    hooks:
      - id: ruff
        args: [--fix]

  - repo: https://github.com/pre-commit/pre-commit-hooks
    hooks:
      - id: end-of-file-fixer
      - id: check-yaml
      - id: trailing-whitespace
```

```bash
poetry run pre-commit install      # Установка хуков
poetry run pre-commit run --all-files  # Запуск на всех файлах
```

**Зачем pre-commit:**
- Автоматическая проверка перед коммитом
- Единообразный код в команде
- Предотвращение "грязных" коммитов

---

## Контейнеризация

### Шаг 14: Dockerfile

```dockerfile
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1          # Не буферизовать вывод
ENV MLFLOW_TRACKING_URI=file:./mlruns

WORKDIR /app

# Сначала зависимости (кеширование слоев)
RUN pip install poetry==1.8.3
RUN poetry config virtualenvs.create false

COPY pyproject.toml poetry.lock ./
RUN poetry install --no-interaction --no-ansi

# Потом код
COPY src src
COPY configs configs

CMD ["python", "-m", "src.app.train"]
```

**Оптимизации Docker:**
- **Multi-stage builds** для меньшего размера
- **Кеширование слоев** - зависимости меняются реже кода
- **Специфичные теги** вместо latest
- **.dockerignore** для исключения ненужных файлов

**Подводные камни Docker:**
- poetry.lock может не работать на другой архитектуре
- Разные результаты на разных ОС даже в контейнерах
- Монтирование volumes для MLflow артефактов

### Шаг 15: .dockerignore

```
.venv/
.git/
mlruns/        # Будем монтировать как volume
data/          # Большие данные не в контейнер
.pytest_cache/
__pycache__/
```

---

## Автоматизация

### Шаг 16: Makefile

```makefile
.PHONY: install lint test train mlflow-ui docker-build docker-run

install:
	poetry install

lint:
	poetry run ruff check .
	poetry run black --check .

test:
	poetry run pytest -q

train:
	poetry run python -m src.app.train

mlflow-ui:
	poetry run mlflow ui --backend-store-uri ./mlruns

docker-build:
	docker build -t lesson1-mlops:0.1 .

docker-run:
	docker run --rm -v "$(PWD)/mlruns:/app/mlruns" lesson1-mlops:0.1
```

**Альтернативы Makefile:**
- **Just** - более современный make
- **Task** (Go) - кроссплатформенный
- **npm scripts** - если уже используете Node.js
- **Poetry scripts** в pyproject.toml

---

## Подводные камни

### Детерминизм и воспроизводимость

**Основные проблемы:**
1. **Разные ОС** → разные результаты даже с одинаковыми seeds
2. **Многопоточность** → race conditions в рандоме
3. **GPU** → недетерминированные операции по умолчанию
4. **Системное время** → влияние на некоторые алгоритмы

**Решения:**
- Фиксируйте ВСЕ источники рандома
- Используйте `stratify` в train_test_split
- Документируйте версии библиотек
- Тестируйте детерминизм автоматически

### MLflow в продакшене

**🔥 Проблемы:**
- File storage не масштабируется
- UI падает на больших объемах данных
- Нет встроенной аутентификации
- Сложности с очисткой старых экспериментов

**Решения:**
- S3/GCS для artifact store
- PostgreSQL для metadata store
- Регулярная очистка экспериментов
- Альтернативы: wandb, neptune

### Управление зависимостями

**🔥 Проблемы:**
- Конфликты версий
- Security vulnerabilities в зависимостях
- Разные результаты с "гибкими" версиями
- Медленная установка большого количества пакетов

**Решения:**
- Регулярные обновления безопасности
- Dependabot для автообновлений
- Точные версии для критичных зависимостей
- Слои Docker для кеширования установки

### Тестирование ML кода

**🔥 Проблемы:**
- Нестабильные тесты из-за рандома
- Сложность тестирования качества модели
- Долгие тесты обучения
- Зависимость от данных

**Решения:**
- Мокайте тяжелые операции
- Тестируйте на маленьких синтетических данных
- Smoke tests вместо точных значений
- Property-based тестирование

---

## Следующие шаги

### Что можно улучшить:

1. **CI/CD Pipeline**
   - GitHub Actions для автотестов
   - Автоматический деплой в staging
   - Matrix builds на разных Python версиях

2. **Мониторинг и логирование**
   - Структурированные логи (JSON)
   - Метрики производительности
   - Алерты на деградацию модели

3. **Data Pipeline**
   - DVC для версионирования данных
   - Автоматическая валидация данных
   - Feature store

4. **Model Serving**
   - REST API с FastAPI
   - Model registry
   - A/B тестирование моделей

5. **Scalability**
   - Kubernetes для оркестрации
   - Apache Airflow для пайплайнов
   - Distributed training

---

## Полезные ресурсы

### Книги
- "Designing Machine Learning Systems" - Chip Huyen
- "Building Machine Learning Pipelines" - Hannes Hapke
- "ML Engineering" - Andriy Burkov

### Инструменты
- **MLflow** - experiment tracking
- **DVC** - data versioning
- **Great Expectations** - data validation
- **Evidently** - ML monitoring
- **BentoML** - model serving

### Best Practices
- [Google's Rules of Machine Learning](https://developers.google.com/machine-learning/guides/rules-of-ml)
- [ML Test Score Rubric](https://storage.googleapis.com/pub-tools-public-publication-data/pdf/aad9f93b86b7addfea4c419b9100c6cdd26cacea.pdf)
- [Hidden Technical Debt in ML Systems](https://papers.nips.cc/paper/2015/file/86df7dcfd896fcaf2674f757a2463eba-Paper.pdf)

---

**Главный совет:** Начинайте просто, добавляйте сложность постепенно. Лучше работающий простой пайплайн, чем сломанный сложный!
