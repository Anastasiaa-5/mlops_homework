# Шаг 3: Prefect с cron триггером

## Описание

Финальный шаг демонстрирует полностью автоматизированный ML пайплайн с использованием Prefect cron триггеров. Пайплайн запускается каждые 2 минуты, имитируя поступление новых данных с продакшена и автоматическое переобучение модели.

## Структура

```
step3_prefect_cron/
├── data/              # Данные
├── models/            # Модели
├── metrics/           # Метрики
├── src/              # Задачи Prefect
│   ├── data_tasks.py     # Задачи для данных
│   ├── model_tasks.py    # Задачи для моделей
│   └── batch_manager.py  # Управление батчами
├── flows/            # Потоки Prefect
│   └── automated_training_flow.py  # Автоматический поток
├── deployments/      # Настройка деплойментов
│   └── setup_deployment.py
├── batch_state.json  # Состояние обработки батчей
└── Makefile          # Команды для управления
```

## Быстрый старт

```bash
# 1. Установка зависимостей
poetry install

# 2. Настройка проекта
make setup

# 3. Запуск сервисов (в отдельных терминалах)
# Терминал 1: Prefect сервер
poetry run prefect server start --host 0.0.0.0

# Терминал 2: MLflow UI
poetry run mlflow ui --host 0.0.0.0 --port 5000

# 4. Создание деплойментов с cron расписанием
poetry run python deployments/setup_deployment.py

# 5. Запуск worker (в отдельном терминале)
poetry run prefect worker start --pool default-process-pool
```

**🚀 После этого автоматический пайплайн будет запускаться каждые 2 минуты!**

## Детальная настройка

### 1. Установка зависимостей

```bash
poetry install
poetry shell
```

### 2. Запуск инфраструктуры

```bash
# Терминал 1: Prefect server
poetry run prefect server start --host 0.0.0.0

# Терминал 2: MLflow UI
poetry run mlflow ui --host 0.0.0.0 --port 5000
```

### 3. Создание деплойментов

```bash
poetry run python deployments/setup_deployment.py
```

### 4. Запуск worker

```bash
# Prefect 3.0 использует workers вместо agents
poetry run prefect worker start --pool default-process-pool

# Альтернативно можно использовать agent (устаревшая версия)
poetry run prefect agent start -q default
```

## Использование

### Автоматический режим

После настройки пайплайн работает полностью автоматически:

1. Каждые 2 минуты запускается автоматический поток
2. Определяется следующий батч для обработки
3. Проверяется наличие данных
4. Выполняется полный цикл: данные → обучение → оценка
5. Результаты сохраняются и логируются
6. Обновляется состояние для следующего запуска

### Ручной режим

```bash
# Запуск конкретного батча
poetry run python flows/automated_training_flow.py 3

# Автоматический режим (определяет следующий батч сам)
poetry run python flows/automated_training_flow.py

# Через Makefile
make run-manual BATCH=4
```

### Мониторинг

1. **Prefect UI**: http://localhost:4200
   - Статус выполнения потоков
   - Расписание запусков
   - Логи выполнения

2. **MLflow UI**: http://localhost:5000
   - Эксперименты и модели
   - Сравнение метрик
   - Артефакты

3. **Состояние батчей**: `batch_state.json`
   - История обработанных батчей
   - Метрики по каждому батчу

## Особенности автоматизации

### Управление батчами

- **Автоматическое определение**: Следующий батч определяется автоматически
- **Проверка данных**: Проверяется наличие данных перед обработкой
- **Ограничения**: Настраиваемое максимальное количество батчей
- **Состояние**: Персистентное состояние между запусками

### Расписание

```yaml
# В params.yaml
automation:
  cron_schedule: "*/2 * * * *"  # Каждые 2 минуты
  max_batches: 8
```

### Обработка ошибок

- Автоматические повторы при сбоях
- Логирование всех ошибок
- Graceful handling отсутствующих данных

## Команды управления

```bash
make help              # Показать все команды
make setup            # Настроить проект (DVC init, создать папки)
make start-services    # Запустить Prefect и MLflow (ЧАСТИЧНО РАБОТАЕТ)
make stop-services     # Остановить сервисы
make clean           # Очистить данные

# Основные команды (обновленные):
poetry install                                    # Установить зависимости
poetry run python deployments/setup_deployment.py # Создать деплойменты
poetry run prefect worker start --pool default-process-pool # Запустить worker
make run-manual BATCH=N                          # Ручной запуск для батча N
```

## Демонстрация работы

### Полная автоматизация

1. **Запуск сервисов**: Следуйте разделу "Быстрый старт"
2. **Открыть интерфейсы**:
   - Prefect UI: http://localhost:4200
   - MLflow UI: http://localhost:5000
3. **Автоматизация**: Через 2 минуты начнется автоматическое выполнение
4. **Мониторинг**: Следить за прогрессом в обоих UI
5. **Результаты**: Новые модели и метрики каждые 2 минуты

### Ручная демонстрация

```bash
# Запустить несколько батчей для демонстрации
poetry run python flows/automated_training_flow.py 1
poetry run python flows/automated_training_flow.py 2
poetry run python flows/automated_training_flow.py 3

# Посмотреть состояние батчей
cat batch_state.json

# Проверить результаты
ls -la models/
ls -la metrics/
```

## Устранение неполадок

### Проблемы с деплойментами

**Ошибка**: `A work queue can only be provided when registering a deployment with a work pool`
**Решение**: Обновленный скрипт deployments/setup_deployment.py уже исправляет эту проблему

**Ошибка**: `prefect.deployments:Deployment has been removed`
**Решение**: Используется новый API с `flow.to_deployment()` для Prefect 3.0

### Проблемы с MLflow

Если видите ошибки подключения к MLflow:

1. **Проверьте статус**: `curl http://localhost:5000/health`
2. **Запустите сервер**: `poetry run mlflow ui --host 0.0.0.0 --port 5000`
3. **Используйте локальное хранилище**: Измените `tracking_uri: "./mlruns"` в params.yaml

### Предупреждения Pydantic

Предупреждения о `pyproject_toml_table_header` безопасны и не влияют на работу.

### Рекомендуемый порядок запуска

1. **Терминал 1**: `poetry run prefect server start --host 0.0.0.0`
2. **Терминал 2**: `poetry run mlflow ui --host 0.0.0.0 --port 5000`
3. **Терминал 3**: `poetry run python deployments/setup_deployment.py`
4. **Терминал 4**: `poetry run prefect worker start --pool default-process-pool`

Этот подход демонстрирует **production-ready ML пайплайн** с полной автоматизацией и мониторингом!
