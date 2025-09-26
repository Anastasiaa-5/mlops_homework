# Шаг 4: Интегрированная система FastAPI + Мониторинг

## Описание

Полнофункциональная система мониторинга которая запускает **реальный FastAPI сервис** из шага 2 и мониторит его производительность с помощью HTTP запросов в реальном времени.

## Архитектура системы

```
step4_monitoring/
├── main.py                     # Интегрированный запуск FastAPI + мониторинг
├── config/
│   └── monitoring_config.yaml  # Конфигурация мониторинга и алертов
├── src/
│   ├── api.py                  # FastAPI сервис (из step2)
│   ├── model_service.py        # ONNX сервис (из step2)
│   ├── service_monitor.py      # HTTP мониторинг сервиса
│   └── logger.py              # Цветное логирование
├── models/                     # ONNX модель (копировать из step1)
├── test_images/               # Тестовые изображения (копировать из step2)
└── logs/                      # Логи мониторинга (генерируются)
```

## Возможности

### 🚀 **Интегрированный сервис**
- Автоматический запуск FastAPI сервиса на порту 8000
- Мониторинг сервиса через HTTP клиент
- Graceful shutdown при Ctrl+C

### 📊 **Мониторинг в реальном времени**
- **Health checks** каждые 30 секунд
- **Inference тесты** с реальными изображениями
- **P95 latency** мониторинг
- **Error rate** отслеживание
- **Quality assessment** предсказаний модели

### 🎨 **Цветное логирование**
- 🟢 **Зеленый** - нормальные метрики (OK)
- 🟡 **Желтый** - предупреждения (WARNING)
- 🔴 **Красный** - критические проблемы (CRITICAL)
- 📊 **Сводки** производительности в реальном времени

### 🚨 **Умные алерты**
- Настраиваемые пороговые значения
- Cooldown период для предотвращения спама
- Отслеживание consecutive failures

## Установка и запуск

### 1. Подготовка зависимостей
```bash
poetry install
poetry shell
```

### 2. Подготовка ресурсов
```bash
# Копируем ONNX модель из step1
cp ../step1_onnx_model/models/blip_model.onnx models/

# Копируем тестовые изображения из step2
cp ../step2_fastapi_inference/test_images/img.jpg test_images/
```

### 3. Запуск интегрированного сервиса
```bash
python main.py
```

## Что происходит при запуске

### Phase 1: Запуск FastAPI
```
🚀 Запуск FastAPI сервиса...
📡 FastAPI запущен с PID: 12345
⏳ Ожидание готовности сервиса...
✅ FastAPI сервис готов к работе

🌐 Сервис доступен по адресам:
   • API: http://localhost:8000
   • Health: http://localhost:8000/health
   • Docs: http://localhost:8000/docs
```

### Phase 2: Мониторинг
```
📊 ЗАПУСК МОНИТОРИНГА
🔍 Запуск цикла мониторинга...
✅ Test 1/5: 3842.1ms
✅ Test 2/5: 3791.6ms
📈 P95 response time: 3850.23 ms [OK]
📈 Error rate: 0.00 % [OK]

📊 PERFORMANCE SUMMARY
Total tests: 5
Success rate: 100.0%
P95 response time: 3850.2ms
Latest prediction: 'a man fishing on a boat at sunset'
```

## Мониторинговые метрики

### 🎯 **Основные метрики**
- **P95 Response Time** - 95-й перцентиль времени ответа
- **Error Rate** - процент неудачных запросов
- **Health Status** - статус health endpoint
- **Consecutive Failures** - количество последовательных ошибок

### 🎨 **Качество предсказаний**
- Проверка наличия ключевых слов: `["man", "fishing", "boat", "sunset"]`
- **Quality Score** - процент найденных ключевых слов
- **Latest Prediction** - последнее предсказание модели

### ⏱️ **Производительность**
- **Inference Time** - время ONNX инференса
- **Total Response Time** - полное время HTTP запроса
- **Request Throughput** - количество запросов в секунду

## Конфигурация алертов

```yaml
thresholds:
  p95_latency_ms:
    warning: 4000   # 4 секунды
    critical: 8000  # 8 секунд
  error_rate_percent:
    warning: 10     # 10% ошибок
    critical: 25    # 25% ошибок
  consecutive_failures:
    warning: 3      # 3 подряд
    critical: 5     # 5 подряд
```

## Примеры алертов

### ⚠️ Warning алерты
```
🚨 ALERT: p95_latency = 4200.5ms (Warning threshold exceeded)
⚠️ 3 consecutive failures detected
```

### 🚨 Critical алерты
```
🚨🚨 CRITICAL ALERT: error_rate = 30.0% (Critical threshold exceeded)
🚨 5 consecutive failures detected!
```

## Результаты мониторинга

### 📁 **Файлы логов**
- `logs/service_monitoring.log` - структурированные логи
- `logs/service_metrics.jsonl` - метрики в JSON Lines формате

### 📊 **Итоговая статистика**
```
📈 ИТОГОВАЯ СТАТИСТИКА
Циклов мониторинга: 25
Средняя P95 response time: 3847.2 ms
Средний error rate: 2.1%
Последний статус: healthy
Последнее предсказание: 'a man fishing on a boat at sunset'
```

## Использование в продакшене

Эта система готова для продакшен мониторинга:

- ✅ **Автоматический restart** сервиса при сбоях
- ✅ **Graceful shutdown** при получении сигналов
- ✅ **Structured logging** для агрегации логов
- ✅ **Configurable thresholds** для разных сред
- ✅ **Real-time alerting** для быстрого реагирования

## Расширение функциональности

Система легко расширяется для добавления:
- Отправки алертов в Slack/Telegram
- Интеграции с Prometheus/Grafana
- Автоматического масштабирования
- A/B тестирования разных версий модели
- Мониторинга drift модели

---

**🎯 Итог**: Полнофункциональная система мониторинга production-ready FastAPI сервиса с ONNX моделью!
