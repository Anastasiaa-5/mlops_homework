# Lesson 6: LLMOps - RAG + Function Calling

Практическое занятие по работе с большими языковыми моделями (LLM) в production.

## 🎯 Цель занятия

Создать систему для генерации анимаций человеческих поз с помощью:
- **LLM** для понимания текстовых запросов
- **Function Calling** для взаимодействия с внешними сервисами
- **RAG** для улучшения качества результатов

## 📚 Структура

### Step 1: LLM Inference Service
**Технологии**: Ollama, Qwen2.5:1.5b, OpenAI API format

**Что делаем**:
- Запуск локального LLM сервера (Ollama)
- Создание Python клиента
- Интерактивный Jupyter notebook для тестирования

**Особенности**:
- ✅ CPU-only (без GPU)
- ✅ Быстрое развертывание
- ✅ GGUF quantized модели
- ✅ OpenAI-совместимый API

**Результат**: Работающий локальный LLM сервис на порту 11434

---

### Step 2: Function Calling
**Технологии**: FastAPI, Matplotlib, Function Calling

**Что делаем**:
- Создаем Pose Visualization API (FastAPI)
- Учим LLM вызывать функции
- Генерируем анимации из последовательности поз

**Архитектура**:
```
User Query → LLM Agent → Function Call → Pose API → Images → GIF
```

**API эндпоинт**:
- `POST /visualize` - принимает JSON с координатами, возвращает изображение

**Формат позы**:
```json
{
  "Torso": [0, 0],
  "Head": [0, 60],
  "RH": [30, 35],    // Right Hand
  "LH": [-30, 35],   // Left Hand
  "RK": [15, -50],   // Right Knee
  "LK": [-15, -50]   // Left Knee
}
```

**Результат**: Агент создает GIF анимации по текстовым запросам

---

### Step 3: RAG System
**Технологии**: TF-IDF, scikit-learn, RAG pattern

**Проблема Step 2**:
- LLM плохо генерирует точные координаты
- Нет знаний о реалистичных позах
- Результат выглядит неестественно

**Решение через RAG**:
1. База данных с 96 описанными позами
2. TF-IDF векторизация описаний
3. Поиск по семантическому сходству
4. Использование проверенных поз

**Workflow**:
```
Text Query → LLM (описания движений) → RAG Search → Poses → GIF
```

**База данных** (`poses_database.json`):
- 96 различных поз
- Описания только позиций тела (без названий действий)
- Примеры: "Руки вытянуты вперед", "Правая рука согнута к плечу"

**Демонстрация**:
- **Макарена** (16 кадров) - сложная хореография
- **Танец рук** (8 кадров) - простые движения

**Результат**:
- Similarity 1.00 для точных совпадений
- Качественные реалистичные анимации
- Зацикленные GIF файлы

---

## 🚀 Быстрый старт

### Шаг 1: Запуск LLM
```bash
cd lesson6/seminar/step1_vllm_inference
make install
make start          # Запуск Ollama
make pull          # Скачать модель qwen2.5:1.5b
make test          # Проверить работу
make notebook      # Открыть демо
```

### Шаг 2: Function Calling
```bash
cd lesson6/seminar/step2_function_calling
make install
make start-pose    # Запуск Pose API (порт 8001)
make start-ollama  # Запуск Ollama (порт 11434)
make test          # Тестирование
make notebook      # Интерактивное демо
```

### Шаг 3: RAG
```bash
cd lesson6/seminar/step3_rag
make install
make demo          # Создать оба танца
```

**Результат**:
- `output/macarena.gif` (16 кадров, 32KB)
- `output/hand_dance.gif` (8 кадров, 19KB)

---

## 📊 Результаты

### Метрики качества

**Step 2 (без RAG)**:
- Координаты генерируются случайно
- Позы неестественные
- Нет согласованности между кадрами

**Step 3 (с RAG)**:
- ✅ Similarity 0.90-1.00
- ✅ Проверенные позы из БД
- ✅ Плавные переходы
- ✅ Реалистичные движения

### Примеры

**Макарена** (16 шагов):
1. Руки вперед ладони вниз/вверх
2. Руки к плечу (правая/левая)
3. Руки за голову
4. Руки к бедрам
5. Движение бедрами
6. Руки вверх
7. Прыжок

**Танец рук** (8 шагов):
1. Руки на уровне плеч
2. Руки вверх
3. Правая вверх, левая вниз
4. Левая вверх, правая вниз
5. Руки в стороны
6. Руки вперед
7-8. Возврат в исходную

---

## 🛠 Технический стек

| Компонент | Технология | Версия |
|-----------|------------|--------|
| LLM Runtime | Ollama | latest |
| Model | Qwen2.5 | 1.5b |
| API Format | OpenAI Compatible | v1 |
| Pose Service | FastAPI | 0.115+ |
| Visualization | Matplotlib | 3.9+ |
| RAG Search | scikit-learn (TF-IDF) | 1.5+ |
| Images | Pillow | 11.0+ |
| Orchestration | Docker Compose | - |
| Dependencies | Poetry | - |

---

## 🎓 Ключевые концепции

### 1. Function Calling
Позволяет LLM вызывать внешние функции:
```python
tools = [{
    "type": "function",
    "function": {
        "name": "create_animation",
        "description": "Создать анимацию из поз",
        "parameters": {...}
    }
}]
```

### 2. RAG Pattern
Улучшение качества через внешние знания:
```
Query → Retrieval (поиск в БД) → Augmentation (обогащение) → Generation
```

### 3. Semantic Search
TF-IDF для поиска похожих описаний:
```python
vectorizer = TfidfVectorizer()
similarities = cosine_similarity(query_vec, db_vecs)
```

---

## 📈 Сравнение подходов

| Метрика | Step 2 (No RAG) | Step 3 (RAG) |
|---------|-----------------|--------------|
| Качество поз | ⚠️ Случайное | ✅ Проверенное |
| Реалистичность | ❌ Низкая | ✅ Высокая |
| Согласованность | ❌ Нет | ✅ Да |
| Similarity | N/A | 0.90-1.00 |
| Сложность | Простая | Средняя |

---

## 🎯 Практические навыки

После прохождения занятия вы умеете:

1. ✅ Развертывать локальные LLM (Ollama)
2. ✅ Работать с OpenAI-совместимым API
3. ✅ Реализовывать Function Calling
4. ✅ Создавать FastAPI сервисы
5. ✅ Применять RAG для улучшения качества
6. ✅ Использовать TF-IDF для семантического поиска
7. ✅ Генерировать анимации программно
8. ✅ Оркестрировать микросервисы через Docker Compose

---

## 💡 Область применения

Подобные системы используются в:
- 🎮 Игровой индустрии (генерация анимаций персонажей)
- 🏃 Спортивной аналитике (анализ движений)
- 🏥 Медицине (физиотерапия, реабилитация)
- 🤖 Робототехнике (планирование движений)
- 🎬 Анимации и кинематографе

---

## 📦 Структура файлов

```
lesson6/
├── README.md
├── SUMMARY.md                    # Этот файл
└── seminar/
    ├── step1_vllm_inference/
    │   ├── docker-compose.yml    # Ollama service
    │   ├── src/llm_client.py     # OpenAI client
    │   ├── notebooks/
    │   │   └── chat_demo.ipynb   # Интерактивное демо
    │   └── Makefile
    │
    ├── step2_function_calling/
    │   ├── docker-compose.yml    # Ollama + Pose API
    │   ├── Dockerfile.pose       # Pose service image
    │   ├── src/
    │   │   ├── pose_api.py       # FastAPI service
    │   │   └── pose_agent.py     # Function calling agent
    │   ├── notebooks/
    │   │   └── pose_demo.ipynb   # Демо анимаций
    │   ├── test_scripts/         # Тестовые скрипты
    │   └── Makefile
    │
    └── step3_rag/
        ├── poses_database.json   # 96 поз с описаниями
        ├── src/
        │   ├── pose_retriever.py # TF-IDF RAG
        │   └── rag_agent.py      # RAG-based agent
        ├── demo_dance.py         # Главное демо
        ├── output/
        │   ├── macarena.gif      # 16 кадров, 32KB
        │   └── hand_dance.gif    # 8 кадров, 19KB
        └── Makefile
```

---

## 🔧 Troubleshooting

### LLM не отвечает
```bash
curl http://localhost:11434/api/tags  # Проверить модели
ollama ps                              # Проверить процессы
```

### Pose API не работает
```bash
curl http://localhost:8001/health     # Health check
docker-compose logs pose_api          # Логи
```

### Низкий similarity в RAG
- Проверьте описания в БД
- Используйте более точные формулировки
- Добавьте больше поз в базу

---

## 🎉 Итог

**Создали полноценную LLMOps систему**:
- ✅ Локальный LLM inference
- ✅ Function calling для взаимодействия с API
- ✅ RAG для улучшения качества
- ✅ Генерация анимаций из текста
- ✅ Микросервисная архитектура

**Применили на практике**:
- Ollama для CPU-only inference
- OpenAI API format
- FastAPI для сервисов
- TF-IDF для семантического поиска
- Docker Compose для оркестрации

**Результат**:
- 2 танца с качественными анимациями
- Similarity 0.90+ для всех кадров
- Работает на обычном CPU без GPU

---

## 📚 Дополнительные материалы

- [Ollama Documentation](https://ollama.ai/docs)
- [OpenAI Function Calling](https://platform.openai.com/docs/guides/function-calling)
- [RAG Pattern Overview](https://docs.anthropic.com/claude/docs/retrieval-augmented-generation)
- [FastAPI Best Practices](https://fastapi.tiangolo.com/tutorial/)

---

**Lesson 6 Complete! 🎉💃🤖**
