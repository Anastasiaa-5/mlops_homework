# Lesson 6: LLMOps - Animation Generation with RAG

## Описание

Урок посвящен **LLMOps** и созданию **анимаций действий человека** с использованием:
- **LLM** (Large Language Model) для генерации
- **Function Calling** для взаимодействия с API
- **RAG** (Retrieval-Augmented Generation) для улучшения качества

## Структура

### Step 1: Ollama Inference
Базовый LLM сервис с OpenAI-совместимым API:
- Ollama для инференса
- qwen2.5:1.5b (квантованная GGUF ~1GB)
- OpenAI SDK клиент
- Jupyter notebook для демонстрации

### Step 2: Function Calling
Генерация анимаций через function calling:
- Pose Visualization API (FastAPI)
- LLM Agent с function calling
- Создание GIF из последовательности поз
- **Проблема**: маленькая модель делает ошибки в координатах

### Step 3: RAG (Retrieval-Augmented Generation)
Улучшение качества через RAG:
- База данных с 75 проверенными позами
- TF-IDF поиск по описаниям
- LLM генерирует описания → RAG находит позы
- **Результат**: всегда корректные координаты
- **Демо**: танец Макарена (12 кадров)

## Основная идея

### Без RAG (Step 2):
```
LLM → JSON координаты → Pose API → GIF
❌ Ошибки в координатах
❌ Нужна большая модель
```

### С RAG (Step 3):
```
LLM → Описания → RAG → Проверенные позы → GIF
✅ Всегда корректные координаты
✅ Работает с маленькой моделью
✅ Легко расширять базу
```

## Быстрый старт

### 1. Запуск сервисов (Step 2 или Step 3)

```bash
cd lesson6/seminar/step2_function_calling
make start-all    # Ollama + Pose API
make pull         # Загрузка модели ~1GB
```

### 2. Step 2: Function Calling (базовый)

```bash
cd lesson6/seminar/step2_function_calling
poetry install
poetry run python test_scripts/demo_animation.py
```

### 3. Step 3: RAG Макарена 🎬

```bash
cd lesson6/seminar/step3_rag
poetry install
poetry run python demo_macarena_direct.py
open output/macarena.gif
```

## Демонстрация

### Макарена (12 кадров, зацикленная)

RAG система:
1. Получает описания движений Макарены
2. Находит похожие позы из базы (TF-IDF similarity)
3. Генерирует 12 кадров
4. Создает зацикленный GIF

```
🔍 Searching for 12 poses...
  1. руки вперед на уровне плеч... (sim: 0.72)
  2. правая рука вверх левая вперед... (sim: 0.58)
  ...
🎬 Creating GIF...
✅ Saved: output/macarena.gif
```

## Архитектура

```
┌─────────────────┐
│  User Request   │  "Создай танец макарена"
└────────┬────────┘
         ↓
┌─────────────────┐
│   LLM (Ollama)  │  Генерирует описания движений
└────────┬────────┘
         ↓
┌─────────────────┐
│  RAG Retriever  │  TF-IDF поиск в базе (75 поз)
└────────┬────────┘
         ↓
┌─────────────────┐
│   Pose API      │  Визуализация поз (FastAPI)
└────────┬────────┘
         ↓
┌─────────────────┐
│  GIF Generator  │  Создание анимации (Pillow)
└────────┬────────┘
         ↓
    macarena.gif 🎬
```

## Технологии

- **Ollama**: LLM inference на CPU
- **qwen2.5:1.5b**: Квантованная модель (GGUF)
- **FastAPI**: Pose Visualization API
- **scikit-learn**: TF-IDF для RAG
- **Pillow**: Создание GIF
- **Matplotlib**: Рисование человечков

## Сравнение подходов

| Критерий | Function Calling | RAG |
|----------|-----------------|-----|
| Качество координат | ⚠️ Зависит от модели | ✅ Всегда корректно |
| Скорость | 🐌 Медленно | ⚡ Быстро |
| Надежность | ⚠️ Нестабильно | ✅ Стабильно |
| Требования к LLM | 🔴 Нужна большая модель | 🟢 Любая модель |
| Расширяемость | ⚠️ Нужны примеры | ✅ Просто добавить позы |

## Результаты

- ✅ 3 шага с прогрессивным улучшением
- ✅ Макарена: 12 кадров, зацикленная
- ✅ 75 поз в базе данных
- ✅ RAG показывает преимущество над function calling
- ✅ Работает на CPU (без GPU)

## Обучающие материалы

Каждый step содержит:
- `README.md` - документация
- `Makefile` - команды для запуска
- `notebooks/` - Jupyter демонстрации
- `test_scripts/` - тестовые скрипты
- `src/` - исходный код

## Требования

- Docker + Colima (для Ollama)
- Python 3.10+
- Poetry (для управления зависимостями)
- ~1GB для модели
- ~2GB RAM

## Следующие шаги

1. Расширить базу поз до 150-200
2. Добавить больше действий (приветствие, упражнения)
3. Использовать embedding модели для better RAG
4. Добавить интерполяцию между кадрами
5. Поддержка видео (MP4) вместо GIF

## Выводы

- **Function calling** работает, но не всегда надежно с маленькими моделями
- **RAG** значительно улучшает качество и надежность
- **Verified database** важнее чем сложная генерация
- **Small models** + **RAG** = хорошие результаты
