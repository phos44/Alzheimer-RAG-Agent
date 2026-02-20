# Тестовое задание — стажеры 2026

## Часть 1. Подготовка данных

### Источники и сбор
- Источники: PubMed + bioRxiv (открытые API).
- Запросы:
  - `Alzheimer's disease targets`
  - `Alzheimer therapeutic targets`
  - `Alzheimer drug targets`
- Итоговый датасет после очистки и дедупликации: **108 статей**.
  - PubMed: 85
  - bioRxiv: 23

### Извлечение и очистка
- Для каждой статьи сохраняются поля: title, abstract, introduction, conclusion, doi, url, journal, authors, published_at.
- Формируется объединенный `text` (title + abstract + introduction + conclusion).
- Очистка: удаление HTML/спецсимволов, нормализация пробелов, дедупликация по `title + doi + source`.

### EDA
- Артефакты:
  - `artifacts/eda/top_tokens.png`
  - `artifacts/eda/wordcloud.png`
  - `artifacts/eda/top_gene_tokens.png`
  - `artifacts/eda/eda_summary.json`
- Основные статистики (по `artifacts/eda/eda_summary.json`):
  - rows: 108
  - mean_len: 1645.68
  - median_len: 1603.5
  - min_len: 19
  - max_len: 4574

### Подготовка к моделированию
- Создан parquet/csv датасет:
  - `data/processed/articles_clean.parquet`
  - `data/processed/articles_clean.csv`
- Текст разбивается на чанки и индексируется в FAISS:
  - `data/index/faiss/index.faiss`
  - `data/index/faiss/index.pkl`

## Часть 2. Реализация RAG системы

### Pipeline
1. Retrieval:
   - Embeddings: `BAAI/bge-base-en-v1.5`
   - Vector DB: FAISS
   - Top-k: 4
2. Generation:
   - LLM: `microsoft/phi-3-mini-4k-instruct`
   - Промпт ограничивает ответ только retrieved-контекстом.
3. Вывод:
   - Ответ + список источников (title/doi/url/source).

### Метрики и оценка
- Метрики retrieval:
  - Hit Rate@k
  - MRR
- Результаты (`artifacts/evaluation_metrics.json`):
  - hit_rate: **0.8**
  - mrr: **0.5**
  - num_questions: 5

### Вывод
- Система стабильно находит релевантные источники по большинству тестовых вопросов.
- Основное ограничение качества сейчас не в retrieval-инфраструктуре, а в качестве/полноте текста (часть статей только на abstract-level).

## Часть 3. Интерфейс

Реализован **Streamlit интерфейс**: `app.py`.

Обязательные поля выполнены:
- Ввод запроса исследователя.
- Отображение найденных источников.
- Сгенерированный ответ с указанием источников.

Запуск:
```bash
streamlit run app.py
```

## Часть 4. Вопросы

### На какие модальности данных можно расширить решение
- Transcriptomics (bulk/single-cell RNA-seq)
- Протеомика и метаболомика
- Графы взаимодействий (PPI, pathway graphs)
- Изображения (патоморфология/микроскопия)
- Клинические данные и биомаркеры

### Как это можно сделать
- Multi-index retrieval: отдельные индексы по модальностям + rank fusion.
- Для графов: добавить GNN-scoring как reranker.
- Для omics: формировать структурированные gene-set evidence chunks.
- Для изображений: совместные мультимодальные эмбеддинги (biomed-CLIP) и ссылки на figure/caption evidence.

### Какие модели выбраны и почему
- `BAAI/bge-base-en-v1.5`:
  - сильный baseline для semantic retrieval;
  - быстрый и удобный для локального прототипа.
- `microsoft/phi-3-mini-4k-instruct`:
  - компактная instruction-tuned модель;
  - практична для локального запуска и достаточно качественна для суммаризации evidence.

## Итог

Прототип выполнен end-to-end: сбор данных, EDA, индексация, retrieval+generation, метрики и интерфейс.  
Код и артефакты готовы для демонстрации и дальнейшего улучшения (reranking, full-text ingestion, multimodal extension).
