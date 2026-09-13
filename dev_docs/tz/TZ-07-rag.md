> **Статус: 🔨 ядро RAG готово (47 тестов зелёные).**
> ✅ LLM-слой (per-request роутинг, 15 тестов).
> ✅ ingestion.py: chunk_markdown (по заголовкам, max_chunk_chars), ingest_docs (white-list),
> render_manifest_text (для промпта), strategy_to_case (для retrieval).
> ✅ generation.py: GENERATION_PROMPT (манифест + доки + few-shot + task),
> REPAIR_PROMPT, generate_dsl() с repair-loop ≤ 2, markdown fence stripping,
> mocked LLM transport для тестов.
> ✅ волна 2: vectorstore.py (VectorStore protocol, InMemoryVectorStore с cosine +
> payload-фильтром + идемпотентным upsert, QdrantVectorStore с ленивым импортом
> qdrant-client, query_points API); embeddings.py (EmbeddingFunction protocol,
> MockEmbedding — детерминированный sha256-хеш, unit-norm, dim=64; OllamaEmbedding —
> bge-m3, dim=1024); retrieval.py (Retriever, коллекции dsl_docs/strategy_cases);
> pipeline.py (RAGPipeline: ingest_docs/ingest_strategies/generate с retrieval→
> generate_dsl, PipelineMetrics с pass@1/pass@N/failed/avg_iterations, QueryLog).
> ⬜ pass@1 evaluation-скрипт над query-set (нужен живой LLM).
> ⬜ rag_integration маркер: живые Qdrant/Ollama (skip без инфраструктуры).

# TZ-07. RAG-контур (инжест, retrieval, генерация DSL)

## 1. Контекст

Инфраструктура выбрана (Qdrant, Ollama, LlamaIndex, sentence-transformers — в
pyproject/compose), прикладного слоя нет. RAG нужен ровно для одного: подсунуть LLM
релевантный контекст (спецификация DSL, доки индикаторов, похожие стратегии, бэктесты),
чтобы модель не галлюцинировала несуществующие индикаторы. Риск-лимиты в индекс не
попадают по построению (сквозной принцип №1).

## 2. Почему именно так

### 2.1. Гибрид «детерминированный манифест + retrieval + few-shot» — главный рычаг
- **Манифест рендерится в промпт программно** (`Manifest.to_dict()` → текст), а не
  эмбеддится: векторизация `{"type": "integer", "min": 2}` бессмысленна и врёт в retrieval.
  Отвечает за синтаксическую корректность.
- **Retrieval по `dsl/docs/*.md` и `dev_docs/`** — за семантику («как писать трендовые
  условия»). Чанки по заголовкам, 400–800 токенов, перекрытие ~15%.
- **Few-shot прецеденты**: «найди 2–3 похожие валидированные стратегии и подложи в промпт» —
  самый эффективный способ заставить 8B-модель писать валидный DSL. Одна стратегия = один
  point, без чанкирования.

### 2.2. Две коллекции Qdrant, не одна
`dsl_docs` (чанки статических документов, payload `{source, heading, doc_type, lang}`) и
`strategy_cases` (одна стратегия = один point, payload `{dsl_entry, dsl_exit,
indicators_used, metrics, manifest_hash}`). **Почему:** это принципиально разные данные
с разным жизненным циклом (доки обновляются по git hash; стратегии — по результатам
бэктестов) и разным поиском (семантический vs поиск прецедентов).

### 2.3. `indicators_used` — обход AST, не regex
AST-сериализация (`to_dict`) уже есть. Regex по тексту DSL промахивается на let-биндингах
и алиасах.

### 2.4. White-list инжеста — enforcement, а не просьба
Инжест-пайплайн принимает явную конфигурацию путей (`dsl/docs/**`, `dev_docs/**` минус
риск-документация). Риск-лимиты не могут попасть в коллекцию, потому что для них нет
ingestion-маршрута. **Почему не «попросим LLM не использовать»:** единственная
архитектурная защита — физическое отсутствие данных в контексте.

### 2.5. Repair-loop через собственный парсер
Выход LLM машинно-проверяем: parse → ManifestValidator → при ошибке текст ошибки + манифест
возвращаются модели (макс. 2 итерации; далее `status: failed`). **Почему 2, а не 5:**
после 2 итераций 8B-модель начинает деградировать и «чинить» работающий код.
Контракт результата: `{status, dsl, errors[], iterations, chunks_used[]}` — failed
не покидает rag-слой.

### 2.6. Эмбеддинги через Ollama, не sentence-transformers в процессе
sentence-transformers тянет PyTorch (~1–2 ГБ RAM) в процесс. Для локального GPU-узла
это ок, но правильнее единая точка эмбеддингов Ollama (`/api/embeddings`) — модель
(bge-m3 или мультиязычная) в контейнере. **Языковой нюанс:** запросы будут на русском,
`dsl/docs` — на английском; берём мультиязычную модель и/или англ. аннотации к
русскоязычным чанкам при инжесте.

### 2.7. Idempotent-инжест
Повторный запуск не дублирует точки: ключ = hash(chunk) / strategy.id; старые точки
с устаревшим git-hash удаляются.

## 3. Требования

0. **LLM-слой с per-request роутингом — ✅ реализован** (`rag/llm.py`):
   `LLM_PROVIDER` из env задаёт только провайдера **по умолчанию** для воркера;
   каждый запрос может переопределить и провайдера (`router.complete(prompt,
   provider=...)`), и модель (`CompletionOptions(model=...)`). Для Ollama и
   OpenAI-compatible бэкендов `model` — поле запроса, поэтому один воркер
   обслуживает несколько моделей одновременно — мультитенантные сценарии
   поддержаны. Транспорты инъектируются → тесты без сети (11 тестов
   `rag/tests/test_llm.py`). Проект лицензируется под **MIT** (файл `LICENSE`,
   поля `license`/`license-files` в pyproject).

1. `rag/ingestion`: чанкер + white-list путей + рендер манифеста + две коллекции.
2. `rag/retrieval`: docs top-k (5–8) + cases top-k (2–3, фильтр по актуальному
   manifest_hash из TZ-02).
3. `rag/generation`: промпт-шаблоны (краткая грамматика DSL, запрет выдумывать индикаторы,
   температура ≤ 0.3) → Ollama DeepSeek-R1:8b → validate_strategy → repair-loop ≤ 2.
4. Метрика pass@1 (доля ок с первой попытки) с логированием query/chunks/iterations/status —
   это одновременно метрика качества retrieval (repaired-запросы ссылаются на индикаторы,
   чьих доков не было в контексте → retrieval виноват).
5. Кросс-проверка: перед индексацией `ai/docs` синхронизировать с реальностью (TZ-06 п.2.7),
   иначе проиндексируется выдуманная спецификация.

## 4. Критерии приёмки

- Инжест идемпотентен (повторный запуск не дублирует точки).
- По запросу «перепроданность с подтверждением тренда» в контексте — доки RSI/rising.
- ≥ 70% запросов из эталонного набора из 10 фраз → валидный DSL ≤ 2 итераций.
- В коллекциях отсутствуют чанки, содержащие риск-лимиты (тест по white-list).