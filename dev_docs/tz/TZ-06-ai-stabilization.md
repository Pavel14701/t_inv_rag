# TZ-06. Стабилизация ai/ (EntryExitTransformer)

## 1. Контекст

`ai/` — самодостаточный, но изолированный «островной» модуль (~3.5k строк): не знает про
dsl/ta/strategies, привязан к концепции order blocks, которой больше нигде нет. Ядро
(transformer, losses, self-training, contracts) качественное, но интеграции мешают
проблемы ниже. Порядок в roadmap — сразу после TZ-01, до TZ-02/03: torch и утечка
валидации блокируют любое использование модели.

## 2. Проблемы и обоснование решений

### 2.1. torch отсутствует в зависимостях (критично)
`pyproject.toml` не содержит torch — он приходит транзитивно через sentence-transformers.
Версия и вариант (CPU/CUDA) не зафиксированы. **Решение:** добавить torch явно
с индекс-конфигурацией. **Почему нельзя оставить транзитивным:** uv-lock не управляет
им напрямую; обновление sentence-transformers молча сменит torch.

### 2.2. Утечка валидации через random-сплит (критично)
`_split_train_val` перемешивает, а окна seq_len=128 перекрываются на 127 баров — один и
тот же кусок рынка в train и val, метрики завышены. Совпадает с п.5 quant_checklist.
**Решение:** хронологический сплит по диапазонам баров (train < T_val ≤ val) без
перекрытия окон на границе. **Почему не K-fold:** временная структура данных — валидация
«из будущего» бессмысленна для трейдинга.

### 2.3. Self-training деструктивно мутирует датасет
`_update_labels_parquet` перезаписывает labels.parquet псевдо-лейблами без пометки
происхождения → накопление ошибок необратимо. **Решение:** отдельный файл/колонка
`is_pseudo`, бэкап предыдущего состояния, откат раунда. **Почему важно:** self-training —
итеративный процесс; без возможности отката первый неудачный раунд портит датасет навсегда.

### 2.4. Model bundle (критично для инференса)
`torch.save(state_dict)` без seq_len, списка колонок, atr_global, нормировочных статистик —
модель нельзя корректно загрузить. **Решение:** bundle
`{state_dict, config, feature_columns, seq_len, atr_global, norm_stats}` + загрузчик.
Без bundle TZ-05 (`--ml`) невозможен технически.

### 2.5. Инференс-контракт
Функция `predict_p_win(model, bundle, window, ...) -> float` — единственная точка
взаимодействия Risk Engine и скрипта инференса с моделью. Сейчас её нет: forward отдаёт
per-bar logits батчами. Требование ТЗ «< 1 мс на сделку» не достижимо ещё и потому, что
forward перебирает order blocks питоновским циклом с созданием тензора на каждый —
нужна батчеризация OB-энкодера.

### 2.6. O(окна × OB) сканирование
`TradingDataset.__getitem__` линейно проходит все блоки на каждый сэмпл. На реальных
объёмах — часы. **Решение:** индекс OB по бару (sorted + bisect).

### 2.7. Пакет-идентификация
Докстринги/примеры ссылаются на `trading.*`; `quickstart.py` импортирует из
`trading.quickstart` (битый путь). Фиксируется имя пакета `ai`, чистится документация
(включая мифический Lark-парсер из `ai/docs/README.md` — см. TZ-01 п.2.7).

### 2.8. Нет ни одного теста
dsl и ta покрыты, ai — нет, хотя ai принимает самое дорогое решение. Минимальный набор:
лейбл-генератор (look-ahead-тесты: лейбл на баре t не зависит от баров > t), лоссы,
сплит-утечка, паритет лейблов с движком TZ-04 (golden-тест).

### 2.9. Мелочи
`contracts.py` печатает предупреждения через print (включая «Batch validation passed.»
на каждый батч) → logging; `quickstart` хардкодит `close_idx=3`; `compute_atr` — питоновский
цикл (есть numba); pattern-head объявлен, но «unused in training» — реализовать данные или
убрать из доков.

## 3. Требования (сводно)

1. ✅ torch в pyproject (п.2.1) — объявлен явно (`torch>=2.4.1`), не только
   через [gpu]-экстры; uv.lock закрепляет 2.4.1 (win) / 2.10.0.
2. ✅ Хронологический train/val сплит (п.2.2) — `_split_train_val` отдаёт
   валидации **самые последние** окна и усекает train до границы
   (`val_start - (seq_len - 1)`), т.е. ни одно окно train не пересекается
   по барам с val. Тест: `test_split_train_val_no_window_overlap`.
3. ✅ Self-training: is_pseudo + бэкап + откат (п.2.3) —
   `_update_labels_parquet(labels_path, pseudo, mode, round_idx)`:
   пер-раундный бэкап `<path>.bak_roundN.parquet`, булева колонка
   `is_pseudo`, уже-псевдо бары не перезаписываются; `_rollback_labels`
   восстанавливает бэкап (FileNotFoundError без него). Тесты:
   `test_update_labels_adds_is_pseudo_and_backup`,
   `test_update_labels_never_overwrites_pseudo`,
   `test_rollback_labels_restores_backup`,
   `test_rollback_labels_missing_backup_raises`.
4. ✅ Model bundle + загрузчик (п.2.4) — `ai/src/bundle.py`:
   `ModelBundle{state_dict, model_config, feature_columns, seq_len,
   atr_global, norm_stats, version}`, `build_bundle / save_bundle /
   load_bundle / rebuild_model`. ONNX-экспорт — в `device.py`
   (`export_onnx` с dynamic_axes), дубль из bundle удалён.
5. ✅ `predict_p_win` — контракт инференса (п.2.5) —
   `EntryExitPredictor(bundle).predict_proba(...) -> {p_entry, p_exit,
   p_win}` и `predict_p_win(...) -> float` (последний бар окна,
   `torch.inference_mode`). Пороговые потребители (Risk Engine, TZ-05)
   никогда не касаются тензоров напрямую.
6. ✅ OB-индекс (bisect) (п.2.6) — `TradingDataset.__getitem__` ищет
   блоки через `np.searchsorted` по отсортированным `end_idx`
   (префикс + фильтр по `start_bar`) вместо полного прохода.
   Батчеризация OB-энкодера в forward — ⬜ остаётся (профилировать
   на реальном объёме; текущий тестовый объём мал).
7. ✅ Имя пакета `ai`, чистка доков (п.2.7) — пакет `ai`, относительные
   импорты в тестах, Lark-легенда убрана (см. TZ-01 п.2.7).
8. ✅ Тесты (п.2.8) — `ai/src/tests/`: look-ahead лейблов
   (`test_features`, `test_label_generation`), лоссы (`test_losses`),
   утечка сплита + self-training изоляция (`test_stabilization`),
   bundle/предикт (`test_bundle`). Итог набора: **68 passed**.
9. ✅ logging вместо print (п.2.9) — `training.py` и `contracts.py`
   используют модульный `logger` (warning для контрактов, info для
   прогресса обучения; 'Batch validation passed.' → debug).
10. ✅ **Конфигурация (YAML)** — реализовано: `configs/ai.yaml` + `ai/src/config.py`
    (секции risk/model/training/compute, `seed`, дефолты = прежние хардкоды,
    конфиг протаскивается аргументами, глобального состояния нет; `risk_kwargs()`
    мапит RiskConfig на генератор лейблов).
11. **Compute backends** — реализовано: `ai/src/device.py`.
    - Обучение: **только CUDA или CPU** (не-CUDA обучение отброшено);
      `train_backend: auto | cuda | cpu`, неверный бэкенд = ValueError.
    - Инференс: `infer_backend: auto | cuda | cpu | onnx_directml | vulkan`;
      `vulkan` — алиас `onnx_directml` (DX12: AMD/Intel/NVIDIA).
    - Чистый Vulkan и GGUF **отвергнуты**: backward-pass на Vulkan не существует;
      GGUF — формат llama.cpp под конкретные LLM-архитектуры, кастомный
      EntryExitTransformer конвертером не покрывается; переписывание forward на
      GGML несоразмерно задаче.
    - Путь не-CUDA-инференса: `export_onnx()` → ONNX Runtime DirectML EP;
      optional-группа `[gpu]` (torch-directml, onnxruntime-directml, Windows).
    - Smoke-тест для non-CUDA бэкендов обязателен (риск тихих NaN).

## 4. Критерии приёмки

- ✅ Набор ai-тестов зелёный: **68 passed, 3 skipped** (skips —
  платформозависимые DirectML-тесты без `[gpu]`).
- ✅ `quick_train` end-to-end на синтетических данных
  (`test_quickstart`).
- ⬜ predict_p_win < 5 мс на окне 128 (CPU) — замерить при первом
  прогоне на реальном железе; тест-инвариант корректности есть.
- ✅ Тест-инвариант лейблов и сплита зелёные.
- ✅ Повторный запуск с тем же seed — идентичные метрики
  (`set_seed` в конфиге; `test_config` проверяет seed).

## 5. Сверка с quant_checklist

Пункты чек-листа, закрываемые этим ТЗ:

| Пункт чек-листа | Статус | Где |
|---|---|---|
| п.5: валидация только по времени (без перемешивания) | ✅ | `_split_train_val`, тест на отсутствие перекрытия окон |
| п.5: TP/SL из данных ≤ t (ATR предыдущих баров) | ✅ (было) | `features.compute_tp_sl` |
| п.6: воспроизводимость (seed, конфиг) | ✅ | `configs/ai.yaml`, `set_seed` |
| п.4: метрика max drawdown в валидации | ⬜ | TZ-04 (бэктест), не ai |
| п.1: онлайн-ZigZag / фиксированные OB | ⬜ | TZ-04 п.4.3 |
| п.2: исполнение open[t+1], комиссии | ✅ (было) | `features.py` (переносится в TZ-04 движок) |
| п.3: базлайны RF/XGBoost | ⬜ | TZ-04 п.4.6 |