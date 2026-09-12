> **Статус: ⬜ не начат** (частично: `infer/provider.py` BarSeriesProvider — цены
> через DSL-контракт; TaProvider поверх ta-индикаторов отсутствует).

# TZ-03. TaProvider: прокидывание ta через DSL

## 1. Контекст

Контракт `ta` единый: каждый модуль имеет триплет `x_numpy / x_ind / x_polars`;
`x_ind` принимает `np.ndarray | pl.Series`; общие параметры `offset, fillna,
nan_policy, trim`; дисциплина IEEE 754 (Inf→NaN, NaN propagates); TA-Lib только при
точном совпадении семантики. Примеры по группам: `rsi_ind(close, length, ...)`,
`atr_ind(high, low, close, length, mamode, ...)`, `adx_ind` (multi-output),
`entropy_ind`, `vwma_ind(close, volume, ...)`, `ott_ind` → кортеж 5 массивов
(ma, long_stop, short_stop, **direction**, ott).

DSL-резолвер же ожидает `(indicator, params, attributes, offset) -> float`. Задача —
мост без четырёх классов ошибок (Т1–Т4 ниже).

## 2. Почему именно так

### Т1. Семантика offset — ложный друг
`offset` в ta — сдвиг выходного ряда (для отрисовки). DSL `rsi.value[3]` — «3 бара назад».
**Правило: DSL-offset никогда не передаётся в ta** — реализуется индексацией кэша
`arr[t - offset]`. Иначе — тихо сдвинутые сигналы. Отвергнутая альтернатива
(использовать ta-offset «потому что удобно») даёт неявную зависимость между двумя
разными семантиками одного слова.

### Т2. Compute-once + кэш (бар-за-баром пересчитывать нельзя)
Наивный resolve с полным пересчётом RSI на каждый бар — O(n²) на бэктесте.
**Модель:** один раз вычислить ряд на доступном срезе → кэш по ключу
`(dsl_name, frozenset(params.items()), slice_end)` → `resolve(offset)` = O(1),
`resolve_history(n)` = срез кэша. Numba-ядра уже с `cache=True`, полный пересчёт ряда
5000 баров — миллисекунды, но не 5000 раз. В live кэш инвалидируется на новом баре.

### Т3. Multi-output через IndicatorBinding
DSL-атрибуты (`rsi.value`, `ott.direction`, adx/plus_di/minus_di) требуют явной карты
«атрибут → выход». OTT-direction живёт в отдельной нумба-функции — без адаптера он в DSL
не попадёт, а эталонная стратегия SIV (`dev_docs/strategy.md`) требует `ott.direction == 1`.

**Реестр биндингов** (ядро дизайна, не хардкод резолверов):
```python
@dataclass(frozen=True)
class IndicatorBinding:
    dsl_name: str                     # 'rsi'
    func: Callable[..., np.ndarray]   # rsi_ind
    params: dict[str, ParamSpec]      # length: int(1..), ...
    outputs: dict[str, OutputSpec]    # 'value' -> выход 0; 'direction' -> выход 3
    sources: tuple[str, ...]          # ('close',) | ('high','low','close') | ('close','volume')
    min_bars: Callable[[dict], int]   # warm-up как функция параметров
```
Манифест генерируется из биндингов (`build_manifest`) → `ManifestValidator` работает
бесплатно, и RAG (TZ-07) рендерит из него детерминированный контекст. Реестр собирается
из групповых `__all__` + явной таблицы одобренных индикаторов (не всё ta экспонируем).

### Т4. Warm-up/NaN — контракт, а не поведение по умолчанию
Внутри провайдера `nan_policy='ignore'` (не 'raise' — иначе прогрев убьёт всё), но:
- `t < min_bars(params)` → NotReady: сигнал False + счётчик warmup-пропусков в отчёте;
- NaN после прогрева → `EvaluationError` (аномалия данных).
Единый контракт с TZ-01 п.5. Отвергнутая альтернатива (fillna нулями) искажает значения
индикаторов в первые бары — сигналы на прогреве будут мусорными.

## 3. Требования

1. `IndicatorBinding` + реестр по 7 группам (по 1–2 индикатора на группу сначала).
2. `build_manifest(bindings) -> Manifest`.
3. `TaProvider(IndicatorProvider)`: compute-once/кэш/offset-индексация/multi-output/
   NotReady-контракт; батчевый `resolve_history` (переопределение из TZ-01 п.4).
4. Look-ahead-безопасность наследуется от движка (срез `[:t+1]`); тест-инвариант:
   подмена «будущих» баров мусором не меняет ни одного resolve(t).
5. Все индикаторы из AST вычисляются батчево один раз до прогона (обход AST уже возможен
   через `to_dict`).

## 4. Критерии приёмки

- **Паритет**: `evaluate_dsl("rsi(period=14).value[2] < 30")` на баре t ==
  `rsi_ind(close, 14)[t-2]` — для представителя каждой из 7 групп.
- Warm-up: ровно `min_bars` баров NotReady, дальше значения.
- Multi-output: `ott.direction` совпадает с `_compute_trend_direction_numba`.
- Offset: `x[3] == arr[t-3]` (не ta-сдвиг).
- Перформанс: 2 выражения × 5000 баров < 1 с.