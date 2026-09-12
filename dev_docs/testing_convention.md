# Testing Convention (конвенция тестов)

> Обязательна к прочтению до правки/написания любого теста. Введена TZ-14 (нулевой этап).
> Нарушение = отклонение PR.

## 1. Структура и имена

```
<package>/tests/                     # все тесты пакета (infer, rag, dsl, ai, ta, ...)
<package>/tests/conftest.py          # fixtures ТОЛЬКО этого пакета
<package>/tests/test_<module>.py     # один файл на тестируемый модуль/фичу
<package>/pytest.ini                 # унифицированный шаблон (см. §6)
```

- Имя файла: `test_<модуль>.py` (например `test_interpreter.py`); подгруппы —
  каталогом `tests_<domain>/` (как `ta/src/tests/tests_candle/`), не суффиксом.
- Тест-функция: `test_<behavior>_<expected>()` — «что проверяем» + «что ожидаем»:
  `test_rising_with_insufficient_history_returns_notready`.
- Классы-группировки (`Test*`) — только когда параметризация/фикстуры общие для группы.
- Никаких numbered-тестов (`test_1`, `test_new_2`).

## 2. Правила написания

1. **AAA**: Arrange / Act / Assert — три блока, между ними пустая строка.
   Один Act, несколько Assert допустимы для одного поведения.
2. **Один тест — одно поведение.** Тест «и парсинг, и валидация, и метрики» запрещён.
3. **No sleeps, no network.** `time.sleep()` запрещён; сеть — только в `@pytest.mark.integration`
   с `skipif` по отсутствию сервиса/токена.
4. **Только явные fixtures из conftest**; дублирующиеся фикстуры двух пакетов
   (синтетические OHLC, mock-провайдеры) — признак, что фикстуру надо параметризовать
   в conftest пакета, а не копировать.
5. **Моки**: `pytest-mock` (`mocker`), не ручные классы-обёртки, если есть Protocol
   (например, `LLMProvider`, `IndicatorProvider`) — мокается Protocol.
6. **Не тестируйте приватное.** Тест через `_private()` — сигнал переработки API,
   а не теста.
7. **Параметризация вместо копипасты**: `@pytest.mark.parametrize` для таблиц случаев;
   ID кейсов (`ids=`) — обязательны при > 3 случаях.
8. **Exception**: `pytest.raises(...)` c `match=` на стабильное подмножество текста;
   в DSL-пакете — только типы иерархии `DSLError` (TZ-01 контракт).

## 3. Маркеры (унифицированный набор)

| Маркер | Значение | CI |
|--------|----------|----|
| `unit` | быстрый, изолированный | всегда |
| `integration` | сервисы/сеть/файлы | всегда, с skipif |
| `slow` | > 5 c | по флагу (`-m slow`), в CI — отдельной job |
| `deprecated` | под удаление | не запускается по умолчанию (`-m "not deprecated"`) |

Допустимы пакетные маркеры (tokenizer, parser, …) — только из тех, что уже объявлены
в `pytest.ini` пакета; новые регистрируются там же (`--strict-markers` иного не позволит).

## 4. Детерминизм и look-ahead

- Тесты со случайными данными фиксируют seed (`np.random.default_rng(0)` / `torch.manual_seed`)
  — тесты обязаны быть воспроизводимыми (TZ-00 п.4.6).
- Для каузальных компонентов (ta/, движок исполнения, лейблы) обязателен
  look-ahead-инвариант-тест: подмена баров > t мусором не меняет результат на t.
- Референсные значения не хардкодятся из «текущего вывода» без понимания —
  если значение из вывода, в комментарии указывается происхождение (документ/расчёт).

## 5. Чего нельзя

- Тесты, проходящие всегда (`assert True` в конце, smoke-`pass`).
- Тесты, зависящие от порядка запуска или глобального состояния.
- Сравнение float без `pytest.approx` / `assert_allclose`.
- Тесты в корне репозитория или вне `tests/` пакета.
- `print()`-отладка в коммитнутых тестах (caplog/assert — да).

## 6. Единый шаблон pytest.ini

```ini
[pytest]
minversion = 7.0
addopts = -v --tb=short --strict-markers -m "not deprecated"
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*

markers =
    unit: fast isolated tests
    integration: tests touching services/network/files
    slow: > 5s tests (opt-in)
    deprecated: scheduled for removal (excluded by default)
```

Различия между пакетами — только в `testpaths` и дополнительных уже существующих
маркерах. Прогресс унификации существующих файлов — часть TZ-14.
