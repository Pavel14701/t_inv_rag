> **Статус: ✅ реализован** (`.github/workflows/ci.yml`: lint job (ruff + format),
> mypy job (risk/infer/ai strict), EN-only guard job, 8 pytest matrix jobs по пакетам).

# TZ-13. CI (GitHub Actions)

## 1. Контекст

~2200 тестов и линтеры существуют, но не запускаются автоматически. Регресс
монорепозитория (например, разрыв импортов члена workspace) не обнаруживается.

## 2. Требования

1. `.github/workflows/ci.yml`: push/PR на dev/main.
2. Джобы:
   - `lint`: ruff check + ruff format --check (корень).
   - `test-matrix`: матрица по членам workspace — ta, dsl, ai, infer, rag, strategies,
     main: `uv sync --package dte-<x>` → `uv run --package dte-<x> pytest <tests>`.
     ai — с кэшем pip/uv для torch (или отдельный маркер `slow` исключить в CI).
3. mypy: отдельная job (пока не блокирующая — `continue-on-error: true` до приведения).
4. Python 3.12, ubuntu-latest; ta-lib — через apt (`libta-lib`) или wheel.

## 3. Критерии приёмки

- Зелёный прогон на пустом PR без ручных шагов; время CI < 15 мин.
- Красный CI блокирует merge (branch protection — настраивается вручную на GitHub).
