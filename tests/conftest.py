"""pytest conftest: делает пакет src/ импортируемым из тестов.
Работает и для `pytest`, и для прямого запуска `python3 tests/test_*.py`
(каждый тест-файл тоже добавляет src/ в путь — см. его шапку)."""
import sys
import os

SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src')
if SRC not in sys.path:
    sys.path.insert(0, SRC)
