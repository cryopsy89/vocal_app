"""
Генератор синтетических f0-контуров с ИЗВЕСТНОЙ ground truth.

Смысл Phase 1: не петь, а сгенерировать сигнал, про который мы точно знаем
правильный ответ (нота ровно -100 центов, вибрато ±30 центов на 6 Гц и т.д.),
и проверить, что формула из compare.py возвращает именно это.

Работаем сразу в области f0 (Гц), а не в waveform: Phase 1 проверяет
СРАВНЕНИЕ, а не pitch tracking. Трекер придёт в Phase 2.

Все контуры — numpy-массивы f0 по фреймам. NaN = unvoiced (нет высоты).
"""
import numpy as np
from config import CFG


# --- перевод нот в частоту ---

_A4_HZ = 440.0
_A4_MIDI = 69

def note_to_hz(name: str) -> float:
    """'C4' -> 261.63 Гц. Поддержка # и b."""
    names = {'C': 0, 'D': 2, 'E': 4, 'F': 5, 'G': 7, 'A': 9, 'B': 11}
    n = name[0].upper()
    i = 1
    semi = names[n]
    if i < len(name) and name[i] in ('#', 'b'):
        semi += 1 if name[i] == '#' else -1
        i += 1
    octave = int(name[i:])
    midi = semi + (octave + 1) * 12   # C4 = MIDI 60
    return _A4_HZ * 2 ** ((midi - _A4_MIDI) / 12)


def cents_to_ratio(cents: float) -> float:
    """Отклонение в центах -> множитель частоты."""
    return 2 ** (cents / 1200)


def _n_frames(dur_s: float) -> int:
    return int(round(dur_s * 1000 / CFG.frame_ms))


# --- строительные блоки контуров ---

def steady(note: str, dur_s: float, offset_cents: float = 0.0) -> np.ndarray:
    """Ровная нота, опционально смещённая на offset_cents от идеала."""
    hz = note_to_hz(note) * cents_to_ratio(offset_cents)
    return np.full(_n_frames(dur_s), hz, dtype=float)


def silence(dur_s: float) -> np.ndarray:
    """Unvoiced: NaN."""
    return np.full(_n_frames(dur_s), np.nan, dtype=float)


def vibrato(note: str, dur_s: float, rate_hz: float = 6.0,
            extent_cents: float = 30.0) -> np.ndarray:
    """Нота с вибрато: синус вокруг центра ноты. Центр = идеальная высота."""
    n = _n_frames(dur_s)
    t = np.arange(n) * CFG.frame_ms / 1000
    center = note_to_hz(note)
    dev = extent_cents * np.sin(2 * np.pi * rate_hz * t)  # центы
    return center * cents_to_ratio(dev)


def portamento(note_from: str, note_to: str, dur_s: float) -> np.ndarray:
    """Плавный скользящий переход между двумя нотами (в лог-области)."""
    n = _n_frames(dur_s)
    lo = np.log2(note_to_hz(note_from))
    hi = np.log2(note_to_hz(note_to))
    return 2 ** np.linspace(lo, hi, n)


def concat(*parts: np.ndarray) -> np.ndarray:
    """Склейка блоков в один контур."""
    return np.concatenate(parts)
