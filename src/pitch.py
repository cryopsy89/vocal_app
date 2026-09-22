"""
Phase 2: реальный pitch tracking через SwiftF0 + генератор синтетического wav.

Два блока:
1. synth_wav — превращает ИЗВЕСТНЫЙ f0-контур (Гц по фреймам) в waveform.
   Нужен, чтобы проверить трекер на честной ground truth: подаём wav, знаем
   правильный f0, смотрим что трекер вернул то же самое.
2. track — обёртка над SwiftF0: wav -> (f0_hz, confidence, timestamps) по фреймам.

SwiftF0: MIT, CPU-only, per-frame confidence, диапазон G1..C7, работает на 16кГц
(ресемплит сам). Выбран в decisions.md после currency-check (сентябрь 2026).

Выход track() совместим со входом compare.analyze(): f0 в Гц (NaN=unvoiced),
confidence 0..1. Трекер — сменная деталь за этим интерфейсом.
"""
import numpy as np
from config import CFG

TRACKER_SR = 16000   # SwiftF0 работает на 16кГц


# ---------- генератор синтетического wav из известного f0 ----------

def synth_wav(f0_frames: np.ndarray, sr: int = TRACKER_SR,
              n_harmonics: int = 5, frame_ms: float = None) -> np.ndarray:
    """f0-контур (Гц по фреймам, NaN=тишина) -> waveform (float32, [-1,1]).

    Пилим не чистый синус, а несколько гармоник — это ближе к голосу и честнее
    для трекера (на чистом синусе легко, на гармониках вылезают октавные срывы).
    Фаза интегрируется непрерывно, чтобы не было щелчков на стыках фреймов.
    """
    if frame_ms is None:
        frame_ms = CFG.frame_ms
    samples_per_frame = int(round(sr * frame_ms / 1000))

    # разворачиваем пофреймовый f0 в пофреймовый на уровне сэмплов
    f0_samples = np.repeat(f0_frames, samples_per_frame)
    voiced = ~np.isnan(f0_samples)
    f_inst = np.where(voiced, f0_samples, 0.0)

    # непрерывная фаза = кумулятивный интеграл мгновенной частоты
    phase = np.cumsum(2 * np.pi * f_inst / sr)

    wav = np.zeros(len(f0_samples), dtype=np.float64)
    # спад амплитуды гармоник 1/k — грубая модель голосового спектра
    for k in range(1, n_harmonics + 1):
        wav += (1.0 / k) * np.sin(k * phase)

    wav[~voiced] = 0.0                      # тишина там где unvoiced
    peak = np.max(np.abs(wav)) or 1.0
    return (0.9 * wav / peak).astype(np.float32)


# ---------- обёртка над SwiftF0 ----------

def track(wav: np.ndarray, sr: int, voiced_thr: float = 0.5):
    """wav -> (f0_hz, confidence, timestamps).
    f0_hz: NaN там где confidence < voiced_thr (SwiftF0 не отдаёт готовый
    voiced-флаг, выводим сами из confidence — README: voiced when >= 0.5).
    Массивы в сетке SwiftF0 (~16 мс, НЕ наши 10 мс — ресэмплинг под CFG.frame_ms
    сделаем при интеграции в compare; здесь сырой выход трекера)."""
    from swift_f0 import SwiftF0
    detector = SwiftF0()
    res = detector.detect(wav, sr)
    f0 = np.asarray(res.pitch_hz, dtype=float)
    conf = np.asarray(res.confidence, dtype=float)
    ts = np.asarray(res.timestamps, dtype=float)
    f0_masked = np.where(conf >= voiced_thr, f0, np.nan)   # unvoiced -> NaN
    return f0_masked, conf, ts


def hz_to_cents_error(f0_tracked: np.ndarray, f0_true: np.ndarray) -> np.ndarray:
    """Отклонение трекера от истины в центах, по совпадающим voiced-фреймам.
    Для оценки САМОГО трекера (не пользователя)."""
    with np.errstate(divide='ignore', invalid='ignore'):
        return 1200.0 * np.log2(f0_tracked / f0_true)
