"""
Ядро: сравнение спетого f0-контура с целевой мелодией.
Реализует контракт формулы, зафиксированный в архитектуре.

ВХОД (транспоз и align считаются УЖЕ отработавшими до этого модуля —
здесь f0_user и f0_target уже приведены по высоте и выровнены по времени,
фрейм в фрейм):
    f0_user     : np.ndarray  — Гц по фреймам, NaN=unvoiced
    f0_target   : np.ndarray  — Гц по фреймам, NaN=unvoiced (стабилизированные ноты)
    conf_user   : np.ndarray  — 0..1 уверенность трекера (Phase 1: единицы)
    conf_target : np.ndarray  — 0..1
    note_bounds : list[(start_frame, end_frame, note_name)] — границы target-нот

ВЫХОД: Result — агрегаты + ПОФРЕЙМОВЫЙ ряд error/масок (нужен графику в Phase 4).

Единицы: всё внутреннее — центы и фреймы. Секунды только на выходе для UI.
"""
from dataclasses import dataclass, field
import numpy as np
from config import CFG


# ---------- элементарные ----------

def error_cents(f0_user: np.ndarray, f0_target: np.ndarray) -> np.ndarray:
    """Знаковое отклонение в центах. минус=ниже цели, плюс=выше.
    NaN там, где любая из сторон unvoiced (сравнивать нечего."""
    with np.errstate(divide='ignore', invalid='ignore'):
        e = 1200.0 * np.log2(f0_user / f0_target)
    return e  # NaN протекает сам, где был NaN во входе


def voiced(f0: np.ndarray) -> np.ndarray:
    """True где есть высота."""
    return ~np.isnan(f0)


# ---------- маски: что участвует в счёте ----------

def transition_mask(note_bounds, n_frames: int) -> np.ndarray:
    """True в ±transition_ms вокруг КАЖДОЙ границы ноты — эти фреймы
    исключаются из intonation (там легитимное портаменто/атака)."""
    half = int(round(CFG.transition_ms / CFG.frame_ms))
    m = np.zeros(n_frames, dtype=bool)
    for (start, end, _name) in note_bounds:
        for edge in (start, end):
            lo = max(0, edge - half)
            hi = min(n_frames, edge + half)
            m[lo:hi] = True
    return m


def vibrato_mask(f0_target: np.ndarray, note_bounds) -> np.ndarray:
    """True на фреймах нот, где ЦЕЛЬ содержит вибрато.
    Phase 1: цель — стабилизированные ноты, поэтому вибрато определяем
    по разбросу внутри сегмента. Реальная логика вынесется в notes.py (Phase 2)."""
    m = np.zeros(len(f0_target), dtype=bool)
    for (start, end, _name) in note_bounds:
        seg = f0_target[start:end]
        seg = seg[~np.isnan(seg)]
        if len(seg) < 3:
            continue
        # размах сегмента в центах относительно медианы
        med = np.median(seg)
        spread = 1200.0 * np.log2(np.max(seg) / np.min(seg))
        if spread >= CFG.vibrato_min_extent_cents and med > 0:
            m[start:end] = True
    return m


def score_mask(f0_user, f0_target, trans_m, vib_m) -> np.ndarray:
    """Фреймы, идущие в intonation: обе стороны voiced, не переход, не вибрато."""
    return voiced(f0_user) & voiced(f0_target) & ~trans_m & ~vib_m


# ---------- агрегаты интонации ----------

def _weighted_fraction(err_abs, w, thr) -> float:
    """Доля веса, где |error| <= thr.
    EPS защищает от floating-point дрожи на границе: ошибка ровно на пороге
    (напр. −100.0000000001 центов от log2) должна считаться попаданием, а не мимо."""
    EPS = 1e-6  # центы; на порядки меньше слышимого, но гасит fp-шум
    total = w.sum()
    if total == 0:
        return float('nan')
    return float((w * (err_abs <= thr + EPS)).sum() / total)


def _weighted_median(values, w) -> float:
    """Взвешенная медиана знаковых значений."""
    if len(values) == 0 or w.sum() == 0:
        return float('nan')
    order = np.argsort(values)
    v, ws = values[order], w[order]
    cum = np.cumsum(ws)
    cutoff = cum[-1] / 2
    return float(v[np.searchsorted(cum, cutoff)])


# ---------- coverage (вторая ось) ----------

def coverage(f0_user, f0_target) -> float:
    """Сколько из того, что надо было спеть, реально спето.
    voiced и там и там / voiced в цели. Отдельно от интонации."""
    tgt = voiced(f0_target)
    if tgt.sum() == 0:
        return float('nan')
    both = voiced(f0_user) & tgt
    return float(both.sum() / tgt.sum())


# ---------- octave flag ----------

def octave_flags(err: np.ndarray) -> np.ndarray:
    """True где |error| близко к 1200 — вероятная октавная ошибка (твоя или трекера).
    НЕ исключаем из счёта, только помечаем."""
    return np.abs(np.abs(err) - 1200.0) <= CFG.octave_tol_cents


# ---------- note-level (для UI) ----------

@dataclass
class NoteResult:
    name: str
    start_s: float
    dur_s: float
    median_cents: float     # знаковое медианное отклонение
    coverage: float         # доля длительности, реально спетая
    correct: bool
    direction: str          # 'ok' | 'low' | 'high' | 'missed'


def note_results(err, f0_user, f0_target, note_bounds) -> list:
    out = []
    for (start, end, name) in note_bounds:
        sl = slice(start, end)
        seg_err = err[sl]
        clean = seg_err[~np.isnan(seg_err)]
        sung = voiced(f0_user[sl])
        cov = float(sung.sum() / max(1, (end - start)))

        if cov < CFG.note_coverage_min or len(clean) == 0:
            out.append(NoteResult(name, start * CFG.frame_ms / 1000,
                                  (end - start) * CFG.frame_ms / 1000,
                                  float('nan'), cov, False, 'missed'))
            continue

        med = float(np.median(clean))
        correct = abs(med) <= CFG.note_correct_cents + 1e-6  # EPS: граница = correct
        if correct:
            direction = 'ok'
        else:
            direction = 'low' if med < 0 else 'high'
        out.append(NoteResult(name, start * CFG.frame_ms / 1000,
                              (end - start) * CFG.frame_ms / 1000,
                              med, cov, correct, direction))
    return out


# ---------- confidence gate ----------

def confidence(cov, conf_user, conf_target, score_m,
               c_align: float = 1.0, c_transpose: float = 1.0) -> float:
    """Overall confidence = MIN компонент (слабое звено рвёт цепь).
    c_align / c_transpose в Phase 1 = 1.0 (эти стадии ещё не подключены)."""
    c_coverage = min(1.0, cov / CFG.coverage_min_for_score) if not np.isnan(cov) else 0.0
    used = score_m
    c_tracker = float(np.mean((conf_user * conf_target)[used])) if used.any() else 0.0
    return float(min(c_coverage, c_tracker, c_align, c_transpose))


# ---------- сборка ----------

@dataclass
class Result:
    # главные цифры
    p50: float
    p25: float
    p100: float
    coverage: float
    bias_cents: float
    conf: float
    low_confidence: bool
    # note-level
    notes: list = field(default_factory=list)
    notes_correct: int = 0
    notes_total: int = 0
    n_low: int = 0
    n_high: int = 0
    n_octave: int = 0
    worst: list = field(default_factory=list)   # (time_s, note, cents)
    # пофреймовый ряд — для графика (Phase 4)
    error_per_frame: np.ndarray = None
    score_mask_per_frame: np.ndarray = None
    octave_mask_per_frame: np.ndarray = None


def analyze(f0_user, f0_target, note_bounds,
            conf_user=None, conf_target=None,
            c_align: float = 1.0, c_transpose: float = 1.0,
            vibrato_mask_ext=None) -> Result:
    """vibrato_mask_ext: готовая per-frame маска вибрато, посчитанная по ИСХОДНОМУ
    контуру (notes.vibrato_frame_mask). Если дана — используется вместо внутренней
    оценки по стабилизированному target (которая не видит вибрато на плоских нотах)."""
    n = len(f0_target)
    if conf_user is None:
        conf_user = np.ones(n)
    if conf_target is None:
        conf_target = np.ones(n)

    err = error_cents(f0_user, f0_target)
    trans_m = transition_mask(note_bounds, n)
    if vibrato_mask_ext is not None:
        vib_m = vibrato_mask_ext[:n]
    else:
        vib_m = vibrato_mask(f0_target, note_bounds)
    score_m = score_mask(f0_user, f0_target, trans_m, vib_m)
    oct_m = octave_flags(err) & score_m

    # веса и очищенные ошибки для intonation
    w = (conf_user * conf_target)[score_m]
    e_clean = err[score_m]
    e_abs = np.abs(e_clean)

    cov = coverage(f0_user, f0_target)
    p25 = _weighted_fraction(e_abs, w, CFG.thr_tight)
    p50 = _weighted_fraction(e_abs, w, CFG.thr_main)
    p100 = _weighted_fraction(e_abs, w, CFG.thr_loose)
    bias = _weighted_median(e_clean, w)

    notes = note_results(err, f0_user, f0_target, note_bounds)
    n_correct = sum(1 for x in notes if x.correct)
    n_low = sum(1 for x in notes if x.direction == 'low')
    n_high = sum(1 for x in notes if x.direction == 'high')

    conf = confidence(cov, conf_user, conf_target, score_m, c_align, c_transpose)
    low = (conf < CFG.gate) or (not np.isnan(cov) and cov < CFG.coverage_min_for_score)

    # worst: топ-N по |median|*sqrt(dur), только некорректные и покрытые
    ranked = sorted(
        [x for x in notes if not x.correct and x.direction != 'missed'],
        key=lambda x: abs(x.median_cents) * (x.dur_s ** 0.5),
        reverse=True,
    )[:CFG.worst_top_n]
    worst = [(x.start_s, x.name, x.median_cents) for x in ranked]

    return Result(
        p50=p50, p25=p25, p100=p100, coverage=cov, bias_cents=bias,
        conf=conf, low_confidence=low,
        notes=notes, notes_correct=n_correct, notes_total=len(notes),
        n_low=n_low, n_high=n_high, n_octave=int(oct_m.sum() > 0),
        worst=worst,
        error_per_frame=err, score_mask_per_frame=score_m,
        octave_mask_per_frame=oct_m,
    )
