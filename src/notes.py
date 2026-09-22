"""
Phase 3: note segmentation — превращение непрерывного f0-контура в стабильные
целевые ноты. Это самый рискованный кусок проекта (бриф п.6).

Задача: из дёрганого f0 (вибрато, портаменто, атаки, придыхание) получить
"target notes" — что вокалист ДОЛЖЕН петь, без того чтобы вибрато плодило
десятки ложных нот.

Метод (осознанно простой и настраиваемый, без магии):
  1. f0 (Гц) -> полутоны (непрерывные, дробные).
  2. медианный фильтр по окну — сглаживает вибрато/джиттер, держит сустейн.
  3. сегментация с гистерезисом: новая нота начинается только если сглаженная
     высота стабильно (>= min длительности) отходит от текущей больше порога.
  4. короткие сегменты (< note_min_ms) поглощаются соседями — это украшения/переходы.
  5. центр ноты = медиана f0 по её voiced-фреймам (устойчива к атаке).

⚠️ Границы "нота vs украшение" — НЕ математика, а пороги из config.py. Вынесены
наружу, калибруются. Ничего не захардкожено в логике.
"""
import numpy as np
from dataclasses import dataclass
from config import CFG


def hz_to_semitones(f0):
    """Гц -> полутоны от A4 (дробные). NaN сохраняется."""
    return 12 * np.log2(f0 / 440.0)


def semitones_to_hz(s):
    return 440.0 * 2 ** (s / 12)


def median_smooth(x, win_frames):
    """Медианный фильтр по окну, игнорируя NaN. Сглаживает вибрато, держит ступени."""
    n = len(x)
    out = np.full(n, np.nan)
    half = win_frames // 2
    for i in range(n):
        lo, hi = max(0, i - half), min(n, i + half + 1)
        seg = x[lo:hi]
        seg = seg[~np.isnan(seg)]
        if len(seg) > 0:
            out[i] = np.median(seg)
    return out


def reject_outliers(f0, frame_ms=None):
    """Выкидывает f0-точки, отскочившие от локальной медианы больше порога
    (октавные срывы трекера на согласных/шипящих). Возвращает копию с NaN на выбросах.

    Это НЕ сглаживание: убираем именно грубые одиночные скачки (~пол-октавы+),
    которые медианный фильтр окном 200мс не давит, если их несколько подряд.
    Вибрато (±<1 полутона) этот шаг не трогает."""
    if frame_ms is None:
        frame_ms = CFG.frame_ms
    semis = hz_to_semitones(f0)
    win = max(3, int(round(CFG.outlier_win_ms / frame_ms)))
    local_med = median_smooth(semis, win)
    dev = np.abs(semis - local_med)                      # полутона
    out = f0.copy()
    out[dev > CFG.outlier_semitones] = np.nan            # выброс -> unvoiced
    return out


@dataclass
class Note:
    start: int          # фрейм начала
    end: int            # фрейм конца (не включая)
    hz: float           # центр ноты (Гц)
    semitone: float     # центр в полутонах от A4
    name: str           # ближайшее имя ноты (C4 и т.д.)


_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

def nearest_note_name(hz):
    midi = round(12 * np.log2(hz / 440.0) + 69)
    return _NAMES[midi % 12] + str(midi // 12 - 1)


def segment(f0, frame_ms=None):
    """f0 (Гц по фреймам, NaN=unvoiced) -> список Note.

    Возвращает СТАБИЛИЗИРОВАННЫЕ ноты + границы, готовые как target для compare.
    """
    if frame_ms is None:
        frame_ms = CFG.frame_ms

    semis = hz_to_semitones(f0)
    # Сначала выкидываем грубые октавные выбросы трекера (согласные/шипящие),
    # иначе медиана сегмента и центр ноты уедут.
    f0_clean = reject_outliers(f0, frame_ms)
    semis = hz_to_semitones(f0_clean)
    # Окно сглаживания >= периода вибрато. Вибрато 5-6 Гц => период ~170-200мс,
    # берём окно ~200мс, иначе качание вибрато пробивает порог смены ноты.
    smooth_win = max(1, int(round(CFG.smooth_ms / frame_ms)))
    smooth = median_smooth(semis, smooth_win)

    min_frames = max(1, int(round(CFG.note_min_ms / frame_ms)))
    change_thr = CFG.note_change_semitones   # порог смены ноты в полутонах

    # --- шаг 1: кандидатные границы там, где сглаженная высота уходит > порога ---
    # но смена ПОДТВЕРЖДАЕТСЯ только если новый уровень держится >= min_frames
    # (иначе это проходящий тон/качание вибрато, а не новая нота).
    segments = []
    i = 0
    n = len(f0)
    while i < n:
        if np.isnan(smooth[i]):
            i += 1
            continue
        start = i
        ref = smooth[i]
        i += 1
        while i < n and not np.isnan(smooth[i]):
            if abs(smooth[i] - ref) >= change_thr:
                # потенциальная смена: смотрим, держится ли новый уровень
                j = i
                new_level = smooth[i]
                while (j < n and not np.isnan(smooth[j])
                       and abs(smooth[j] - new_level) < change_thr):
                    j += 1
                if j - i >= min_frames:
                    break          # подтверждённая новая нота -> режем здесь
                else:
                    # короткий заскок (украшение/качание) — не режем, идём дальше
                    i = j
            else:
                ref = np.median(smooth[start:i + 1])   # обновляем центр по накопленному
                i += 1
        segments.append((start, i))

    # --- шаг 2: короткие сегменты -> украшения, поглощаются (пропускаем как ноты) ---
    notes = []
    for (s, e) in segments:
        if e - s < min_frames:
            continue   # слишком коротко для целевой ноты (глиссандо/украшение)
        seg_f0 = f0_clean[s:e]                     # по ОЧИЩЕННОМУ (без выбросов) контуру
        seg_f0 = seg_f0[~np.isnan(seg_f0)]
        if len(seg_f0) == 0:
            continue
        center_hz = float(np.median(seg_f0))      # центр = медиана (устойчива к атаке/вибрато)
        center_st = float(hz_to_semitones(np.array([center_hz]))[0])
        notes.append(Note(s, e, center_hz, center_st, nearest_note_name(center_hz)))

    return notes


def vibrato_frame_mask(f0, note_bounds, frame_ms=None):
    """Per-frame маска вибрато по ИСХОДНОМУ контуру (до стабилизации).

    Фрейм = вибрато, если в окне ~smooth_ms вокруг него (внутри своей ноты) сырой
    f0 колеблется с размахом в диапазоне [vibrato_min, vibrato_max] центов:
      - меньше vibrato_min — это стабильная нота (не вибрато);
      - больше vibrato_max — это переход/глиссандо/смена ноты (не вибрато).
    Вибрато-фреймы потом исключаются из intonation (нельзя честно оценить
    пофреймово участок, который эталон сам качал вокруг центра).

    Работает по ОЧИЩЕННОМУ от выбросов контуру, иначе октавный глюк раздует размах."""
    if frame_ms is None:
        frame_ms = CFG.frame_ms
    clean = reject_outliers(f0, frame_ms)
    semis = hz_to_semitones(clean)
    win = max(3, int(round(CFG.smooth_ms / frame_ms)))
    half = win // 2
    vmin = CFG.vibrato_min_extent_cents / 100.0     # полутона
    vmax = CFG.vibrato_max_extent_cents / 100.0
    mask = np.zeros(len(f0), dtype=bool)
    for (s, e, _name) in note_bounds:
        for i in range(s, e):
            lo, hi = max(s, i - half), min(e, i + half + 1)
            seg = semis[lo:hi]
            seg = seg[~np.isnan(seg)]
            if len(seg) < 3:
                continue
            extent = np.max(seg) - np.min(seg)      # полутона
            if vmin <= extent <= vmax:
                mask[i] = True
    return mask


def notes_to_target(notes, n_frames, frame_ms=None):
    """Список Note -> (f0_target СТУПЕНЬКИ, note_bounds).
    ВНИМАНИЕ: ступеньки (плоский центр ноты) — ТОЛЬКО для note-level вердикта и UI
    (кликабельные блоки в training). Для пофреймового расчёта центов НЕЛЬЗЯ:
    плоский центр против живого контура штрафует естественный дрейф/вибрато
    (потолок ~83% на копии-vs-копии). Для расчёта используй smoothed_target()."""
    f0_target = np.full(n_frames, np.nan)
    bounds = []
    for nt in notes:
        f0_target[nt.start:nt.end] = nt.hz
        bounds.append((nt.start, nt.end, nt.name))
    return f0_target, bounds


def prepare_contour(f0, frame_ms=None):
    """Единый препроцессинг контура для СРАВНЕНИЯ: отброс выбросов -> сглаживание.
    Применяется ОДИНАКОВО к target и к user, иначе вибрато одного бьёт по счёту.
    Возвращает сглаженный f0 (Гц), NaN сохраняется."""
    if frame_ms is None:
        frame_ms = CFG.frame_ms
    clean = reject_outliers(f0, frame_ms)
    semis = hz_to_semitones(clean)
    smooth = median_smooth(semis, max(1, int(round(CFG.smooth_ms / frame_ms))))
    return semitones_to_hz(smooth)


def smoothed_target(f0, notes, n_frames, frame_ms=None):
    """f0-target для РАСЧЁТА ЦЕНТОВ: сглаженный контур эталона, но только внутри нот
    (между нотами — NaN, там петь нечего). Сравнение формы-с-формой, а не с плоским
    центром: копия-vs-копия даёт ~100%, реальная фальшь ловится честно."""
    contour = prepare_contour(f0, frame_ms)
    in_note = np.zeros(n_frames, dtype=bool)
    bounds = []
    for nt in notes:
        in_note[nt.start:nt.end] = True
        bounds.append((nt.start, nt.end, nt.name))
    out = contour.copy()
    out[~in_note] = np.nan
    return out, bounds
