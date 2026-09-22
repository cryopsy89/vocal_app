"""
Phase 3: DTW alignment — выравнивание user-мелодии к target по ВРЕМЕНИ.

Зачем: живой дубль не совпадает со студийным эталоном по секундам — вступил
раньше/позже, темп чуть плывёт. Нельзя сравнивать фрейм-в-фрейм без выравнивания,
иначе задержка вокала превратится в кучу ложных pitch-ошибок (бриф п.8).

Метод: DTW по КОНТУРУ ПОЛУТОНОВ (не по waveform, не по MFCC — они реагируют на
тембр/аранжировку). Сравниваем мелодические контуры. Ограничение Sakoe-Chiba band —
макс. рассинхрон, чтобы DTW не убегал на повторяющихся мотивах.

Транспоз тут НЕ делается (в кейсе Виктора один голос). Если понадобится —
транспонировать надо ДО align (иначе контуры не совпадут по высоте).

Выход: user-мелодия, ресэмплированная в сетку target (фрейм-в-фрейм), готовая
для compare.analyze.
"""
import numpy as np
from config import CFG


def _to_semitones(f0):
    """Гц -> полутоны от A4. NaN -> специальное значение для DTW (см. ниже)."""
    return 12 * np.log2(f0 / 440.0)


def _local_cost(a, b, unvoiced_pen):
    """Матрица стоимостей |a_i - b_j| в полутонах.
    NaN (unvoiced) с любой стороны -> фиксированный штраф unvoiced_pen,
    чтобы паузы матчились с паузами, а не давали бесконечную стоимость."""
    A = a[:, None]
    B = b[None, :]
    cost = np.abs(A - B)
    nan_mask = np.isnan(A) | np.isnan(B)
    both_nan = np.isnan(A) & np.isnan(B)
    cost[nan_mask] = unvoiced_pen     # voiced vs unvoiced — штраф
    cost[both_nan] = 0.0              # пауза против паузы — ок
    return cost


def dtw_path(target_st, user_st, band_frames=None, unvoiced_pen=None):
    """DTW с Sakoe-Chiba band. Возвращает путь [(i_target, j_user), ...].

    band_frames: АБСОЛЮТНАЯ ширина коридора в фреймах (макс. рассинхрон). НЕ доля
    длины — иначе на длинном фрагменте DTW уходит на секунды и маскирует
    систематический pitch-сдвиг, пересопоставляя чужие ноты по высоте.
    """
    from config import CFG
    if band_frames is None:
        band_frames = max(1, int(round(CFG.dtw_band_ms / CFG.frame_ms)))
    if unvoiced_pen is None:
        unvoiced_pen = CFG.dtw_unvoiced_penalty
    n, m = len(target_st), len(user_st)
    cost = _local_cost(target_st, user_st, unvoiced_pen)

    band = band_frames
    INF = np.inf
    D = np.full((n + 1, m + 1), INF)
    D[0, 0] = 0.0

    for i in range(1, n + 1):
        # ограничиваем j коридором вокруг диагонали
        j_center = int(i * m / n)
        j_lo = max(1, j_center - band)
        j_hi = min(m, j_center + band)
        for j in range(j_lo, j_hi + 1):
            c = cost[i - 1, j - 1]
            D[i, j] = c + min(D[i - 1, j],      # user отстаёт (растяжение target)
                              D[i, j - 1],      # user спешит (растяжение user)
                              D[i - 1, j - 1])  # шаг по диагонали

    # обратный проход — восстановление пути
    path = []
    i, j = n, m
    while i > 0 and j > 0:
        path.append((i - 1, j - 1))
        step = np.argmin([D[i - 1, j - 1], D[i - 1, j], D[i, j - 1]])
        if step == 0:
            i, j = i - 1, j - 1
        elif step == 1:
            i -= 1
        else:
            j -= 1
    path.reverse()
    return path, D[n, m]


def warp_user_to_target(f0_target, f0_user, band_frames=None):
    """Выравнивает user к сетке target по DTW-пути.
    Возвращает f0_user_aligned той же длины что f0_target (фрейм-в-фрейм).

    КРИТИЧНО: путь ищется по ДЕ-МЕАНИРОВАННЫМ контурам (у каждого вычтена своя
    медианная высота) — DTW матчит по ФОРМЕ мелодии (ходы вверх-вниз), а не по
    абсолютной высоте. Иначе систематический pitch-сдвиг (напр. весь дубль на -60c)
    DTW маскирует, пересопоставляя чужие ноты ради сближения высот. А warp
    применяется к ИСХОДНОМУ f0_user, поэтому сдвиг сохраняется и ловится scoring.

    Для каждого target-фрейма берём f0_user из сопоставленного пути.
    Если одному target соответствует несколько user-фреймов — берём медиану (voiced)."""
    t_st = _to_semitones(f0_target)
    u_st = _to_semitones(f0_user)
    # де-меанирование ТОЛЬКО для поиска пути (не для scoring)
    t_dm = t_st - np.nanmedian(t_st)
    u_dm = u_st - np.nanmedian(u_st)
    path, dist = dtw_path(t_dm, u_dm, band_frames=band_frames)

    # для каждого i (target) собираем сопоставленные j (user)
    mapping = {}
    for (i, j) in path:
        mapping.setdefault(i, []).append(j)

    n = len(f0_target)
    aligned = np.full(n, np.nan)
    for i in range(n):
        js = mapping.get(i, [])
        vals = f0_user[js]
        vals = vals[~np.isnan(vals)]
        if len(vals) > 0:
            aligned[i] = np.median(vals)
    # нормируем стоимость пути на длину — для confidence align
    norm_cost = dist / max(len(path), 1)
    return aligned, norm_cost


def align_confidence(norm_cost, good=1.0, bad=4.0):
    """norm_cost (средняя стоимость DTW-пути в полутонах) -> 0..1.
    good: путь почти идеальный (контуры совпали). bad: рассинхрон/несовпадение.
    Идёт в общий confidence как C_align."""
    return float(np.clip((bad - norm_cost) / (bad - good), 0.0, 1.0))
