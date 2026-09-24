"""
Транспоз — определение и снятие общего высотного сдвига между оригиналом и дублем.

Зачем: если дубль/кавер спет в другой тональности или октаве (мужик поёт женское
на -12), scoring без транспоза покажет ложный полный провал (весь вокал "мимо").
Транспоз убирает ОБЩИЙ сдвиг, оставляя только реальные промахи.

Порядок в пайплайне жёсткий (Phase 0): транспоз ДО align. Сначала снимаем сдвиг
по высоте, потом DTW выравнивает по времени. Иначе DTW на несовпадающих по высоте
контурах наделает бреда.

Ключ (semitones):
  - Auto: алгоритм определяет сдвиг сам (estimate_shift).
  - Ручной: пользователь задаёт сдвиг в полутонах (-12..+12), ПРИОРИТЕТ над auto
    (бриф п.7 — ручное значение всегда важнее авто).
Октава = частный случай ключа (-12 / +12), отдельной логики не нужно.
"""
import numpy as np
from config import CFG


def apply_shift(f0, semitones):
    """Сдвигает контур f0 на semitones полутонов. NaN сохраняется."""
    if semitones == 0:
        return f0.copy()
    return f0 * (2 ** (semitones / 12))


def _median_semitone(f0):
    """Медианная высота контура в полутонах от A4 (по voiced-фреймам)."""
    v = f0[~np.isnan(f0)]
    if len(v) == 0:
        return None
    return float(np.median(12 * np.log2(v / 440.0)))


def estimate_shift(f0_target, f0_user, search_semitones=None):
    """AUTO-определение сдвига дубля относительно оригинала, в ЦЕЛЫХ полутонах.

    Метод: перебор кандидатов сдвига (в т.ч. ±октавы), для каждого — насколько
    совпадают РАСПРЕДЕЛЕНИЯ высот после сдвига (по voiced-фреймам, без выравнивания
    по времени — нам нужен только высотный сдвиг). Берём сдвиг с минимальным
    рассогласованием медиан + формы. Возвращает (best_shift, confidence).

    confidence 0..1: насколько уверенно один сдвиг выделяется среди других.
    Низкая уверенность -> честно сказать пользователю "проверь ключ вручную".

    НЕ определяет дробный тюнинг — это делает scoring (bias). Здесь только целые
    полутоны тональности/октавы.
    """
    if search_semitones is None:
        search_semitones = CFG.transpose_search_semitones

    mt = _median_semitone(f0_target)
    mu = _median_semitone(f0_user)
    if mt is None or mu is None:
        return 0, 0.0

    # грубая оценка от медиан -> старт поиска вокруг неё
    rough = mu - mt   # на сколько полутонов user выше target (в среднем)
    center = int(round(rough))

    # кандидаты: вокруг грубой оценки + явные октавы от неё
    cands = set()
    for d in range(-search_semitones, search_semitones + 1):
        cands.add(center + d)
    for octv in (-24, -12, 0, 12, 24):
        cands.add(center + octv)
    cands = sorted(c for c in cands if -CFG.transpose_max <= c <= CFG.transpose_max)

    tgt_semi = 12 * np.log2(f0_target[~np.isnan(f0_target)] / 440.0)
    usr_semi = 12 * np.log2(f0_user[~np.isnan(f0_user)] / 440.0)

    # Метрика на кандидат: доля user-фреймов, которые после сдвига попадают близко
    # (в пределах tol полутона) к КАКОЙ-НИБУДЬ target-высоте. Прямо меряет "ноты
    # совпали", устойчива к одной фальшивой ноте и неравномерности мелодии —
    # в отличие от гистограммы, которую сбивает распределение длительностей.
    # Считаем расстояние каждого сдвинутого user-фрейма до ближайшей target-высоты.
    tgt_sorted = np.sort(tgt_semi)
    tol = 0.5   # полутона (±50 центов) — фрейм "попал" в целевую высоту

    def frac_matched(shift):
        shifted = usr_semi - shift
        # ближайшая target-высота для каждого user-фрейма
        idx = np.searchsorted(tgt_sorted, shifted)
        idx = np.clip(idx, 1, len(tgt_sorted) - 1)
        left = tgt_sorted[idx - 1]
        right = tgt_sorted[np.clip(idx, 0, len(tgt_sorted) - 1)]
        dist = np.minimum(np.abs(shifted - left), np.abs(shifted - right))
        return float(np.mean(dist <= tol))

    scores = {c: frac_matched(c) for c in cands}
    best = max(scores, key=scores.get)
    vals = sorted(scores.values(), reverse=True)
    if len(vals) >= 2 and vals[0] > 0:
        conf = float((vals[0] - vals[1]) / vals[0])
    else:
        conf = 1.0 if vals and vals[0] > 0 else 0.0
    return int(best), conf


def transpose_user(f0_target, f0_user, key='auto'):
    """Приводит user к тональности target.

    key:
      'auto' -> определить сдвиг автоматически (estimate_shift)
      int    -> ручной сдвиг в полутонах, ПРИОРИТЕТ над auto

    Возвращает (f0_user_transposed, applied_shift, confidence).
    Для ручного ключа confidence = 1.0 (пользователь знает лучше)."""
    if key == 'auto':
        shift, conf = estimate_shift(f0_target, f0_user)
        # КРИТИЧНО: если auto не уверен — НЕ применять транспоз. Низкая уверенность
        # значит, что явного целого сдвига тональности нет, а есть мелкая систематика,
        # которая может быть РЕАЛЬНОЙ фальшью певца (стабильно занижает). Съесть её
        # транспозом = замаскировать фальшь. Оставляем 0, пусть scoring покажет bias.
        if conf < CFG.transpose_min_conf:
            return f0_user.copy(), 0, conf
    else:
        shift, conf = int(key), 1.0
    # приводим user К target: вычитаем найденный сдвиг
    return apply_shift(f0_user, -shift), shift, conf
