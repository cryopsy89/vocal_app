"""
Phase 2 -> integration: цельный конвейер wav -> track -> compare.
Плюс честный ресэмплинг сетки трекера (~16 мс) под сетку compare (CFG.frame_ms, 10 мс).

Почему ресэмплинг нетривиален:
  f0 НЕЛЬЗЯ интерполировать линейно через границы voiced/unvoiced. Между 200 Гц
  и NaN линейная интерполяция даст мусорную частоту на переходе. Поэтому:
  - интерполируем в ЛОГ-области (полутоны/центы) — слух логарифмичен, и это корректно;
  - voiced-маску переносим ОТДЕЛЬНО: целевой фрейм voiced только если он опирается
    на voiced-соседей трекера (не «дотянут» из тишины).
"""
import numpy as np
import soundfile as sf
from config import CFG
import pitch
import compare
import notes as notes_mod
import align as align_mod


# ---------- ресэмплинг f0 на сетку CFG.frame_ms ----------

def resample_f0(f0_src, ts_src, dur_s, frame_ms=None, voiced_gap_ms=None):
    """f0 трекера (в его сетке ts_src) -> f0 на равномерной сетке frame_ms.

    f0_src: Гц, NaN=unvoiced.  ts_src: времена фреймов трекера (сек).
    Возвращает (f0_dst, ts_dst) на сетке frame_ms от 0 до dur_s.

    Логика:
      - voiced-точки трекера переводим в полутоны, линейно интерполируем В ЛОГ-области;
      - целевой фрейм считаем voiced только если он лежит внутри voiced-сегмента
        трекера и не дальше voiced_gap_ms от реальной voiced-точки (не тянем из тишины).
    """
    if frame_ms is None:
        frame_ms = CFG.frame_ms
    if voiced_gap_ms is None:
        voiced_gap_ms = 1.5 * (ts_src[1] - ts_src[0]) * 1000 if len(ts_src) > 1 else frame_ms

    n_dst = int(round(dur_s * 1000 / frame_ms))
    ts_dst = np.arange(n_dst) * frame_ms / 1000

    voiced_src = ~np.isnan(f0_src)
    if voiced_src.sum() < 2:
        return np.full(n_dst, np.nan), ts_dst

    # интерполяция в полутонах по voiced-точкам
    semis_src = 12 * np.log2(f0_src[voiced_src] / 440.0)
    t_voiced = ts_src[voiced_src]
    semis_dst = np.interp(ts_dst, t_voiced, semis_src)   # линейно в лог-области
    f0_dst = 440.0 * 2 ** (semis_dst / 12)

    # voiced-маска: целевой фрейм voiced, только если ближайшая voiced-точка трекера
    # не дальше voiced_gap_ms (иначе это интерполяция через паузу -> unvoiced)
    gap_s = voiced_gap_ms / 1000
    nearest_dist = np.min(np.abs(ts_dst[:, None] - t_voiced[None, :]), axis=1)
    f0_dst[nearest_dist > gap_s] = np.nan
    return f0_dst, ts_dst


def resample_confidence(conf_src, ts_src, ts_dst):
    """Confidence на целевую сетку — ИСТИННО ближайшим соседом (не интерполируем,
    это вероятность, а не частота).

    searchsorted даёт индекс ВСТАВКИ (правый сосед), из-за чего near-boundary фрейму
    доставалась бы уверенность соседнего сегмента (напр. конца паузы вместо ноты) и
    искажался бы LOW CONFIDENCE. Поэтому сравниваем расстояние до левого и правого
    соседа и берём реально ближайший."""
    if len(ts_src) == 1:
        return np.full(len(ts_dst), conf_src[0])
    right = np.clip(np.searchsorted(ts_src, ts_dst), 0, len(ts_src) - 1)
    left = np.clip(right - 1, 0, len(ts_src) - 1)
    d_right = np.abs(ts_src[right] - ts_dst)
    d_left = np.abs(ts_src[left] - ts_dst)
    idx = np.where(d_left <= d_right, left, right)
    return conf_src[idx]


# ---------- цельный конвейер ----------

def wav_to_f0_grid(wav, sr, dur_s=None):
    """wav -> (f0, conf) на сетке CFG.frame_ms. Готово к подаче в compare.analyze."""
    if dur_s is None:
        dur_s = len(wav) / sr
    f0_src, conf_src, ts_src = pitch.track(wav, sr)
    f0_dst, ts_dst = resample_f0(f0_src, ts_src, dur_s)
    conf_dst = resample_confidence(conf_src, ts_src, ts_dst)
    conf_dst[np.isnan(f0_dst)] = 0.0     # unvoiced -> нулевая уверенность
    return f0_dst, conf_dst, ts_dst


def analyze_wavs(user_wav, sr_user, f0_target, note_bounds,
                 conf_target=None):
    """END-TO-END: пользовательский wav + УЖЕ готовая target-мелодия (f0 по нашей
    сетке) -> Result. target приходит из Phase 3 (пока в тестах задаём вручную).

    ВАЖНО: транспоз и align тут НЕ делаются — они между Phase 3 и этим вызовом.
    Здесь предполагается, что user и target уже в одной тональности и выровнены."""
    dur = len(f0_target) * CFG.frame_ms / 1000
    f0_user, conf_user, _ = wav_to_f0_grid(user_wav, sr_user, dur_s=dur)
    # выровнять длины (ресэмплинг может дать ±1 фрейм)
    n = min(len(f0_user), len(f0_target))
    f0_user, conf_user = f0_user[:n], conf_user[:n]
    f0_target = f0_target[:n]
    if conf_target is not None:
        conf_target = conf_target[:n]
    note_bounds = [(s, min(e, n), nm) for (s, e, nm) in note_bounds if s < n]
    return compare.analyze(f0_user, f0_target, note_bounds,
                           conf_user=conf_user, conf_target=conf_target)


def score_take(f0_target_raw, f0_user_raw, conf_user=None, conf_target=None,
               do_align=True):
    """ПОЛНЫЙ путь сравнения (Phase 3): сырой target-f0 + сырой user-f0 -> Result.

    Шаги:
      1. сегментация target -> ноты (для note-вердикта и UI-границ);
      2. smoothed_target: сглаженный контур эталона внутри нот (для расчёта центов);
      3. user проходит тот же prepare_contour (форма-с-формой);
      4. DTW выравнивает user к target по времени;
      5. compare.analyze без внутренней вибрато-маски (вибрато уже погашено сглаживанием).

    Транспоз НЕ делается (кейс Виктора — один голос). Если нужен — до этого вызова.
    """
    n = len(f0_target_raw)
    seg = notes_mod.segment(f0_target_raw)
    tgt, bounds = notes_mod.smoothed_target(f0_target_raw, seg, n)

    user = notes_mod.prepare_contour(f0_user_raw)
    # выровнять длины
    m = min(len(user), n)
    user, tgt = user[:m], tgt[:m]
    bounds = [(s, min(e, m), nm) for (s, e, nm) in bounds if s < m]

    c_align = 1.0
    if do_align:
        user, ncost = align_mod.warp_user_to_target(tgt, user)
        c_align = align_mod.align_confidence(ncost)

    no_vib = np.zeros(m, dtype=bool)   # target уже сглажен — отдельная маска не нужна
    if conf_user is not None:
        conf_user = conf_user[:m]
    if conf_target is not None:
        conf_target = conf_target[:m]
    return compare.analyze(user, tgt, bounds, conf_user=conf_user,
                           conf_target=conf_target, c_align=c_align,
                           vibrato_mask_ext=no_vib)


def load_wav_mono(path, start_s=None, dur_s=None):
    """Читает wav/mp3 -> (mono float32, sr)."""
    info = sf.info(path)
    kw = {}
    if start_s is not None:
        kw['start'] = int(start_s * info.samplerate)
    if dur_s is not None:
        kw['frames'] = int(dur_s * info.samplerate)
    y, sr = sf.read(path, **kw)
    if y.ndim > 1:
        y = y.mean(axis=1)
    return y.astype(np.float32), sr


def wav_path_to_f0(path, start_s=None, dur_s=None):
    """wav/mp3 файл (+ опц. участок) -> (f0, conf) на сетке CFG.frame_ms."""
    y, sr = load_wav_mono(path, start_s=start_s, dur_s=dur_s)
    seg_dur = len(y) / sr
    f0, conf, _ts = wav_to_f0_grid(y, sr, dur_s=seg_dur)
    return f0, conf


def score_wavs(target_path, user_path,
               target_start=None, target_dur=None,
               user_start=None, user_dur=None):
    """Движок для UI: два wav-файла (эталон + дубль) + опциональные участки -> Result.

    Участки задаются в секундах (start, dur) для каждого файла отдельно — эталон и
    дубль могут стоять на разных таймкодах. DTW потом выравнивает их внутри участка.
    UI вызывает эту функцию; сам UI логику не содержит.
    """
    f0_t, conf_t = wav_path_to_f0(target_path, target_start, target_dur)
    f0_u, conf_u = wav_path_to_f0(user_path, user_start, user_dur)
    return score_take(f0_t, f0_u, conf_user=conf_u, conf_target=conf_t, do_align=True)
