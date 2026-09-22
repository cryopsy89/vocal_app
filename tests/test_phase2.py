"""
Phase 2: проверка САМОГО pitch tracker (SwiftF0) на синтетике с известной ground truth.
Генерим wav из известного f0 -> гоняем трекер -> меряем сколько он врёт в центах.

Это НЕ про пользователя и НЕ про формулу (это Phase 1). Здесь один вопрос:
можно ли вообще доверять связке "wav -> f0"? Если трекер сам врёт на чистом
сигнале больше нашего порога интонации, вся оценка бессмысленна.

Запуск: python3 tests/test_phase2.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))
import numpy as np
import synth
import pitch


def _true_f0_on_grid(f0_frames_10ms, ts_tracker):
    """Наш ground-truth f0 задан в сетке 10мс, трекер отдаёт в своей (~16мс).
    Возвращаем истинный f0 в точках трекера — ближайшим по времени фреймом."""
    idx = np.clip((ts_tracker * 1000 / synth.CFG.frame_ms).astype(int),
                  0, len(f0_frames_10ms) - 1)
    return f0_frames_10ms[idx]


def _median_abs_cents(f0_frames_10ms, note='A4', dur=2.0):
    """Прогнать одну ноту через трекер, вернуть медиану |ошибки| в центах
    по voiced-фреймам (исключая края, где атака/затухание)."""
    wav = pitch.synth_wav(f0_frames_10ms)
    f0_tr, conf, ts = pitch.track(wav, pitch.TRACKER_SR)
    true_on_grid = _true_f0_on_grid(f0_frames_10ms, ts)
    err = pitch.hz_to_cents_error(f0_tr, true_on_grid)
    # берём только voiced (не NaN) и отрезаем 3 фрейма с краёв
    voiced = ~np.isnan(err)
    voiced[:3] = False
    voiced[-3:] = False
    return np.median(np.abs(err[voiced])), voiced.sum()


# ---------- трекер на ровной ноте ----------

def test_tracker_steady_note():
    f0 = synth.steady('A4', 2.0)
    med, n = _median_abs_cents(f0)
    print(f'  steady A4: median |err| = {med:.1f}c over {n} voiced frames')
    assert med < 30.0, f'трекер врёт {med:.1f}c на ровной ноте — слишком много'


# ---------- трекер ловит СМЕЩЁННУЮ ноту (не подтягивает к "правильной") ----------

def test_tracker_follows_offset():
    # нота на -100 центов от A4. Трекер должен вернуть ИМЕННО смещённую частоту,
    # а не "исправить" её к A4. Иначе он маскирует фальшь пользователя.
    f0 = synth.steady('A4', 2.0, offset_cents=-100.0)
    wav = pitch.synth_wav(f0)
    f0_tr, conf, ts = pitch.track(wav, pitch.TRACKER_SR)
    true_on_grid = _true_f0_on_grid(f0, ts)
    err = pitch.hz_to_cents_error(f0_tr, true_on_grid)
    voiced = ~np.isnan(err); voiced[:3]=False; voiced[-3:]=False
    med = np.median(np.abs(err[voiced]))
    print(f'  offset -100c: median |err vs true| = {med:.1f}c')
    assert med < 30.0, f'трекер не следует за смещением, врёт {med:.1f}c'


# ---------- трекер на разных высотах (низ/верх диапазона) ----------

def test_tracker_range():
    for note in ['G2', 'C4', 'A4', 'E5', 'C6']:
        f0 = synth.steady(note, 1.5)
        med, n = _median_abs_cents(f0)
        print(f'  {note}: median |err| = {med:.1f}c ({n} frames)')
        assert med < 40.0, f'{note}: {med:.1f}c — трекер плывёт на этой высоте'


# ---------- НЕ должно быть октавных срывов на ровной ноте ----------

def test_no_octave_slips():
    f0 = synth.steady('A4', 2.0)
    wav = pitch.synth_wav(f0)
    f0_tr, conf, ts = pitch.track(wav, pitch.TRACKER_SR)
    true_on_grid = _true_f0_on_grid(f0, ts)
    err = pitch.hz_to_cents_error(f0_tr, true_on_grid)
    voiced = ~np.isnan(err)
    octave_slips = np.sum(np.abs(np.abs(err[voiced]) - 1200) < 100)
    print(f'  octave slips: {octave_slips} of {voiced.sum()} voiced frames')
    assert octave_slips == 0, f'{octave_slips} октавных срывов на ровной ноте'


# ---------- тишина -> unvoiced (низкий confidence) ----------

def test_silence_unvoiced():
    f0 = synth.concat(synth.steady('A4', 0.8),
                      synth.silence(0.6),
                      synth.steady('A4', 0.8))
    wav = pitch.synth_wav(f0)
    f0_tr, conf, ts = pitch.track(wav, pitch.TRACKER_SR)
    # в середине (пауза) confidence должен просесть
    mid = len(ts) // 2
    mid_conf = conf[mid-2:mid+2].mean()
    edge_conf = conf[3:8].mean()
    print(f'  conf: voiced~{edge_conf:.2f}  silence~{mid_conf:.2f}')
    assert mid_conf < edge_conf, 'трекер не отличает тишину от ноты по confidence'


if __name__ == '__main__':
    import sys
    fns = [v for k, v in sorted(globals().items()) if k.startswith('test_')]
    failed = 0
    for fn in fns:
        try:
            print(f'{fn.__name__}:')
            fn()
            print('  ok')
        except AssertionError as e:
            failed += 1
            print(f'  FAIL: {e}')
        except Exception as e:
            failed += 1
            print(f'  ERR {type(e).__name__}: {e}')
    print(f'\n{len(fns)-failed}/{len(fns)} passed')
    sys.exit(1 if failed else 0)
