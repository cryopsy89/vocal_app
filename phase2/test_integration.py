"""
Integration test: цельный конвейер wav -> track -> resample -> compare
на синтетике с известной ground truth.

Отличие от test_phase2 (там проверялся ТОЛЬКО трекер): здесь проверяется весь
путь, включая ресэмплинг сетки и стыковку с compare.analyze. Заложили смещение
X центов через wav -> ожидаем, что итоговый Result покажет то же X.

Запуск: python3 test_integration.py
"""
import numpy as np
import synth
import pitch
import pipeline
from config import CFG


def _target_from_contour(f0_contour_10ms):
    """Наш ground-truth f0 (сетка 10мс) как target напрямую (он уже стабилизирован)."""
    return f0_contour_10ms


# ---------- ресэмплинг: voiced не течёт через паузу ----------

def test_resample_no_leak_through_silence():
    # нота - пауза - нота. После ресэмплинга пауза должна остаться unvoiced,
    # а не заполниться интерполяцией между двумя нотами.
    f0 = synth.concat(synth.steady('A4', 0.5),
                      synth.silence(0.4),
                      synth.steady('E4', 0.5))
    wav = pitch.synth_wav(f0)
    dur = len(f0) * CFG.frame_ms / 1000
    f0_g, conf_g, ts_g = pipeline.wav_to_f0_grid(wav, pitch.TRACKER_SR, dur)
    # середина (пауза ~0.5-0.9с) должна быть NaN
    mid = (ts_g > 0.55) & (ts_g < 0.85)
    voiced_in_pause = np.sum(~np.isnan(f0_g[mid]))
    print(f'  voiced frames inside pause: {voiced_in_pause} of {mid.sum()}')
    assert voiced_in_pause <= 2, f'{voiced_in_pause} фреймов протекли через паузу'


# ---------- end-to-end: идеальное пение -> ~0 центов ----------

def test_e2e_perfect():
    f0 = synth.steady('A4', 3.0)
    wav = pitch.synth_wav(f0)
    target = _target_from_contour(f0)
    bounds = [(0, len(target), 'A4')]
    r = pipeline.analyze_wavs(wav, pitch.TRACKER_SR, target, bounds)
    print(f'  perfect: p50={r.p50*100:.0f}% bias={r.bias_cents:+.1f}c conf={"LOW" if r.low_confidence else "OK"}')
    assert abs(r.bias_cents) < 15, f'bias {r.bias_cents:.1f}c на идеале — трекер/ресэмпл шумит'
    assert r.p50 > 0.85, f'p50={r.p50:.2f} — слишком много промахов на идеале'
    assert not r.low_confidence


# ---------- end-to-end: смещение -100 центов проходит через весь конвейер ----------

def test_e2e_offset_minus100():
    # target = ровная A4; user = та же нота но -100 центов, через wav.
    target = synth.steady('A4', 3.0)
    user_f0 = synth.steady('A4', 3.0, offset_cents=-100.0)
    user_wav = pitch.synth_wav(user_f0)
    bounds = [(0, len(target), 'A4')]
    r = pipeline.analyze_wavs(user_wav, pitch.TRACKER_SR, target, bounds)
    print(f'  offset -100c: bias={r.bias_cents:+.1f}c (ждём ~-100)  p50={r.p50*100:.0f}%  n_low={r.n_low}')
    # конвейер должен УВИДЕТЬ смещение, не съесть его ресэмплингом
    assert abs(r.bias_cents + 100) < 20, f'bias {r.bias_cents:.1f}c, потеряли смещение'
    assert r.n_low == 1, 'нота не помечена как low'


# ---------- end-to-end: +40 центов, должно быть в допуске ±50 но не ±25 ----------

def test_e2e_offset_plus40():
    target = synth.steady('C4', 3.0)
    user_f0 = synth.steady('C4', 3.0, offset_cents=+40.0)
    user_wav = pitch.synth_wav(user_f0)
    bounds = [(0, len(target), 'C4')]
    r = pipeline.analyze_wavs(user_wav, pitch.TRACKER_SR, target, bounds)
    print(f'  offset +40c: bias={r.bias_cents:+.1f}c p25={r.p25*100:.0f}% p50={r.p50*100:.0f}%')
    assert abs(r.bias_cents - 40) < 20
    assert r.p50 > 0.8, 'должно попадать в ±50'


# ---------- end-to-end: мелодия из нескольких нот с разными промахами ----------

def test_e2e_multi_note():
    notes = ['C4', 'E4', 'G4', 'E4', 'C4']
    offsets = [0, -60, +10, -70, +5]   # E4 промахнуты вниз
    parts_t = [synth.steady(n, 0.7) for n in notes]
    parts_u = [synth.steady(n, 0.7, o) for n, o in zip(notes, offsets)]
    target = synth.concat(*parts_t)
    user_wav = pitch.synth_wav(synth.concat(*parts_u))
    nf = synth._n_frames(0.7)
    bounds = [(i*nf, (i+1)*nf, n) for i, n in enumerate(notes)]
    r = pipeline.analyze_wavs(user_wav, pitch.TRACKER_SR, target, bounds)
    print(f'  multi: {r.notes_correct}/{r.notes_total} correct, LOW={r.n_low} HIGH={r.n_high}, p50={r.p50*100:.0f}%')
    for nr in r.notes:
        print(f'      {nr.name}: med={nr.median_cents:+.0f}c {nr.direction}')
    # две E4 с -60/-70 должны выпасть в low; остальные correct
    assert r.n_low == 2, f'ждём 2 low ноты, получили {r.n_low}'
    assert r.notes_correct == 3


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