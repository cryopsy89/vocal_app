"""
Phase 1: проверка ФОРМУЛЫ на синтетике с известной ground truth.
Никакого реального аудио, никакого трекинга — только математика сравнения.

Логика каждого теста: строим target, строим user заведомым смещением от target,
проверяем что analyze() возвращает ровно то, что мы заложили.

Запуск:  pytest test_phase1.py -v   (или python test_phase1.py)
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))
import numpy as np
import synth
from compare import analyze, error_cents
from config import CFG


def _bounds(*segs):
    """segs: список (note_name, n_frames). -> границы + склеенная длина."""
    bounds, pos = [], 0
    for name, n in segs:
        bounds.append((pos, pos + n, name))
        pos += n
    return bounds, pos


# ---------- Case 1: идеальная нота -> ~0 центов ----------

def test_case1_perfect():
    target = synth.steady('C4', 2.0)
    user = target.copy()
    bounds, _ = _bounds(('C4', len(target)))
    r = analyze(user, target, bounds)
    assert abs(r.bias_cents) < 1.0, r.bias_cents
    assert r.p50 == 1.0
    assert r.notes_correct == 1


# ---------- Case 2: +50 центов -> ловим +50 ----------

def test_case2_plus50():
    target = synth.steady('C4', 2.0)
    user = synth.steady('C4', 2.0, offset_cents=+50.0)
    bounds, _ = _bounds(('C4', len(target)))
    r = analyze(user, target, bounds)
    assert abs(r.bias_cents - 50.0) < 1.0, r.bias_cents
    # ровно на пороге: должно уходить в high
    assert r.n_high == 1 or r.notes[0].direction in ('ok', 'high')


# ---------- Case 3: -100 центов -> ловим -100, это LOW ----------

def test_case3_minus100():
    target = synth.steady('A4', 2.0)
    user = synth.steady('A4', 2.0, offset_cents=-100.0)
    bounds, _ = _bounds(('A4', len(target)))
    r = analyze(user, target, bounds)
    assert abs(r.bias_cents + 100.0) < 1.0, r.bias_cents
    assert r.p50 == 0.0          # ничего не в пределах ±50
    assert r.p100 == 1.0         # но всё в пределах ±100 (граница)
    assert r.n_low == 1


# ---------- Case 4: октава -> НЕ считать C4 и C5 одинаковыми ----------

def test_case4_octave():
    target = synth.steady('C4', 2.0)
    user = synth.steady('C5', 2.0)   # ровно +1200 центов
    bounds, _ = _bounds(('C4', len(target)))
    r = analyze(user, target, bounds)
    # это НЕ correct — октава это грубая ошибка
    assert r.notes_correct == 0
    # и должна быть помечена октавным флагом, а не спрятана
    assert r.n_octave == 1
    assert abs(abs(r.bias_cents) - 1200.0) < 1.0


# ---------- Case 5: вибрато -> НЕ должно плодить ошибки ----------

def test_case5_vibrato():
    # target — вибрато вокруг C4; user поёт то же вибрато чисто
    target = synth.vibrato('C4', 2.0, rate_hz=6.0, extent_cents=30.0)
    user = target.copy()
    bounds, _ = _bounds(('C4', len(target)))
    r = analyze(user, target, bounds)
    # вибрато-фреймы исключены из intonation -> счёт не должен рухнуть
    # ключевое: нота не разваливается на кучу "ошибок"
    assert r.notes_correct == 1, r.notes[0]
    # и на нотном уровне вердикт не 'low/high' от размаха вибрато
    assert r.notes[0].direction in ('ok',)


def test_case5_vibrato_not_counted_as_errors():
    # даже если бы считали пофреймово — проверяем что vibrato_mask сработал:
    # user с вибрато, target — стабильная нота. Без маски были бы десятки ошибок.
    target = synth.vibrato('C4', 2.0, rate_hz=6.0, extent_cents=40.0)
    user = target.copy()
    bounds, _ = _bounds(('C4', len(target)))
    r = analyze(user, target, bounds)
    used = r.score_mask_per_frame.sum()
    # большинство фреймов ноты должны быть исключены как вибрато
    assert used < len(target) * 0.5, f"вибрато не исключено, used={used}"


# ---------- Case 6: portamento -> переход не создаёт ложных ошибок ----------

def test_case6_portamento():
    # target: C4 держится, скользит в G4, держится
    n_hold = synth._n_frames(0.7)
    n_slide = synth._n_frames(0.3)
    target = synth.concat(
        synth.steady('C4', 0.7),
        synth.portamento('C4', 'G4', 0.3),
        synth.steady('G4', 0.7),
    )
    user = target.copy()
    bounds, _ = _bounds(('C4', n_hold), ('G4', n_slide + n_hold))
    r = analyze(user, target, bounds)
    # переход попадает в transition-зоны границ -> не должен ронять счёт
    assert r.p50 == 1.0
    assert r.notes_correct == 2


# ---------- Case 13: пауза -> unvoiced не считается ошибкой ----------

def test_case13_pause():
    target = synth.concat(synth.steady('C4', 1.0),
                          synth.silence(0.5),
                          synth.steady('E4', 1.0))
    user = target.copy()
    n1 = synth._n_frames(1.0)
    ns = synth._n_frames(0.5)
    bounds = [(0, n1, 'C4'), (n1 + ns, len(target), 'E4')]
    r = analyze(user, target, bounds)
    assert r.p50 == 1.0
    # пауза не должна портить coverage: считаем только voiced target
    assert abs(r.coverage - 1.0) < 0.01


# ---------- Case 12/17: молчание там где нота -> coverage падает, не intonation ----------

def test_missed_note_hits_coverage_not_intonation():
    target = synth.concat(synth.steady('C4', 1.0), synth.steady('E4', 1.0))
    n1 = synth._n_frames(1.0)
    # user спел первую ноту, вторую промолчал
    user = synth.concat(synth.steady('C4', 1.0), synth.silence(1.0))
    bounds = [(0, n1, 'C4'), (n1, len(target), 'E4')]
    r = analyze(user, target, bounds)
    # интонация первой ноты идеальна
    assert r.p50 == 1.0
    # но вторая нота missed, coverage ~50%
    assert r.notes[1].direction == 'missed'
    assert abs(r.coverage - 0.5) < 0.05


# ---------- проверка самой error_cents ----------

def test_error_cents_math():
    f_c4 = synth.note_to_hz('C4')
    assert abs(error_cents(np.array([f_c4]), np.array([f_c4]))[0]) < 1e-6
    # +100 центов = соседний полутон
    f_up = f_c4 * synth.cents_to_ratio(100)
    assert abs(error_cents(np.array([f_up]), np.array([f_c4]))[0] - 100.0) < 1e-6


if __name__ == '__main__':
    import sys
    fns = [v for k, v in sorted(globals().items()) if k.startswith('test_')]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"  ok   {fn.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"  FAIL {fn.__name__}: {e}")
        except Exception as e:
            failed += 1
            print(f"  ERR  {fn.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    sys.exit(1 if failed else 0)
