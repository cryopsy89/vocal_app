"""
Тесты краевых случаев и устойчивости к битому входу.
Главная цель: НЕ выдавать красивую цифру там, где оценивать нечего
(принцип "честность важнее красивого UX" из брифа п.17).

Запуск: python3 tests/test_edge_cases.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))
import numpy as np
import warnings
import compare
import pipeline
import align
import notes


# ---------- ложные высокие оценки на пустом входе ----------

def test_no_notes_is_low_confidence():
    # нет ни одной target-ноты -> оценивать нечего -> LOW CONFIDENCE
    r = compare.analyze(np.full(100, 261.6), np.full(100, 261.6), [])
    assert r.low_confidence, 'пустой note_bounds должен давать LOW CONFIDENCE'


def test_almost_no_singing_is_low_confidence():
    # спел ~2% песни -> нельзя говорить "100%"
    target = np.full(500, 261.6)
    user = np.concatenate([np.full(10, 261.6), np.full(490, np.nan)])
    r = pipeline.score_take(target, user)
    assert r.low_confidence, 'при мизерном покрытии должно быть LOW CONFIDENCE'
    assert r.coverage < 0.1


def test_all_silence_user_low_confidence():
    # молчал всю песню -> LOW, не 100%
    r = compare.analyze(np.full(200, np.nan), np.full(200, 261.6), [(0, 200, 'C4')])
    assert r.low_confidence


# ---------- битый вход ----------

def test_length_mismatch_raises_clearly():
    # разная длина user/target -> явная ValueError с внятным текстом, не broadcast-фейл
    try:
        compare.analyze(np.full(50, 261.6), np.full(100, 261.6), [(0, 100, 'C4')])
        assert False, 'должно было бросить ValueError'
    except ValueError as e:
        assert 'длины' in str(e) or 'length' in str(e).lower()


def test_zero_and_negative_f0_no_crash():
    # битые f0 (0 и <0) не должны ронять — становятся nan
    r = compare.analyze(np.array([0.0, -100.0, 261.6]),
                        np.array([261.6, 261.6, 261.6]), [(0, 3, 'C4')])
    assert r is not None  # не упало


def test_empty_arrays_no_crash():
    r = compare.analyze(np.array([]), np.array([]), [])
    assert r.low_confidence  # пусто -> точно LOW


# ---------- align на краевых ----------

def test_align_empty_no_warning():
    # пустой/всё-NaN align не должен сорить RuntimeWarning
    with warnings.catch_warnings():
        warnings.simplefilter('error', RuntimeWarning)
        aligned, cost = align.warp_user_to_target(np.array([]), np.array([261.6]))
        assert cost == float('inf')
        aligned2, cost2 = align.warp_user_to_target(np.full(50, np.nan), np.full(50, 261.6))
        assert cost2 == float('inf')


def test_segment_all_nan_empty():
    # сегментация тишины -> ноль нот, не падение
    assert len(notes.segment(np.full(100, np.nan))) == 0


# ---------- контроль: нормальный вход НЕ помечается LOW ----------

def test_normal_input_not_low():
    # ровная спетая нота = target -> НЕ должно быть LOW (иначе guard'ы слишком строгие)
    import synth
    f0 = synth.steady('C4', 3.0)
    seg = notes.segment(f0)
    tgt, bounds = notes.smoothed_target(f0, seg, len(f0))
    user = notes.prepare_contour(f0)
    r = compare.analyze(user, tgt, bounds, vibrato_mask_ext=np.zeros(len(f0), bool))
    assert not r.low_confidence, 'нормальный вход ошибочно помечен LOW'
    assert r.p50 > 0.9


if __name__ == '__main__':
    fns = [v for k, v in sorted(globals().items()) if k.startswith('test_')]
    failed = 0
    for fn in fns:
        try:
            print(f'{fn.__name__}:', end=' ')
            fn()
            print('ok')
        except AssertionError as e:
            failed += 1
            print(f'FAIL: {e}')
        except Exception as e:
            failed += 1
            print(f'ERR {type(e).__name__}: {e}')
    print(f'\n{len(fns)-failed}/{len(fns)} passed')
    sys.exit(1 if failed else 0)
