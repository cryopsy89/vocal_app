"""
Тесты транспоза: определение и снятие высотного сдвига оригинал/дубль.

Ключевые инварианты:
  - auto находит заложенный сдвиг (тональность и октаву)
  - после транспоза идеальный кавір даёт ~100%, а НЕ 0%
  - несмещённый контур НЕ получает ложный сдвиг
  - ручной ключ имеет приоритет над auto

Запуск: python3 tests/test_transpose.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))
import numpy as np
import synth
import transpose
import pipeline


def _melody(dur_each=0.6):
    """Небольшая мелодия из нескольких нот (как реальная фраза)."""
    notes = ['C4', 'E4', 'G4', 'E4', 'A4', 'G4', 'C4']
    return synth.concat(*[synth.steady(n, dur_each) for n in notes])


# ---------- auto находит заложенный сдвиг ----------

def test_auto_detects_semitone_shift():
    target = _melody()
    for true_shift in [-12, -5, -3, 3, 7, 12]:
        user = transpose.apply_shift(target, true_shift)
        found, conf = transpose.estimate_shift(target, user)
        assert found == true_shift, f'заложил {true_shift}, нашёл {found}'


def test_auto_detects_octave():
    # мужик поёт на октаву ниже
    target = _melody()
    user = transpose.apply_shift(target, -12)
    found, conf = transpose.estimate_shift(target, user)
    assert found == -12, f'октава вниз не определена: {found}'


# ---------- после транспоза идеальный кавер -> ~100%, не 0% ----------

def test_octave_cover_scores_high_after_transpose():
    target = _melody(0.7)
    # идеальный кавер на октаву ниже
    user = transpose.apply_shift(target, -12)
    # БЕЗ транспоза (key=0 отключает) -> провал
    r_no = pipeline.score_take(target, user, do_align=True, key=0)
    # С транспозом (auto)
    r_tr = pipeline.score_take(target, user, do_align=True, key='auto')
    print(f'  октава-кавер: key=0 p50={r_no.p50*100:.0f}%, '
          f'auto p50={r_tr.p50*100:.0f}% (shift={r_tr.transpose_shift})')
    assert r_no.p50 < 0.2, 'без транспоза должен быть провал (контроль)'
    assert r_tr.p50 > 0.9, f'после транспоза должно быть ~100%, а {r_tr.p50*100:.0f}%'
    assert r_tr.transpose_shift == -12


def test_key_change_cover_scores_high():
    # кавер в другой тональности (+4 полутона), спетый идеально
    target = _melody(0.7)
    user = transpose.apply_shift(target, 4)
    r = pipeline.score_take(target, user, do_align=True, key='auto')
    print(f'  +4 тональность: shift={r.transpose_shift}, p50={r.p50*100:.0f}%')
    assert r.transpose_shift == 4
    assert r.p50 > 0.9


# ---------- нет ложного сдвига на несмещённом ----------

def test_no_false_shift():
    target = _melody()
    user = target.copy()   # тот же голос, та же тональность
    found, conf = transpose.estimate_shift(target, user)
    assert found == 0, f'ложный сдвиг {found} на несмещённом контуре'


# ---------- ручной ключ имеет приоритет ----------

def test_manual_key_overrides_auto():
    target = _melody()
    user = transpose.apply_shift(target, -12)   # реально на октаву ниже
    # но пользователь вручную ставит 0 (не хочет сворачивать октаву)
    u_tr, shift, conf = transpose.transpose_user(target, user, key=0)
    assert shift == 0, 'ручной ключ 0 должен переопределить auto'
    assert conf == 1.0, 'ручной ключ = полная уверенность'
    # а с ручным -12 -> свернётся
    u_tr2, shift2, conf2 = transpose.transpose_user(target, user, key=-12)
    assert shift2 == -12


# ---------- реальная фальшь ПОВЕРХ транспоза остаётся видна ----------

def test_real_offpitch_survives_transpose():
    # кавер на +3 полутона, но одна нота спета мимо на -60c
    target = _melody(0.7)
    user = transpose.apply_shift(target, 3)
    nf = synth._n_frames(0.7)
    user[nf:2*nf] = user[nf:2*nf] * 2 ** (-60/1200)   # 2-я нота фальшивая
    r = pipeline.score_take(target, user, do_align=True, key='auto')
    print(f'  +3 с фальшивой нотой: shift={r.transpose_shift}, p50={r.p50*100:.0f}%, n_low={r.n_low}')
    assert r.transpose_shift == 3, 'транспоз должен найти +3, не спутать с фальшью'
    assert r.n_low >= 1, 'реальная фальшь поверх транспоза должна остаться видна'


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
