"""
Phase 3 test: полный путь сравнения через pipeline.score_take на РЕАЛЬНОМ
студийном вокале Виктора (target melody из его же записи).

Проверяем ключевые инварианты на реальном материале с известными манипуляциями:
  - идеал vs идеал -> ~100% (сглаженный контур, не плоские ступеньки)
  - сдвиг по времени -> DTW убирает, счёт не падает
  - реальная фальшь (-60c) -> ловится жёстко, не маскируется

Требует studio_f0.npz (f0 студийного вокала, извлечённый в Phase 3).
Запуск: python3 tests/test_phase3.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))
import numpy as np
import pipeline

# Фикстура — производный файл (f0 студийного вокала). Лежит в tests/data/.
# Если его нет (чужой чекаут без аудио) — тесты честно скипаются, а не падают.
NPZ = os.path.join(os.path.dirname(__file__), 'data', 'studio_f0.npz')
HAVE_NPZ = os.path.exists(NPZ)

try:
    import pytest
    _skip = pytest.mark.skipif(not HAVE_NPZ, reason="нет studio_f0.npz (fixture)")
except ImportError:
    def _skip(f):
        return f


def _load_target():
    d = np.load(NPZ)
    f0, conf, ts = d['f0'], d['conf'], d['ts']
    m = (ts >= 60) & (ts < 80)          # фрагмент 1:00-1:20
    return f0[m].copy(), conf[m].copy()


@_skip
def test_ideal_vs_ideal():
    f0_t, conf_t = _load_target()
    r = pipeline.score_take(f0_t, f0_t.copy(), conf_user=conf_t, conf_target=conf_t,
                            do_align=False)
    print(f'  ideal: p50={r.p50*100:.0f}% notes={r.notes_correct}/{r.notes_total} bias={r.bias_cents:+.0f}c')
    assert r.p50 > 0.95, f'идеал vs идеал должен быть >95%, а {r.p50*100:.0f}%'
    assert r.notes_correct >= r.notes_total - 1


@_skip
def test_time_shift_absorbed_by_align():
    f0_t, conf_t = _load_target()
    n = len(f0_t)
    shifted = np.concatenate([np.full(30, np.nan), f0_t])[:n]   # +300ms
    r = pipeline.score_take(f0_t, shifted, conf_user=conf_t, conf_target=conf_t,
                            do_align=True)
    print(f'  shift+align: p50={r.p50*100:.0f}% bias={r.bias_cents:+.0f}c')
    assert r.p50 > 0.90, f'DTW должен убрать сдвиг, а p50={r.p50*100:.0f}%'


@_skip
def test_real_flat_error_caught():
    f0_t, conf_t = _load_target()
    bad = f0_t * 2 ** (-60 / 1200)      # весь дубль на -60 центов ниже
    r = pipeline.score_take(f0_t, bad, conf_user=conf_t, conf_target=conf_t,
                            do_align=True)
    print(f'  -60c flat: p50={r.p50*100:.0f}% bias={r.bias_cents:+.0f}c notes={r.notes_correct}/{r.notes_total}')
    assert r.p50 < 0.15, f'реальная фальшь должна ронять p50, а {r.p50*100:.0f}%'
    assert r.bias_cents < -40, 'должен показать систематический минус'
    assert -75 < r.bias_cents < -45, 'bias должен быть около -60'


@_skip
def test_partial_error_localized():
    # первая половина чисто, вторая на -70c -> средний p50, bias отрицательный
    f0_t, conf_t = _load_target()
    n = len(f0_t)
    user = f0_t.copy()
    user[n // 2:] *= 2 ** (-70 / 1200)
    r = pipeline.score_take(f0_t, user, conf_user=conf_t, conf_target=conf_t,
                            do_align=True)
    n_low = sum(1 for x in r.notes if x.direction == 'low')
    print(f'  half -70c: p50={r.p50*100:.0f}% bias={r.bias_cents:+.0f}c LOW={n_low}')
    assert 0.3 < r.p50 < 0.85, f'частичная ошибка -> средний p50, а {r.p50*100:.0f}%'


if __name__ == '__main__':
    import sys
    fns = [v for k, v in sorted(globals().items()) if k.startswith('test_')]
    if not HAVE_NPZ:
        print(f'SKIP: нет фикстуры {NPZ} (нужен f0 студийного вокала)')
        sys.exit(0)
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
