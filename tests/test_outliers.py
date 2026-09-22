"""
Тест устойчивости к артефактам трекинга/separation (reject_outliers).
Кейс из реального прогона: Demucs оставил иглу D6 (1239 Гц) в студийном эталоне
посреди C4-фразы. Она НЕ должна штрафовать пользователя, НО реальная фальшь
на терцию-кварту (2-5 полутонов) должна ОСТАВАТЬСЯ.

Критерий (из данных «90 белых дней»): артефакт = отскок >9 полутонов от
глобальной медианы песни + короткий пробег; реальная фальшь <=5.5 полутона, устойчива.

Запуск: python3 tests/test_outliers.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))
import numpy as np
import notes
import synth


def _contour_c4_phrase(n=200):
    """Ровная фраза вокруг C4 (~262 Гц) с лёгким вибрато."""
    return synth.vibrato('C4', n * 0.01, rate_hz=5.5, extent_cents=30.0)


def test_octave_spike_removed():
    # фраза C4 + вставленная игла D6 на 6 фреймов (артефакт)
    f0 = _contour_c4_phrase(200)
    d6 = synth.note_to_hz('D6')
    f0[100:106] = d6                       # 60мс игла ~27 полутонов вверх
    clean = notes.reject_outliers(f0)
    # игла должна стать NaN
    assert np.all(np.isnan(clean[100:106])), 'артефакт D6 не убран'
    # остальное — цело
    assert np.sum(~np.isnan(clean[:100])) > 90, 'порезал нормальную фразу до иглы'
    assert np.sum(~np.isnan(clean[106:])) > 90, 'порезал нормальную фразу после иглы'


def test_real_offpitch_kept():
    # реальная фальшь: кусок фразы на -4 полутона (400 центов) ниже, УСТОЙЧИВО
    f0 = _contour_c4_phrase(200)
    f0[80:140] = f0[80:140] * 2 ** (-4 / 12)   # 600мс на кварту ниже — это НЕ артефакт
    clean = notes.reject_outliers(f0)
    # фальшь НЕ должна быть вырезана (это реальное пение мимо)
    kept = np.sum(~np.isnan(clean[80:140]))
    assert kept >= 55, f'реальная фальшь вырезана как артефакт: осталось {kept}/60'


def test_short_high_run_removed_but_not_sustained():
    # короткий заскок вверх на 12 полутонов (артефакт) режется,
    # а устойчивый высокий заход на 12 полутонов на 400мс — НЕ режется (реальная нота)
    f0 = _contour_c4_phrase(300)
    c5 = synth.note_to_hz('C5')            # +12 полутонов
    f0[50:54] = c5                         # 40мс игла -> артефакт
    f0[150:190] = c5                       # 400мс устойчиво -> реальный высокий заход
    clean = notes.reject_outliers(f0)
    assert np.all(np.isnan(clean[50:54])), 'короткая игла не убрана'
    assert np.sum(~np.isnan(clean[150:190])) > 30, 'устойчивый высокий заход ошибочно вырезан'


def test_clean_contour_untouched():
    # чистая фраза без артефактов — ничего не должно вырезаться
    f0 = _contour_c4_phrase(200)
    clean = notes.reject_outliers(f0)
    lost = np.sum(np.isnan(clean) & ~np.isnan(f0))
    assert lost == 0, f'на чистом контуре вырезано {lost} фреймов'


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