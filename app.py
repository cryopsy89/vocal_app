"""
Phase 4 — UI (Streamlit) для Vocal Pitch Accuracy Analyzer.

Запуск (локально):  streamlit run app.py

Дизайн по брифу (п.10, п.17):
  - ГЛАВНОЕ — цифра, не график. p50 крупно, рядом coverage/confidence/ноты/worst.
  - графики (наложение + deviation) — под цифрами, на одной оси времени.
  - LOW CONFIDENCE вместо красивого процента, когда оценка ненадёжна.

Вход: два ГОТОВЫХ вокал-wav (оригинал/эталон + твой дубль). Separation (Demucs) —
отдельный ручной шаг до загрузки. UI логику НЕ содержит — вся оценка в
pipeline.score_wavs.

Языки: переключатель RU/EN в сайдбаре. Все тексты — в словаре STRINGS.
"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))

import numpy as np
import streamlit as st
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import pipeline
from config import CFG


# ================== i18n ==================
# Все тексты UI. Ключ -> {ru, en}. Никаких строк по коду мимо этого словаря.
STRINGS = {
    'page_title':   {'ru': 'Точность попадания в ноты', 'en': 'Pitch Accuracy'},
    'title':        {'ru': '🎯 Точность попадания в ноты', 'en': '🎯 Pitch Accuracy'},
    'subtitle':     {'ru': 'Загрузи оригинальный вокал и свой дубль (готовые вокал-дорожки, '
                           'без минуса). Отделение от минуса — отдельным шагом до загрузки.',
                     'en': 'Upload the original vocal and your take (ready vocal stems, no '
                           'backing track). Separation is a manual step before upload.'},
    'lang_label':   {'ru': 'Язык / Language', 'en': 'Язык / Language'},
    'original_hdr': {'ru': 'Оригинал (как надо)', 'en': 'Original (reference)'},
    'user_hdr':     {'ru': 'Твой дубль (что спел)', 'en': 'Your take (what you sang)'},
    'upl_original': {'ru': 'вокал-wav оригинала', 'en': 'original vocal wav'},
    'upl_user':     {'ru': 'вокал-wav дубля', 'en': 'your vocal wav'},
    'start_s':      {'ru': 'начало участка, с', 'en': 'segment start, s'},
    'dur_s':        {'ru': 'длина участка, с (0 = весь файл)', 'en': 'segment length, s (0 = whole file)'},
    'key_hdr':      {'ru': 'Тональность (ключ)', 'en': 'Key (transpose)'},
    'key_mode':     {'ru': 'Как задать сдвиг дубля по высоте',
                     'en': 'How to set the take’s pitch shift'},
    'key_auto':     {'ru': 'Авто (определить самому)', 'en': 'Auto (detect)'},
    'key_auto_note':{'ru': '⚠️ авто может ошибаться на реальном вокале — сверь результат, '
                           'при сомнении задай сдвиг вручную', 'en': '⚠️ auto can be wrong on '
                           'real vocals — check the result, set the shift manually if unsure'},
    'key_manual':   {'ru': 'Вручную (полутоны)', 'en': 'Manual (semitones)'},
    'key_semitones':{'ru': 'сдвиг, полутонов (−12…+12; ±12 = октава)',
                     'en': 'shift, semitones (−12…+12; ±12 = octave)'},
    'key_detected': {'ru': 'Определённый сдвиг', 'en': 'Detected shift'},
    'key_st':       {'ru': 'полутонов', 'en': 'semitones'},
    'key_low':      {'ru': 'не уверен в тональности — если дубль в другом ключе/октаве, '
                           'задай сдвиг вручную', 'en': 'unsure about the key — if the take is '
                           'in a different key/octave, set the shift manually'},
    'btn_score':    {'ru': 'Оценить', 'en': 'Score'},
    'spinner':      {'ru': 'Извлекаю f0, выравниваю, считаю…', 'en': 'Extracting f0, aligning, scoring…'},
    'main_caption': {'ru': 'попадание в ноты (доля времени в пределах ±50 центов)',
                     'en': 'in-tune rate (share of time within ±50 cents)'},
    'm_25':         {'ru': 'в ±25c', 'en': 'within ±25c'},
    'm_100':        {'ru': 'в ±100c', 'en': 'within ±100c'},
    'm_cov':        {'ru': 'покрытие', 'en': 'coverage'},
    'm_bias':       {'ru': 'тюнинг (bias)', 'en': 'tuning (bias)'},
    'notes_line':   {'ru': 'Ноты', 'en': 'Notes'},
    'notes_clean':  {'ru': 'чисто', 'en': 'clean'},
    'worst_hdr':    {'ru': 'Где сильнее всего мимо', 'en': 'Biggest misses'},
    'target_word':  {'ru': 'цель', 'en': 'target'},
    'you_are':      {'ru': 'ты на', 'en': 'you are'},
    'flat':         {'ru': 'ниже', 'en': 'flat'},
    'sharp':        {'ru': 'выше', 'en': 'sharp'},
    'graphs_cap':   {'ru': 'Ниже — где именно промахи по времени:',
                     'en': 'Below — where the misses are over time:'},
    'ov_title':     {'ru': 'Наложение: где твоя линия совпадает с оригиналом',
                     'en': 'Overlay: where your line matches the original'},
    'ov_original':  {'ru': 'оригинал (эталон)', 'en': 'original (reference)'},
    'ov_you':       {'ru': 'ты', 'en': 'you'},
    'ov_ylabel':    {'ru': 'нота (MIDI)', 'en': 'note (MIDI)'},
    'ov_xlabel':    {'ru': 'время, с', 'en': 'time, s'},
    'dev_title':    {'ru': 'Где промахи: линия вне зелёного коридора = мимо ноты',
                     'en': 'Misses: line outside the green band = off the note'},
    'dev_ylabel':   {'ru': 'отклонение, центы', 'en': 'deviation, cents'},
    'low_conf':     {'ru': '### LOW CONFIDENCE\nОценка ненадёжна — слишком мало данных или '
                           'плохое выравнивание. Проверь, что дубль спет под тот же участок и '
                           'это чистый вокал.',
                     'en': '### LOW CONFIDENCE\nUnreliable result — too little data or poor '
                           'alignment. Check the take covers the same segment and is a clean vocal.'},
    'err_soxr':     {'ru': "Не хватает пакета **soxr** (нужен SwiftF0 для ресемплинга аудио "
                           "не-16кГц).\n\nПоставь один раз:\n\n```\npip install \"swift-f0[audio]\" "
                           "soxr\n```\n\nПотом перезапусти — больше ставить не придётся.",
                     'en': "Missing package **soxr** (SwiftF0 needs it to resample non-16kHz "
                           "audio).\n\nInstall once:\n\n```\npip install \"swift-f0[audio]\" soxr"
                           "\n```\n\nThen restart — no need to install again."},
    'err_dep':      {'ru': 'Не хватает зависимости', 'en': 'Missing dependency'},
    'err_dep_hint': {'ru': 'Поставь всё разом: `pip install -r requirements.txt`',
                     'en': 'Install everything: `pip install -r requirements.txt`'},
    'err_generic':  {'ru': 'Не смог обработать', 'en': 'Could not process'},
}


def T(key, lang):
    """Строка по ключу и языку."""
    return STRINGS[key][lang]


st.set_page_config(page_title="Pitch Accuracy", layout="wide")

# --- переключатель языка (в сайдбаре) ---
lang = st.sidebar.radio(STRINGS['lang_label']['ru'], ['ru', 'en'],
                        format_func=lambda x: {'ru': 'Русский', 'en': 'English'}[x])


def _save_upload(uploaded):
    suffix = os.path.splitext(uploaded.name)[1] or '.wav'
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(uploaded.getbuffer())
    tmp.close()
    return tmp.name


def _fmt_time(sec):
    m = int(sec // 60)
    s = sec % 60
    return f"{m}:{s:04.1f}"


def _deviation_figure(result, lang):
    """deviation-график: отклонение от цели по времени, зелёный коридор ±50c.
    Из error_per_frame (тот же источник, что p50)."""
    err = result.error_per_frame
    ts = np.arange(len(err)) * CFG.frame_ms / 1000
    fig, ax = plt.subplots(figsize=(12, 3.2))
    ax.axhspan(-CFG.thr_main, CFG.thr_main, color='green', alpha=0.12)
    ax.axhline(0, color='k', lw=0.5)
    ax.plot(ts, err, color='#6a4ca5', lw=0.7)
    ax.set_ylim(-700, 700)
    ax.set_ylabel(T('dev_ylabel', lang))
    ax.set_xlabel(T('ov_xlabel', lang))
    ax.set_title(T('dev_title', lang))
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def _overlay_figure(result, lang):
    """Верхний график: наложение контуров оригинал/ты в MIDI. Из ТЕХ ЖЕ выровненных
    контуров (после сглаживания+DTW), что и deviation — иначе кривые разъедутся."""
    tgt = result.target_f0_per_frame
    usr = result.user_f0_per_frame
    if tgt is None or usr is None:
        return None
    ts = np.arange(len(tgt)) * CFG.frame_ms / 1000

    def to_midi(f):
        with np.errstate(divide='ignore', invalid='ignore'):
            return np.where(~np.isnan(f), 69 + 12 * np.log2(np.where(np.isnan(f), 1, f) / 440), np.nan)

    fig, ax = plt.subplots(figsize=(12, 3.6))
    ax.plot(ts, to_midi(tgt), color='#2c6fbb', lw=1.6, label=T('ov_original', lang))
    ax.plot(ts, to_midi(usr), color='#d1495b', lw=0.9, alpha=0.8, label=T('ov_you', lang))
    ax.set_ylabel(T('ov_ylabel', lang))
    ax.set_xlabel(T('ov_xlabel', lang))
    ax.legend(loc='upper right')
    ax.grid(alpha=0.3)
    ax.set_title(T('ov_title', lang))
    fig.tight_layout()
    return fig


def _render_result(result, lang):
    if result.low_confidence:
        st.warning(T('low_conf', lang))
        st.caption(f"confidence={result.conf:.2f}, coverage={result.coverage*100:.0f}%")
        return

    st.markdown(f"<h1 style='font-size:5rem;margin:0'>{result.p50*100:.0f}%</h1>",
                unsafe_allow_html=True)
    st.caption(T('main_caption', lang))

    c1, c2, c3, c4 = st.columns(4)
    c1.metric(T('m_25', lang), f"{result.p25*100:.0f}%")
    c2.metric(T('m_100', lang), f"{result.p100*100:.0f}%")
    c3.metric(T('m_cov', lang), f"{result.coverage*100:.0f}%")
    c4.metric(T('m_bias', lang), f"{result.bias_cents:+.0f}c")

    st.markdown(f"**{T('notes_line', lang)}:** {result.notes_correct} / "
                f"{result.notes_total} {T('notes_clean', lang)} "
                f"&nbsp;&nbsp; LOW: {result.n_low} &nbsp; HIGH: {result.n_high}",
                unsafe_allow_html=True)

    # применённый транспоз
    if result.transpose_shift != 0:
        st.caption(f"{T('key_detected', lang)}: {result.transpose_shift:+d} {T('key_st', lang)}")
    elif result.transpose_conf < CFG.transpose_min_conf:
        # авто не уверено И сдвиг 0 — подсказать про ручной ключ
        st.caption(f"ℹ️ {T('key_low', lang)}")

    if result.worst:
        st.subheader(T('worst_hdr', lang))
        for t, name, cents in result.worst:
            direction = T('flat', lang) if cents < 0 else T('sharp', lang)
            st.write(f"`{_fmt_time(t)}`  {T('target_word', lang)} **{name}**, "
                     f"{T('you_are', lang)} **{abs(cents):.0f}c {direction}**")

    st.divider()
    overlay = _overlay_figure(result, lang)
    if overlay is not None:
        st.pyplot(overlay)
    st.caption(T('graphs_cap', lang))
    st.pyplot(_deviation_figure(result, lang))


# ================== UI ==================

st.title(T('title', lang))
st.caption(T('subtitle', lang))

col_a, col_b = st.columns(2)
with col_a:
    st.subheader(T('original_hdr', lang))
    target_file = st.file_uploader(T('upl_original', lang), type=['wav', 'mp3'], key='target')
    t_start = st.number_input(T('start_s', lang), 0.0, value=0.0, key='ts')
    t_dur = st.number_input(T('dur_s', lang), 0.0, value=0.0, key='td')
with col_b:
    st.subheader(T('user_hdr', lang))
    user_file = st.file_uploader(T('upl_user', lang), type=['wav', 'mp3'], key='user')
    u_start = st.number_input(T('start_s', lang), 0.0, value=0.0, key='us')
    u_dur = st.number_input(T('dur_s', lang), 0.0, value=0.0, key='ud')

# --- ключ (транспоз) ---
# По умолчанию РУЧНОЙ со сдвигом 0 (не транспонируем — не гадаем). Auto ненадёжен
# на реальном вокале (не различает октаву и фальшь по уверенности), поэтому
# включается явно, не дефолт.
st.subheader(T('key_hdr', lang))
key_mode = st.radio(T('key_mode', lang), ['manual', 'auto'],
                    format_func=lambda x: T('key_manual', lang) if x == 'manual' else T('key_auto', lang),
                    horizontal=True, key='keymode')
if key_mode == 'manual':
    key_val = st.slider(T('key_semitones', lang), -12, 12, 0, key='keyval')
else:
    key_val = 'auto'
    st.caption(T('key_auto_note', lang))

if st.button(T('btn_score', lang), type="primary", disabled=not (target_file and user_file)):
    with st.spinner(T('spinner', lang)):
        tp = _save_upload(target_file)
        up = _save_upload(user_file)
        try:
            result = pipeline.score_wavs(
                tp, up,
                target_start=(t_start if t_start > 0 else None),
                target_dur=(t_dur if t_dur > 0 else None),
                user_start=(u_start if u_start > 0 else None),
                user_dur=(u_dur if u_dur > 0 else None),
                key=key_val,
            )
            _render_result(result, lang)
        except ImportError as e:
            msg = str(e)
            if 'soxr' in msg.lower():
                st.error(T('err_soxr', lang))
            else:
                st.error(f"{T('err_dep', lang)}: {msg}\n\n{T('err_dep_hint', lang)}")
        except Exception as e:
            st.error(f"{T('err_generic', lang)}: {type(e).__name__}: {e}")
        finally:
            for p in (tp, up):
                try:
                    os.unlink(p)
                except OSError:
                    pass
