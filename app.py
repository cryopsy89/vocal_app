"""
Phase 4 — UI (Streamlit) для Vocal Pitch Accuracy Analyzer.

Запуск (локально):  streamlit run app.py

Дизайн по брифу (п.10, п.17):
  - ГЛАВНОЕ — цифра, не график. Пользователь не интерпретирует кривые.
  - p50 крупно, рядом coverage + confidence + ноты + worst с таймкодами.
  - deviation-график — ВСПОМОГАТЕЛЬНЫЙ, внизу (где именно промахи).
  - LOW CONFIDENCE вместо красивого процента, когда оценка ненадёжна.

Вход: два ГОТОВЫХ вокал-wav (эталон + твой дубль). Separation (Demucs) —
отдельный ручной шаг до загрузки (веса качаются только на машине пользователя).
UI логику НЕ содержит — вся оценка в pipeline.score_wavs.
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
import compare
import notes as notes_mod
import align as align_mod
from config import CFG


st.set_page_config(page_title="Pitch Accuracy", layout="wide")


def _save_upload(uploaded):
    """Сохраняет загруженный файл во временный, возвращает путь."""
    suffix = os.path.splitext(uploaded.name)[1] or '.wav'
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(uploaded.getbuffer())
    tmp.close()
    return tmp.name


def _fmt_time(sec):
    m = int(sec // 60)
    s = sec % 60
    return f"{m}:{s:04.1f}"


def _deviation_figure(result, target_f0_for_plot=None, user_f0_for_plot=None):
    """Вспомогательный deviation-график: отклонение от цели по времени,
    зелёный коридор ±50c. Строится из error_per_frame (тот же источник, что p50)."""
    err = result.error_per_frame
    ts = np.arange(len(err)) * CFG.frame_ms / 1000
    fig, ax = plt.subplots(figsize=(12, 3.2))
    ax.axhspan(-CFG.thr_main, CFG.thr_main, color='green', alpha=0.12)
    ax.axhline(0, color='k', lw=0.5)
    ax.plot(ts, err, color='#6a4ca5', lw=0.7)
    ax.set_ylim(-700, 700)
    ax.set_ylabel('отклонение, центы')
    ax.set_xlabel('время, с')
    ax.set_title('Где промахи: линия вне зелёного коридора = мимо ноты')
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def _render_result(result):
    """Главный экран результата: цифры сверху (крупно), график снизу."""
    if result.low_confidence:
        st.warning("### LOW CONFIDENCE\n"
                   "Оценка ненадёжна — слишком мало данных или плохое выравнивание. "
                   "Проверь, что дубль спет под тот же участок и это чистый вокал.")
        st.caption(f"confidence={result.conf:.2f}, coverage={result.coverage*100:.0f}%")
        return

    # --- главная цифра ---
    st.markdown(f"<h1 style='font-size:5rem;margin:0'>{result.p50*100:.0f}%</h1>",
                unsafe_allow_html=True)
    st.caption("попадание в ноты (доля времени в пределах ±50 центов)")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("в ±25c", f"{result.p25*100:.0f}%")
    c2.metric("в ±100c", f"{result.p100*100:.0f}%")
    c3.metric("покрытие", f"{result.coverage*100:.0f}%")
    c4.metric("тюнинг (bias)", f"{result.bias_cents:+.0f}c")

    st.markdown(f"**Ноты:** {result.notes_correct} / {result.notes_total} чисто "
                f"&nbsp;&nbsp; LOW: {result.n_low} &nbsp; HIGH: {result.n_high}",
                unsafe_allow_html=True)

    # --- худшие места ---
    if result.worst:
        st.subheader("Где сильнее всего мимо")
        for t, name, cents in result.worst:
            direction = "ниже" if cents < 0 else "выше"
            st.write(f"`{_fmt_time(t)}`  цель **{name}**, ты на **{abs(cents):.0f}c "
                     f"{direction}**")

    # --- вспомогательный график ---
    st.divider()
    st.caption("Дополнительно — где именно промахи по времени:")
    st.pyplot(_deviation_figure(result))


# ================== UI ==================

st.title("🎯 Pitch Accuracy — попадание в ноты")
st.caption("Загрузи эталонный вокал и свой дубль (готовые вокал-дорожки, "
           "без минуса). Отделение от минуса — отдельным шагом до загрузки.")

col_a, col_b = st.columns(2)
with col_a:
    st.subheader("Эталон (как надо)")
    target_file = st.file_uploader("вокал-wav эталона", type=['wav', 'mp3'],
                                   key='target')
    t_start = st.number_input("начало участка, с", 0.0, value=0.0, key='ts')
    t_dur = st.number_input("длина участка, с (0 = весь файл)", 0.0, value=0.0,
                            key='td')
with col_b:
    st.subheader("Твой дубль (что спел)")
    user_file = st.file_uploader("вокал-wav дубля", type=['wav', 'mp3'], key='user')
    u_start = st.number_input("начало участка, с", 0.0, value=0.0, key='us')
    u_dur = st.number_input("длина участка, с (0 = весь файл)", 0.0, value=0.0,
                            key='ud')

if st.button("Оценить", type="primary", disabled=not (target_file and user_file)):
    with st.spinner("Извлекаю f0, выравниваю, считаю…"):
        tp = _save_upload(target_file)
        up = _save_upload(user_file)
        try:
            result = pipeline.score_wavs(
                tp, up,
                target_start=(t_start if t_start > 0 else None),
                target_dur=(t_dur if t_dur > 0 else None),
                user_start=(u_start if u_start > 0 else None),
                user_dur=(u_dur if u_dur > 0 else None),
            )
            _render_result(result)
        except ImportError as e:
            # частый случай: не хватает soxr (ресемплинг SwiftF0). Даём точную команду,
            # а не голый traceback.
            msg = str(e)
            if 'soxr' in msg.lower():
                st.error("Не хватает пакета **soxr** (нужен SwiftF0 для ресемплинга "
                         "аудио не-16кГц).\n\nПоставь один раз:\n\n"
                         "```\npip install \"swift-f0[audio]\" soxr\n```\n\n"
                         "После этого перезапусти приложение — больше ставить не придётся.")
            else:
                st.error(f"Не хватает зависимости: {msg}\n\n"
                         "Поставь всё разом: `pip install -r requirements.txt`")
        except Exception as e:
            st.error(f"Не смог обработать: {type(e).__name__}: {e}")
        finally:
            for p in (tp, up):
                try:
                    os.unlink(p)
                except OSError:
                    pass
