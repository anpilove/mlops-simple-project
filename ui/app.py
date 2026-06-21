"""Streamlit UI — RuSentiTweet sentiment via ClearML Serving HTTP."""

from __future__ import annotations

import os
import time

import requests
import streamlit as st

from src.common import LABEL_RU

SERVING_URL = os.getenv(
    "SERVING_URL",
    "http://localhost:8088/serve/sentiment",
)

EXAMPLES = {
    "Позитив": "Отличный сервис, всё быстро и удобно, рекомендую!",
    "Нейтрал": "Нормально, без восторга, но и без претензий.",
    "Негатив": "Ужасное качество, деньги на ветер, больше не куплю.",
}

EMOJI = {"negative": "😞", "neutral": "😐", "positive": "😊", "негатив": "😞", "нейтрал": "😐", "позитив": "😊"}

st.set_page_config(page_title="mlops-simple-project", page_icon="🐦")
st.title("mlops-simple-project")
st.subheader("Тональность русскоязычного текста")
st.caption("Streamlit → ClearML Serving → Model Registry · датасет RuSentiTweet")

if "text_input" not in st.session_state:
    st.session_state.text_input = ""

cols = st.columns(3)
for col, (name, sample) in zip(cols, EXAMPLES.items()):
    if col.button(name, use_container_width=True):
        st.session_state.text_input = sample

text = st.text_area(
    "Текст (твит или короткое сообщение)",
    height=140,
    placeholder="Введите текст на русском...",
    key="text_input",
)

if st.button("Predict", type="primary"):
    if not text.strip():
        st.warning("Введите текст.")
    else:
        try:
            started = time.perf_counter()
            resp = requests.post(SERVING_URL, json={"text": text}, timeout=30)
            latency_ms = (time.perf_counter() - started) * 1000
            resp.raise_for_status()
            data = resp.json()
            label = str(data.get("label", "")).lower()
            label_id = data.get("label_id")
            emoji = EMOJI.get(label, "")
            ru = LABEL_RU.get(label_id, label) if label_id is not None else label
            st.success(f"{emoji} **{ru}** (`{label}`)")
            st.caption(f"Latency: **{latency_ms:.1f} ms**")
            with st.expander("Ответ API"):
                st.json(data)
        except requests.RequestException as exc:
            st.error(f"Serving недоступен: {exc}")
            st.info(
                "Подними ClearML Serving или задай SERVING_URL. "
                "Для локальной проверки: `python -m scripts.local_infer --text '...'`"
            )
