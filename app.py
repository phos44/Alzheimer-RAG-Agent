from __future__ import annotations

import streamlit as st

from alz_ras.config import DEFAULT_CONFIG
from alz_ras.service import answer_question

st.set_page_config(page_title="Alzheimer RAS Agent", layout="wide")

st.title("Alzheimer Target Discovery RAS")
st.caption("Retrieval-Augmented System for biomedical literature exploration")

with st.sidebar:
    st.header("Model Info")
    st.write(f"Embedding: `{DEFAULT_CONFIG.rag.embedding_model_name}`")
    st.write(f"Generator: `{DEFAULT_CONFIG.rag.generator_model_name}`")
    st.write(f"Top-K Retriever: `{DEFAULT_CONFIG.rag.top_k}`")

query = st.text_area(
    "Введите вопрос исследователя",
    placeholder="Какие мишени (genes/proteins) чаще всего обсуждаются для модификации прогрессии Alzheimer's disease?",
    height=140,
)

if st.button("Поиск", type="primary"):
    if not query.strip():
        st.warning("Введите вопрос перед запуском поиска.")
    else:
        with st.spinner("Ищу релевантные фрагменты и генерирую ответ..."):
            result = answer_question(query)

        tab_answer, tab_sources = st.tabs(["Ответ", "Источники"])

        with tab_answer:
            st.markdown(result.answer)

        with tab_sources:
            if not result.sources:
                st.info("Источники не найдены.")
            for idx, source in enumerate(result.sources, start=1):
                st.markdown(f"**{idx}. {source['title'] or 'Untitled'}**")
                st.write(f"DOI: {source.get('doi', '') or 'N/A'}")
                if source.get("url"):
                    st.markdown(f"[Открыть публикацию]({source['url']})")
                st.caption(f"Source: {source.get('source', 'unknown')}")
