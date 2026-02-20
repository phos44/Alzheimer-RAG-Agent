from __future__ import annotations

from functools import lru_cache
import re

from langchain_core.documents import Document

from alz_ras.config import DEFAULT_CONFIG
from alz_ras.rag.embeddings import SentenceTransformerEmbeddings
from alz_ras.rag.llm import build_generation_llm
from alz_ras.rag.pipeline import RagAnswer, build_retrieval_chain, run_query
from alz_ras.rag.vector_store import load_faiss_index


@lru_cache(maxsize=1)
def _build_chain():
    cfg = DEFAULT_CONFIG
    embeddings = SentenceTransformerEmbeddings(cfg.rag.embedding_model_name)
    store = load_faiss_index(cfg.rag.index_path, embeddings)
    retriever = store.as_retriever(search_kwargs={"k": cfg.rag.top_k})

    if not cfg.rag.use_generation:
        return RetrievalOnlyChain(retriever)

    llm = build_generation_llm(
        cfg.rag.generator_model_name,
        max_new_tokens=cfg.rag.max_new_tokens,
        temperature=cfg.rag.temperature,
        device_map=cfg.rag.device_map,
    )
    return build_retrieval_chain(llm, retriever)


def _build_fast_answer(context_docs: list[Document]) -> str:
    if not context_docs:
        return "Недостаточно данных в контексте."

    target_patterns = [
        r"amyloid[- ]?beta",
        r"abeta",
        r"\bAPP\b",
        r"\bBACE1\b",
        r"\bPSEN1\b",
        r"\bPSEN2\b",
        r"\bAPOE\b",
        r"\btau\b",
        r"\bMAPT\b",
        r"\bTREM2\b",
        r"\bCD33\b",
        r"\bNLRP3\b",
    ]
    context = " ".join(doc.page_content for doc in context_docs)
    found = []
    for pattern in target_patterns:
        if re.search(pattern, context, flags=re.IGNORECASE):
            found.append(pattern.replace(r"[- ]?", "-").replace(r"\b", "").upper())

    uniq = []
    seen = set()
    for item in found:
        if item not in seen:
            seen.add(item)
            uniq.append(item)

    if uniq:
        return (
            "1) Ключевые мишени:\n"
            f"- {', '.join(uniq[:8])}\n\n"
            "2) Что про них сказано в источниках:\n"
            "- Эти мишени встречаются в retrieved-публикациях как потенциально значимые для терапии/патогенеза AD.\n\n"
            "3) Ограничения/пробелы:\n"
            "- Это быстрый retrieval-only режим: без генеративного анализа полной аргументации по каждому таргету."
        )

    return (
        "Найдены релевантные источники, но явные target-токены из базового словаря не обнаружены. "
        "См. вкладку источников для ручного анализа."
    )


class RetrievalOnlyChain:
    def __init__(self, retriever: object) -> None:
        self.retriever = retriever

    def invoke(self, payload: dict[str, str]) -> dict[str, object]:
        docs: list[Document] = self.retriever.invoke(payload["input"])
        return {"answer": _build_fast_answer(docs), "context": docs}


def answer_question(query: str) -> RagAnswer:
    chain = _build_chain()
    return run_query(chain, query)
