from __future__ import annotations

from dataclasses import dataclass
import re

from langchain_core.documents import Document
from langchain_core.messages import BaseMessage
from langchain_core.prompts import PromptTemplate
from langchain_core.retrievers import BaseRetriever


PROMPT = PromptTemplate.from_template(
    """Ты — научный ассистент по биомедицине.
Отвечай ТОЛЬКО по контексту ниже. Не выдумывай факты.
Если данных мало, явно напиши: "Недостаточно данных в контексте".

Контекст:
{context}

Вопрос исследователя: {input}

Дай ответ на русском и в формате:
1) Ключевые мишени
2) Что про них сказано в контексте
3) Ограничения/пробелы
"""
)


@dataclass
class RagAnswer:
    answer: str
    sources: list[dict[str, str]]


class RetrievalGenerationChain:
    def __init__(self, llm: object, retriever: BaseRetriever) -> None:
        self.llm = llm
        self.retriever = retriever

    def invoke(self, payload: dict[str, str]) -> dict[str, object]:
        query = payload["input"]
        context_docs: list[Document] = self.retriever.invoke(query)
        context_text = "\n\n".join(doc.page_content for doc in context_docs)
        if not context_text.strip():
            return {"answer": "Недостаточно данных в контексте.", "context": context_docs}

        prompt = PROMPT.format(context=context_text, input=query)
        answer = self.llm.invoke(prompt)
        if isinstance(answer, BaseMessage):
            answer_text = str(answer.content)
        else:
            answer_text = str(answer)
        answer_text = _cleanup_answer(answer_text)
        if _looks_noisy(answer_text):
            answer_text = _extractive_fallback(context_docs)
        return {"answer": answer_text, "context": context_docs}


def build_retrieval_chain(llm: object, retriever: BaseRetriever) -> RetrievalGenerationChain:
    return RetrievalGenerationChain(llm=llm, retriever=retriever)


def _cleanup_answer(text: str) -> str:
    cleaned = text.strip()
    # Sometimes model echoes the prompt sections; keep only the actual answer tail.
    markers = ["Дай ответ на русском и в формате:", "Вопрос исследователя:", "Контекст:"]
    for marker in markers:
        if marker in cleaned:
            cleaned = cleaned.split(marker)[-1].strip()
    return cleaned


def _looks_noisy(text: str) -> bool:
    words = re.findall(r"[A-Za-zА-Яа-я0-9\\-]+", text)
    if len(words) < 20:
        return False
    punctuation = re.findall(r"[\\.,;:!?]", text)
    if len(punctuation) <= 1 and len(words) > 50:
        return True
    long_weird = sum(1 for w in words if len(w) > 20)
    return (long_weird / max(len(words), 1)) > 0.08


def _extractive_fallback(context_docs: list[Document]) -> str:
    target_patterns = [
        r"amyloid[- ]?beta",
        r"abeta",
        r"APP",
        r"BACE1",
        r"PSEN1",
        r"PSEN2",
        r"APOE",
        r"tau",
        r"MAPT",
        r"TREM2",
        r"CD33",
        r"NLRP3",
    ]
    context = " ".join(doc.page_content for doc in context_docs)
    found = []
    for pattern in target_patterns:
        if re.search(pattern, context, flags=re.IGNORECASE):
            found.append(pattern.replace(r"[- ]?", "-").upper())

    if not found:
        return "Недостаточно данных в контексте."

    uniq = []
    seen = set()
    for x in found:
        if x not in seen:
            seen.add(x)
            uniq.append(x)

    return (
        "1) Ключевые мишени:\n"
        f"- {', '.join(uniq[:8])}\n\n"
        "2) Что сказано в контексте:\n"
        "- Упоминаются как потенциальные терапевтические или патогенетические цели при болезни Альцгеймера.\n\n"
        "3) Ограничения/пробелы:\n"
        "- Нужны дополнительные полные тексты/экспериментальные данные для уверенных выводов."
    )


def run_query(chain: object, query: str) -> RagAnswer:
    response = chain.invoke({"input": query})
    context_docs: list[Document] = response.get("context", [])

    sources: list[dict[str, str]] = []
    seen_titles: set[str] = set()
    for doc in context_docs:
        title = str(doc.metadata.get("title", ""))
        if title in seen_titles:
            continue
        seen_titles.add(title)
        sources.append(
            {
                "title": title,
                "doi": str(doc.metadata.get("doi", "")),
                "url": str(doc.metadata.get("url", "")),
                "source": str(doc.metadata.get("source", "")),
            }
        )

    return RagAnswer(answer=str(response.get("answer", "")), sources=sources)
