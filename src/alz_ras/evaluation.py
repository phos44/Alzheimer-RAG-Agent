from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever


@dataclass
class EvalQuestion:
    question: str
    expected_title_keywords: list[str]


def load_eval_questions(path: Path) -> list[EvalQuestion]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    return [EvalQuestion(**row) for row in rows]


def _find_relevant_rank(documents: list[Document], expected_keywords: list[str]) -> int | None:
    expected_lower = [k.lower() for k in expected_keywords]
    for idx, doc in enumerate(documents, start=1):
        title = str(doc.metadata.get("title", "")).lower()
        if any(k in title for k in expected_lower):
            return idx
    return None


def evaluate_retrieval(
    retriever: BaseRetriever,
    questions: list[EvalQuestion],
) -> dict[str, float]:
    hits = 0
    reciprocal_ranks: list[float] = []

    for item in questions:
        docs = retriever.invoke(item.question)
        rank = _find_relevant_rank(docs, item.expected_title_keywords)
        if rank is not None:
            hits += 1
            reciprocal_ranks.append(1 / rank)
        else:
            reciprocal_ranks.append(0.0)

    total = len(questions)
    return {
        "hit_rate": hits / total if total else 0.0,
        "mrr": sum(reciprocal_ranks) / total if total else 0.0,
        "num_questions": float(total),
    }


def save_metrics(metrics: dict[str, float], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
