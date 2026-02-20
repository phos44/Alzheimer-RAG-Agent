from __future__ import annotations

from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings


def build_faiss_index(documents: list[Document], embeddings: Embeddings) -> FAISS:
    return FAISS.from_documents(documents, embeddings)


def save_faiss_index(store: FAISS, path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    store.save_local(str(path))


def load_faiss_index(path: Path, embeddings: Embeddings) -> FAISS:
    return FAISS.load_local(str(path), embeddings, allow_dangerous_deserialization=True)
