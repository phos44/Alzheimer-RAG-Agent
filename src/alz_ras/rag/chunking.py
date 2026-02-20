from __future__ import annotations

from typing import Any

import pandas as pd
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


def build_documents(frame: pd.DataFrame) -> list[Document]:
    docs: list[Document] = []
    for _, row in frame.iterrows():
        docs.append(
            Document(
                page_content=str(row["text"]),
                metadata={
                    "title": str(row.get("title", "")),
                    "doi": str(row.get("doi", "")),
                    "url": str(row.get("url", "")),
                    "source": str(row.get("source", "")),
                    "authors": str(row.get("authors", "")),
                    "journal": str(row.get("journal", "")),
                },
            )
        )
    return docs


def split_documents(
    documents: list[Document],
    chunk_size: int,
    chunk_overlap: int,
    separators: list[str] | None = None,
) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=separators or ["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )
    return splitter.split_documents(documents)
