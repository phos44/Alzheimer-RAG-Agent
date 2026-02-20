from __future__ import annotations

import pandas as pd

from alz_ras.config import DEFAULT_CONFIG
from alz_ras.rag.chunking import build_documents, split_documents
from alz_ras.rag.embeddings import SentenceTransformerEmbeddings
from alz_ras.rag.vector_store import build_faiss_index, save_faiss_index


def main() -> None:
    frame = pd.read_parquet(DEFAULT_CONFIG.data.processed_data_path)
    docs = build_documents(frame)
    chunks = split_documents(
        docs,
        chunk_size=DEFAULT_CONFIG.rag.chunk_size,
        chunk_overlap=DEFAULT_CONFIG.rag.chunk_overlap,
    )

    embedding_model = SentenceTransformerEmbeddings(DEFAULT_CONFIG.rag.embedding_model_name)
    index = build_faiss_index(chunks, embedding_model)
    save_faiss_index(index, DEFAULT_CONFIG.rag.index_path)

    print(f"Indexed {len(chunks)} chunks into {DEFAULT_CONFIG.rag.index_path}")


if __name__ == "__main__":
    main()
