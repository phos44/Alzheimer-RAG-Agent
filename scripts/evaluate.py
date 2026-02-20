from __future__ import annotations

from alz_ras.config import DEFAULT_CONFIG
from alz_ras.evaluation import evaluate_retrieval, load_eval_questions, save_metrics
from alz_ras.rag.embeddings import SentenceTransformerEmbeddings
from alz_ras.rag.vector_store import load_faiss_index


def main() -> None:
    cfg = DEFAULT_CONFIG

    embeddings = SentenceTransformerEmbeddings(cfg.rag.embedding_model_name)
    vector_store = load_faiss_index(cfg.rag.index_path, embeddings)
    retriever = vector_store.as_retriever(search_kwargs={"k": cfg.rag.top_k})

    questions = load_eval_questions(cfg.evaluation.eval_questions_path)
    metrics = evaluate_retrieval(retriever, questions)
    save_metrics(metrics, cfg.evaluation.eval_output_path)
    print(metrics)


if __name__ == "__main__":
    main()
