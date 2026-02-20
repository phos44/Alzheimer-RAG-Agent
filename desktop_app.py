from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from alz_ras.config import DEFAULT_CONFIG


def _format_sources(sources: list[dict[str, str]], limit: int = 3) -> str:
    """Сделать вывод источников компактным и удобным для чтения."""
    if not sources:
        return "Источники не найдены."

    shown = sources[:limit]
    lines: list[str] = []
    for idx, source in enumerate(shown, start=1):
        title = source.get("title") or "Untitled"
        url = source.get("url") or ""
        doi = source.get("doi") or ""

        if url:
            lines.append(f"{idx}. {title} ({url})")
        elif doi:
            lines.append(f"{idx}. {title} (DOI: {doi})")
        else:
            lines.append(f"{idx}. {title}")

    if len(sources) > limit:
        lines.append(f"... и ещё {len(sources) - limit} источник(ов)")

    return "\n".join(lines)


def self_check() -> int:
    try:
        __import__("sentence_transformers")
        __import__("langchain")
        __import__("faiss")
        __import__("transformers")
        __import__("torch")
    except Exception as exc:  # noqa: BLE001
        print(f"[ERROR] Required dependency is missing: {exc}")
        print("Run: pip install -r requirements.txt")
        return 1

    index_path = DEFAULT_CONFIG.rag.index_path
    index_dir = Path(index_path)
    required_files = [index_dir / "index.faiss", index_dir / "index.pkl"]
    if not index_dir.exists() or not all(path.exists() for path in required_files):
        print(f"[ERROR] FAISS index path does not exist: {index_path}")
        print("Run: python scripts/build_index.py")
        return 1

    print("[OK] Config loaded")
    print(f"[OK] Embedding model: {DEFAULT_CONFIG.rag.embedding_model_name}")
    print(f"[OK] Generator model: {DEFAULT_CONFIG.rag.generator_model_name}")
    print(f"[OK] Index path exists: {index_path}")
    return 0


def run_once(query: str) -> int:
    try:
        from alz_ras.service import answer_question

        result = answer_question(query)
    except Exception as exc:  # noqa: BLE001
        print(f"[ERROR] Failed to answer query: {exc}")
        return 1

    print("\nОтвет:\n")
    print(result.answer.strip())
    print("\nКраткий список источников:\n")
    print(_format_sources(result.sources))
    return 0


def repl() -> int:
    print("Alzheimer RAS Standalone App")
    print(f"Embedding: {DEFAULT_CONFIG.rag.embedding_model_name}")
    print(f"LLM: {DEFAULT_CONFIG.rag.generator_model_name}")
    print("Type a question and press Enter. Type 'exit' to quit.\n")

    while True:
        query = input("Question> ").strip()
        if not query:
            continue
        if query.lower() in {"exit", "quit"}:
            return 0
        run_once(query)
        print()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Standalone Alzheimer RAS application")
    parser.add_argument("--query", type=str, help="Single query mode")
    parser.add_argument("--self-check", action="store_true", help="Validate runtime prerequisites")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.self_check:
        return self_check()

    if args.query:
        return run_once(args.query)

    return repl()


if __name__ == "__main__":
    raise SystemExit(main())
