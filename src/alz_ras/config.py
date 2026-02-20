from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence


@dataclass(frozen=True)
class DataConfig:
    queries: Sequence[str] = (
        "Alzheimer's disease targets",
        "Alzheimer therapeutic targets",
        "Alzheimer drug targets",
    )
    max_pubmed_results_per_query: int = 40
    max_biorxiv_results_per_query: int = 20
    raw_data_path: Path = Path("data/raw/articles_raw.parquet")
    processed_data_path: Path = Path("data/processed/articles_clean.parquet")


@dataclass(frozen=True)
class RagConfig:
    chunk_size: int = 1200
    chunk_overlap: int = 120
    embedding_model_name: str = "BAAI/bge-base-en-v1.5"
    generator_model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    index_path: Path = Path("data/index/faiss")
    top_k: int = 4
    device_map: str = "auto"
    max_new_tokens: int = 180
    temperature: float = 0.1
    use_generation: bool = False


@dataclass(frozen=True)
class EvaluationConfig:
    eval_questions_path: Path = Path("data/eval_questions.json")
    eval_output_path: Path = Path("artifacts/evaluation_metrics.json")


@dataclass(frozen=True)
class AppConfig:
    data: DataConfig = field(default_factory=DataConfig)
    rag: RagConfig = field(default_factory=RagConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)


DEFAULT_CONFIG = AppConfig()
