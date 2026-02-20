# Alzheimer RAS Agent

Prototype **Retrieval-Augmented System (RAS)** for helping researchers identify Alzheimer's disease therapeutic targets from biomedical literature.

## Features
- Data collection from PubMed (`pymed`) + bioRxiv (`requests`)
- Text cleaning and dataset export (`.parquet` + `.csv`)
- EDA: text-length stats, Top-30 tokens, WordCloud, frequent gene/protein-like mentions
- RAG pipeline (LangChain + FAISS + Sentence-Transformers + HF LLM)
- Retrieval evaluation (Hit Rate, MRR)
- Standalone CLI app (no localhost)

## Project Structure

```text
.
├── desktop_app.py
├── app.py
├── scripts/
│   ├── build_dataset.py
│   ├── run_eda.py
│   ├── build_index.py
│   └── evaluate.py
├── src/alz_ras/
│   ├── config.py
│   ├── data_pipeline.py
│   ├── eda.py
│   ├── evaluation.py
│   ├── service.py
│   └── rag/
│       ├── chunking.py
│       ├── embeddings.py
│       ├── llm.py
│       ├── pipeline.py
│       └── vector_store.py
├── data/
│   ├── raw/
│   ├── processed/
│   ├── index/
│   └── eval_questions.json
└── REPORT.md
```

## Setup

```bash
python3.10 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
pip install -e .
```

## Run Pipeline

1. Build dataset (~45+ papers):

```bash
python scripts/build_dataset.py
```

2. Run EDA:

```bash
python scripts/run_eda.py
```

3. Build FAISS index:

```bash
python scripts/build_index.py
```

4. Evaluate retrieval metrics:

```bash
python scripts/evaluate.py
```

5. Launch standalone app:

```bash
python3 desktop_app.py
```

Single-query mode:
```bash
python3 desktop_app.py --query "What targets are associated with amyloid-beta?"
```

Runtime self-check:
```bash
python3 desktop_app.py --self-check
```

## Notes
- Default embedding: `BAAI/bge-base-en-v1.5`
- Default generator: `microsoft/phi-3-mini-4k-instruct`
- 4-bit quantization via `BitsAndBytesConfig`
- If no CUDA is available, use a smaller model or disable quantized loading in `src/alz_ras/rag/llm.py`.
- `app.py` (Streamlit) is kept as optional legacy web interface; primary interface is `desktop_app.py`.

## Expected Artifacts
- `data/processed/articles_clean.parquet`
- `artifacts/eda/top_tokens.png`
- `artifacts/eda/wordcloud.png`
- `artifacts/evaluation_metrics.json`

## Current Run Snapshot
- Articles collected: `108` (PubMed + bioRxiv)
- Retrieval metrics: `Hit Rate@4 = 0.8`, `MRR = 0.5`
