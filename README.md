# Alzheimer RAS Agent

Prototype **Retrieval-Augmented System (RAS)** for helping researchers identify Alzheimer's disease therapeutic targets from biomedical literature.
## Setup

```bash
python3.10 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
pip install -e .
```

## Notes
- Default embedding: `BAAI/bge-base-en-v1.5`
- Default generator: `microsoft/phi-3-mini-4k-instruct`
- 4-bit quantization via `BitsAndBytesConfig`
- If no CUDA is available, use a smaller model or disable quantized loading in `src/alz_ras/rag/llm.py`.
- `app.py` (Streamlit) is kept as optional legacy web interface; primary interface is `desktop_app.py`.

## Run
- After building the project, you need to go to the root folder and run run_app.bat.
