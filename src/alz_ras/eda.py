from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from wordcloud import WordCloud

STOPWORDS = {
    "the",
    "and",
    "for",
    "with",
    "that",
    "from",
    "this",
    "are",
    "was",
    "were",
    "have",
    "has",
    "into",
    "about",
    "their",
    "these",
    "also",
    "using",
    "alzheimer",
    "disease",
    "target",
    "targets",
}

GENE_PATTERN = re.compile(r"\b[A-Z0-9]{3,10}\b")


def _tokenize(text: str) -> list[str]:
    tokens = re.findall(r"[a-zA-Z][a-zA-Z\-]{2,}", text.lower())
    return [t for t in tokens if t not in STOPWORDS]


def build_eda_artifacts(frame: pd.DataFrame, output_dir: Path) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)

    stats = {
        "rows": int(frame.shape[0]),
        "mean_len": float(frame["char_len"].mean()),
        "median_len": float(frame["char_len"].median()),
        "min_len": int(frame["char_len"].min()),
        "max_len": int(frame["char_len"].max()),
    }

    all_text = " ".join(frame["text"].fillna(""))
    token_counts = Counter(_tokenize(all_text))
    top_tokens = token_counts.most_common(30)

    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(12, 6))
    sns.barplot(x=[t[1] for t in top_tokens], y=[t[0] for t in top_tokens], orient="h")
    plt.title("Top-30 Tokens")
    plt.tight_layout()
    plt.savefig(output_dir / "top_tokens.png", dpi=160)
    plt.close()

    wc = WordCloud(width=1400, height=900, background_color="white").generate_from_frequencies(token_counts)
    plt.figure(figsize=(14, 8))
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(output_dir / "wordcloud.png", dpi=160)
    plt.close()

    proteins = Counter()
    for text in frame["text"].fillna(""):
        proteins.update([t for t in GENE_PATTERN.findall(text) if not t.isdigit()])
    top_proteins = proteins.most_common(20)

    if top_proteins:
        plt.figure(figsize=(12, 6))
        sns.barplot(x=[p[1] for p in top_proteins], y=[p[0] for p in top_proteins], orient="h")
        plt.title("Top Mentioned Gene/Protein Tokens")
        plt.tight_layout()
        plt.savefig(output_dir / "top_gene_tokens.png", dpi=160)
        plt.close()

    artifact = {
        "length_stats": stats,
        "top_tokens": top_tokens,
        "top_gene_tokens": top_proteins,
    }
    (output_dir / "eda_summary.json").write_text(json.dumps(artifact, indent=2), encoding="utf-8")
    return artifact
