from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from html import unescape
from typing import Any

import pandas as pd
import requests
from pymed import PubMed

from alz_ras.config import DataConfig


BIBLIORXIV_URL = "https://api.biorxiv.org/details/biorxiv"


@dataclass
class ArticleRecord:
    source: str
    query: str
    title: str
    abstract: str
    introduction: str
    conclusion: str
    doi: str
    url: str
    journal: str
    authors: str
    published_at: str


def clean_text(text: str | None) -> str:
    if not text:
        return ""
    text = unescape(text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"&[a-zA-Z0-9#]+;", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def split_sections(text: str) -> tuple[str, str, str]:
    normalized = text.lower()
    intro_marker = normalized.find("introduction")
    conclusion_marker = max(normalized.find("conclusion"), normalized.find("discussion"))

    intro = ""
    conclusion = ""
    if intro_marker != -1:
        intro = text[intro_marker: conclusion_marker if conclusion_marker > intro_marker else len(text)]
    if conclusion_marker != -1:
        conclusion = text[conclusion_marker:]

    return clean_text(text), clean_text(intro), clean_text(conclusion)


def fetch_pubmed_articles(config: DataConfig) -> list[ArticleRecord]:
    client = PubMed(tool="alz-rag-agent", email="demo@example.com")
    records: list[ArticleRecord] = []

    for query in config.queries:
        pubmed_results = client.query(query, max_results=config.max_pubmed_results_per_query)
        for article in pubmed_results:
            title = clean_text(getattr(article, "title", ""))
            abstract_text = clean_text(getattr(article, "abstract", ""))
            full_text, intro, conclusion = split_sections(abstract_text)

            doi = ""
            if getattr(article, "doi", None):
                doi = str(article.doi)

            pub_date = getattr(article, "publication_date", None)
            published_at = pub_date.isoformat() if isinstance(pub_date, datetime) else ""

            authors: list[str] = []
            for author in getattr(article, "authors", []) or []:
                firstname = getattr(author, "firstname", "")
                lastname = getattr(author, "lastname", "")
                name = f"{firstname} {lastname}".strip()
                if name:
                    authors.append(name)

            url = f"https://pubmed.ncbi.nlm.nih.gov/{getattr(article, 'pubmed_id', '')}/"
            records.append(
                ArticleRecord(
                    source="pubmed",
                    query=query,
                    title=title,
                    abstract=full_text,
                    introduction=intro,
                    conclusion=conclusion,
                    doi=doi,
                    url=url,
                    journal=clean_text(getattr(article, "journal", "")),
                    authors=", ".join(authors),
                    published_at=published_at,
                )
            )

    return records


def _extract_biorxiv_response(response_json: dict[str, Any]) -> list[dict[str, Any]]:
    collection = response_json.get("collection")
    if isinstance(collection, list):
        return collection
    return []


def fetch_biorxiv_articles(config: DataConfig) -> list[ArticleRecord]:
    records: list[ArticleRecord] = []
    for query in config.queries:
        response = requests.get(f"{BIBLIORXIV_URL}/2020-01-01/3000-01-01", timeout=30)
        response.raise_for_status()
        payload = _extract_biorxiv_response(response.json())

        matched = 0
        query_tokens = {token.lower() for token in query.replace("'", "").split()}
        for item in payload:
            if matched >= config.max_biorxiv_results_per_query:
                break

            title = clean_text(item.get("title", ""))
            abstract = clean_text(item.get("abstract", ""))
            searchable = f"{title} {abstract}".lower()
            if not any(token in searchable for token in query_tokens):
                continue

            full_text, intro, conclusion = split_sections(abstract)
            doi = clean_text(item.get("doi", ""))
            url = f"https://www.biorxiv.org/content/{doi}v1" if doi else ""

            records.append(
                ArticleRecord(
                    source="biorxiv",
                    query=query,
                    title=title,
                    abstract=full_text,
                    introduction=intro,
                    conclusion=conclusion,
                    doi=doi,
                    url=url,
                    journal="bioRxiv",
                    authors=clean_text(item.get("authors", "")),
                    published_at=clean_text(item.get("date", "")),
                )
            )
            matched += 1

    return records


def build_dataset(config: DataConfig) -> pd.DataFrame:
    pubmed_records = fetch_pubmed_articles(config)
    biorxiv_records = fetch_biorxiv_articles(config)

    all_records = pubmed_records + biorxiv_records
    if not all_records:
        raise ValueError("No records fetched from PubMed/bioRxiv.")

    frame = pd.DataFrame([r.__dict__ for r in all_records])
    frame["text"] = (
        frame["title"].fillna("")
        + "\n\n"
        + frame["abstract"].fillna("")
        + "\n\n"
        + frame["introduction"].fillna("")
        + "\n\n"
        + frame["conclusion"].fillna("")
    ).apply(clean_text)
    frame = frame.drop_duplicates(subset=["title", "doi", "source"]).reset_index(drop=True)
    frame["char_len"] = frame["text"].str.len()

    return frame


def save_dataset(frame: pd.DataFrame, config: DataConfig) -> None:
    config.raw_data_path.parent.mkdir(parents=True, exist_ok=True)
    config.processed_data_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(config.raw_data_path, index=False)
    frame.to_parquet(config.processed_data_path, index=False)
    frame.to_csv(config.processed_data_path.with_suffix(".csv"), index=False)
