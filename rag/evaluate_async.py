"""Async evaluation of a RAG pipeline against a JSON Q&A dataset.

This script loads a set of *reference* Question/Answer pairs from a JSON file,
uses a RAG pipeline (built via `rag.simple.pipeline_gemma.build_rag_pipeline`)
to generate *predicted* answers **concurrently**, and then evaluates the
predictions with industry-standard extractive-QA metrics:

1. Exact-Match (EM)
2. Token-level F1 (SQuAD style)

Usage
-----
python rag/evaluate_async.py \
    --data fine_tuning2/reasoned_qa_output/allfours_test_data.json \
    --llm gemma:7b \
    --embedding all-minilm \
    --k 4 \
    --concurrency 4

Notes
-----
• Requires Python ≥3.9.
• Runs purely locally using the Ollama model specified via ``--llm``.
• Make sure your Chroma vector store is already populated (see
  ``preprocessing/data_preprocessing.ipynb``).
"""
from __future__ import annotations

import argparse
import asyncio
import json
import re
from collections import Counter
from functools import partial
from pathlib import Path
from typing import List, Tuple

from rag.simple.pipeline_gemma import build_rag_pipeline, RAGPipeline  # Re-use existing builder

# ---------------------------------------------------------------------------
# Utility: text normalisation ------------------------------------------------
# ---------------------------------------------------------------------------

def _normalize(text: str) -> str:
    """Lower-case, strip, and collapse whitespace."""
    return re.sub(r"\s+", " ", text.strip().lower())


def _tokenize(text: str) -> List[str]:
    """Whitespace tokeniser after normalisation."""
    return _normalize(text).split()


# ---------------------------------------------------------------------------
# Metrics -------------------------------------------------------------------
# ---------------------------------------------------------------------------

def squad_em(pred: str, ref: str) -> float:
    """Exact-match score (case- and whitespace-insensitive)."""
    return float(_normalize(pred) == _normalize(ref))


def squad_f1(pred: str, ref: str) -> float:
    """Token-level F1 (SQuAD style)."""
    pred_tokens = _tokenize(pred)
    ref_tokens = _tokenize(ref)
    common = Counter(pred_tokens) & Counter(ref_tokens)
    num_same = sum(common.values())

    if num_same == 0:
        return 0.0

    precision = num_same / len(pred_tokens)
    recall = num_same / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


# ---------------------------------------------------------------------------
# Async inference ------------------------------------------------------------
# ---------------------------------------------------------------------------

async def _ask_async(pipe: RAGPipeline, question: str, k: int) -> str:
    """Run ``pipe.ask`` inside a ThreadPool because it's blocking."""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, partial(pipe.ask, question, k=k))


async def gather_predictions(
    pipe: RAGPipeline,
    questions: List[str],
    *,
    k: int,
    concurrency: int = 4,
) -> List[str]:
    """Fetch answers concurrently with the desired level of parallelism."""

    semaphore = asyncio.Semaphore(concurrency)

    async def _sema_ask(q: str) -> str:
        async with semaphore:
            return await _ask_async(pipe, q, k)

    return await asyncio.gather(*[_sema_ask(q) for q in questions])


# ---------------------------------------------------------------------------
# Main evaluation routine ----------------------------------------------------
# ---------------------------------------------------------------------------

def evaluate(
    refs: List[Tuple[str, str]],
    *,
    llm_model: str,
    embedding_name: str,
    k: int,
    concurrency: int,
) -> None:
    """Run the end-to-end evaluation and print aggregate metrics."""

    questions = [q for q, _ in refs]
    gold_answers = [a for _, a in refs]

    pipe = build_rag_pipeline(
        llm_model=llm_model,
        embedding_name=embedding_name,
        k=k,
    )

    print(f"\nGenerating predictions with model '{llm_model}' …")
    preds = asyncio.run(gather_predictions(pipe, questions, k=k, concurrency=concurrency))

    # Metric computation
    em_scores = [squad_em(p, g) for p, g in zip(preds, gold_answers)]
    f1_scores = [squad_f1(p, g) for p, g in zip(preds, gold_answers)]

    mean_em = sum(em_scores) / len(em_scores)
    mean_f1 = sum(f1_scores) / len(f1_scores)

    print("\n=== Evaluation Results ===")
    print(f"Samples evaluated : {len(refs)}")
    print(f"Exact-Match (EM) : {mean_em * 100:.2f}%")
    print(f"Token F1         : {mean_f1 * 100:.2f}%")


# ---------------------------------------------------------------------------
# CLI -----------------------------------------------------------------------
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Async RAG evaluation")
    p.add_argument("--data", required=True, help="Path to JSON file with Question/Answer pairs")
    p.add_argument("--llm", default=None, help="Ollama model name (default: env OLLAMA_LLM or gemma:7b)")
    p.add_argument("--embedding", default="all-minilm", help="Embedding key (default: all-minilm)")
    p.add_argument("--k", type=int, default=4, help="Retriever top-k (default: 4)")
    p.add_argument("--concurrency", type=int, default=4, help="Number of concurrent requests (default: 4)")
    return p.parse_args()


def _load_refs(path: Path) -> List[Tuple[str, str]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    refs: List[Tuple[str, str]] = [(item["Question"], item["Answer"]) for item in data]
    return refs


if __name__ == "__main__":
    args = _parse_args()
    # Resolve the dataset path. If the user passes a relative path, first
    # interpret it relative to the current working directory; if that fails,
    # fall back to interpreting it relative to the project *root* (the parent
    # of this "rag" package).
    dataset_path = Path(args.data)

    if not dataset_path.exists():
        root_dir = Path(__file__).resolve().parent.parent  # project root
        alt_path = root_dir / args.data
        if alt_path.exists():
            dataset_path = alt_path
        else:
            raise FileNotFoundError(args.data)

    references = _load_refs(dataset_path)

    evaluate(
        references,
        llm_model=args.llm,
        embedding_name=args.embedding,
        k=args.k,
        concurrency=args.concurrency,
    ) 