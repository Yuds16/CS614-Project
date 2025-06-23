"""
Accuracy evaluation script for Multiple-Choice QA datasets using our RAG pipeline.

Given a JSON file of the shape:
[
  {
    "question": "…",
    "options": {"A": "…", "B": "…", "C": "…", "D": "…"},
    "answer" : "B"
  },
  …
]

The script builds a RAG pipeline (defaults to `rag.simple.pipeline_qwen`) and
queries it for every question (question + options). Predicted answers are
compared against the ground-truth letter (A/B/C/D) and overall accuracy is
reported.

A live progress-bar (tqdm) shows evaluation progress.

Usage
-----
python rag/accuracy_eval.py \
    --data rag/data/qa_data.json \
    --llm qwen3:8b \
    --embedding all-minilm \
    --k 4
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import List, Tuple

from tqdm import tqdm

# Re-use the existing simple RAG builder
# from simple.pipeline_qwen import build_rag_pipeline, RAGPipeline
from simple.pipeline_gemma import build_rag_pipeline, RAGPipeline
from rerank.rerank_gemma import build_rag_rerank_pipeline,RerankRAGPipeline
from rerank.rerank_qwen import build_rag_rerank_pipeline,RerankRAGPipeline
from contextual.contextual_gemma import build_rag_contextual_pipeline,ContextualRAGPipeline
from contextual.contextual_qwen import build_rag_contextual_pipeline,ContextualRAGPipeline

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_prompt(q: str, opts: dict[str, str]) -> str:
    """Format *q* and *opts* into a single prompt understood by the pipeline."""
    return (
        f"{q}\n"
        f"A. {opts['A']}\n"
        f"B. {opts['B']}\n"
        f"C. {opts['C']}\n"
        f"D. {opts['D']}"
    )


def _extract_choice(text: str, llm_model: str) -> str:
    """Return the first choice letter (A–D) **after** any </think> tag.

    Many model outputs are of the form:
        <think> … internal reasoning … </think>\nA
    We ignore everything up to and including the closing tag so that stray
    capital letters inside the reasoning block don't interfere with the
    extraction.
    """
    with open(f"{llm_model}_results.txt", "a") as f:
        f.write(text)
    # Keep only the substring *after* the reasoning marker, if present.

    if "</think>" in text and llm_model == "qwen3:8b":
        text = text.split("</think>")[-1]

    match = re.search(r"\b([ABCD])\b", text, flags=re.I)
    return match.group(1).upper() if match else ""


# ---------------------------------------------------------------------------
# Core evaluation
# ---------------------------------------------------------------------------

def evaluate(
    refs: List[Tuple[str, str]],
    *,
    llm_model: str | None,
    embedding_name: str,
    k: int,
) -> None:
    """Run evaluation and print aggregate accuracy."""

    pipe: RAGPipeline = build_rag_pipeline(
        llm_model=llm_model,
        embedding_name=embedding_name,
        k=k,
    )

    correct = 0
    total = len(refs)

    for prompt, gold in tqdm(refs, desc="Evaluating", unit="q"):
        pred_raw = pipe.ask(prompt, k=k)
        pred = _extract_choice(pred_raw,llm_model)
        correct += int(pred == gold)

    accuracy = correct / total if total else 0.0
    print("\n=== Accuracy Report ===")
    print(f"Samples evaluated : {total}")
    print(f"Correct           : {correct}")
    print(f"Accuracy          : {accuracy * 100:.2f}%")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="MCQ accuracy evaluation with progress bar")
    p.add_argument("--data", required=True, help="Path to JSON dataset")
    p.add_argument("--llm", default=None, help="Ollama model name (default: env OLLAMA_LLM or qwen3:8b)")
    p.add_argument("--embedding", default="all-minilm", help="Embedding model key (default: all-minilm)")
    p.add_argument("--k", type=int, default=4, help="Retriever top-k (default: 4)")
    return p.parse_args()


def _load_mcq_refs(path: Path) -> List[Tuple[str, str]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return [(_build_prompt(item["question"], item["options"]), item["answer"].strip().upper()) for item in data]


if __name__ == "__main__":
    args = _parse_args()

    ds_path = Path(args.data)
    if not ds_path.exists():
        raise FileNotFoundError(ds_path)

    references = _load_mcq_refs(ds_path)

    evaluate(
        references,
        llm_model=args.llm,
        embedding_name=args.embedding,
        k=args.k,
    ) 