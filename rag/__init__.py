"""Retrieval-Augmented Generation (RAG) package.

This package provides helpers to quickly construct a LangChain-based RAG
pipeline over the vector data generated in the project (see `preprocessing/`).

Typical usage:

>>> from rag import build_rag_pipeline
>>> rag = build_rag_pipeline()
>>> rag.ask("Who wrote Sunrise on the Reaping?")
"""

from .pipeline import build_rag_pipeline  # noqa
from .rerank_pipeline import build_rag_rerank_pipeline  # noqa
from .contextual_pipeline import build_rag_contextual_pipeline  # noqa 