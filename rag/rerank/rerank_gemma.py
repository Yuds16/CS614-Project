"""An upgraded RAG pipeline that re-ranks retrieved documents with a cross-encoder.

This module introduces `build_rag_rerank_pipeline`, which enhances basic vector
similarity retrieval with a secondary re-ranking stage. It uses an
open-source cross-encoder from `sentence_transformers` so it can run fully
offline.

The overall flow:
1. Vector search (Chroma) → top `top_n` candidates.
2. Cross-encoder scores each (query, doc) pair.
3. Keep the highest-scoring `rerank_k` docs.
4. Feed those docs into an LLM (Ollama) via LangChain `RetrievalQA`.
"""
from __future__ import annotations

import os
import sys
import asyncio
from typing import List, Optional
# Add project root to path to allow imports from `modules`.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from langchain.chains import RetrievalQA
from langchain_ollama import OllamaLLM
from langchain.prompts import PromptTemplate
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain.schema import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from sentence_transformers import CrossEncoder


from modules.accessor import EMBEDDING_FUNCTIONS, get_collection_names_from_dim

__all__ = [
    "RerankRAGPipeline",
    "build_rag_rerank_pipeline",
]

# ---------------------------------------------------------------------------
# Custom retriever with rerank
# ---------------------------------------------------------------------------


class _RerankRetriever(BaseRetriever):
    """Wrap a base retriever with a cross-encoder re-ranking step."""

    def __init__(
        self,
        base_retriever: BaseRetriever,
        cross_encoder: CrossEncoder,
        *,
        top_n: int = 20,
        rerank_k: int = 4,
    ) -> None:
        super().__init__()
        # Use `object.__setattr__` to bypass Pydantic's immutability checks.
        object.__setattr__(self, "_base_retriever", base_retriever)
        object.__setattr__(self, "_cross_encoder", cross_encoder)
        object.__setattr__(self, "top_n", top_n)
        object.__setattr__(self, "rerank_k", rerank_k)

    # ------------------------- sync -------------------------------------

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun | None = None,
        **kwargs,
    ) -> List[Document]:
        docs = self._base_retriever.invoke(query, **kwargs)
        # Limit to top_n candidates before scoring to avoid OOM
        docs = docs[: self.top_n]
        return self._rerank(query, docs)

    # ------------------------- async ------------------------------------

    async def _aget_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun | None = None,
        **kwargs,
    ) -> List[Document]:
        docs = await self._base_retriever.ainvoke(query, **kwargs)
        docs = docs[: self.top_n]
        return await asyncio.to_thread(self._rerank, query, docs)

    # ------------------------- helpers ----------------------------------

    def _rerank(self, query: str, docs: List[Document]) -> List[Document]:
        if not docs:
            return []
        # Prepare inputs for cross-encoder: (query, doc)
        pairs = [(query, d.page_content[: 1024]) for d in docs]
        scores = self._cross_encoder.predict(pairs)
        scored = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
        return [doc for doc, _ in scored[: self.rerank_k]]


# ---------------------------------------------------------------------------
# High-level pipeline wrapper
# ---------------------------------------------------------------------------


class RerankRAGPipeline:
    """RAG pipeline that performs similarity search + cross-encoder re-rank."""

    def __init__(
        self,
        chain: RetrievalQA,
        *,
        retriever_collection: str,
        embedding_name: str,
        k: int,
    ) -> None:
        self._chain = chain
        self.retriever_collection = retriever_collection
        self.embedding_name = embedding_name
        self.k = k  # final k after rerank

    def ask(self, question: str) -> str:
        return self._chain.invoke({"query": question})["result"]

    def ask_with_sources(self, question: str) -> dict[str, str]:
        result = self._chain.invoke({"query": question})
        sources: List[Document] = result["source_documents"]  # type: ignore
        return {
            "answer": result["result"],
            "sources": "\n---\n".join(d.page_content for d in sources),
        }


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------


def _build_chat_prompt_template() -> ChatPromptTemplate:
    """Create a chat prompt where *context* is placed inside the system role.

    The caller may inject a custom *system_prompt* containing the ``{context}`` and
    ``{question}`` placeholders. When *system_prompt* is *None* a sensible
    default is provided.
    """

    system_prompt = (
        "You are an expert assistant. Answer the user's MCQ question based on the provided context.\n"
        "Give me ONLY the letter corresponding to the chosen answer in the format of 'A', 'B', 'C', 'D'.\n"
        "No need to explain anything. No need to print anything else.\n"
        "Your output response should ONLY contain 1 letter. Example: 'A'.\n"
        "Context: {context}"
    )
    system_msg = SystemMessagePromptTemplate.from_template(system_prompt)
    human_msg = HumanMessagePromptTemplate.from_template("{question}")

    return ChatPromptTemplate.from_messages([system_msg, human_msg])


def build_rag_rerank_pipeline(
    *,
    llm_model: str | None = None,
    embedding_name: str = "all-minilm",
    vector_suffix: str = "512_chunks",
    top_n: int = 20,
    rerank_k: int = 4,
    cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L12-v2",
) -> RerankRAGPipeline:
    """Construct a RAG pipeline with a cross-encoder re-ranking stage.

    Parameters
    ----------
    llm_model
        Ollama model name to generate answers.
    embedding_name
        Key matching what was used during preprocessing.
    vector_suffix
        Suffix identifying the vector collection.
    top_n
        Number of docs fetched from vector store before re-ranking.
    rerank_k
        Final number of docs kept after re-ranking. Must be ≤ top_n.
    cross_encoder_model
        HuggingFace model name for cross-encoder (defaults to a small MS-Marco
        model). Larger models generally rerank better but are slower.
    """

    if rerank_k > top_n:
        raise ValueError("`rerank_k` must be less than or equal to `top_n`.")

    if embedding_name not in EMBEDDING_FUNCTIONS:
        raise ValueError(
            f"Embedding '{embedding_name}' not found. Available: {list(EMBEDDING_FUNCTIONS)}"
        )

    collection = get_collection_names_from_dim(embedding_name, vector_suffix)

    db_location = "../../data/chroma_db"
    # Vector store retriever
    vector_store = Chroma(
        collection_name=collection,
        persist_directory=db_location,
        embedding_function=EMBEDDING_FUNCTIONS[embedding_name],
    )
    
    base_retriever = vector_store.as_retriever(search_kwargs={"k": top_n})

    # Cross-encoder
    cross_encoder = CrossEncoder(cross_encoder_model)

    retriever = _RerankRetriever(
        base_retriever=base_retriever,
        cross_encoder=cross_encoder,
        top_n=top_n,
        rerank_k=rerank_k,
    )

    llm_model = llm_model or os.environ.get("OLLAMA_LLM", "gemma:7b")
    llm = OllamaLLM(model=llm_model)

    prompt = _build_chat_prompt_template()

    chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt},
    )

    return RerankRAGPipeline(
        chain,
        retriever_collection=collection,
        embedding_name=embedding_name,
        k=rerank_k,
    )


# ---------------------------------------------------------------------------
# CLI helper
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    import textwrap

    parser = argparse.ArgumentParser(
        description="Interactive console for RAG pipeline with cross-encoder rerank",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--llm", help="Ollama model name (default: llama2)")
    parser.add_argument("--embedding", help="Embedding key (default: all-minilm)")
    parser.add_argument("--top-n", type=int, default=20, help="Initial docs to retrieve (default: 20)")
    parser.add_argument("--rerank-k", type=int, default=4, help="Docs kept after rerank (default: 4)")
    parser.add_argument(
        "--cross-encoder",
        default="cross-encoder/ms-marco-MiniLM-L12-v2",
        help="HuggingFace cross-encoder model name",
    )
    args = parser.parse_args()

    pipeline = build_rag_rerank_pipeline(
        llm_model=args.llm,
        embedding_name=args.embedding or "all-minilm",
        top_n=args.top_n,
        rerank_k=args.rerank_k,
        cross_encoder_model=args.cross_encoder,
    )

    print(
        textwrap.dedent(
            """
            RAG + Rerank interactive console. Type your questions and press <enter>.
            Ctrl-D or Ctrl-C to exit.
            """
        )
    )

    try:
        while True:
            try:
                question = input("\n> ")
            except EOFError:
                break
            answer = pipeline.ask_with_sources(question)
            print("\nAnswer:\n" + answer["answer"])
            print("\nSources:\n" + answer["sources"])
    except KeyboardInterrupt:
        pass 