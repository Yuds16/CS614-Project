"""RAG pipeline that applies contextual compression on retrieved documents.

It leverages LangChain's `ContextualCompressionRetriever`, which calls an LLM to
reduce each candidate document to only the text relevant to the user's query.
This improves both answer quality and token efficiency, especially when
retrieved documents are long.
"""
from __future__ import annotations

import os
import sys

sys.path.append("../../")

from typing import List

from langchain.chains import RetrievalQA
from langchain_ollama import OllamaLLM
from langchain.prompts import PromptTemplate
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor

from modules.accessor import EMBEDDING_FUNCTIONS, get_collection_names_from_dim

__all__ = [
    "ContextualRAGPipeline",
    "build_rag_contextual_pipeline",
]


class ContextualRAGPipeline:
    """Pipeline that compresses retrieved docs to query-relevant snippets."""

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
        self.k = k

    def ask(self, question: str) -> str:
        return self._chain.invoke({"query": question})["result"]

    def ask_with_sources(self, question: str) -> dict[str, str]:
        result = self._chain.invoke({"query": question})
        sources: List[Document] = result["source_documents"]  # type: ignore
        return {
            "answer": result["result"],
            "sources": "\n---\n".join(doc.page_content for doc in sources),
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_prompt_template() -> PromptTemplate:
    template = (
        "You are an expert assistant. Answer the user's question using ONLY the provided context.\n"
        "If the answer is not present, say you don't know.\n\n"
        "Context:\n{context}\n\nQuestion: {question}\nAnswer:"
    )
    return PromptTemplate.from_template(template)


def build_rag_contextual_pipeline(
    *,
    llm_model: str | None = None,
    embedding_name: str = "all-minilm",
    vector_suffix: str = "512_chunks",
    k: int = 6,
    compression_llm_model: str | None = None,
) -> ContextualRAGPipeline:
    """Create a RAG pipeline that uses contextual compression retrieval.

    Parameters
    ----------
    llm_model
        LLM for final answer generation (Ollama).
    compression_llm_model
        LLM used by the compressor. Defaults to the same model as `llm_model`.
    k
        Number of documents to retrieve from the vector store before compression.
    """

    if embedding_name not in EMBEDDING_FUNCTIONS:
        raise ValueError(
            f"Embedding '{embedding_name}' not found. Available: {list(EMBEDDING_FUNCTIONS)}"
        )

    collection = get_collection_names_from_dim(embedding_name, vector_suffix)

    # Vector store
    db_location = "../../data/chroma_db"
    vector_store = Chroma(
        collection_name=collection,
        persist_directory=db_location,
        embedding_function=EMBEDDING_FUNCTIONS[embedding_name],
    )
    base_retriever = vector_store.as_retriever(search_kwargs={"k": k})

    # Compression retriever setup
    compression_model_name = compression_llm_model or llm_model or os.environ.get("OLLAMA_LLM", "gemma3:4b")
    compression_llm = OllamaLLM(model=compression_model_name)
    compressor = LLMChainExtractor.from_llm(compression_llm)
    retriever = ContextualCompressionRetriever(
        base_compressor=compressor, base_retriever=base_retriever
    )

    # Final answer LLM
    answer_llm_model = llm_model or os.environ.get("OLLAMA_LLM", "gemma:7b")
    answer_llm = OllamaLLM(model=answer_llm_model)

    chain = RetrievalQA.from_chain_type(
        llm=answer_llm,
        retriever=retriever,
        chain_type="stuff",
        return_source_documents=True,
        chain_type_kwargs={"prompt": _build_prompt_template()},
    )

    return ContextualRAGPipeline(
        chain,
        retriever_collection=collection,
        embedding_name=embedding_name,
        k=k,
    )


# ---------------------------------------------------------------------------
# CLI helper
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    import textwrap

    parser = argparse.ArgumentParser(
        description="Interactive console for RAG pipeline with contextual retrieval",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--llm", help="Ollama model name (default: llama2)")
    parser.add_argument("--embedding", help="Embedding key (default: all-minilm)")
    parser.add_argument("--k", type=int, default=6, help="Documents to retrieve before compression (default: 6)")
    args = parser.parse_args()

    pipeline = build_rag_contextual_pipeline(
        llm_model=args.llm,
        embedding_name=args.embedding or "all-minilm",
        k=args.k,
    )

    print(
        textwrap.dedent(
            """
            RAG + Contextual Retrieval interactive console. Type your question and press <enter>.
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