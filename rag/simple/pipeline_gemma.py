"""Utilities for building and interacting with a Retrieval-Augmented Generation (RAG) pipeline.

The pipeline relies on:
1.  A Chroma vector store already populated by the preprocessing notebook.
2.  Ollama models for both embeddings (handled in `modules.accessor`) and LLM generation.

If the vector store is empty, run `preprocessing/data_preprocessing.ipynb` first.
"""
from __future__ import annotations

import os
import sys

# Add project root to path to allow imports from `modules`.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from typing import List, Optional

from langchain.chains import RetrievalQA
from langchain_ollama import OllamaLLM
from langchain.prompts import PromptTemplate
from langchain_chroma import Chroma
from langchain_core.documents import Document

from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

from modules.accessor import EMBEDDING_FUNCTIONS, get_collection_names_from_dim
__all__ = [
    "RAGPipeline",
    "build_rag_pipeline",
]


class RAGPipeline:
    """A thin wrapper around a LangChain `RetrievalQA` chain."""

    def __init__(
        self,
        chain: RetrievalQA,
        *,
        retriever_collection: str,
        embedding_name: str,
        k: int = 4,
    ) -> None:
        self._chain = chain
        self.retriever_collection = retriever_collection
        self.embedding_name = embedding_name
        self.k = k

    # ---------------------------------------------------------------------
    # Public helpers
    # ---------------------------------------------------------------------

    def ask(self, question: str, *, k: Optional[int] = None) -> str:
        """Run a question through the RAG pipeline and return the answer."""
        if not question:
            raise ValueError("Question cannot be empty.")
        return self._chain.invoke({"query": question, "k": k or self.k})[
            "result"
        ]

    def ask_with_sources(self, question: str, *, k: Optional[int] = None) -> dict[str, str]:
        """Return answer plus a concatenated string of sources."""
        result = self._chain.invoke({"query": question, "k": k or self.k})
        sources: List[Document] = result["source_documents"]  # type: ignore[assignment]
        source_text = "\n---\n".join(doc.page_content for doc in sources)
        return {"answer": result["result"], "sources": source_text}


# -------------------------------------------------------------------------
# Builder
# -------------------------------------------------------------------------

def _build_prompt_template() -> PromptTemplate:
    # template = (
    #     "You are an expert assistant. Answer the user's question based on the provided documents.\n"
    #     "It's MCQ question, please give me the answer in the format of 'A', 'B', 'C', 'D'.If you are not sure, please give me the answer in the format of 'I don't know'.\n"
    #     "Context:\n{context}\n\nQuestion: {question}\nAnswer:"  # noqa: E501
    # )
    # return PromptTemplate.from_template(template)

    system_prompt = (
    "You are an expert assistant. Answer the user's question based on the provided documents.\n"
        "It's MCQ question, please give me the answer in the format of 'A', 'B', 'C', 'D'.No need to explain the answer. Context: {context}"
)
#     system_prompt = (
#     "You are an expert assistant. Answer the user's question based on the provided documents. "
#         "If the answer is not in the documents, say you don't know.\ Context: {context}"
# )
    # user_prompt=(
    #     "You are an expert assistant. Answer the user's question based on the provided documents.\n"
    #     "It's MCQ question, please give me the answer in the format of 'A', 'B', 'C', 'D'. Context: {context}"
    # )
    system_msg = SystemMessagePromptTemplate.from_template(system_prompt)
    human_msg = HumanMessagePromptTemplate.from_template("{question}")
    # human_msg = HumanMessagePromptTemplate.from_template(user_prompt+"Question: {question}")

    return ChatPromptTemplate.from_messages([system_msg, human_msg])
    # return ChatPromptTemplate.from_messages([human_msg])


def build_rag_pipeline(
    *,
    llm_model: str | None = None,
    embedding_name: str = "all-minilm",
    vector_suffix: str = "512_chunks",
    k: int = 4,
) -> RAGPipeline:
    """Create a ready-to-use `RAGPipeline` instance.

    Parameters
    ----------
    llm_model
        Name of the Ollama model to use for generation. Defaults to the value of
        the `OLLAMA_LLM` env-var or to "llama2".
    embedding_name
        Key used in :data:`modules.accessor.EMBEDDING_FUNCTIONS` that identifies
        the embedding model that was used to create the vectors. This *must*
        match what was used during preprocessing.
    vector_suffix
        Custom suffix used when persisting the chunks ("full_text", "512_chunks", …).
    k
        Number of documents to retrieve.
    """

    if embedding_name not in EMBEDDING_FUNCTIONS:
        raise ValueError(
            f"Embedding '{embedding_name}' not found. Available: {list(EMBEDDING_FUNCTIONS)}"
        )

    collection = get_collection_names_from_dim(embedding_name, vector_suffix)

    # ------------------------------------------------------------------
    # Vector store + retriever
    # ------------------------------------------------------------------
    db_location = "../../data/chroma_db"
    vector_store = Chroma(
        collection_name=collection,
        persist_directory=db_location,
        embedding_function=EMBEDDING_FUNCTIONS[embedding_name],
    )

    print(vector_store.similarity_search(query='a',k=1))
    retriever = vector_store.as_retriever(search_kwargs={"k": k})

    # ------------------------------------------------------------------
    # LLM and RAG chain
    # ------------------------------------------------------------------
    llm_model = llm_model or os.environ.get("OLLAMA_LLM", "gemma:7b")
    llm = OllamaLLM(model=llm_model)  # Ensure the model is pulled in your Ollama server

    prompt = _build_prompt_template()

    chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt},
    )

    return RAGPipeline(chain, retriever_collection=collection, embedding_name=embedding_name, k=k)


# -------------------------------------------------------------------------
# CLI helper
# -------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    import textwrap

    parser = argparse.ArgumentParser(
        description="Simple interactive CLI for the RAG pipeline",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--llm", help="Ollama model name (default: llama2)")
    parser.add_argument(
        "--embedding",
        help="Embedding key as defined in modules.accessor (default: all-minilm)",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=4,
        help="Number of documents to retrieve (default: 4)",
    )
    args = parser.parse_args()

    pipeline = build_rag_pipeline(
        llm_model=args.llm,
        embedding_name=args.embedding or "all-minilm",
        k=args.k,
    )

    print(
        textwrap.dedent(
            """
            Interactive RAG console. Type your questions and hit <enter>.
            Press Ctrl-D to exit.
            """
        )
    )

    try:
        while True:
            try:
                question = input("\n> ")
            except EOFError:
                break
            answer_data = pipeline.ask_with_sources(question)
            print("\nAnswer:\n" + answer_data["answer"])
            print("\nSources:\n" + answer_data["sources"])
    except KeyboardInterrupt:
        pass 