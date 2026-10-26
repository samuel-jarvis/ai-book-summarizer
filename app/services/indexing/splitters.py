"""Token-based document splitting for the summary and retrieval pipelines."""

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.core.config import settings

SUMMARY_SPLITTER = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    encoding_name=settings.CHUNK_ENCODING,
    chunk_size=settings.SUMMARY_CHUNK_TOKENS,
    chunk_overlap=settings.SUMMARY_CHUNK_OVERLAP_TOKENS,
)


def split_summary_documents(source_documents: list[Document]) -> list[Document]:
    return SUMMARY_SPLITTER.split_documents(source_documents)


RAG_SPLITTER = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    encoding_name=settings.CHUNK_ENCODING,
    chunk_size=settings.RAG_CHUNK_TOKENS,
    chunk_overlap=settings.RAG_CHUNK_OVERLAP_TOKENS,
)


def split_rag_documents(source_documents: list[Document]) -> list[Document]:
    return RAG_SPLITTER.split_documents(source_documents)

