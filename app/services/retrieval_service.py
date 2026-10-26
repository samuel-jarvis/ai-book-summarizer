"""Project evidence retrieval for conversation runs.

Deliberately simple: the nearest `k` chunks are the sources. Deduplication,
distance cutoffs, and per-book coverage can be layered on once we have evals.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from uuid import UUID

from langchain_core.documents import Document as LangChainDocument
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.conversation import CandidateSnapshot, RunSnapshot, SourceSnapshot
from app.services.indexing.vector_store import VectorStoreService

EXCERPT_CHARS = 600


@dataclass(frozen=True)
class RetrievalResult:
    # Full chunk text for the context builder.
    documents: list[LangChainDocument]
    # Citation-labelled sources for the run snapshot and sources panel.
    sources: list[SourceSnapshot]

    def to_run_snapshot(self, previous: RunSnapshot | None = None) -> RunSnapshot:
        """Copy retrieval into a snapshot, keeping its other captured inputs."""
        snapshot = previous.model_copy(deep=True) if previous else RunSnapshot()
        snapshot.retrieval_candidates = [
            CandidateSnapshot(
                chunk_id=source.chunk_id,
                document_id=source.document_id,
                cosine_distance=source.cosine_distance,
                selected=True,
            )
            for source in self.sources
        ]
        snapshot.selected_sources = [
            source.model_copy(deep=True) for source in self.sources
        ]
        return snapshot


def to_source(document: LangChainDocument, label: str) -> SourceSnapshot:
    metadata = document.metadata
    return SourceSnapshot(
        chunk_id=document.id,
        document_id=metadata["document_id"],
        generation_id=metadata["generation_id"],
        document_title=metadata["document_title"],
        chunk_index=metadata["chunk_index"],
        page_start=metadata.get("page_start"),
        page_end=metadata.get("page_end"),
        cosine_distance=metadata["cosine_distance"],
        citation_label=label,
        excerpt=document.page_content[:EXCERPT_CHARS],
    )


class RetrievalService:
    def __init__(self, db: AsyncSession, user_id: UUID) -> None:
        self.db = db
        self.user_id = user_id

    async def retrieve(
        self,
        query: str,
        *,
        project_id: UUID,
        document_ids: Sequence[UUID] | None = None,
        k: int = 6,
    ) -> RetrievalResult:
        documents = await VectorStoreService(self.db, self.user_id).similarity_search(
            query, project_id=project_id, k=k, document_ids=document_ids
        )
        sources = [
            to_source(document, f"S{index}")
            for index, document in enumerate(documents, start=1)
        ]
        return RetrievalResult(documents=documents, sources=sources)
