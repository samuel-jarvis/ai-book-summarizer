"""Project evidence retrieval for conversation runs.

Deliberately simple: the nearest `k` chunks are the sources, except that in a
multi-book project each book keeps a minimum share so one book cannot fill every
slot (see `VectorStoreService.balanced_search`). Deduplication and distance
cutoffs can be layered on later; evals/EXPERIMENTS.md tracks what each change did.
"""

from collections.abc import Sequence
from dataclasses import dataclass, field
from uuid import UUID

from langchain_core.documents import Document as LangChainDocument
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.models.conversation import CandidateSnapshot, RunSnapshot, SourceSnapshot
from app.services.indexing.vector_store import VectorStoreService

EXCERPT_CHARS = 600


@dataclass(frozen=True)
class RetrievalResult:
    # Full chunk text for the context builder.
    documents: list[LangChainDocument]
    # Citation-labelled sources for the run snapshot and sources panel.
    sources: list[SourceSnapshot]
    # The searches that produced these chunks.
    queries: list[str] = field(default_factory=list)

    def to_run_snapshot(self, previous: RunSnapshot | None = None) -> RunSnapshot:
        """Copy retrieval into a snapshot, keeping its other captured inputs."""
        snapshot = previous.model_copy(deep=True) if previous else RunSnapshot()
        snapshot.search_queries = list(self.queries)
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
        min_per_document: int | None = None,
        queries: Sequence[str] | None = None,
    ) -> RetrievalResult:
        """Return the `k` best chunks for the question.

        With two or more `queries` (a question split into parts) each is searched on
        its own and gets an equal share of the results. Otherwise `query` is searched
        once and each book keeps a minimum share: `min_per_document` defaults to
        `RAG_MIN_CHUNKS_PER_DOCUMENT`, and 0 means plain nearest-k. The per-book
        minimum only matters in projects with more than one book.
        """
        store = VectorStoreService(self.db, self.user_id)
        if queries is not None and len(queries) > 1:
            used = list(queries)
            documents = await store.multi_query_search(
                used, project_id=project_id, k=k, document_ids=document_ids
            )
        else:
            used = [query]
            floor = (
                settings.RAG_MIN_CHUNKS_PER_DOCUMENT
                if min_per_document is None
                else min_per_document
            )
            documents = await store.balanced_search(
                query,
                project_id=project_id,
                k=k,
                min_per_document=floor,
                document_ids=document_ids,
            )
        sources = [
            to_source(document, f"S{index}")
            for index, document in enumerate(documents, start=1)
        ]
        return RetrievalResult(documents=documents, sources=sources, queries=used)
