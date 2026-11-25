"""Document embeddings stored through SQLAlchemy and PostgreSQL pgvector."""

import math
from collections import Counter
from collections.abc import Sequence
from uuid import UUID, uuid4

import tiktoken
from langchain_core.documents import Document as LangChainDocument
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.exception import NotFoundError
from app.models.document import Document, EmbeddingStatus
from app.models.document_embedding import DocumentEmbedding, DocumentEmbeddingMetadata
from app.models.embedding_generation import EmbeddingGeneration
from app.models.project import Project
from app.models.project_documents import ProjectDocument
from app.services.indexing.embedder import embed_query, embed_texts


def merge_balanced(
    candidates: list[LangChainDocument], k: int, min_per_document: int
) -> list[LangChainDocument]:
    """Pick up to `k` chunks, nearest first, keeping each book's best `min_per_document`.

    Each book's closest chunks are reserved (the closest overall if reservations
    alone exceed `k`); the remaining slots go to the nearest other candidates.
    """
    ranked = sorted(
        {doc.id: doc for doc in candidates}.values(),
        key=lambda doc: (doc.metadata["cosine_distance"], doc.id),
    )
    taken: Counter[str] = Counter()
    chosen: set[str] = set()
    for doc in ranked:
        book = doc.metadata["document_id"]
        if taken[book] < min_per_document and len(chosen) < k:
            taken[book] += 1
            chosen.add(doc.id)
    for doc in ranked:
        if len(chosen) >= k:
            break
        chosen.add(doc.id)
    return [doc for doc in ranked if doc.id in chosen]


class VectorStoreService:
    """User-scoped ORM storage. The caller commits or rolls back the session.

    Each nonempty batch is the complete replacement index for one document.
    Failed builds leave the previous generation active. Historical generations
    remain stored until the document is deleted.
    """

    def __init__(self, db: AsyncSession, user_id: UUID) -> None:
        self.db = db
        self.user_id = user_id

    @staticmethod
    def _validate_vectors(vectors: list[list[float]], count: int) -> None:
        if len(vectors) != count or any(
            len(vector) != settings.EMBEDDING_DIMENSION
            or not all(math.isfinite(value) for value in vector)
            or not any(vector)
            for vector in vectors
        ):
            raise ValueError(
                "Embedding response has invalid count, dimensions, or values."
            )

    async def add_document_chunks(
        self,
        chunks: list[LangChainDocument],
        *,
        document_id: UUID,
        project_id: UUID,
        chunking_version: str | None = None,
        vectors: list[list[float]] | None = None,
    ) -> list[str]:
        """Embed a complete document batch and atomically activate its generation.

        Explicit IDs determine the batch scope, independently of chunk metadata.
        The caller supplies chunks for that document. It must belong to the caller and
        be attached to the owned project. A savepoint protects the previous
        generation on failure; no partial replacement is committed here.

        Pass `vectors` (one per chunk, already embedded) to keep the provider
        call outside the row lock this method takes; otherwise it embeds here.
        """
        if not chunks:
            return []

        document_id = UUID(str(document_id))
        project_id = UUID(str(project_id))

        if any(not chunk.page_content.strip() for chunk in chunks):
            raise ValueError("Document chunks must contain non-empty text.")
        chunking_version = chunking_version or settings.rag_chunking_version

        if not chunking_version.strip() or len(chunking_version) > 100:
            raise ValueError("chunking_version must contain 1 to 100 characters.")

        async with self.db.begin_nested():
            # Serialize replacement builds for this document, including across projects.
            document = await self.db.scalar(
                select(Document)
                .join(ProjectDocument, ProjectDocument.document_id == Document.id)
                .join(Project, Project.id == ProjectDocument.project_id)
                .where(
                    Document.id == document_id,
                    Document.user_id == self.user_id,
                    Project.id == project_id,
                    Project.user_id == self.user_id,
                )
                .with_for_update(of=Document)
            )
            if document is None:
                raise NotFoundError("Document not found in this project.")

            if vectors is None:
                vectors = await embed_texts([chunk.page_content for chunk in chunks])
            self._validate_vectors(vectors, len(chunks))
            generation = EmbeddingGeneration(
                id=uuid4(),
                document_id=document_id,
                model=settings.EMBEDDING_MODEL,
                dimensions=settings.EMBEDDING_DIMENSION,
                chunking_version=chunking_version,
                status=EmbeddingStatus.COMPLETED,
                is_active=False,
            )
            self.db.add(generation)
            await self.db.flush()

            encoding = tiktoken.get_encoding(settings.CHUNK_ENCODING)
            rows = []
            for index, (chunk, vector) in enumerate(zip(chunks, vectors, strict=True)):
                # Identity/order live in ORM columns; project membership lives in
                # ProjectDocument. Keep source metadata, without duplicate scope.
                metadata = {
                    key: value
                    for key, value in chunk.metadata.items()
                    if key not in {"document_id", "project_id", "chunk_index"}
                }
                metadata.setdefault("page_start", metadata.get("page"))
                metadata.setdefault("page_end", metadata.get("page"))
                metadata["token_count"] = len(
                    encoding.encode(chunk.page_content, disallowed_special=())
                )
                rows.append(
                    DocumentEmbedding(
                        id=uuid4(),
                        document_id=document_id,
                        generation_id=generation.id,
                        content=chunk.page_content,
                        chunk_index=index,
                        embedding=vector,
                        embedding_metadata=DocumentEmbeddingMetadata(**metadata),
                    )
                )
            self.db.add_all(rows)
            await self.db.flush()
            # Separate statements satisfy the unique partial index on active generations.
            await self.db.execute(
                update(EmbeddingGeneration)
                .where(
                    EmbeddingGeneration.document_id == document_id,
                    EmbeddingGeneration.is_active.is_(True),
                )
                .values(is_active=False)
            )
            generation.is_active = True
            document.embedding_status = EmbeddingStatus.COMPLETED
            await self.db.flush()
            return [str(row.id) for row in rows]

    async def _query_vector(self, query: str, project_id: UUID, k: int) -> list[float]:
        """Validate the request, confirm the project is owned, and embed the query once."""
        if not query.strip() or k < 1:
            raise ValueError("Provide a non-empty query and a positive k.")
        project = await self.db.scalar(
            select(Project.id).where(
                Project.id == project_id,
                Project.user_id == self.user_id,
            )
        )
        if project is None:
            raise NotFoundError("Project not found.")
        vector = await embed_query(query)
        self._validate_vectors([vector], 1)
        return vector

    async def similarity_search(
        self,
        query: str,
        *,
        project_id: UUID,
        k: int = 4,
        document_ids: Sequence[UUID] | None = None,
    ) -> list[LangChainDocument]:
        """Return nearest chunks from active, compatible generations in an owned project.

        `document_ids` narrows the search; ids outside the project match nothing.
        """
        vector = await self._query_vector(query, project_id, k)
        return await self._nearest(
            vector, project_id=project_id, k=k, document_ids=document_ids
        )

    async def balanced_search(
        self,
        query: str,
        *,
        project_id: UUID,
        k: int = 6,
        min_per_document: int = 2,
        document_ids: Sequence[UUID] | None = None,
    ) -> list[LangChainDocument]:
        """Nearest chunks, but each book in the project keeps at least `min_per_document`.

        Plain nearest-k lets the best-matching book fill every slot, which starves
        the other books on questions that span several. This embeds the query
        once, runs the usual nearest-k search, and searches a book separately only
        when it fell short. A project with one book, or `min_per_document` of 0,
        behaves exactly like `similarity_search`.
        """
        vector = await self._query_vector(query, project_id, k)
        nearest = await self._nearest(
            vector, project_id=project_id, k=k, document_ids=document_ids
        )
        if min_per_document < 1:
            return nearest
        scope = select(ProjectDocument.document_id).where(
            ProjectDocument.project_id == project_id
        )
        if document_ids is not None:
            scope = scope.where(ProjectDocument.document_id.in_(document_ids))
        books = list(await self.db.scalars(scope))
        if len(books) < 2:
            return nearest
        have = Counter(doc.metadata["document_id"] for doc in nearest)
        extra: list[LangChainDocument] = []
        for book in books:
            if have[str(book)] < min_per_document:
                extra += await self._nearest(
                    vector,
                    project_id=project_id,
                    k=min_per_document,
                    document_ids=[book],
                )
        return merge_balanced(nearest + extra, k, min_per_document)

    async def _nearest(
        self,
        vector: list[float],
        *,
        project_id: UUID,
        k: int,
        document_ids: Sequence[UUID] | None = None,
    ) -> list[LangChainDocument]:
        """Run the owner-, project- and generation-scoped nearest-chunk query."""
        distance = DocumentEmbedding.embedding.cosine_distance(vector)
        statement = (
            select(DocumentEmbedding, Document.title, distance.label("distance"))
            .join(
                EmbeddingGeneration,
                EmbeddingGeneration.id == DocumentEmbedding.generation_id,
            )
            .join(Document, Document.id == DocumentEmbedding.document_id)
            .join(ProjectDocument, ProjectDocument.document_id == Document.id)
            .join(Project, Project.id == ProjectDocument.project_id)
            .where(
                Project.id == project_id,
                Project.user_id == self.user_id,
                Document.user_id == self.user_id,
                EmbeddingGeneration.is_active.is_(True),
                EmbeddingGeneration.status == EmbeddingStatus.COMPLETED,
                EmbeddingGeneration.model == settings.EMBEDDING_MODEL,
                EmbeddingGeneration.dimensions == settings.EMBEDDING_DIMENSION,
            )
            .order_by(distance, DocumentEmbedding.id)
            .limit(k)
        )
        if document_ids is not None:
            statement = statement.where(Document.id.in_(document_ids))
        rows = (await self.db.execute(statement)).all()
        return [
            LangChainDocument(
                id=str(chunk.id),
                page_content=chunk.content,
                metadata={
                    **(
                        chunk.embedding_metadata.model_dump()
                        if chunk.embedding_metadata
                        else {}
                    ),
                    "document_id": str(chunk.document_id),
                    "document_title": title,
                    "project_id": str(project_id),
                    "generation_id": str(chunk.generation_id),
                    "chunk_index": chunk.chunk_index,
                    "cosine_distance": float(distance),
                },
            )
            for chunk, title, distance in rows
        ]
