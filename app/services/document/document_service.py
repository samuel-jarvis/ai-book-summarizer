import hashlib
import uuid
from pathlib import Path

import anyio
from fastapi import UploadFile
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.exception import ValidationError
from app.models.document import ContentType, Document, StorageProvider

UPLOAD_DIR = Path("temp_uploads").resolve()
MAX_UPLOAD_BYTES = 50 * 1024 * 1024  # 50 MB
CHUNK_SIZE = 1024 * 1024  # 1 MB
PDF_MIME_TYPE = "application/pdf"


class DocumentService:
    """Persists uploaded files and their library rows.

    Methods flush but never commit, so the caller controls the transaction.
    Extraction, summarization, and embedding are handled elsewhere.
    """

    def __init__(self, db: AsyncSession) -> None:
        self.db = db

    async def create_document(
        self, *, user_id: uuid.UUID, file: UploadFile, title: str | None = None
    ) -> Document:
        """Stream an upload to disk and record it as an owned document.

        The file is hashed and measured while streaming, so an oversized upload
        is abandoned without being read into memory. A caller that fails after
        this returns is responsible for removing `storage_path(document)`.
        """
        if not file or not file.filename:
            raise ValidationError("No file uploaded")
        if not file.filename.lower().endswith(".pdf"):
            raise ValidationError("Only PDF files are supported")

        UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        storage_key = f"{uuid.uuid4()}.pdf"
        file_path = UPLOAD_DIR / storage_key

        digest = hashlib.sha256()
        total_bytes = 0
        too_large = False
        try:
            # anyio keeps the 50 MB write off the event loop thread.
            async with await anyio.open_file(file_path, "wb") as target:
                while chunk := await file.read(CHUNK_SIZE):
                    total_bytes += len(chunk)
                    if total_bytes > MAX_UPLOAD_BYTES:
                        too_large = True
                        break
                    digest.update(chunk)
                    await target.write(chunk)
        except Exception:
            file_path.unlink(missing_ok=True)
            raise

        if too_large:
            file_path.unlink(missing_ok=True)
            raise ValidationError(
                f"File exceeds the {MAX_UPLOAD_BYTES // (1024 * 1024)} MB limit"
            )
        if total_bytes == 0:
            file_path.unlink(missing_ok=True)
            raise ValidationError("The uploaded file is empty")

        original_filename = Path(file.filename).name
        document = Document(
            user_id=user_id,
            title=(title or Path(original_filename).stem or original_filename)[:255],
            file_size_bytes=total_bytes,
            content_hash=digest.hexdigest(),
            mime_type=file.content_type or PDF_MIME_TYPE,
            content_type=ContentType.PDF,
            original_filename=original_filename[:255],
            storage_provider=StorageProvider.LOCAL,
            storage_key=storage_key,
        )
        self.db.add(document)
        try:
            await self.db.flush()
        except Exception:
            file_path.unlink(missing_ok=True)
            raise
        return document

    @staticmethod
    def storage_path(document: Document) -> Path | None:
        """Resolve a locally stored document to a path, or None if remote."""
        if document.storage_provider != StorageProvider.LOCAL or not document.storage_key:
            return None
        return UPLOAD_DIR / document.storage_key
