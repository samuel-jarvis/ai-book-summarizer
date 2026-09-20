from pathlib import Path

from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
)
from langchain_core.document_loaders import BaseLoader
from langchain_core.documents import Document

SUPPORTED_EXTENSIONS = [".pdf", ".txt"]
type PathLike = str | Path


def get_document_loader(file_path: PathLike) -> BaseLoader:
    path = Path(file_path)

    extension = path.suffix.lower()

    match extension:
        case ".pdf":
            return PyPDFLoader(str(path))

        case ".txt":
            return TextLoader(
                str(path),
                encoding="utf-8",
                autodetect_encoding=True,
            )

        case _:
            raise ValueError(
                f"Unsupported document type: {extension}"
            )

def load_document(file_path: Path, *, document_id: str | None = None, project_id: str | None = None) -> list[Document]:
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(
            f"Document not found: {path}"
        )

    if not path.is_file():
        raise ValueError(
            f"Expected a file but received: {path}"
        )

    loader = get_document_loader(path)

    documents = loader.load()

    for doc in documents:
        doc.metadata.update (
            {
                "document_id": document_id,
                "project_id": project_id,
                "filename": str(path),
            }
        )

    return documents
