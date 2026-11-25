"""Index the fixture PDFs and exercise the normal conversation services."""

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from uuid import NAMESPACE_URL, UUID, uuid4, uuid5

from langchain_core.documents import Document as Page
from openai import APIError
from sqlalchemy import func, select

from app.core.config import settings
from app.core.database import AsyncSessionLocal, async_engine
from app.models import Document, DocumentEmbedding, Project, ProjectDocument, User
from app.models.conversation import ConversationRun, RunStatus
from app.models.document import ContentType
from app.services.conversation_pipeline import ConversationPipeline
from app.services.conversation_service import ConversationService
from app.services.indexing.loaders import load_document
from app.services.indexing.splitters import split_rag_documents
from app.services.indexing.vector_store import VectorStoreService

from .cases import Case, words
from .judge import Judge
from .run_info import run_info
from .summary import format_summary, summarize

CITATION_GROUP = re.compile(r"\[\s*(S[0-9]+\b[^\[\]\r\n]*)\]|\(\s*(S[0-9]+\b[^()\r\n]*)\)")
CITATION_LABEL = re.compile(r"\bS[0-9]+\b")


@dataclass
class Fixture:
    path: Path
    sha256: str
    size: int
    pages: list[Page]


def load_fixtures(cases: list[Case], paths: list[Path]) -> list[Fixture]:
    fixtures = []
    page_words = {}
    for path in paths:
        data = path.read_bytes()
        try:
            pages = load_document(path)
        except Exception as exc:
            raise ValueError(f"Could not read fixture {path} ({type(exc).__name__}).") from exc
        fixtures.append(Fixture(path, hashlib.sha256(data).hexdigest(), len(data), pages))
        for page in pages:
            page_words[path.name, page.metadata["page"] + 1] = words(page.page_content)

    mismatches = []
    for case in cases:
        for fact in case.required_facts:
            for anchor in fact.evidence:
                if not anchor.matches(page_words.get((anchor.document, anchor.pdf_page), "")):
                    mismatches.append(f"{case.id}: {anchor.document} p. {anchor.pdf_page}")
    if mismatches:
        raise ValueError(
            "Evidence quotes were missing from their labeled PDF pages. "
            "Check the fixture editions and labels:\n  " + "\n  ".join(mismatches)
        )
    return fixtures


async def indexed_project(fixtures: list[Fixture], index_settings: dict) -> tuple[UUID, UUID]:
    # Identical PDFs and index settings reuse the same project across runs.
    project_id = uuid5(NAMESPACE_URL, "lumen-chat-eval:" + json.dumps(index_settings, sort_keys=True))
    lock_key = int.from_bytes(project_id.bytes[:8], signed=True)
    async with AsyncSessionLocal.begin() as db:
        await db.execute(select(func.pg_advisory_xact_lock(lock_key)))
        project = await db.get(Project, project_id)
        if project is not None:
            print("Reusing the indexed fixture project.", flush=True)
            return project_id, project.user_id

        user_id = uuid4()
        db.add(User(id=user_id, email=f"chat-eval-{user_id}@example.invalid"))
        await db.flush()
        db.add(Project(id=project_id, user_id=user_id, title="Chat eval fixtures"))
        await db.flush()
        store = VectorStoreService(db, user_id)
        for fixture in fixtures:
            print(f"Indexing {fixture.path.name}...", flush=True)
            document = Document(
                user_id=user_id,
                title=fixture.path.stem.replace("-", " ").title(),
                original_filename=fixture.path.name,
                file_size_bytes=fixture.size,
                content_hash=fixture.sha256,
                content_type=ContentType.PDF,
                mime_type="application/pdf",
            )
            db.add(document)
            await db.flush()
            db.add(ProjectDocument(project_id=project_id, document_id=document.id))
            await db.flush()
            chunks = split_rag_documents(fixture.pages)
            if not chunks:
                raise ValueError(f"No indexable text in {fixture.path}.")
            await store.add_document_chunks(
                chunks, document_id=document.id, project_id=project_id
            )
    return project_id, user_id


async def turn_sources(run: ConversationRun) -> list[dict]:
    candidates = run.snapshot.retrieval_candidates
    if not candidates:
        return []
    async with AsyncSessionLocal() as db:
        rows = await db.execute(
            select(
                DocumentEmbedding.id,
                DocumentEmbedding.content,
                DocumentEmbedding.embedding_metadata,
                Document.original_filename,
            )
            .join(Document, Document.id == DocumentEmbedding.document_id)
            .where(DocumentEmbedding.id.in_([hit.chunk_id for hit in candidates]))
        )
        chunks = {row.id: row for row in rows}
    labels = {source.chunk_id: source.citation_label for source in run.snapshot.selected_sources}
    sources = []
    for hit in candidates:
        chunk = chunks[hit.chunk_id]
        metadata = chunk.embedding_metadata
        start = metadata.page_start if metadata else None
        end = metadata.page_end if metadata else None
        sources.append({
            "document": chunk.original_filename,
            "pdf_page_start": start + 1 if start is not None else None,
            "pdf_page_end": end + 1 if end is not None else None,
            "chunk_id": str(hit.chunk_id),
            "cosine_distance": hit.cosine_distance,
            "citation_label": labels.get(hit.chunk_id),
            "text": chunk.content,
        })
    return sources


def score(case: Case, sources: list[dict], answer: str) -> dict:
    normalized = [words(source["text"]) for source in sources]
    evidence = []
    for fact in case.required_facts:
        for anchor in fact.evidence:
            hits = [
                source for source, text in zip(sources, normalized, strict=True)
                if source["document"] == anchor.document
                and source["pdf_page_start"] is not None
                and source["pdf_page_end"] is not None
                and source["pdf_page_start"] <= anchor.pdf_page <= source["pdf_page_end"]
                and anchor.matches(text)
            ]
            evidence.append({
                "fact": fact.fact,
                **anchor.model_dump(),
                "retrieved": bool(hits),
                "in_prompt": any(source["citation_label"] for source in hits),
            })

    cited = {
        label
        for bracketed, parenthesized in CITATION_GROUP.findall(answer)
        for label in CITATION_LABEL.findall(bracketed or parenthesized)
    }
    selected = {source["citation_label"] for source in sources}
    unknown = sorted(cited - selected)
    return {
        "evidence": evidence,
        "retrieval": {"found": sum(item["retrieved"] for item in evidence), "total": len(evidence)}
        if evidence else None,
        "prompt_evidence": {
            "found": sum(item["in_prompt"] for item in evidence), "total": len(evidence)
        } if evidence else None,
        "cited_labels": sorted(cited),
        "unknown_citation_labels": unknown,
        "citation_labels_valid": bool(cited) and not unknown
        if cited or case.expected_behavior == "answer" else None,
        "answer_correct": None,
        "citations_support_claims": None,
    }


async def run_case(case: Case, project_id: UUID, user_id: UUID) -> dict:
    pipeline = ConversationPipeline()
    async with AsyncSessionLocal.begin() as db:
        conversation = await ConversationService(db, user_id).create(project_id, title=case.id)

    turns = []
    for question in case.questions:
        async with AsyncSessionLocal.begin() as db:
            reserved = await ConversationService(db, user_id).reserve_turn(
                conversation.id, content=question, client_request_id=uuid4()
            )
        await pipeline.process(reserved.run.id, user_id)
        async with AsyncSessionLocal() as db:
            run = await ConversationService(db, user_id).get_run(reserved.run.id)
        turns.append({
            "question": question,
            "run_id": str(run.id),
            "status": run.status.value,
            "error_code": run.error_code,
            "answer": run.response_text,
            "rewritten_query": run.rewritten_query,
            "pipeline_version": run.pipeline_version,
            "model": run.model,
            "rewrite_model": run.rewrite_model,
            "metrics": run.metrics.model_dump(exclude_none=True),
        })
        if run.status != RunStatus.COMPLETED:
            break

    sources = await turn_sources(run)
    completed = run.status == RunStatus.COMPLETED
    return {
        "case": case.model_dump(),
        "turns": turns,
        "last_turn_sources": sources,
        "scores": score(case, sources, run.response_text or "") if completed else None,
    }


def print_result(result: dict) -> None:
    case_id = result["case"]["id"]
    scores = result["scores"]
    if scores is None:
        last = result["turns"][-1]
        print(f"{case_id} {last['status']} on turn {len(result['turns'])}: {last['error_code']}")
        return
    retrieval = scores["retrieval"]
    prompt = scores["prompt_evidence"]
    retrieved = f"{retrieval['found']}/{retrieval['total']}" if retrieval else "N/A"
    included = f"{prompt['found']}/{prompt['total']}" if prompt else "N/A"
    valid = scores["citation_labels_valid"]
    citations = {True: "pass", False: "fail", None: "N/A"}[valid]
    judged = {True: "ok", False: "WRONG", None: "ungraded"}[scores["answer_correct"]]
    print(
        f"{case_id} retrieved {retrieved} | in prompt {included} "
        f"| citation labels {citations} | answer {judged}"
    )


async def grade(judge: Judge, case: Case, result: dict) -> None:
    """Add the judge's verdict to a completed case. A judge failure leaves scores ungraded."""
    try:
        answer = result["turns"][-1]["answer"] or ""
        result["scores"].update(await judge.grade(case, answer, result["last_turn_sources"]))
    except Exception as exc:  # noqa: BLE001 - provider text is not recorded, only the type.
        result["scores"]["judge"] = {"model": judge.llm.model, "error": type(exc).__name__}
        print(f"{case.id} judge failed ({type(exc).__name__}); answer left ungraded", flush=True)


async def run_benchmark(
    cases: list[Case], paths: list[Path], output: Path, judge: Judge | None = None
) -> int:
    try:
        if output.exists():
            raise ValueError(f"Report already exists: {output}")
        fixtures = load_fixtures(cases, paths)
        index_settings = {
            "fixtures": {fixture.path.name: fixture.sha256 for fixture in fixtures},
            "embedding_model": settings.EMBEDDING_MODEL,
            "embedding_dimensions": settings.EMBEDDING_DIMENSION,
            "chunking_version": settings.rag_chunking_version,
        }
        project_id, user_id = await indexed_project(fixtures, index_settings)
        output.parent.mkdir(parents=True, exist_ok=True)
        failed = 0
        rows = []
        info = run_info(cases, judge.llm.model if judge else None)
        with output.open("x", encoding="utf-8") as report:
            for case in cases:
                result = await run_case(case, project_id, user_id)
                if judge and result["scores"]:
                    await grade(judge, case, result)
                row = {**result, "index": index_settings, "run": info}
                rows.append(row)
                report.write(json.dumps(row, ensure_ascii=False) + "\n")
                report.flush()
                print_result(result)
                failed += result["scores"] is None
        print()
        print(format_summary(summarize(rows)))
        print(f"Completed {len(cases) - failed}/{len(cases)} cases. Report: {output.resolve()}")
        return int(failed > 0)
    except APIError as exc:
        raise ValueError(f"Fixture indexing failed at the embedding provider ({type(exc).__name__}).") from exc
    finally:
        await async_engine.dispose()
