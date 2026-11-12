"""Run the complete conversation workflow using the existing small services."""

import asyncio
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.core.config import settings
from app.core.database import AsyncSessionLocal
from app.core.exception import ConflictError, NotFoundError
from app.models.conversation import ConversationRun, RunStatus
from app.services.answer_context_service import AnswerContextService
from app.services.answer_generation_service import (
    AnswerGenerationError,
    AnswerGenerationService,
)
from app.services.conversation_service import ConversationService
from app.services.jobs.lease import LeaseLost, job_lease, row_renewer
from app.services.llm.llm_service import LLMService
from app.services.query_rewrite_service import QueryRewriteService
from app.services.retrieval_service import RetrievalService


class ConversationPipeline:
    """Claim a pending attempt once, checkpoint each stage, and record safe failures."""

    def __init__(
        self,
        *,
        sessions: async_sessionmaker[AsyncSession] = AsyncSessionLocal,
        llm: LLMService | None = None,
    ) -> None:
        """Share a session factory and optionally inject a provider for local tests."""
        self.sessions = sessions
        self.llm = llm

    async def process(self, run_id: UUID, user_id: UUID) -> None:
        """Execute rewrite, retrieval, context assembly, and answer generation in order."""
        try:
            async with self.sessions.begin() as db:
                service = ConversationService(db, user_id)
                run = await service.start_run(
                    run_id,
                    model=self.llm.model
                    if self.llm is not None
                    else settings.CONVERSATION_MODEL,
                    pipeline_version="conversation-v1",
                )
                conversation = await service.get(run.conversation_id)
                project_id = conversation.project_id
        except (ConflictError, NotFoundError):
            # Redelivery of a claimed task and deletion before pickup are harmless.
            return

        # The heartbeat keeps the attempt's lease alive while stages run, so
        # the sweeper can tell a slow answer from a dead worker.
        renew = row_renewer(
            ConversationRun,
            ConversationRun.lease_expires_at,
            ConversationRun.id == run_id,
            ConversationRun.status == RunStatus.RUNNING,
            sessions=self.sessions,
        )
        error_code = "rewrite_failed"
        try:
            async with job_lease(renew):
                rewrite = await QueryRewriteService(self.llm).rewrite_run(
                    run_id, user_id, sessions=self.sessions
                )
                error_code = "retrieval_failed"
                async with self.sessions.begin() as db:
                    await ConversationService(db, user_id).checkpoint(
                        run_id, stage="retrieving"
                    )
                async with self.sessions() as db:
                    retrieval = await RetrievalService(db, user_id).retrieve(
                        rewrite.query, project_id=project_id
                    )
                async with self.sessions.begin() as db:
                    service = ConversationService(db, user_id)
                    run = await service.get_run(run_id)
                    await service.checkpoint(
                        run_id,
                        stage="retrieved",
                        snapshot=retrieval.to_run_snapshot(run.snapshot),
                    )
                error_code = "context_failed"
                await AnswerContextService().assemble_run(
                    run_id, user_id, retrieval, sessions=self.sessions
                )
                error_code = "answer_generation_failed"
                await AnswerGenerationService(self.llm).generate_run(
                    run_id, user_id, sessions=self.sessions
                )
        except LeaseLost:
            # The sweeper already failed this attempt as worker_lost (or the
            # user cancelled it); there is nothing left for us to record.
            return
        except asyncio.CancelledError:
            # Worker shutdown. Fail rather than cancel, so the client shows
            # "interrupted" with Retry instead of "you stopped this answer".
            async with self.sessions.begin() as db:
                try:
                    await ConversationService(db, user_id).fail_if_active(
                        run_id, "worker_lost"
                    )
                except NotFoundError:
                    pass
            raise
        except AnswerGenerationError:
            # Generation already recorded a specific safe error code.
            return
        except Exception:  # noqa: BLE001 -- background failures are recorded as safe stage codes.
            async with self.sessions.begin() as db:
                try:
                    await ConversationService(db, user_id).fail_if_active(
                        run_id, error_code
                    )
                except NotFoundError:
                    pass
