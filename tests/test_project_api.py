"""Project HTTP contract tests: processing status rides along with each project."""

import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch
from uuid import uuid4

import httpx

from app.api.deps import get_current_user
from app.core.database import get_db, utcnow
from app.core.exception import ConflictError
from app.main import app
from app.models.document import DocumentSummaryStatus, EmbeddingStatus
from app.models.project import Project
from app.services.project.project_service import DocumentJobStatus, RetryPlan


class ProjectApiTests(unittest.IsolatedAsyncioTestCase):
    """Exercise the real project routes with a faked service and no database."""

    async def asyncSetUp(self):
        """Create two owned projects and override authentication and storage."""
        self.user_id = uuid4()
        timestamps = {"created_at": utcnow(), "updated_at": utcnow()}
        self.ready = Project(id=uuid4(), user_id=self.user_id, title="Ready", **timestamps)
        self.empty = Project(id=uuid4(), user_id=self.user_id, title="Empty", **timestamps)
        self.service = AsyncMock()
        self.service.get_all.return_value = [self.ready, self.empty]
        self.service.get_by_id.return_value = self.ready
        self.service.get_first_document_summary.return_value = "# Summary"
        self.service.get_document_statuses.return_value = {
            self.ready.id: DocumentJobStatus(
                DocumentSummaryStatus.COMPLETED, EmbeddingStatus.FAILED, None, "worker_lost"
            )
        }
        self.original_overrides = app.dependency_overrides.copy()

        async def database():
            """Supply a per-request fake database session."""
            yield AsyncMock()

        async def current_user():
            """Supply an authenticated owner without invoking the auth database."""
            return SimpleNamespace(id=self.user_id)

        app.dependency_overrides[get_db] = database
        app.dependency_overrides[get_current_user] = current_user
        self.service_patch = patch(
            "app.api.v1.project.ProjectService", return_value=self.service
        )
        self.service_patch.start()
        self.client = httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test/api/v1"
        )

    async def asyncTearDown(self):
        """Close the client and restore global dependencies and patched services."""
        await self.client.aclose()
        self.service_patch.stop()
        app.dependency_overrides.clear()
        app.dependency_overrides.update(self.original_overrides)

    async def test_list_attaches_statuses_with_one_lookup(self):
        """Every listed project carries its first document's status, or null without one."""
        response = await self.client.get("/project")
        assert response.status_code == 200
        ready, empty = response.json()["data"]
        assert ready["summary_status"] == "completed"
        assert ready["embedding_status"] == "failed"
        assert empty["summary_status"] is None
        assert empty["embedding_status"] is None
        self.service.get_document_statuses.assert_awaited_once_with(
            [self.ready.id, self.empty.id], self.user_id
        )

    async def test_detail_includes_summary_and_statuses(self):
        """The detail view keeps the summary and adds the processing state."""
        response = await self.client.get(f"/project/{self.ready.id}")
        assert response.status_code == 200
        data = response.json()["data"]
        assert data["summary"] == "# Summary"
        assert data["summary_status"] == "completed"
        assert data["embedding_status"] == "failed"
        assert data["embedding_error_code"] == "worker_lost"
        assert data["summary_error_code"] is None

    def retry_patches(self):
        """Pretend the queue is up and capture what the retry route dispatches."""
        self.summary_task = SimpleNamespace(kiq=AsyncMock())
        self.embedding_task = SimpleNamespace(kiq=AsyncMock())
        for target, value in (
            ("app.api.v1.project.is_task_queue_available", AsyncMock(return_value=True)),
            ("app.api.v1.project.summarize_document_task", self.summary_task),
            ("app.api.v1.project.embed_document_task", self.embedding_task),
        ):
            patcher = patch(target, value)
            patcher.start()
            self.addCleanup(patcher.stop)

    async def test_retry_requeues_only_the_failed_job_with_its_new_fence(self):
        """A failed index is queued again under the fresh job id; the finished summary is not."""
        self.retry_patches()
        document = SimpleNamespace(
            id=uuid4(),
            summary_status=DocumentSummaryStatus.COMPLETED,
            embedding_status=EmbeddingStatus.PENDING,
            summary_error_code=None,
            embedding_error_code=None,
            summary_job_id=uuid4(),
            embedding_job_id=uuid4(),
        )
        self.service.retry_failed_jobs.return_value = RetryPlan(document, False, True)

        response = await self.client.post(f"/project/{self.ready.id}/retry")

        assert response.status_code == 202, response.text
        data = response.json()["data"]
        assert data["embedding_status"] == "pending"
        assert data["embedding_error_code"] is None
        self.embedding_task.kiq.assert_awaited_once_with(
            str(document.id), str(self.ready.id), str(document.embedding_job_id)
        )
        self.summary_task.kiq.assert_not_awaited()

    async def test_retry_with_nothing_failed_is_a_conflict(self):
        """The service's 409 reaches the client through the AppError envelope."""
        self.retry_patches()
        self.service.retry_failed_jobs.side_effect = ConflictError("Nothing to retry.")
        response = await self.client.post(f"/project/{self.ready.id}/retry")
        assert response.status_code == 409
        assert response.json()["success"] is False
        self.embedding_task.kiq.assert_not_awaited()
