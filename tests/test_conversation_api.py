"""Conversation HTTP contract tests with dependency overrides and no external services."""

import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch
from uuid import uuid4

import httpx
import pytest

from app.api.deps import get_current_user
from app.core.database import get_db, utcnow
from app.core.exception import ConflictError, NotFoundError
from app.main import app
from app.models.conversation import (
    Conversation,
    ConversationMessage,
    ConversationRun,
    MessageRole,
    RunMetrics,
    RunSnapshot,
    RunStatus,
)
from app.services.conversation_service import ReservedTurn


class ConversationApiTests(unittest.IsolatedAsyncioTestCase):
    """Exercise real routes, response validation, authentication, and safe error envelopes."""

    async def asyncSetUp(self):
        """Create owned response rows and override only authentication, storage, and dispatch."""
        self.user_id, self.project_id, self.conversation_id = uuid4(), uuid4(), uuid4()
        timestamps = {"created_at": utcnow(), "updated_at": utcnow()}
        self.conversation = Conversation(
            id=self.conversation_id,
            project_id=self.project_id,
            title="Books",
            **timestamps,
        )
        self.user_message = ConversationMessage(
            id=uuid4(),
            conversation_id=self.conversation_id,
            position=1,
            role=MessageRole.USER,
            content="  Explain Book B.  ",
            client_request_id=uuid4(),
            **timestamps,
        )
        self.assistant = ConversationMessage(
            id=uuid4(),
            conversation_id=self.conversation_id,
            position=2,
            role=MessageRole.ASSISTANT,
            content="",
            **timestamps,
        )
        self.run = ConversationRun(
            id=uuid4(),
            conversation_id=self.conversation_id,
            user_message_id=self.user_message.id,
            assistant_message_id=self.assistant.id,
            attempt_number=1,
            status=RunStatus.PENDING,
            stage="reserved",
            snapshot=RunSnapshot(),
            metrics=RunMetrics(),
            **timestamps,
        )
        self.db = AsyncMock()
        self.service = AsyncMock()
        self.service.create.return_value = self.conversation
        self.service.list_conversations.return_value = [self.conversation]
        self.service.get.return_value = self.conversation
        self.service.rename.return_value = self.conversation
        self.service.list_messages.return_value = [self.user_message, self.assistant]
        self.service.latest_message_runs.return_value = {self.assistant.id: self.run}
        self.service.list_runs.return_value = [self.run]
        self.service.get_run.return_value = self.run
        self.service.reserve_turn.return_value = ReservedTurn(
            self.user_message, self.assistant, self.run, False
        )
        self.service.cancel.return_value = self.run
        self.service.retry.return_value = self.run
        self.original_overrides = app.dependency_overrides.copy()

        async def database():
            """Supply a per-request fake database session."""
            yield self.db

        async def current_user():
            """Supply an authenticated owner without invoking the auth database."""
            return SimpleNamespace(id=self.user_id)

        app.dependency_overrides[get_db] = database
        app.dependency_overrides[get_current_user] = current_user
        self.service_patch = patch(
            "app.api.v1.conversations.ConversationService", return_value=self.service
        )
        self.service_patch.start()
        self.dispatch_patch = patch(
            "app.api.v1.conversations.dispatch_conversation", new=AsyncMock()
        )
        self.dispatch = self.dispatch_patch.start()
        self.client = httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test/api/v1"
        )

    async def asyncTearDown(self):
        """Close the client and restore global dependencies and patched services."""
        await self.client.aclose()
        self.service_patch.stop()
        self.dispatch_patch.stop()
        app.dependency_overrides.clear()
        app.dependency_overrides.update(self.original_overrides)

    async def test_conversation_crud_returns_standard_envelopes(self):
        """Create, list, read, rename, and delete are available to the frontend."""
        created = await self.client.post(
            "/conversations",
            json={"project_id": str(self.project_id), "title": "Books"},
        )
        assert created.status_code == 201
        assert created.json()["data"]["id"] == str(self.conversation_id)
        listed = await self.client.get(
            "/conversations", params={"project_id": str(self.project_id)}
        )
        assert listed.json()["data"][0]["title"] == "Books"
        read = await self.client.get(f"/conversations/{self.conversation_id}")
        assert read.status_code == 200
        renamed = await self.client.patch(
            f"/conversations/{self.conversation_id}", json={"title": "New title"}
        )
        assert renamed.status_code == 200
        self.service.rename.assert_awaited_once_with(self.conversation_id, "New title")
        deleted = await self.client.delete(f"/conversations/{self.conversation_id}")
        assert deleted.json() == {
            "message": "Conversation deleted.",
            "data": None,
            "success": True,
        }

    async def test_send_preserves_content_and_commits_before_dispatch(self):
        """Accepted messages include both stable IDs and a run that can be polled."""

        async def dispatch(*args):
            """Assert reservation has been committed before background work is queued."""
            self.db.commit.assert_awaited_once()

        self.dispatch.side_effect = dispatch
        response = await self.client.post(
            f"/conversations/{self.conversation_id}/messages",
            json={
                "content": self.user_message.content,
                "client_request_id": str(self.user_message.client_request_id),
            },
        )
        assert response.status_code == 202
        data = response.json()["data"]
        assert data["user_message"]["content"] == "  Explain Book B.  "
        assert data["assistant_message"]["content"] == ""
        assert data["run"]["status"] == "pending"
        assert not data["reused"]
        self.service.reserve_turn.assert_awaited_once_with(
            self.conversation_id,
            content=self.user_message.content,
            client_request_id=self.user_message.client_request_id,
        )
        self.dispatch.assert_awaited_once_with(self.db, self.run.id, self.user_id)

    async def test_duplicate_message_returns_existing_run_without_redispatch(self):
        """Resending the same client request is safe even after completion or queue failure."""
        self.service.reserve_turn.return_value = ReservedTurn(
            self.user_message, self.assistant, self.run, True
        )
        response = await self.client.post(
            f"/conversations/{self.conversation_id}/messages",
            json={
                "content": self.user_message.content,
                "client_request_id": str(self.user_message.client_request_id),
            },
        )
        assert response.status_code == 202
        assert response.json()["data"]["reused"]
        self.dispatch.assert_not_awaited()

    async def test_history_contains_latest_attempt_and_cursor(self):
        """Reloaded assistant placeholders carry the status needed to render pending and failed turns."""
        response = await self.client.get(
            f"/conversations/{self.conversation_id}/messages",
            params={"limit": 2, "after_position": 0},
        )
        assert response.status_code == 200
        data = response.json()["data"]
        assert data["next_after_position"] == 2
        assert data["messages"][0]["latest_run"] is None
        assert data["messages"][1]["latest_run"]["id"] == str(self.run.id)
        self.service.list_messages.assert_awaited_once_with(
            self.conversation_id, after_position=0, limit=2
        )

    async def test_poll_sources_attempt_history_cancel_and_retry(self):
        """Polling and attempt actions expose stable IDs and recorded diagnostics."""
        polled = await self.client.get(f"/conversation-runs/{self.run.id}")
        assert polled.status_code == 200
        assert polled.json()["data"]["status"] == "pending"
        assert "snapshot" not in polled.json()["data"]
        debug = await self.client.get(f"/conversation-runs/{self.run.id}/debug")
        assert debug.json()["data"]["snapshot"]["prompt_messages"] == []
        sources = await self.client.get(f"/conversation-runs/{self.run.id}/sources")
        assert sources.json()["data"] == []
        runs = await self.client.get(f"/conversations/{self.conversation_id}/runs")
        assert runs.json()["data"][0]["attempt_number"] == 1
        cancelled = await self.client.post(f"/conversation-runs/{self.run.id}/cancel")
        assert cancelled.status_code == 200
        retry = await self.client.post(f"/conversation-runs/{self.run.id}/retry")
        assert retry.status_code == 202
        self.service.retry.assert_awaited_once_with(self.run.id)
        self.dispatch.assert_awaited_once_with(self.db, self.run.id, self.user_id)

    async def test_missing_authentication_returns_401(self):
        """Conversation routes require the existing access-cookie or Bearer authentication."""
        app.dependency_overrides.pop(get_current_user)
        response = await self.client.get(f"/conversations/{self.conversation_id}")
        assert response.status_code == 401
        assert not response.json()["success"]

    async def test_not_found_and_conflict_use_safe_error_envelopes(self):
        """Ownership failures are 404 and active-turn or idempotency conflicts are 409."""
        self.service.get.side_effect = NotFoundError("Conversation not found.")
        response = await self.client.get(f"/conversations/{self.conversation_id}")
        assert response.status_code == 404
        assert response.json()["data"] is None
        self.service.reserve_turn.side_effect = ConflictError(
            "This conversation already has an active run."
        )
        response = await self.client.post(
            f"/conversations/{self.conversation_id}/messages",
            json={"content": "Q", "client_request_id": str(uuid4())},
        )
        assert response.status_code == 409
        self.dispatch.assert_not_awaited()

    async def test_request_validation_rejects_empty_oversized_and_unknown_fields(self):
        """Body and pagination validation reject invalid input before database mutations."""
        for payload in (
            {"content": " \n\t", "client_request_id": str(uuid4())},
            {"content": "x" * 20001, "client_request_id": str(uuid4())},
            {"content": "Q"},
            {
                "content": "Q",
                "client_request_id": str(uuid4()),
                "user_id": str(uuid4()),
            },
        ):
            with self.subTest(payload_keys=list(payload)):
                response = await self.client.post(
                    f"/conversations/{self.conversation_id}/messages", json=payload
                )
                assert response.status_code == 422
        for params in ({"limit": 0}, {"limit": 201}, {"after_position": -1}):
            response = await self.client.get(
                f"/conversations/{self.conversation_id}/messages", params=params
            )
            assert response.status_code == 422
        self.service.reserve_turn.assert_not_awaited()


class ConversationDispatchTests(unittest.IsolatedAsyncioTestCase):
    """Check broker failures without connecting to Redis."""

    async def test_send_failure_only_fails_an_unclaimed_attempt(self):
        """An uncertain broker acknowledgement never overwrites a worker's claimed attempt."""
        from app.services.conversation_dispatcher import (
            ConversationQueueError,
            dispatch_conversation,
        )

        db = AsyncMock()
        service = AsyncMock()
        run_id, user_id = uuid4(), uuid4()
        with (
            patch(
                "app.services.conversation_dispatcher.answer_conversation_task.kiq",
                new=AsyncMock(side_effect=RuntimeError("private broker details")),
            ),
            patch(
                "app.services.conversation_dispatcher.ConversationService",
                return_value=service,
            ),
            pytest.raises(ConversationQueueError) as error,
        ):
            await dispatch_conversation(db, run_id, user_id)
        assert "private" not in str(error.value)
        service.fail_if_active.assert_awaited_once_with(
            run_id, "queue_unavailable", pending_only=True
        )
        db.commit.assert_awaited_once()
