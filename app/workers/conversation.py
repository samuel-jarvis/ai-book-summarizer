"""Taskiq entry point for a reserved conversation attempt."""

from uuid import UUID

from app.services.conversation_pipeline import ConversationPipeline
from app.taskiq_broker import broker


@broker.task
async def answer_conversation_task(run_id: str, user_id: str) -> None:
    """Process one committed attempt; the pipeline handles redelivery and safe failures."""
    await ConversationPipeline().process(UUID(run_id), UUID(user_id))
