"""Async text generation using the configured DeepSeek chat endpoint."""

from collections.abc import AsyncIterator
from contextlib import aclosing
from functools import cached_property
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import SecretStr

from app.core.config import settings


class LLMService:
    """Reuse an instance for generation and streaming; model creation is lazy.

    Supply a chat model for testing or another provider. Provider errors and
    cancellation propagate to the caller. Streaming yields text deltas, not SSE
    frames; the HTTP layer is responsible for transport and disconnect handling.
    """

    def __init__(
        self,
        *,
        model: str = "deepseek-v4-flash",
        timeout: float | None = None,
        temperature: float = 0.3,
        chat_model: BaseChatModel | None = None,
    ) -> None:
        self.model = model
        self.timeout = (
            timeout if timeout is not None else settings.DEEPSEEK_TIMEOUT_SECONDS
        )
        self.temperature = temperature
        self._chat_model = chat_model

    @cached_property
    def chat_model(self) -> BaseChatModel:
        if self._chat_model is not None:
            return self._chat_model
        if not settings.DEEPSEEK_API_KEY:
            raise ValueError("Set DEEPSEEK_API_KEY before invoking the LLM.")
        return ChatOpenAI(
            model=self.model,
            api_key=SecretStr(settings.DEEPSEEK_API_KEY),
            base_url=settings.DEEPSEEK_BASE_URL,
            timeout=self.timeout,
            max_retries=settings.DEEPSEEK_MAX_RETRIES,
            temperature=self.temperature,
            use_responses_api=False,
            stream_usage=False,
        )

    async def generate(self, prompt: ChatPromptTemplate, values: dict[str, Any]) -> str:
        """Render a prompt and return the complete generated text."""
        messages = prompt.format_messages(**values)
        response = await self.chat_model.ainvoke(messages)
        return StrOutputParser().invoke(response)

    async def stream(
        self, prompt: ChatPromptTemplate, values: dict[str, Any]
    ) -> AsyncIterator[str]:
        """Yield incremental text and close the provider stream on exit.

        Callers that stop early should close this iterator (e.g. with aclosing).
        A mid-stream error propagates; emitted text is not retried or replayed.
        """
        messages = prompt.format_messages(**values)
        parser = StrOutputParser()
        async with aclosing(self.chat_model.astream(messages)) as chunks:
            async for chunk in chunks:
                text = parser.invoke(chunk)
                if text:
                    yield text
