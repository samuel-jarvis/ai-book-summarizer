import re

from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AsyncOpenAI,
)
from app.core.config import settings

LOG_PREVIEW_CHARS = 100

USER_ID_PATTERN = re.compile(r"^[a-zA-Z0-9\-_]{1,512}$")

OPERATOR_HINTS = {
    400: "Malformed request body; likely a bug in how the call is built.",
    401: "Check DEEPSEEK_API_KEY in .env; the key is wrong, revoked or expired.",
    402: "DEEPSEEK ACCOUNT IS OUT OF BALANCE. Top up; every summary fails until then.",
    422: "Invalid parameters; check the payload against the DeepSeek API docs.",
    429: "Rate limited despite SDK retries. Concurrency is well under the account cap, "
         "so suspect account quota rather than SUMMARY_MAX_CONCURRENCY.",
    500: "DeepSeek server error; transient, already retried by the SDK.",
    503: "DeepSeek overloaded; transient, already retried by the SDK.",
}


class DeepSeekAIService:
    def __init__(
        self,
        model: str = "deepseek-v4-flash",
        timeout: float | None = None,
    ):
        self.model = model
        self.timeout = timeout if timeout is not None else settings.DEEPSEEK_TIMEOUT_SECONDS
        _api_key = settings.DEEPSEEK_API_KEY
        if not _api_key:
            raise ValueError(
                "DEEPSEEK_API_KEY is empty. Set it in your .env file before running summaries."
            )

        self.client = AsyncOpenAI(
            api_key=_api_key,
            base_url=settings.DEEPSEEK_BASE_URL,
            timeout=self.timeout,
            max_retries=settings.DEEPSEEK_MAX_RETRIES,
        )

    async def summarize_text(self, prompt: str, user_id: str | None = None) -> str:
        prompt_text = prompt.strip() if prompt else ""

        if not prompt_text:
            print("No text to summarize. Returning empty string.")
            return ""

        extra_body = {}
        if user_id:
            if USER_ID_PATTERN.match(user_id):
                extra_body["user_id"] = user_id

        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You summarize books clearly and concisely.",
                    },
                    {
                        "role": "user",
                        "content": prompt_text,
                    },
                ],
                stream=False,
                temperature=0.3,
                extra_body=extra_body,
            )
        # APITimeoutError subclasses APIConnectionError, so it has to be caught first.
        except APITimeoutError as exc:
            raise RuntimeError(
                f"DeepSeek call timed out after {settings.DEEPSEEK_MAX_RETRIES + 1} attempts "
                f"at {self.timeout:.0f}s each (model={self.model}, prompt={len(prompt_text):,} chars). "
                "The timeout is inter-byte, not total: DeepSeek's keep-alive empty lines "
                "reset it, so this means silence, not slowness."
            ) from exc
        except APIConnectionError as exc:
            raise RuntimeError(
                f"Could not reach DeepSeek (model={self.model}): {exc}"
            ) from exc
        except APIStatusError as exc:
            raise RuntimeError(
                f"DeepSeek {exc.status_code} (model={self.model}): {exc.message} "
                f"{OPERATOR_HINTS.get(exc.status_code, '')}".strip()
            ) from exc

        choice = response.choices[0] if response.choices else None
        content = (choice.message.content or "").strip() if choice else ""
        usage = response.usage

        preview = " ".join(content[:LOG_PREVIEW_CHARS].split())
        if len(content) > LOG_PREVIEW_CHARS:
            preview += " ..."

        print(
            f"[deepseek] model={response.model} finish={choice.finish_reason if choice else 'none'} "
            f"in={usage.prompt_tokens if usage else '?'} out={usage.completion_tokens if usage else '?'} "
            f"chars={len(content)} :: {preview or '<empty>'}"
        )

        return content
