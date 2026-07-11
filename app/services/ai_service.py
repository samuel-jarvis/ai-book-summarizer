from openai import AsyncOpenAI, AuthenticationError
from app.core.config import settings


class DeepSeekAIService:
    def __init__(self, model: str = "deepseek-v4-flash"):
        self.model = model
        _api_key = settings.DEEPSEEK_API_KEY
        if not _api_key:
            raise ValueError(
                "DEEPSEEK_API_KEY is empty. Set it in your .env file before running summaries."
            )

        self.client = AsyncOpenAI(
            api_key=_api_key,
            base_url=settings.DEEPSEEK_BASE_URL,
        )

    async def summarize_text(self, prompt: str) -> str:
        prompt_text = prompt.strip() if prompt else ""

        if not prompt_text:
            print("No text to summarize. Returning empty string.")
            return ""

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
            )
        except AuthenticationError as exc:
            raise RuntimeError(
                "DeepSeek authentication failed (401). Check DEEPSEEK_API_KEY in .env and ensure it is a valid active key."
            ) from exc

        content = response.choices[0].message.content if response.choices else None
        return content.strip() if content else ""
