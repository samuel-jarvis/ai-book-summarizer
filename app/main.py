import asyncio
from contextlib import asynccontextmanager
import logging
import sys

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from redis.exceptions import RedisError

from app.api.router import api_router
from app.core.config import settings
from app.core.database import init_db
from app.taskiq_broker import broker

logger = logging.getLogger(__name__)

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())


@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()
    app.state.task_queue_started = False

    if not broker.is_worker_process:
        try:
            await broker.startup()
        except RedisError:
            logger.warning(
                "Redis is unavailable; starting the API without background "
                "task processing. Summary uploads will return 503 until Redis "
                "is reachable."
            )
        else:
            app.state.task_queue_started = True

    yield

    if not broker.is_worker_process and app.state.task_queue_started:
        await broker.shutdown()

app = FastAPI(
    title="AI Book Summarizer",
    description="An API that summarizes books using AI.",
    version="1.0.0",
    lifespan=lifespan,
    contact={
        "name": "Jarvis 2077",
        "email": "jarvisdev2077@gmail.com"
    }
)

if settings.BACKEND_CORS_ORIGINS:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[str(origin)
                       for origin in settings.BACKEND_CORS_ORIGINS],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

app.add_middleware(GZipMiddleware, minimum_size=1000)

app.include_router(api_router, prefix=settings.API_V1_PREFIX)


@app.get("/",  tags=["Root"])
async def home():
    return {
        "name": settings.PROJECT_NAME,
        "docs": "/docs",
        "health": f"{settings.API_V1_PREFIX}/health",
        "summary": f"{settings.API_V1_PREFIX}/summary"
    }

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        reload=True,
        port=8000,
    )
