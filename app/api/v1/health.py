from fastapi import APIRouter, Response, status
from sqlalchemy import text
from app.api.deps import DbSession
from app.core.config import settings
from app.taskiq_broker import is_task_queue_available

router = APIRouter()


@router.get("/health")
async def get_health(db: DbSession, response: Response):
    # Perform a simple database query to check the connection

    db_ok = True
    try:
        await db.execute(text("SELECT 1"))
    except Exception:
        db_ok = False

    queue_ok = await is_task_queue_available()

    if not db_ok:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE

    return {
        "status": "healthy" if db_ok and queue_ok else "degraded" if db_ok else "unhealthy",
        "environment": settings.ENVIRONMENT,
        "database": "reachable" if db_ok else "unreachable",
        "task_queue": "reachable" if queue_ok else "unreachable",
    }
