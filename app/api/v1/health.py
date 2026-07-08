from fastapi import APIRouter
from sqlalchemy import text
from app.api.deps import DbSession
from app.core.config import settings

router = APIRouter()


@router.get("/health")
async def get_health(db: DbSession):
    # Perform a simple database query to check the connection

    db_ok = True
    try:
        await db.execute(text("SELECT 1"))
    except Exception as e:
        db_ok = False
    return {
        "status": "unhealthy",
        "environment": settings.ENVIRONMENT,
        "database": "unreachable" if not db_ok else "reachable"
    }
