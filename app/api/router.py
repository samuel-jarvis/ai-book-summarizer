from fastapi import APIRouter

from app.api.v1 import summary, health

api_router = APIRouter()
api_router.include_router(health.router, tags=["Health"])
api_router.include_router(summary.router, tags=["Summary"])
