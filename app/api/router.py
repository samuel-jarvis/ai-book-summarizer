from fastapi import APIRouter

from app.api.v1 import auth, conversations, health, project, summary

api_router = APIRouter()
api_router.include_router(health.router, tags=["Health"])
api_router.include_router(auth.router, tags=["Auth"])
api_router.include_router(summary.router, tags=["Summary"])

api_router.include_router(project.router, tags=["Project"])
api_router.include_router(conversations.router, tags=["Conversations"])
