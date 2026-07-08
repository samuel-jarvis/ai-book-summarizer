from fastapi import APIRouter

router = APIRouter()


@router.get("/summary")
async def get_summary():
    return {"message": "This is the summary endpoint."}


@router.post("/summary")
async def create_summary():
    return {"message": "This is the create summary endpoint."}
