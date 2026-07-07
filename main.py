from fastapi import FastAPI


app = FastAPI(
    title="AI Book Summarizer",    description="An API that summarizes books using AI.",
    version="1.0.0",
    contact={
        "name": "Jarvis 2077",
        "email": "jarvisdev2077@gmail.com"
    }
)


@app.get("/")
async def home():
    return {"message": "Welcome to the AI Book Summarizer API!"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        reload=True,
        port=8000,
    )
