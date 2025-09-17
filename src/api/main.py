# src/api/main.py
# uvicorn src.api.main:app --reload
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# register routes
from src.api.routes import resume as resume_router  # will import our router

app = FastAPI(title="HireSense API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(resume_router.router, prefix="")


@app.get("/health")
async def health():
    return JSONResponse({"status": "ok", "msg": "HireSense backend healthy!!"})
