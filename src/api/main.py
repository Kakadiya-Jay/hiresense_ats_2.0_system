# src/api/main.py
"""
FastAPI entrypoint for HireSense API.
This file mounts route modules and provides a health endpoint.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Import routers
from src.api.routes import resume  # ensure this package is importable
from src.api.routes import score as score_routes  # new file: src/api/routes/score.py

from dotenv import load_dotenv

load_dotenv()


app = FastAPI(title="HireSense API", version="0.1")

# CORS for local dev / Streamlit frontends
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(resume.router)
app.include_router(score_routes.router) 

# Simple health route
@app.get("/health")
def health():
    return {"status": "ok", "message": "HireSense API is running"}
