from fastapi import FastAPI
from src.api.routes import auth

from src.api.routes import admin as admin_mod

admin_router = admin_mod.router

from dotenv import load_dotenv

load_dotenv()

from src.db.session import Base, engine
import src.models.user as _user  # to register model metadata

app = FastAPI(title="HireSense - Auth & Admin MVP")

# Create DB tables (for dev only)
Base.metadata.create_all(bind=engine)

app.include_router(auth.router)

if admin_router:
    app.include_router(admin_router, tags=["admin"])


@app.get("/")
def root():
    return {"message": "HireSense Auth API is up. Use /docs for API docs."}
