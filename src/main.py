from fastapi import FastAPI
from src.api.routes import auth, admin
from src.db.session import Base, engine
import src.models.user as _user  # to register model metadata

app = FastAPI(title="HireSense - Auth & Admin MVP")

# Create DB tables (for dev only)
Base.metadata.create_all(bind=engine)

app.include_router(auth.router)
app.include_router(admin.router)


@app.get("/")
def root():
    return {"message": "HireSense Auth API is up. Use /docs for API docs."}
