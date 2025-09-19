# src/api/main.py
# uvicorn src.api.main:app --reload
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from fastapi.encoders import jsonable_encoder

# register routes

app = FastAPI(title="HireSense API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """
    Custom handler for validation errors that safely encodes any `bytes`
    found inside the validation error details by decoding them with
    errors='replace' so the server doesn't crash on non-UTF8 bytes.
    """
    try:
        safe_detail = jsonable_encoder(
            exc.errors(), custom_encoder={bytes: lambda b: b.decode(errors="replace")}
        )
    except Exception:
        # fallback: convert to string representation
        safe_detail = str(exc.errors())
    return JSONResponse(status_code=422, content={"detail": safe_detail})


# -- Register other exception handlers --
@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500, content={"detail": "Internal Server Error", "msg": str(exc)}
    )


from src.api.routes import resume as resume_router  # will import our router

app.include_router(resume_router.router, prefix="", tags=["resume"])


@app.get("/health")
async def health():
    return JSONResponse({"status": "ok", "msg": "HireSense backend healthy!!"})
