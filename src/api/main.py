# src/api/main.py
from fastapi import FastAPI
from fastapi.responses import JSONResponse

app = FastAPI(title="HireSense API (skeleton)")

@app.get("/health")
async def health():
    return JSONResponse({"status": "ok", "msg": "HireSense backend healthy"})

# import routes lazily if you add them later
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.api.main:app", host="0.0.0.0", port=8000, reload=True)
