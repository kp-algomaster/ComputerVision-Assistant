"""Image Generation server stub — Stable Diffusion / SDXL via diffusers."""
import os
import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse

app = FastAPI(title="CV Image Generation Server")


@app.get("/health")
async def health():
    return JSONResponse({"status": "ok", "device": os.environ.get("DEVICE", "auto")})


@app.get("/api/info")
async def info():
    loaded = []
    try:
        import torch
        device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    except ImportError:
        device = os.environ.get("DEVICE", "cpu")
    return JSONResponse({"loaded_models": loaded, "device": device, "ready": False})


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=7860, log_level="info")
