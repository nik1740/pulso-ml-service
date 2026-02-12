from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import os
import sys

# Add project root to path so imports work
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.routes import inference
from app.services.model_service import ModelService

# Model Path (adjust if needed)
MODEL_PATH = "/home/csc-project-pulso-2026/pulso-ml-service/models/PULSE-7B"

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Load Model
    print("Starting Pulso ML Service...")
    model_service = ModelService(MODEL_PATH)
    if model_service.load_model():
        app.state.model_service = model_service
        print("Model loaded and attached to app state.")
    else:
        print("Model failed to load. Check logs.")
        app.state.model_service = None
    
    yield
    
    # Shutdown
    print("Shutting down service...")

app = FastAPI(title="Pulso ML Service", lifespan=lifespan)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routes
app.include_router(inference.router, prefix="/api/v1")

@app.get("/health")
def health_check():
    status = "ok" if getattr(app.state, "model_service", None) else "model_not_loaded"
    return {"status": status}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=False)
