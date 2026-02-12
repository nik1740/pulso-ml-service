from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any

class InferenceRequest(BaseModel):
    image: str = Field(..., description="Base64 encoded ECG image")
    prompt: Optional[str] = "Describe this ECG image."
    session_id: Optional[str] = None
    patient_id: Optional[str] = None

class InferenceResponse(BaseModel):
    inference: str
    session_id: Optional[str] = None
    model_version: str = "pulse-7b-nf4"
    processing_time_ms: Optional[float] = None
