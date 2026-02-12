from fastapi import APIRouter, HTTPException, Depends
from ..schemas.ecg import InferenceRequest, InferenceResponse
from ..services.model_service import ModelService
import time

router = APIRouter()



# Correct implementation with Request object
from fastapi import Request

@router.post("/predict", response_model=InferenceResponse)
async def predict_ecg(payload: InferenceRequest, request: Request):
    
    start_time = time.time()
    model_service: ModelService = request.app.state.model_service
    
    if not model_service:
        raise HTTPException(status_code=503, detail="Model service not initialized")

    try:
        result = model_service.predict(payload.image, prompt=payload.prompt)
        
        if "error" in result:
             raise HTTPException(status_code=500, detail=result["error"])

        processing_time = (time.time() - start_time) * 1000
        
        return InferenceResponse(
            inference=result["inference"],
            session_id=payload.session_id,
            processing_time_ms=processing_time
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
