import httpx
import os
import base64
from typing import Optional, Dict

class MECService:
    """
    Service to communicate with the MEC Pulse ECG Model
    """
    def __init__(self):
        # Default to localhost if not set. Update this in production to the MEC server IP.
        self.mec_url = os.getenv("MEC_SERVICE_URL", "http://localhost:8000/api/v1/predict")

    async def analyze_image(self, image_data: bytes, session_id: str = None) -> str:
        """
        Send ECG image to MEC Server for Pulse-7B Analysis
        
        Args:
            image_data: Raw bytes of the ECG image
            session_id: Optional tracking ID
            
        Returns:
            The text inference from the Pulse model, or specific error message.
        """
        if not image_data:
            return "No ECG image available for Pulse analysis."

        try:
            # Encode image to base64 string
            encoded_image = base64.b64encode(image_data).decode('utf-8')
            
            payload = {
                "image": encoded_image,
                "session_id": str(session_id) if session_id else None,
                "prompt": "Describe this ECG image in detail, focusing on rhythm, rate, intervals, and any anomalies."
            }

            # Call MEC API
            async with httpx.AsyncClient(timeout=45.0) as client:
                response = await client.post(self.mec_url, json=payload)
                
                if response.status_code == 200:
                    result = response.json()
                    return result.get("inference", "No inference returned.")
                else:
                    print(f"MEC Service Error: {response.status_code} - {response.text}")
                    return f"Pulse Analysis Unavailable (Status {response.status_code})"
                    
        except httpx.RequestError as e:
            print(f"MEC Connection Error: {e}")
            return "Pulse Analysis Unavailable (Connection Error)"
        except Exception as e:
            print(f"MEC Integration Error: {e}")
            return f"Pulse Analysis Failed: {str(e)}"
