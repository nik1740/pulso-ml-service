# Pulso Backend Integration Guide 🚀

I have generated the **exact files** you need based on your backend code. You don't need to write any code, just copy these files.

## 1. Copy the Integration Files

On your **backend server**, copy these two files into your `app/services/` directory:

1.  **`pulso-ml-service/INTEGRATION_CODE/mec_service.py`**  
    $\rightarrow$ Copy to `backend/app/services/mec_service.py`

2.  **`pulso-ml-service/INTEGRATION_CODE/gemini_service.py`**  
    $\rightarrow$ Copy to `backend/app/services/gemini_service.py`  
    *(This replaces your existing file with the integrated version)*

## 2. Update Environment Variables

Add this line to your backend's `.env` file:

```env
MEC_SERVICE_URL=http://localhost:8000/api/v1/predict
```
*(Replace `localhost` with the MEC server's IP address if they are on different machines)*

## 3. Install Dependencies

Ensure `httpx` is installed in your backend environment (it should already be there):

```bash
pip install httpx
```

## 4. Restart Backend

Restart your FastAPI backend. Now, every time a user requests an analysis, the backend will:
1.  Send the ECG image to this MEC Server (Pulse-7B).
2.  Receive a specialized Rhythm/Morphology report.
3.  Send that report + patient context to Gemini 3.
4.  Generate the final result for the user.
