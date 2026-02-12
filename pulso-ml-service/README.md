# Pulso ML Service (MEC)

This service hosts the **Pulse-7B ECG Analysis Model** on the Mobile Edge Computing (MEC) server.
It provides a FastAPI interface for the main application backend to request ECG inference.

## Features
- **Pulse-7B Model**: LLaVA-based VLM optimized for ECG interpretation.
- **4-bit Quantization**: Runs on 8GB VRAM GPUs (e.g., NVIDIA Quadro RTX 4000).
- **MEC Integration**: Provides low-latency inference at the edge.

## Setup

1. **Install Dependencies**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

2. **Download Model Weights**
   ```bash
   python download_model.py
   ```
   *This downloads ~14GB of weights to `models/PULSE-7B`.*

3. **Run the Service**
   ```bash
   python app/main.py
   ```
   *The service will start on port 8000.*

## API Usage

**POST /api/v1/predict**
```json
{
  "image": "<base64_string>",
  "prompt": "Describe this ECG.",
  "session_id": "optional-id"
}
```

## Backend Integration
See `INTEGRATION_GUIDE.md` for instructions on connecting your backend.
