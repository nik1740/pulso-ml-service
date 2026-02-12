import asyncio
import httpx
import base64
import json

# Configuration
MEC_SERVICE_URL = "http://localhost:8000/api/v1/predict"
IMAGE_PATH = "path/to/sample_ecg.png" # Replace with actual image path

async def test_integration():
    print(f"Testing integration with MEC Service at {MEC_SERVICE_URL}...")
    
    # 1. Load an image (create a dummy one if needed or use existing)
    # For demo, we'll just send a dummy base64 string if file doesn't exist
    try:
        with open(IMAGE_PATH, "rb") as img_file:
            image_data = base64.b64encode(img_file.read()).decode('utf-8')
    except FileNotFoundError:
        # 10x10 Red Square (Valid PNG)
        image_data = "iVBORw0KGgoAAAANSUhEUgAAAAoAAAAKCAIAAAACUFjqAAAAEklEQVR4nGP8z4APMOGVHbHSAEEsAROxCnMTAAAAAElFTkSuQmCC"

    payload = {
        "image": image_data,
        "session_id": "test-session-123",
        "patient_id": "patient-456",
        "prompt": "Describe the rhythm."
    }

    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(MEC_SERVICE_URL, json=payload)
            
            if response.status_code == 200:
                print("\n✅ Success! Response:")
                print(json.dumps(response.json(), indent=2))
            else:
                print(f"\n❌ Error: {response.status_code}")
                print(response.text)
                
    except Exception as e:
        print(f"\n❌ Exception: {e}")

if __name__ == "__main__":
    asyncio.run(test_integration())
