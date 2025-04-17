from fastapi import FastAPI, Request
from pydantic import BaseModel
import requests

app = FastAPI()

# Replace this with the actual URL of your HuggingFace server
HUGGINGFACE_SERVER_URL = "https://arefin-001-t4-pixp2pix.hf.space/generate"

class GenerateInput(BaseModel):
    image_url: str
    prompt: str
    negative_prompt: str
    image_guidance_scale: float
    guidance_scale: float
    steps: int

@app.post("/generate")
def proxy_generate(input: GenerateInput):
    try:
        # Forward request to your actual HuggingFace server
        response = requests.post(HUGGINGFACE_SERVER_URL, json=input.dict())

        # Forward the response back to the caller
        return response.json()
    
    except Exception as e:
        return {"status": "error", "message": str(e)}
