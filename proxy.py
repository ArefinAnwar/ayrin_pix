from fastapi import FastAPI, Request
from pydantic import BaseModel
import requests

app = FastAPI()

# Replace this with the actual URL of your HuggingFace server
HUGGINGFACE_SERVER_URL = "https://arefin-001-t4-pixp2pix.hf.space/generate"

class GenerateInput(BaseModel):
    image_url: str
    prompt: str
    steps: int = 20                  # Default value set to 20
    guidance_scale: float = 7.5      # Default value set to 7.5
    image_guidance_scale: float = 1.5  # Default value set to 1.5
    negative_prompt: Optional[str] = None 

@app.post("/generate")
def proxy_generate(input: GenerateInput):
    try:
        # Forward request to your actual HuggingFace server
        response = requests.post(HUGGINGFACE_SERVER_URL, json=input.dict())

        # Forward the response back to the caller
        return response.json()
    
    except Exception as e:
        return {"status": "error", "message": str(e)}
