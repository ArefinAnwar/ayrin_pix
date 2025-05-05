from fastapi import FastAPI, Request
from pydantic import BaseModel
import requests
import json
from typing import Optional
from openai import OpenAI
from google import genai
from google.genai import types
app = FastAPI()

# Replace this with the actual URL of your HuggingFace server
HUGGINGFACE_SERVER_URL = "https://arefin-001-t4-pixp2pix.hf.space/generate"

# Llama API configuration
# NOVITA_BASE_URL = "https://api.novita.ai/v3/openai"
# API_KEY = "sk_MW70hsHUnUIKGxOShT97HjJ5zf6qOoRzRRrZqLdQUys"  # Replace with your actual API key
# LLAMA_MODEL = "meta-llama/llama-3.1-8b-instruct"


class GenerateInput(BaseModel):
    image_url: str
    prompt: str
    steps: int = 20                  # Default value set to 20
    guidance_scale: float = 7.5      # Default value set to 7.5
    image_guidance_scale: float = 1.5  # Default value set to 1.5
    negative_prompt: Optional[str] = None
    use_llama: bool = True          # Flag to indicate whether to use Llama for prompt refinement

class CountryInfo(BaseModel):
    p: str
    np: str
    
    
def get_refined_prompts(prompt):
    """
    Use Llama to refine the prompt and generate a negative prompt
    """
    # client = OpenAI(
    #     base_url=NOVITA_BASE_URL,
    #     api_key=API_KEY,
    # )

    client = genai.Client(api_key="AIzaSyC-Tw-ccOtZ0y0TAjdAzIJ6N2b8TnbAOzk")

    
    instruction = f"""
    Given the following image generation prompt: {prompt}
    
    Please refine it to make it more detailed and effective for image generation.
    Also create a detailed negative prompt to avoid unwanted elements in the image. Do not change hair color, do not change skin tone, do not alter clothes color, do not distort face, do not change facial structure, do not add heavy makeup, only add lipstick and light pink blush on cheeks, do not change background color, do not change lighting
    When performing any face editing operations, strictly adhere to these non-negotiable constraints to preserve the original appearance and structure:
    - Preserve the exact original hair color, thickness, fullness, and volume; do not introduce any changes to the hair color, density, or volume.
    - Preserve the exact original skin tone, structure, and texture; do not alter the skin tone, pores, or surface details in any way.
    - Preserve the exact original clothing colors; do not change the colors of the clothes.
    - Ensure the face remains natural and free from distortion; avoid any actions that would distort the facial features, overall facial structure, or *any* part of the image.
    - Maintain the original facial structure precisely; do not modify the underlying skeletal or muscular structure of the face.
    - If makeup application is part of the editing request, *only* add subtle lipstick and light pink blush on the cheeks; heavy, dramatic, or any other type of makeup is strictly prohibited.
    - Preserve the exact original background color; do not change the background color.
    - Preserve the exact original lighting conditions and direction; do not alter the lighting.
    
    Return your response in JSON format with two keys:
    - 'p': the refined detailed prompt
    - 'np': the negative prompt
    
    Example format:

    {{
        "p": "detailed refined prompt here",
        "np": "detailed negative prompt here"
    }}
    """
    
    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash-preview-04-17",
            contents=instruction,
            config=types.GenerateContentConfig(
                response_mime_type='application/json',
                response_schema=CountryInfo,
            ),
        )
        
        result = response.text
        print(result)
        return json.loads(result)
    except Exception as e:
        # If there's an error, return the original prompt and a generic negative prompt
        return {
            "p": prompt,
            "np": "low quality, bad anatomy, worst quality, low resolution"
        }

@app.post("/generate")
def proxy_generate(input: GenerateInput):
    try:
        payload = input.dict()
        
        # If the llama refinement is requested
        if input.use_llama:
            refined_prompts = get_refined_prompts(input.prompt)
            payload["prompt"] = input.prompt
            payload["negative_prompt"] = refined_prompts["np"]
            # Remove the use_llama field as it's not needed by the HuggingFace server
            payload.pop("use_llama", None)
        
        # Forward request to your actual HuggingFace server
        response = requests.post(HUGGINGFACE_SERVER_URL, json=payload)
        
        # Forward the response back to the caller
        return response.json()
    
    except Exception as e:
        return {"status": "error", "message": str(e)}
