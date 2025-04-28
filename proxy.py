from fastapi import FastAPI, Request
from pydantic import BaseModel
import requests
import json
from typing import Optional
from openai import OpenAI

app = FastAPI()

# Replace this with the actual URL of your HuggingFace server
HUGGINGFACE_SERVER_URL = "https://arefin-001-t4-pixp2pix.hf.space/generate"

# Llama API configuration
NOVITA_BASE_URL = "https://api.novita.ai/v3/openai"
API_KEY = "sk_MW70hsHUnUIKGxOShT97HjJ5zf6qOoRzRRrZqLdQUys"  # Replace with your actual API key
LLAMA_MODEL = "meta-llama/llama-3.1-8b-instruct"

class GenerateInput(BaseModel):
    image_url: str
    prompt: str
    steps: int = 20                  # Default value set to 20
    guidance_scale: float = 7.5      # Default value set to 7.5
    image_guidance_scale: float = 1.5  # Default value set to 1.5
    negative_prompt: Optional[str] = None
    use_llama: bool = True          # Flag to indicate whether to use Llama for prompt refinement

def get_refined_prompts(prompt):
    """
    Use Llama to refine the prompt and generate a negative prompt
    """
    client = OpenAI(
        base_url=NOVITA_BASE_URL,
        api_key=API_KEY,
    )
    
    instruction = f"""
    Given the following image generation prompt: {prompt}
    
    Please refine it to make it more detailed and effective for image generation.
    Also create a detailed negative prompt to avoid unwanted elements in the image.
    
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
        response = client.chat.completions.create(
            model=LLAMA_MODEL,
            messages=[
                {"role": "user", "content": instruction}
            ],
            stream=False,
            max_tokens=1000,
            response_format={"type": "json_object"}
        )
        
        result = response.choices[0].message.content
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
            payload["prompt"] = input.promt
            payload["negative_prompt"] = refined_prompts["np"]
            # Remove the use_llama field as it's not needed by the HuggingFace server
            payload.pop("use_llama", None)
        
        # Forward request to your actual HuggingFace server
        response = requests.post(HUGGINGFACE_SERVER_URL, json=payload)
        
        # Forward the response back to the caller
        return response.json()
    
    except Exception as e:
        return {"status": "error", "message": str(e)}
