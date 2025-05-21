from fastapi import FastAPI, Request
from pydantic import BaseModel
import requests
import json
from typing import Optional
from openai import OpenAI
from google import genai
from google.genai import types
app = FastAPI()
from io import BytesIO
from supabase import create_client, Client
# Replace this with the actual URL of your HuggingFace server
HUGGINGFACE_SERVER_URL = "https://arefin-001-t4-pixp2pix.hf.space/generate"
import time

from PIL import Image
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
    
SUPABASE_URL = "https://trblhcytzzdsvatxjbds.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InRyYmxoY3l0enpkc3ZhdHhqYmRzIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NDM5MzQ3NjIsImV4cCI6MjA1OTUxMDc2Mn0.o6ysuFzh0GsW5rMvoRCt6LGN61gsjJszuKPxNtflumw" # Keep this secret!
BUCKET_NAME = "generated-images"

try:
    supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
except Exception as e:
    print(f"Error connecting to Supabase: {e}")
    # Decide how to handle this - maybe exit or run without Supabase functionality
    supabase = None # Set to None if connect   
def upload_image_to_supabase(pil_image, filename):
    if not supabase:
        raise Exception("Supabase client is not initialized.") # Prevent proceeding if Supabase connection failed

    try:
        # Convert PIL Image to bytes
        buffer = BytesIO()
        pil_image.save(buffer, format="PNG")
        buffer.seek(0)

        # Upload the file using the Supabase storage client
        file_path = f"{filename}.png"
        # Use buffer.getvalue() which returns bytes
        supabase.storage.from_(BUCKET_NAME).upload(
            path=file_path,
            file=buffer.getvalue(), # Pass the bytes directly
            file_options={"content-type": "image/png", "upsert": "false"} # Don't overwrite existing
        )

        # Check if upload was successful (newer supabase-py versions might not return useful data directly on success)
        # A common pattern is to try getting the public URL, which implies success.
        public_url_response = supabase.storage.from_(BUCKET_NAME).get_public_url(file_path)

        return {"status": "success", "url": public_url_response}
    except Exception as e:
        # More specific error handling could be added here based on Supabase client exceptions
        print(f"Supabase upload error: {e}") # Log the detailed error server-side
        return {"status": "error", "message": f"Failed to upload image to Supabase. {str(e)}"}


def get_refined_prompts(prompt, img_url):
    client = genai.Client(api_key="AIzaSyC-Tw-ccOtZ0y0TAjdAzIJ6N2b8TnbAOzk")
    try:
        print("Running")

        image_path = img_url
        image_bytes = requests.get(image_path).content
        image = types.Part.from_bytes(
            data=image_bytes, mime_type="image/jpeg"
        )
        instruction = f"""
        Given the following image generation prompt: {prompt}

        Please refine it to make it more precise and effective for image generation, focusing **only on enhancing what the user is asking for**, without introducing changes to unrelated features.

        🛑 Do not:
        - Change hair thickness, volume, or texture.
        - Alter skin tone, texture, or facial features.
        - Modify the color or style of clothing.
        - Add or remove background elements or change its color.
        - Change the lighting conditions or direction.
        - Apply any heavy or dramatic makeup.

        ✅ You may:
        - Slightly enhance the description based on the user's intent.
        - Add light pink blush and subtle lipstick if the prompt suggests makeup.
        - Keep improvements minimal, targeted, and relevant to the original request.

        Make it a command, it is necessary!! Like change it ... or update it ... or edit it ... Add the negative promt too after instrcution, make sure to not make any facial shape change, dont make background or lighting change, just edit

        Respond in **JSON** format with only:
        - `"p"`: the refined, enhanced prompt

        Example output format:
        
        {{
        "p": "refined prompt here",
        }}

        """
        # response = client.models.generate_content(
        #     model="gemini-2.5-flash-preview-04-17",
        #     contents=instruction,
        #     config=types.GenerateContentConfig(
        #         response_mime_type='application/json',
        #         response_schema=CountryInfo,
        #     ),
        # )
        
        # result = response.text
        # result_json = json.loads(result)
        # print(result_json['p'])
        # return
        response = client.models.generate_content(
            model="gemini-2.0-flash-preview-image-generation",
            contents=[prompt, "dont change background or lightings, dont distort face", image],
            config=types.GenerateContentConfig(
                temperature=0.25,

                response_modalities=["TEXT", "IMAGE"]
            ),
        )

        # Loop over response parts to find the image
        for part in response.candidates[0].content.parts:
            if part.inline_data is not None:
                image = Image.open(BytesIO(part.inline_data.data))

                # Save to Supabase
                filename = f"generated_image_{int(time.time())}"  # e.g. generated_image_1689439837
                result = upload_image_to_supabase(image, filename)

                if result["status"] == "success":
                    return result
                else:
                    raise Exception(result["message"])
        
        raise Exception("No image found in the response.")
    
    except Exception as e:
        print(f"Error: {e}")
        return None


# if __name__ == "__main__":
    
#     print(get_refined_prompts("Edit to have some makeup", "https://drive.google.com/u/0/drive-viewer/AKGpihbh88_-jmOpahS4h8mcCMy5wrW-dc7KEJP0TgTrr6GdyRCMj2-pOvSgpX0MesSMOzXCncaAvZ9B_zLG27gcCAB9tp3H2-Df58U=s1600-rw-v1"))

@app.post("/generate")
def proxy_generate(input: GenerateInput):
    try:
        payload = input.dict()
        
        # If the llama refinement is requested
        refined_prompts = get_refined_prompts(input.prompt, input.image_url)
        # payload["prompt"] = input.prompt
        # payload["negative_prompt"] = refined_prompts["np"]
        # Remove the use_llama field as it's not needed by the HuggingFace server
        # payload.pop("use_llama", None)
        
        # result_json = json.loads(refined_prompts)
     
        return refined_prompts
    
    except Exception as e:
        return {"status": "error", "message": str(e)}
