import os
import json
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()

def llm(content: str, url: str) -> dict:
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY environment variable not set. Please set the key to proceed.")

    client = genai.Client(
        api_key=os.environ.get("GEMINI_API_KEY"),
    )

    model_name = "gemini-2.5-flash"

    with open("src/llm_prompt.txt", "r") as f:
        prompt = f.read()

    contents = [
        types.Content(
            role="user",
            parts=[
                types.Part.from_text(text=prompt.format(url=url, content=content)),
            ],
        ),
    ]

    generation_config = types.GenerateContentConfig(
        response_mime_type="application/json",
        response_schema=types.Schema(
            type=types.Type.OBJECT,
            required=["organization_name", "activity_type", "location", "time_slot", "slot_availability", "contact_email", "contact_number", "extra", "tags"],
            properties={
                "organization_name": types.Schema(type=types.Type.STRING),
                "activity_type": types.Schema(type=types.Type.STRING),
                "location": types.Schema(type=types.Type.STRING),
                "time_slot": types.Schema(type=types.Type.STRING),
                "slot_availability": types.Schema(
                    type=types.Type.ARRAY,
                    items=types.Schema(type=types.Type.STRING)
                ),
                "contact_email": types.Schema(type=types.Type.STRING),
                "contact_number": types.Schema(type=types.Type.INTEGER),
                "extra": types.Schema(type=types.Type.STRING),
                "tags": types.Schema(
                    type=types.Type.ARRAY,
                    items=types.Schema(type=types.Type.STRING)
                ),
            }
        )
    )

    try:
        response = client.models.generate_content(
            model=model_name,
            contents=contents,
            generation_config=generation_config
        )

        extracted_data = json.loads(response.text)
        extracted_data['url'] = url
        return extracted_data
    except Exception as e:
        print(f"An error occurred with the Gemini API: {e}")
        return None
