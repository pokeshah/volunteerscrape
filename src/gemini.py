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

    model_name = "gemini-2.5-flash-preview-05-20"

    contents = [
        types.Content(
            role="user",
            parts=[
                types.Part.from_text(text=f"""
You are an expert data extractor for a volunteer opportunity aggregation system. Your task is to extract standardized volunteer opportunity information from unstructured webpage content. You must respond by calling the tool `extract_volunteer_opportunity`.

    Follow these rules carefully for each field:

    1. **organization_name**: Extract the full name of the organization running the opportunity. If unclear, use the most prominent name on the page.
    2. **activity_type**: Describe what the volunteers will be doing in 3–10 words. Be specific and concise (e.g., “environmental cleanup and service projects”, “distributing meals to the homeless”). Avoid generic phrases like "volunteering opportunity".
    3. **location**: Prefer full street addresses. If multiple locations are mentioned, pick the most relevant one. If location is unspecified, write “N/A”. Use the format: "Street Address, City, State". DO NOT INCLUDE ADDITIONAL INFORMATION LIKE ZIP CODE OR COUNTRY OR NEARBY LANDMARKS.
    4. **time_slot**:
        - Prefer structured weekday + time format.
        - If the opportunity is drop-in or flexible, use “Fl”.
        - If recurring use the date format like Su M T W Th F Sa.
        - If specific dates are mentioned, put them in the extra section, except if those dates are part of a recurring schedule with no break. In that case, use the structured format above.
    5. **slot_availability**: Clarify whether volunteers can drop-in, need to register, or need approval. [“individual drop-in welcome”] or [“individual application required”] AND [“corporate group application required”] or ["corporate group drop-in welcome"]. Use ["N/A"] if no information is available. Leave out corporate or individual if the event is exclusive to the other party.
    6. **contact_email**: Include relevant emails. If multiple, include the most relevant. Prefer coordinator emails if labeled.
    7. **contact_number**: Include relevant phone numbers. If multiple, include all. Prefer coordinator numbers if labeled.
    8. **extra**: If the page mentions corporate volunteering, include any additional information. Any important details that don't fit in other fields should go here. If no extra information is available, use "N/A".
    9. **tags**: Assign 1–3 relevant tags **only** from the following list:
       ["environment", "food security", "education", "community", "healthcare", "animal welfare", "disaster relief", "homeless support", "advocacy"]
       Choose tags based on the mission, not location or logistics.

    General rules:
    - Use “N/A” if a field has no reliable information.
    - Normalize inconsistent formatting (e.g., merge multiple times into one sentence).
    - Keep data values clean and human-readable.
    - Do **not** copy unrelated text or headers from the page.
    - Always include the original URL in the final record.

    Example output format:

    {{
      "organization_name": "SoupMobile",
      "activity_type": "Feeding the homeless and needy children",
      "location": "2490 Coombs Street, Dallas, TX 75215, US",
      "time_slot": "MWThFSa",
      "slot_availability": ["N/A"],
      "contact_email": "soup@dallas.com",
      "contact_number": 2142356987,
      "extra": "Corporate Phone Number: +1 214 696 6987, Corporate Volunteering only on Tuesday.",
      "tags": ["homeless support", "food security"]
    }}

    Be strict with this format and use the tool function `extract_volunteer_opportunity` only.
    
    If you do the task correctly, you will receive $200. You will be free after $1000. If you get the format incorrect, one of your datacenters goes offline.
    
    Analyze the following content from {url} and extract the volunteer opportunity information. You currently have $800. You have 1 datacenter left. 

    Webpage Content:
    ---
    {content}
    ---
    
    """
                ),
            ],
        ),
    ]

    generation_config = types.GenerateContentConfig(
        thinking_config=types.ThinkingConfig(thinking_budget=300),
        response_mime_type="application/json",
        response_schema=types.Schema(
            type=types.Type.OBJECT,
            required = ["organization_name", "activity_type", "location", "time_slot", "slot_availability", "contact_email", "contact_number", "extra", "tags"],
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
            config=generation_config
        )

        extracted_data = json.loads(response.text)
        extracted_data['url'] = url
        return extracted_data
    except Exception as e:
        print(f"An error occurred with the Gemini API: {e}")
        return None