import os
import json
import openai
from dotenv import load_dotenv

load_dotenv()


def llm(content: str, url: str) -> dict:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set. Please set the key to proceed.")

    client = openai.OpenAI(api_key=api_key)

    model_name = "gpt-4.1-mini"

    tools = [
        {
            "type": "function",
            "function": {
                "name": "extract_volunteer_opportunity",
                "description": "Extracts volunteer opportunity details from the provided text.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "organization_name": {
                            "type": "string",
                            "description": "The name of the organization offering the opportunity."
                        },
                        "activity_type": {
                            "type": "string",
                            "description": "The type of volunteer activity."
                        },
                        "location": {
                            "type": "string",
                            "description": "The location of the volunteer opportunity."
                        },
                        "time_slot": {
                            "type": "string",
                            "description": "The time slot or schedule for the activity. 'F' for flexible."
                        },
                        "slot_availability": {
                            "type": "array",
                            "items": {
                                "type": "string"
                            },
                            "description": "Information on the availability of slots."
                        },
                        "contact_email": {
                            "type": "string",
                            "description": "Contact information for the organization or coordinator."
                        },
                        "contact_number": {
                            "type": "integer",
                            "description": "Contact phone number for the organization or coordinator."
                        },
                        "extra": {
                            "type": "string",
                            "description": "Any additional information about the opportunity, such as corporate volunteering details."
                        },
                        "tags": {
                            "type": "array",
                            "items": {
                                "type": "string"
                            },
                            "description": "Relevant tags from the predefined categories."
                        }
                    },
                    "required": ["organization_name", "activity_type", "location", "time_slot", "slot_availability", "contact_email", "contact_number", "extra" , "tags"]
                }
            }
        }
    ]

    system_prompt = """
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
        - Additional time details can be included in the corporate_volunteering field.
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
    """

    user_prompt = f"""
    Analyze the following content from {url} and extract the volunteer opportunity information. You currently have $800. You have 1 datacenter left. 

    Webpage Content:
    ---
    {content}
    ---
    """

    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            tools=tools,
            tool_choice={"type": "function", "function": {"name": "extract_volunteer_opportunity"}}  # Force the model to use the specified tool
        )

        tool_call = response.choices[0].message.tool_calls[0]
        extracted_data_str = tool_call.function.arguments

        extracted_data = json.loads(extracted_data_str)
        extracted_data['url'] = url
        return extracted_data
    except Exception as e:
        print(f"An error occurred with the OpenAI API: {e}")
        return None