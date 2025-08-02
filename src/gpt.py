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

    with open("src/llm_prompt.txt", "r") as f:
        prompt = f.read()

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

    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": prompt.format(url=url, content=content)}
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
