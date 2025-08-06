import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from langchain.tools import tool
from dotenv import load_dotenv

load_dotenv()

class VolunteerOpportunity(BaseModel):
    organization_name: str = Field(description="The full name of the organization running the opportunity. If unclear, use the most prominent name on the page.")
    activity_type: str = Field(description='A description of what the volunteers will be doing in 3–10 words. Be specific and concise (e.g., "environmental cleanup and service projects", "distributing meals to the homeless"). Avoid generic phrases like "volunteering opportunity".')
    location: str = Field(description='The full street address. If multiple locations are mentioned, pick the most relevant one. If location is unspecified, write "N/A". Use the format: "Street Address, City, State". DO NOT INCLUDE ADDITIONAL INFORMATION LIKE ZIP CODE OR COUNTRY OR NEARBY LANDMARKS.')
    time_slot: str = Field(description='The time slot for the opportunity. If the opportunity is drop-in or flexible, use "F". If recurring use the date format like Su M T W Th F Sa. If specific dates are mentioned, put them in the extra section, except if those dates are part of a recurring schedule with no break. In that case, use the structured format above.')
    slot_availability: list[str] = Field(description='Clarify whether volunteers can drop-in, need to register, or need approval. Use one or more of the following values: ["individual drop-in welcome"], ["individual application required"], ["corporate group application required"], ["corporate group drop-in welcome"]. Use ["N/A"] if no information is available. Leave out corporate or individual if the event is exclusive to the other party.')
    contact_email: str = Field(description="Include relevant emails. If multiple, include the most relevant. Prefer coordinator emails if labeled. Use 'N/A' if no email is available.")
    contact_number: int = Field(description="Include relevant phone numbers. If multiple, include all. Prefer coordinator numbers if labeled. Use 0 if no number is available.")
    extra: str = Field(description="If the page mentions corporate volunteering, include any additional information. Any important details that don't fit in other fields should go here. If no extra information is available, use 'N/A'.")
    tags: list[str] = Field(description='Assign 1–3 relevant tags only from the following list: ["environment", "food security", "education", "community", "healthcare", "animal welfare", "disaster relief", "homeless support", "advocacy"]. Choose tags based on the mission, not location or logistics.')

@tool(args_schema=VolunteerOpportunity)
def extract_volunteer_opportunity(organization_name: str, activity_type: str, location: str, time_slot: str, slot_availability: list[str], contact_email: str, contact_number: int, extra: str, tags: list[str]) -> dict:
    """Extracts volunteer opportunity details from the provided text."""
    return {
        "organization_name": organization_name,
        "activity_type": activity_type,
        "location": location,
        "time_slot": time_slot,
        "slot_availability": slot_availability,
        "contact_email": contact_email,
        "contact_number": contact_number,
        "extra": extra,
        "tags": tags
    }

def llm(content: str, url: str, model_name: str = "gemini") -> dict:
    if model_name == "gemini":
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable not set. Please set the key to proceed.")
        llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=api_key)
    elif model_name == "gpt":
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set. Please set the key to proceed.")
        llm = ChatOpenAI(openai_api_key=api_key, model_name="gpt-4.1-mini")
    else:
        raise ValueError(f"Unknown model: {model_name}")

    with open("src/llm_prompt.txt", "r") as f:
        prompt_template = f.read()

    prompt = ChatPromptTemplate.from_template(prompt_template)
    llm_with_tools = llm.bind_tools([extract_volunteer_opportunity])
    chain = prompt | llm_with_tools

    response = chain.invoke({"url": url, "content": content})
    tool_call = response.tool_calls[0]
    extracted_data = tool_call["args"]
    extracted_data['url'] = url
    return extracted_data

#send full content from extension to flask
#add llm prompt to extract the filter from content
#extract the original content with the filter
#pass the extracted content to llm.py
#return filter, final attributes