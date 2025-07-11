import json
from storage import save_to_tinydb
from crawler import get_webpage_content
from gemini import llm # or use from gpt import llm for OAI model

VOLUNTEER_URL = input("VOLUNTEER URL: ")
OUTPUT_FILENAME = "volunteer_opportunities.json"

print(f"Fetching content from: {VOLUNTEER_URL}")
page_content = get_webpage_content(VOLUNTEER_URL)

if page_content:
    print("Querying LLM...")
    opportunity_data = llm(page_content, VOLUNTEER_URL)

    if opportunity_data:
        print(json.dumps(opportunity_data, indent=2))
        save_to_tinydb(opportunity_data, OUTPUT_FILENAME)
    else:
        print("Failed to extract information using the API.")
else:
    print("Failed to retrieve webpage content. The script will now exit.")