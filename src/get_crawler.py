import requests
import logging
from parser import parse_html


def get_webpage_content(url: str, instructions: dict) -> str:
    """
    Fetches webpage content using GET requests and parses it.
    """
    logging.info(f"Fetching content from {url} with GET request")
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/107.0.0.0 Safari/537.36'
    }
    response = requests.get(url, headers=headers, timeout=10)
    response.raise_for_status()
    logging.info(f"Successfully fetched content from {url}")
    return parse_html(response.content, instructions)
