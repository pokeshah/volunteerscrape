from playwright.sync_api import sync_playwright, Error
import logging
from parser import parse_html


def get_webpage_content_js(url: str, instructions: dict) -> str:
    """
    Fetches webpage content using Playwright and parses it.
    """
    with sync_playwright() as p:
        logging.info(f"Launching browser")
        browser = p.chromium.launch()
        page = browser.new_page()
        logging.info(f"Navigating to {url}")
        page.goto(url)
        if wait_selector := instructions.get('wait'):
            logging.info(f"Waiting for selector: {wait_selector}")
            page.wait_for_selector(wait_selector)
        html_content = page.content()
        logging.info(f"Successfully fetched content from {url}")
        return parse_html(html_content, instructions)