import logging
from bs4 import BeautifulSoup, Tag

def parse_html(html_content: str, instructions: dict) -> str:
    """
    Parses HTML content using BeautifulSoup, applying filtering rules
    and cleaning the resulting text.
    """
    soup = BeautifulSoup(html_content, 'html.parser')

    # Exclude specified elements first
    if exclude_selectors := instructions.get('exclude'):
        for element in soup.select(exclude_selectors):
            element.decompose()

    # If include selectors are provided, focus only on those parts.
    # Otherwise, use the entire document body.
    if include_selectors := instructions.get('include'):
        content_elements = soup.select(include_selectors)
        if not content_elements:
            logging.warning(f"Include selectors '{include_selectors}' did not match any content.")
            return ""
        # Join the text from all matched elements
        text = '\n'.join(el.get_text(separator='\n', strip=True) for el in content_elements)
    else:
        text = soup.get_text(separator='\n', strip=True)

    # Final cleanup to remove any resulting blank lines
    clean_text = '\n'.join(line for line in text.splitlines() if line.strip())

    logging.info(f"Cleaned text:\n{clean_text}")
    return clean_text