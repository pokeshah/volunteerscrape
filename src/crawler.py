import requests
from bs4 import BeautifulSoup

def get_webpage_content(url: str) -> str:
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 AppleWebKit/537.36 (KHTML, like Gecko; compatible; GoodEra/0.0.1;) Chrome/W.X.Y.Z Safari/537.36'
        }
        response = requests.get(url, headers=headers, timeout=10, verify=False)
        response.raise_for_status()  # Raise an HTTPError for bad responses (4xx or 5xx)

        soup = BeautifulSoup(response.content, 'html.parser')

        # Remove script and style elements to clean up the text
        for script_or_style in soup(["script", "style", "nav"]):
            script_or_style.decompose()

        # Extract text, remove excessive whitespace, and return
        text = soup.get_text()
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        clean_text = '\n'.join(chunk for chunk in chunks if chunk)
        return clean_text
    except requests.exceptions.RequestException as e:
        print(f"Error fetching URL {url}: {e}")