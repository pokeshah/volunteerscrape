from tinydb import TinyDB, Query

def save_to_tinydb(data: dict, db_path="volunteer_opportunities.json") -> None:
    db = TinyDB(db_path)
    Opportunity = Query()

    url = data.get("url")
    if db.contains(Opportunity.url == url):
        print(f"URL {url} already exists in {db_path}. Skipping.")
        return

    db.insert(data)
    print(f"Successfully saved opportunity from {url} to {db_path}.")
