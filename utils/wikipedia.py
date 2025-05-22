import requests
import re
def get_foundation_date_from_wikipedia(company_name):
    def search_in_language(lang):
        try:
            search_url = f"https://{lang}.wikipedia.org/w/api.php"
            search_params = {
                "action": "query",
                "list": "search",
                "srsearch": company_name,
                "format": "json"
            }
            response = requests.get(search_url, params=search_params)
            results = response.json().get("query", {}).get("search", [])
            if not results:
                return None, None

            page_title = results[0]["title"]

            content_url = f"https://{lang}.wikipedia.org/w/api.php"
            content_params = {
                "action": "query",
                "prop": "revisions",
                "titles": page_title,
                "rvprop": "content",
                "rvslots": "main",
                "format": "json"
            }
            content_response = requests.get(content_url, params=content_params)
            pages = content_response.json().get("query", {}).get("pages", {})
            content = next(iter(pages.values())).get("revisions", [{}])[0].get("slots", {}).get("main", {}).get("*", "")

            # Regex to find founding date or similar infobox fields
            match = re.search(r"\|\s*(founded|foundation|création|date de création)\s*=\s*(.*)", content, re.IGNORECASE)
            if match:
                raw_date = match.group(2)
                clean_date = re.sub(r"\[\[|\]\]|<.*?>", "", raw_date).strip()
                return clean_date, lang
        except Exception as e:
            print(f"[Wikipedia {lang.upper()} ERROR] {company_name}: {e}")
        return None, None

    # Try French first, then English
    date, lang_used = search_in_language("fr")
    if not date:
        date, lang_used = search_in_language("en")

    return date

print(get_foundation_date_from_wikipedia("Nfinite"))