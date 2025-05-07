import requests

def google_cse_search(company_name, api_key, cx, extra_keywords=None, num_results=10):
    if extra_keywords is None:
        extra_keywords = [
            "site officiel",                    # Official source
            #"présence internationale",          # Global presence
            #"filiales",                         # Subsidiaries
            "nombre d'employés",                # HR scale
            #"CA annuel", "chiffre d'affaires",  # Revenue
            "clients",                          # Customer base
            "événements d'entreprise",          # Company events
            "recrutement", "offres d'emploi",   # Hiring intent
            "actualité", "news",                # Recent developments
            "CSE",                              # Comité Social Économique
            #"partenariats",                     # Collaboration potential
            #"team", "leadership", "dirigeants", # People
            "contact entreprise",               # General contact info
            "LinkedIn",                         # General LinkedIn
            f"{company_name} LinkedIn",         # Direct LinkedIn page
            f"{company_name} équipe LinkedIn",  # Team on LinkedIn
            "responsable communication",        # Key contact for you
            "responsable marketing",
            "office manager",
            "talent acquisition",
            "event manager",
            "expansion", "nouveaux bureaux",    # Growth and expansion
            "lancement", "produit",             # New initiatives
        ]
        
    all_snippets = []
    for keyword in extra_keywords:
        query = f"{company_name} {keyword}"
        params = {
            'key': api_key,
            'cx': cx,
            'q': query,
            'num': num_results,
            'gl': 'fr'
        }
        response = requests.get("https://www.googleapis.com/customsearch/v1", params=params)

        if response.status_code == 200:
            results = response.json()
            items = results.get('items', [])
            print(f"[DEBUG] Query: {query} — Results: {len(items)}")
            snippets = [f"🔹 {item.get('title', '')}\n{item.get('snippet', '')}" for item in items]
            all_snippets.extend(snippets)
        else:
            print(f"[ERROR] Query failed: {query}, Status: {response.status_code}")

    return "\n\n".join(all_snippets)