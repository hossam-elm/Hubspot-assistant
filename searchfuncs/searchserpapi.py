import requests


def serpapi_search(company_name, api_key, extra_keywords=None, num_results=20):
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
        
        # Setup SerpApi search parameters
        params = {
            'q': query,
            'api_key': api_key,
            'num': num_results,
            'engine': 'google',  # Specify to use Google's engine,
            'gl': 'fr',  # Set the location to France
        }
        
        # Send the request to SerpApi
        url = 'https://serpapi.com/search'
        response = requests.get(url, params=params)
        
        # Check if the request was successful
        if response.status_code == 200:
            results = response.json()
            
            # Check for 'organic_results' in the response
            items = results.get('organic_results', [])
            
            # Debugging output to see how many results we get
            print(f"[DEBUG] Query: {query} — Results: {len(items)}")
            
            # Extract the title and snippet for each result
            snippets = [f"🔹 {item.get('title', '')}\n{item.get('snippet', '')}" for item in items]
            all_snippets.extend(snippets)
        else:
            print(f"[ERROR] Failed to retrieve data for query: {query}, Status Code: {response.status_code}")

    # Return all the snippets as a single string
    return "\n\n".join(all_snippets)
