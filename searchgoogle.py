import requests

def google_cse_search(company_name, api_key, cx, num_results=10):
    # Natural queries without question marks
    query_list = [
        f"{company_name} intext:actualité",
        f"{company_name} intext:collaborateurs",
        f"{company_name} intext:anniversaire",
        f"{company_name} intext:bureau",
        f"{company_name} intext:effectif",
        f"{company_name} intext:calendrier",
        f"{company_name} intext:clients",
        f"{company_name} intext:événements",
        f"{company_name} intext:site officiel",
        f"site:linkedin.com '{company_name}' intext:event",
        f"intext:agicap event",
        f"site:linkedin.com '{company_name}' intext:cse",
        f"site:linkedin.com '{company_name}' intext:rh",
        f"site:linkedin.com '{company_name}' intext:marketing",
        f"site:linkedin.com '{company_name}' intext:communication",
        f"site:linkedin.com '{company_name}' intext:office",
        f"site:linkedin.com '{company_name}' intext:events",
        f"site:linkedin.com '{company_name}' intext:talent",
        ]
    
    all_snippets = []

    # Process each query
    for query in query_list:
        params = {
            'key': api_key,
            'cx': cx,
            'q': query,
            'num': num_results,
            'gl': 'fr'  # To filter for French content
        }
        print(f"[DEBUG] Query: {query}")
        response = requests.get("https://www.googleapis.com/customsearch/v1", params=params)

        if response.status_code == 200:
            results = response.json()
            items = results.get('items', [])
            print(f"[INFO] {query} — Results: {len(items)}")
            snippets = [
                f"🔹 {item.get('title', '')}\n{item.get('snippet', '')}\n🔗 {item.get('link', '')}"
                for item in items
            ]
            all_snippets.extend(snippets)
        else:
            print(f"[ERROR] Query failed: {query}, Status: {response.status_code}")
    
    # Return all snippets gathered from the search results
    return "\n\n".join(all_snippets)