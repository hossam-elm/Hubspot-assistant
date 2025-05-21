import time
from duckduckgo_search import DDGS

def duckduckgo_grouped_search(company_name, num_results=5, sleep_between_queries=3):
    # Groupes de mots-clés par thématique
    keyword_groups = {
        "infos_société": [
            "site officiel",
            "nombre d'employés",
            "clients",
            "événements d'entreprise",
            "actualité",
            "CSE",
        ],
        "recrutement_et_contacts": [
            "recrutement",
            "offres d'emploi",
            "LinkedIn",
            f"{company_name} LinkedIn",
            f"{company_name} équipe LinkedIn",
            "contact entreprise",
        ],
        "opportunités_marketing": [
            "responsable communication",
            "responsable marketing",
            "office manager",
            "talent acquisition",
            "event manager",
            "expansion",
            "nouveaux bureaux",
            "lancement produit",
        ]
    }

    all_snippets = []

    with DDGS() as ddgs:
        for group_name, keywords in keyword_groups.items():
            quoted_keywords = [f'"{k}"' if " " in k else k for k in keywords]
            query = f'{company_name} {" OR ".join(quoted_keywords)}'
            print(f"[DEBUG] Groupe `{group_name}` — Query: {query}")

            try:
                results = list(ddgs.text(query, max_results=num_results))
            except Exception as e:
                print(f"[ERROR] {group_name} — Échec DuckDuckGo : {e}")
                continue

            if results:
                print(f"[INFO] {group_name}: {len(results)} résultats")
                for res in results:
                    snippet = f"🔹 [{group_name}] {res.get('title')}\n{res.get('body')}\n🔗 {res.get('href')}"
                    all_snippets.append(snippet)
            else:
                print(f"[WARN] Aucun résultat pour {group_name}")

            time.sleep(sleep_between_queries)

    return "\n\n".join(all_snippets)
