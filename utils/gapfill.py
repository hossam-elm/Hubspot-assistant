import json, re

GAP_PROMPT = """Vous êtes un contrôleur de couverture pour GPT-4o-mini.  
À partir du résumé JSON ci-dessous, identifiez jusqu’à 10 faits non traités de la liste suivante:
Secteur d’activité (ex : FinTech, SaaS, Retail…)
Modèle économique (ex : abonnement B2B/B2C, marketplace, freemium, licensing…)
Taille de l’entreprise
Nombre total de collaborateurs
Nombre de collaborateurs en France
Évolution des effectifs (hypercroissance / stable / décroissance)
Nombre estimé de clients (utilisateurs ou entreprises)
Chiffre d’affaires annuel estimé (dernier exercice)
Siège social (adresse complète)
Autres bureaux (villes + pays)
Présence d’un CSE (Oui / Non / À confirmer)
Levées de fonds : dates, montants, investisseurs
Rebranding ou changement de nom
Ouverture de nouveaux bureaux (dates et lieux)
Recrutement actif : nombre de postes ouverts, profils recherchés
Lancements de produits ou de nouvelles offres
Nom et dates de chaque événement (interne ou externe)
Interlocuteurs clés à contacter
un query par ligne sans rien d'autre

Pour chacun, proposez une requête Google concise et ciblée (une par point), sans commentaire, sans rien juste les requêtes  
- Pour tout événement trouvé, générez une requête pour obtenir détails date, lieu et participants.  
- Ajoutez des requêtes pour contacts LinkedIn en marketing, CSE, RH, office.  
- Adaptez les requêtes à l’année 2025 et aux villes/pays de l’entreprise.  
- Évitez les doublons et ne surchargez pas une requête avec trop d’informations."""


BULLET_RE = re.compile(r"[-•]\s*(.+)")

def _extract_bullets(text: str) -> list[str]:
    return BULLET_RE.findall(text)

def gap_fill_once(
    client,                        # OpenAI client
    merged_summary_json: dict,
    search_fn,                     # callable(company_name, structured=True)
    company_name: str,
) -> list[dict]:
    """
    Ask GPT-4o-mini for coverage gaps, turn them into extra Google queries,
    run one more search cycle, and return NEW structured items.
    """
    gaps_txt = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0,
        messages=[
            {"role": "system", "content": GAP_PROMPT},
            {"role": "user",   "content": json.dumps(merged_summary_json)}
        ]
    ).choices[0].message.content
    MAX_GAP_QUERIES = 10
    print("=== GPT Gaps Output ===")
    print(gaps_txt)
    gap_queries = _extract_bullets(gaps_txt)
    gap_queries = gap_queries[:MAX_GAP_QUERIES] 
    if not gap_queries:
        return []                          # nothing to add

    # Run an extra Google CSE for each gap, concatenate & dedup
    
    extra_items, seen = [], set()
    for q in gap_queries:
        q_str = f"{company_name} {q}"
        print(f"[GAP] searching → {q_str}")
        for it in search_fn(q_str, structured=True, num_results=5):  # ↓ limit hits
            if it["link"] in seen:
                continue
            seen.add(it["link"])
            extra_items.append(it)
        print(f"[GAP] {q} done • new items: {len(extra_items)}")
    return extra_items
