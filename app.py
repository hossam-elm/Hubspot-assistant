from openai import OpenAI
from auths.auth import get_google_sheet
import os
from dotenv import load_dotenv
from datetime import datetime, timezone
from pipe import report
from typing import Dict

load_dotenv()

# Keys
serpapi_key = os.getenv('serpapi_key')
google_key = os.getenv('google_key')
SERVICE_ACCOUNT_FILE = os.getenv('SERVICE_ACCOUNT_FILE')
google_cse_id = os.getenv('google_cse_id')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
hubspot_key = os.getenv('hubspot_key')
sheet_key = os.getenv('sheet_key')


client = OpenAI()

def get_company_profile(company_name: str) -> Dict[str, str]:
    try:
        rpt = report(company_name)
        semantic_ctx = rpt["semantic_items"]
        linkedin_profiles = rpt["linkedin_profiles"]
    except Exception as e:
        print(f"[ERROR] report() failed for {company_name}: {e}")
        semantic_ctx = []
        linkedin_profiles = []

    # Build semantic context for prompt
    if semantic_ctx:
        web_context = "\n".join(
            f"• [{it['chunk_id']}] {it['chunk']} (↪ {it['link']})"
            for it in semantic_ctx
        )
    else:
        web_context = "Aucune information fiable trouvée."
    # 2) Dedupe linkedin_profiles by name or link
    seen = set()
    unique_profiles = []
    for p in linkedin_profiles:
        key = (p['name'].lower(), p['link'].lower())
        if key not in seen:
            seen.add(key)
            unique_profiles.append(p)

    # 3) Rank LinkedIn contacts: Office > HR > Marketing > Talent Acquisition > others
    def rank_key(profile):
        job = profile.get('job', '').lower()
        if 'office' in job:
            return 1
        if 'hr' in job or 'human resources' in job:
            return 2
        if 'people' in job:
            return 3
        if 'marketing' in job:
            return 4
        if 'talent acquisition' in job or 'talent ' in job:
            return 5
        return 6

    unique_profiles.sort(key=rank_key)

    # 4) Build the LinkedIn block from ranked unique_profiles
    if unique_profiles:
        li_block = "\n".join(
            f"• {p['name']} – {p.get('job','N/A')} – {p['link']}"
            for p in unique_profiles
        )
    else:
        li_block = "Aucun contact Linkedin trouvé."

    # --- Build prompt WITHOUT the interlocuteurs section ---
    prompt = f"""
Tu réponds en français, tu priorises les informations récentes (2025), tu es un analyste B2B senior chez Atelier Box, expert en cadeaux d’entreprise personnalisés, welcome packs, textiles premium, goodies écoresponsables et e-shops internes.

Utilise ce contexte web (extraits sémantiques) pour générer la fiche compte :
{web_context}

🏢 FICHE COMPTE – {company_name}
🔹 Secteur & Modèle économique
• Secteur d’activité (ex : FinTech, SaaS, Retail…)
• Modèle économique (ex : abonnement B2B, marketplace…)

👥 Taille de l’entreprise
• Nombre de collaborateurs (total + France si dispo) si tu ne sais pas tu donne une estimation
• Évolution : hypercroissance / stable / décroissance

💰 Clients / Chiffre d'affaires
• Nombre estimé de clients si tu ne sais pas tu donne une estimation
• Chiffre d’affaires annuel, si tu ne sais pas tu donne une estimation

🌍 Présence géographique
• Siège social
• Autres bureaux (villes + pays) + nombre total

🏛️ CSE
• Présence d’un CSE ? (Oui / Non / À confirmer) si tu ne sais pas tu donne une estimation et justification

🔥 Actualités & signaux business
• Levée de fonds / rebranding / lancement de bureaux
• Recrutement actif? tu donnes le nombre de poste ouverts
• Expansion produit ou géographique
• Rebranding, achats/fusion

🎯 Opportunités Atelier Box:
Voici des exemples d’opportunités que nous proposons :  
  – Welcome kits onboarding  
  – Textiles internes (hoodies, polos, vestes)  
  – Goodies pour événements ou packs démo  
  – Coffrets VIP (speakers, direction)  
  – Boxes culture interne (anniversaires, milestones)  
  – E-shop marque blanche (multi-pays) 
  À partir du contexte web, génère **au moins 3** opportunités **spécifiquement personnalisées** pour {company_name}. Donne pour chaque :
    1. Un titre court (emoji optionnel)  
  2. Une phrase expliquant pourquoi c’est pertinent  
  3. Une estimation de fréquence ou volume si possible  

📅 Calendrier opportunités 2025
• Événements internes ou publics avec dates précises, listes tous les événements même-ci si t'es pas sûr

✅ Score Atelier Box (sur 100)
• Potentiel gifting, international, volume, culture événementielle, RSE
• Justification en une phrase claire

Entreprise à analyser : {company_name}
"""

    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0.0,
        messages=[
            {"role": "developer", "content": "You are a company profile generator for Atelier Box…"},
            {"role": "user",      "content": prompt}
        ]
    )
    profile_text = completion.choices[0].message.content

    # Return both the profile and the Linkedin contacts block
    return {
        "profile": profile_text,
        "linkedin_contacts": li_block
    }


def fill_bd():
    sheet = get_google_sheet(SERVICE_ACCOUNT_FILE, sheet_key)
    records = sheet.get_all_records()
    names = sheet.col_values(1)
    existing_profiles = sheet.col_values(2)

    for i, row in enumerate(records, start=2):
        company_name = names[i-1] if i-1 < len(names) else ""
        current_note = existing_profiles[i-1] if i-1 < len(existing_profiles) else ""

        if not company_name.strip() or current_note.strip():
            continue

        print(f"Generating profile for: {company_name}")
        result = get_company_profile(company_name)
        # Combine the generated profile with the LinkedIn block for storage
        combined = (
            f"{result['profile']}\n\n"
            "👥 Interlocuteurs clés à contacter\n"
            f"{result['linkedin_contacts']}"
        )
        sheet.update(range_name=f'B{i}', values=[[combined]])
        print(f"Profile for {company_name} updated in row {i}.")

if __name__ == "__main__":
    fill_bd()