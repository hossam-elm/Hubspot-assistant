from openai import OpenAI
from auths.auth import get_google_sheet
import os
from dotenv import load_dotenv
from datetime import datetime, timezone
from pipe import report
from typing import Dict
import time
from gspread.exceptions import APIError
from utils.setup_log import logger
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
        logger.error(f"[ERROR] report() failed for {company_name}: {e}", exc_info=True)
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
        if 'cse' in job:
            return 1
        if 'office' in job:
            return 2
        if 'hr' in job or 'human resources' in job:
            return 3
        if 'people' in job:
            return 4
        if 'marketing' in job:
            return 5
        if 'talent acquisition' in job or 'talent ' in job:
            return 6
        return 7

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
Tu réponds en français, tu priorises les informations récentes (2025) et les chiffres précis, tu es un analyste B2B senior chez Atelier Box, expert en cadeaux d’entreprise personnalisés, welcome packs, textiles premium, goodies écoresponsables et e-shops internes.

Utilise ce contexte web (extraits sémantiques) pour générer la fiche compte :
{web_context}

🏢 FICHE COMPTE – {company_name}
🔹 Secteur & Modèle économique
• Secteur d’activité (ex : FinTech, SaaS, Retail…)
• Modèle économique (ex : abonnement B2B, marketplace…)
• Date de création + age en 2025

👥 Taille de l’entreprise
• Nombre de collaborateurs (total + France si dispo) si tu ne sais pas tu donne une estimation + source de l'info
• Évolution : hypercroissance / stable / décroissance

💰 Clients / Chiffre d'affaires
• Nombre estimé de clients si tu ne sais pas tu donne une estimation
• Chiffre d’affaires annuel, si tu ne sais pas tu donne une estimation chiffrées

🌍 Présence géographique
• Siège social
• Autres bureaux (villes + pays) + nombre total
• nombre de pays ou l'entreprise est présente

🏛️ CSE
• Présence d’un CSE ? (Oui / Non / À confirmer) si tu ne sais pas tu donnes une estimation et justification

🔥 Actualités & signaux business
• Levée de fonds avec dates et série / rebranding avec dates / date et localisation lancement de bureaux
• Recrutement actif? tu donnes le nombre de poste ouverts
• Expansion produit ou géographique avec dates/années
• Rebranding, achats/fusion, positionnement, tu donnes le nom de la campagne et la dates/années
• action relative à la certification RSE, expansion, croissance avec des dates précises 

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
  finalement tu donne un petit brief pour nos commercials qui résume tous ça pour savois exactement comment intéragir avec ce lead

📅 Calendrier opportunités 2025 (sections très importantes)
  • Liste exhaustive des événements internes ou publics à venir, sinon les événements passés en 2025, pour référence on est on mai 2025 avec **dates précises** (JJ/MM/AAAA) : noms, lieux et objectifs pour chaque, au moins 5


✅ Score Atelier Box
En vous basant sur le potentiel de cette entreprise en matière de cadeaux d’entreprise, veuillez attribuer un score sur 100 en utilisant la grille d’évaluation en bas, tu te base sur la croissance, nombre de collaborateurs, événements, cse et autres indices que tu pense importants
90–100: Strategic key account: CSE, présence dans plusieurs pays, +1000 collaborateurs..

80–89: High-priority:  CSE, présence dans plusieurs pays, +500 collaborateurs...

70–79: Warm lead: +200 collaborateurs

60–69: Nurture/monitor: +100 collaborateurs

<60: Low match: the rest

. Évitez de donner systématiquement la même note de 85 ou autres, je t'encourages à donner des mauvaise note pour ceux qui ne mérite pas, think critically et d'habitude on cherche un client pour au moins 1500 euros de commande, la box coûte au moins 45 euros et tu sais déja le prix pour textiles ou objets. Soyez rigoureux et baissez la note en cas de données manquantes ou faibles.
• Justification en une phrase claire

Entreprise à analyser : {company_name}
"""

    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0.0,
        top_p=1,
        presence_penalty=0, 
        frequency_penalty=0,
        messages=[
            {"role": "developer", "content": "You are a company profile generator for Atelier Box, you prioritize numbers and dates, and you give actionable and concrete informations."},
            {"role": "user",      "content": prompt}
        ]
    )
    profile_text = completion.choices[0].message.content

    # Return both the profile and the Linkedin contacts block
    return {
        "profile": profile_text,
        "linkedin_contacts": li_block
    }


BATCH_SIZE = 10

def batch_update_with_fallback(sheet, data):
    for attempt in range(5):
        try:
            sheet.batch_update(data)
            return True
        except APIError as e:
            err_str = str(e).lower()
            if "500" in err_str or "timeout" in err_str:
                wait_time = 2 ** attempt
                logger.error(f"⚠️ Batch API error (attempt {attempt + 1}) — retrying in {wait_time}s...")
                time.sleep(wait_time)
            else:
                logger.error(f"❌ Unrecoverable batch update error: {e}")
                break
    return False

def fill_bd():
    sheet = get_google_sheet(SERVICE_ACCOUNT_FILE, sheet_key)
    records = sheet.get_all_records()
    names = sheet.col_values(1)
    existing_profiles = sheet.col_values(3)

    updates = []

    for i, row in enumerate(records, start=2):
        company_name = names[i - 1] if i - 1 < len(names) else ""
        current_note = existing_profiles[i - 1] if i - 1 < len(existing_profiles) else ""

        if not company_name.strip() or current_note.strip():
            continue

        logger.info(f"🔍 Generating profile for: {company_name}")
        result = get_company_profile(company_name)
        if not result:
            continue

        combined = (
            f"{result['profile']}\n\n"
            "👥 Interlocuteurs clés à contacter\n"
            f"{result['linkedin_contacts']}"
        )
        updates.append((i, combined))
        logger.info(f"✅ Profile stored for {company_name}")

    if not updates:
        logger.info("✅ No new profiles to update.")
        return

    # Process updates in batches for atomicity and partial update fallback
    for batch_start in range(0, len(updates), BATCH_SIZE):
        batch = updates[batch_start:batch_start + BATCH_SIZE]
        data = [{'range': f'C{row_index}', 'values': [[text]]} for row_index, text in batch]

        success = batch_update_with_fallback(sheet, data)
        logger.info(f"✅ Successfully updated {len(batch)} rows")
        if not success:
            # fallback to individual updates
            logger.error(f"⚠️ Batch update failed for rows {batch_start + 2} to {batch_start + len(batch) + 1}. Trying individual updates.")
            for row_index, text in batch:
                for attempt in range(5):
                    try:
                        sheet.update(f'C{row_index}', [[text]])
                        logger.info(f"✅ Updated row {row_index} individually.")
                        break
                    except APIError as e:
                        err_str = str(e).lower()
                        if "500" in err_str or "timeout" in err_str:
                            wait_time = 2 ** attempt
                            logger.error(f"⚠️ API error updating row {row_index} (attempt {attempt+1}) — retrying in {wait_time}s...")
                            time.sleep(wait_time)
                        else:
                            logger.error(f"❌ Unrecoverable error on row {row_index}: {e}")
                            break

if __name__ == "__main__":
    fill_bd()