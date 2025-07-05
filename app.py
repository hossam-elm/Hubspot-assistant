from openai import OpenAI
from auths.auth import get_google_sheet
import os
from dotenv import load_dotenv
from datetime import datetime, timezone
from pipe import report
from typing import Dict, List
import time
from gspread.exceptions import APIError
from utils.setup_log import logger
import tiktoken
from collections import deque


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
TPM_LIMIT = 200_000
TOKEN_WINDOW_SECONDS = 60
token_timestamps = deque()

encoding = tiktoken.encoding_for_model("gpt-4o-mini")
def count_tokens(text: str) -> int:
    return len(encoding.encode(text))
def throttle_if_needed(tokens_needed: int):
    now = time.time()
    # Remove tokens older than 60 seconds
    while token_timestamps and now - token_timestamps[0][0] > TOKEN_WINDOW_SECONDS:
        token_timestamps.popleft()

    used_tokens = sum(t[1] for t in token_timestamps)

    if used_tokens + tokens_needed > TPM_LIMIT:
        wait_time = TOKEN_WINDOW_SECONDS - (now - token_timestamps[0][0])
        logger.warning(f"⏳ Token limit reached. Waiting {int(wait_time)+1}s...")
        time.sleep(wait_time + 1)

    token_timestamps.append((time.time(), tokens_needed))

def truncate_web_context(semantic_ctx, max_tokens=50000):
    truncated_chunks = []
    total_tokens = 0

    for item in semantic_ctx:
        chunk_text = f"• [{item['chunk_id']}] {item['chunk']} (↪ {item['link']})"
        chunk_tokens = count_tokens(chunk_text)

        if total_tokens + chunk_tokens > max_tokens:
            # Stop adding more chunks when token limit is reached
            break
        truncated_chunks.append(chunk_text)
        total_tokens += chunk_tokens

    return "\n".join(truncated_chunks)

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
        web_context = truncate_web_context(semantic_ctx, max_tokens=50000)
        if not web_context.strip():
            web_context = "Aucune information fiable trouvée."
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
• Date de création ou si l'entreprise est le résultat de fusion tu donne cette date (format JJ/MM/YYYY, faut donner la date complète)  + age en 2025 + source

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
    input_tokens = count_tokens(prompt)
    throttle_if_needed(input_tokens)
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
    output_tokens = count_tokens(profile_text)
    throttle_if_needed(output_tokens)
    logger.info(f"🔢 Input tokens: {input_tokens}, Output tokens: {output_tokens}")

    # Return both the profile and the Linkedin contacts block
    return {
        "profile": profile_text,
        "linkedin_contacts": li_block
    }


BATCH_SIZE = 5

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
    names = sheet.col_values(2)
    existing_profiles = sheet.col_values(3)

    logger.info(f"📋 Found {len(records)} total rows in sheet")
    
    # Count companies that need processing
    companies_to_process = 0
    for i, row in enumerate(records, start=2):
        company_name = names[i - 1] if i - 1 < len(names) else ""
        current_note = existing_profiles[i - 1] if i - 1 < len(existing_profiles) else ""
        if company_name.strip() and not current_note.strip():
            companies_to_process += 1
    
    logger.info(f"🎯 Found {companies_to_process} companies that need profiles generated")
    
    if companies_to_process == 0:
        logger.info("✅ All companies already have profiles - nothing to do!")
        return

    batch_updates: List[tuple[int, str]] = []
    processed_count = 0

    for i, row in enumerate(records, start=2):
        company_name = names[i - 1] if i - 1 < len(names) else ""
        current_note = existing_profiles[i - 1] if i - 1 < len(existing_profiles) else ""

        if not company_name.strip() or current_note.strip():
            continue

        logger.info(f"🔍 Generating profile for: {company_name}")
        result = get_company_profile(company_name)
        if not result:
            logger.warning(f"⚠️ Failed to generate profile for: {company_name}")
            continue

        linkedin_contacts = result.get('linkedin_contacts', '')
        if isinstance(linkedin_contacts, list):
            linkedin_contacts = "\n".join(linkedin_contacts)

        combined = f"{result['profile']}\n\n👥 Interlocuteurs clés à contacter\n{linkedin_contacts}".strip()
        batch_updates.append((i, combined))
        processed_count += 1
        logger.info(f"✅ Profile ready for row {i} ({company_name}) - {processed_count}/{companies_to_process} completed")

        # As soon as we have 10 updates, send them in one batch
        if len(batch_updates) >= BATCH_SIZE:
            data = [
                {'range': f'C{row_index}', 'values': [[text]]}
                for row_index, text in batch_updates
            ]
            success = batch_update_with_fallback(sheet, data)
            if success:
                logger.info(f"✅ Batch of {len(batch_updates)} rows updated.")
            else:
                logger.error(f"⚠️ Batch update failed for rows {batch_updates[0][0]}–{batch_updates[-1][0]}. Trying individually.")
                for row_index, text in batch_updates:
                    for attempt in range(5):
                        try:
                            sheet.update(values=[[text]], range_name=f'C{row_index}')
                            logger.info(f"✅ Updated row {row_index} individually.")
                            break
                        except APIError as e:
                            err_str = str(e).lower()
                            if "500" in err_str or "timeout" in err_str:
                                wait_time = 2 ** attempt
                                logger.warning(f"⚠️ API error on row {row_index} (attempt {attempt+1}) — retrying in {wait_time}s...")
                                time.sleep(wait_time)
                            else:
                                logger.error(f"❌ Unrecoverable error on row {row_index}: {e}")
                                break

            # Clear the batch list and continue
            batch_updates.clear()

    # After the loop, if any updates remain (< BATCH_SIZE), send them as a final batch
    if batch_updates:
        data = [
            {'range': f'C{row_index}', 'values': [[text]]}
            for row_index, text in batch_updates
        ]
        success = batch_update_with_fallback(sheet, data)
        if success:
            logger.info(f"✅ Final batch of {len(batch_updates)} rows updated.")
        else:
            logger.error(f"⚠️ Final batch update failed for rows {batch_updates[0][0]}–{batch_updates[-1][0]}. Trying individually.")
            for row_index, text in batch_updates:
                for attempt in range(5):
                    try:
                        sheet.update(values=[[text]], range_name=f'C{row_index}')
                        logger.info(f"✅ Updated row {row_index} individually.")
                        break
                    except APIError as e:
                        err_str = str(e).lower()
                        if "500" in err_str or "timeout" in err_str:
                            wait_time = 2 ** attempt
                            logger.warning(f"⚠️ API error on row {row_index} (attempt {attempt+1}) — retrying in {wait_time}s...")
                            time.sleep(wait_time)
                        else:
                            logger.error(f"❌ Unrecoverable error on row {row_index}: {e}")
                            break

    logger.info("✅ All profiles processed and written.")


if __name__ == "__main__":
    logger.info("🚀 Starting HubSpot Assistant - Company Profile Generator")
    logger.info(f"📊 Processing Google Sheet: {sheet_key}")
    logger.info(f"🔑 Using OpenAI model: gpt-4o-mini")
    
    try:
        fill_bd()
        logger.info("🎉 HubSpot Assistant completed successfully!")
    except Exception as e:
        logger.error(f"💥 HubSpot Assistant failed: {e}")
        raise