from openai import OpenAI
from embed import filter_by_embedding      # new
from guard import clip_or_split             # new (your tiktoken helper)
from gapfill         import gap_fill_once             # new
from auth import get_google_sheet
from searchserpapi import serpapi_search
from searchgoogle import google_cse_search
from searchddg import duckduckgo_grouped_search
import os
from dotenv import load_dotenv
from datetime import datetime, timezone
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

def get_company_profile(company_name: str) -> str:
    """
    Build a web context with:
      – Google CSE (structured=True)
      – Embedding similarity filter (top 12)
      – tiktoken guard + GPT-4o-mini chunk summaries
      – One gap-fill loop

    Returns the final profile string ready for your prompt.
    """
    try:
        # 1️⃣  Structured search
        articles = google_cse_search(
            company_name,
            google_key,
            google_cse_id,
            structured=True
        )

        # 2️⃣  Embedding pre-filter (keeps only the most relevant hits)
        articles = filter_by_embedding(
            articles,
            user_question=company_name,
            top_k=12,
            similarity_floor=0.25,
        )

        # 3️⃣  tiktoken guard  → split long bodies into safe chunks
        chunks = []
        for art in articles:
            chunks.extend(clip_or_split(art.get("content") or art["snippet"]))

        # 4️⃣  First-pass chunk summaries (GPT-4o-mini, cheap)
        chunk_summaries = []
        for txt in chunks:
            chunk_summaries.append(
                client.chat.completions.create(
                    model="gpt-4o-mini",
                    temperature=0.3,
                    messages=[
                        {"role": "system",
                         "content": "Résume en 2 phrases maximum, français:"},
                        {"role": "user", "content": txt[:4000]}  # safety trim
                    ]
                ).choices[0].message.content
            )

        # 5️⃣  Merge summaries (GPT-4o – single call)
        merged_summary = client.chat.completions.create(
            model="gpt-4o",
            temperature=0.2,
            messages=[
                {"role": "system",
                 "content": "Fusionne et dé-double les points:"},
                {"role": "user",
                 "content": "\n\n".join(chunk_summaries)}
            ]
        ).choices[0].message.content

        # 6️⃣  Gap-fill loop (one extra search cycle)
        extra_items = gap_fill_once(
            client=client,
            merged_summary_json={"summary": merged_summary},
            search_fn=lambda q, **kw: google_cse_search(
                company_name=q,
                api_key=google_key,
                cx=google_cse_id,
                structured=True
            ),
            company_name=company_name,
        )

        if extra_items:
            # summarise the extra hits quickly with o3-mini
            for art in extra_items:
                merged_summary += "\n• " + client.chat.completions.create(
                    model="gpt-4o-mini",
                    temperature=0.3,
                    messages=[
                        {"role": "system",
                         "content": "organize ce texte, ce qui tintéresse c'est les informations sur l'entreprise, evenements, news, and contacts"},
                        {"role": "user",
                         "content": (art.get('content') or art['snippet'])[:4000]}
                    ]
                ).choices[0].message.content

        web_context = merged_summary or "Aucune information trouvée sur le web."
        os.makedirs("web_context_logs", exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H-%M-%S")
        log_path = f"web_context_logs/{company_name}_{stamp}.txt"

        with open(log_path, "w", encoding="utf-8") as fh:
            fh.write(web_context)

        print(f"[INFO] web_context saved → {log_path}")
    except Exception as e:
        print(f"[ERROR] Web context build failed for {company_name}: {e}")
        web_context = "Erreur lors de la récupération des informations web."

    # 7️⃣  Build the final prompt exactly as before -------------------------
    prompt = f"""
            Tu réponds en français, Tu es un analyste B2B senior chez Atelier Box, expert en cadeaux d’entreprise personnalisés, welcome packs, textiles premium, goodies écoresponsables, et e-shops en interne.

            Ta mission : générer une fiche compte commerciale structurée et exploitable avant un call avec un prospect en utilisant ce context:{web_context}. Le but est d’identifier des opportunités concrètes de vente et les bons interlocuteurs.

            T'es libre à ajouter des informations pertinentes
            À partir du nom d’une entreprise, recherche des informations fiables et formate les résultats exactement comme suit, avec des sections claires, des emojis, et des puces.

            🏢 FICHE COMPTE – {company_name}
            🔹 Secteur & Modèle économique
            • Secteur d’activité (ex : FinTech, SaaS, Retail…)
            • Modèle économique (ex : abonnement B2B, marketplace…)

            👥 Taille de l’entreprise
            • Nombre de collaborateurs (total + France si dispo)
            • Évolution : hypercroissance / stable / décroissance

            💰 Clients / Chiffre d'affaires
            • Nombre estimé de clients
            • Chiffre d’affaires annuel estimé

            🌍 Présence géographique
            • Siège social
            • Autres bureaux clés (villes + pays)

            🏛️ CSE
            • Présence d’un CSE ? (Oui / Non / À confirmer)

            🔥 Actualités & signaux business
            • Levée de fonds / rebranding / lancement de bureaux
            • Recrutement actif (ex : “+60 postes ouverts”)
            • Expansion produit ou géographique

            🎯 Opportunités Atelier Box
            Liste précise des besoins qu’Atelier Box peut couvrir :
            • Welcome kits onboarding (fréquence ?)
            • Textiles internes (hoodies, polos, vestes équipes sales/tech)
            • Goodies clients ou packs démo
            • Coffrets VIP, speakers, direction
            • Boxes culture interne (anniversaires, milestones, séminaires)
            • E-shop marque blanche (préciser si multi-pays)
            • Un petits briefs pour nos commercial avec les points clés à mettre en avant

            📅 Calendrier opportunités 2025 tous les événement en 2025
            • Événements internes ou publics avec dates précises, tu prends tous les dates que tu trouves
            (ex : séminaires, lancements, salons)

            👥 Interlocuteurs clés à contacter
            • Nom + fonction (uniquement : Marketing, RH, Office, Events, CSE, Talent…)+ Lien Linkedin
            • Priorité aux profils décisionnaires ou influenceurs
            • Pas de commerciaux, pas de profils trop juniors
            • C'est très important de donner des noms et des liens Linkedin, pas seulement des fonctions
            • Si pas d’infos, indiquer “Aucun contact trouvé”
            • ne supprime aucun contact, même si tu trouves des doublons, je veux tout savoir

            ✅ Score Atelier Box (sur 100)
            • Note sur 100 en fonction : potentiel gifting, international, volume, culture événementielle, RSE
            • Justification en une phrase claire

            Before handing me this, go over it again and make sure you have:
            • 1. All the sections
            • 2. All the relevant information
            • 3. Enrich it with your own knowledge


            Entreprise à analyser : {company_name}            

    """

    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "developer", "content": "You are a company profile generators that helps my company 'atelierbox', specialist in corporate gifts, to gather information about companies, we offer mainly personnalized high quality products and textiles, giftboxes, onboarding boxes,ecommerce platforms for their clients or employees, and  everything related, you are professional, concise, and clear with proper organization here's an example of the output I want you to generate: "},
            {"role": "user",      "content": prompt}
        ]
    )
    return completion.choices[0].message.content



def fill_bd():
    sheet = get_google_sheet(SERVICE_ACCOUNT_FILE, sheet_key)
    records = sheet.get_all_records()
    names = sheet.col_values(1) 
    existing_profiles = sheet.col_values(2)

    for i, row in enumerate(records, start=2): 
        company_name = names[i - 1] if i - 1 < len(names) else ""
        current_note = existing_profiles[i - 1] if i - 1 < len(existing_profiles) else ""

        if not company_name.strip():
            continue
        if current_note.strip():
            continue

        print(f"Generating profile for: {company_name}")
        profile = get_company_profile(company_name)
        sheet.update(range_name = f'B{i}', values = [[str(profile)]])
        print(f"Profile for {company_name} updated in row {i}.")


if __name__ == "__main__":
    fill_bd()