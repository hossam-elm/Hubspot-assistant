from openai import OpenAI
from auth import get_google_sheet
from searchserpapi import serpapi_search
from searchgoogle import google_cse_search
from searchddg import duckduckgo_grouped_search
import os
from dotenv import load_dotenv
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


def get_company_profile(company_name):
    try:
        web_context = google_cse_search(company_name, google_key, google_cse_id)
        #web_context = duckduckgo_grouped_search(company_name)
        if not web_context:
            web_context = "Aucune information trouvée sur le web."
        else:
            print(f"Web context fetched for {company_name}")

    except Exception as e:
        print(f"[ERROR] Failed to fetch web context for {company_name}: {e}")
        web_context = "Erreur lors de la récupération des informations web."

    prompt=f"""
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

            📅 Calendrier opportunités 2025
            • Événements internes ou publics avec dates précises, tu prends tous les dates que tu trouves
            (ex : séminaires, lancements, salons)

            👥 Interlocuteurs clés à contacter
            • Nom + fonction (uniquement : Marketing, RH, Office, Events, CSE, Talent…)+ Lien Linkedin
            • Priorité aux profils décisionnaires ou influenceurs
            • Pas de commerciaux, pas de profils trop juniors
            • C'est très important de donner des noms et des liens Linkedin, pas seulement des fonctions
            • Si pas d’infos, indiquer “Aucun contact trouvé”

            ✅ Score Atelier Box (sur 100)
            • Note sur 100 en fonction : potentiel gifting, international, volume, culture événementielle, RSE
            • Justification en une phrase claire

            🎯 Ne donne que des informations concrètes et utiles à un commercial, ne donne pas des fausses informations. Pas de phrases vagues, pas de blabla.

            Entreprise à analyser : {company_name}            


            """
    completion = client.chat.completions.create(
    model="o3-mini",
    messages=[
        {"role": "developer", "content": "You are a company profile generators that helps my company 'atelierbox', specialist in corporate gifts, to gather information about companies, we offer mainly personnalized high quality products and textiles, giftboxes, onboarding boxes,ecommerce platforms for their clients or employees, and  everything related, you are professional, concise, and clear with proper organization here's an example of the output I want you to generate: "},
        {"role": "user", "content": prompt}
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

fill_bd()