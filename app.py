from openai import OpenAI
from auth import get_google_sheet
from searchserpapi import serpapi_search
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
        web_context = serpapi_search(company_name, serpapi_key)
        if not web_context:
            web_context = "Aucune information trouvée sur le web."
        else:
            print(f"Web context fetched for {company_name}")

    except Exception as e:
        print(f"[ERROR] Failed to fetch web context for {company_name}: {e}")
        web_context = "Erreur lors de la récupération des informations web."

    prompt=f"""
            Please provide a detailed company profile in the following format for the company named {company_name} in french, use your informations and this web context: {web_context}:

            Company Name: [Company Name]

            Sector: [Industry/Sector Information]

            Employees: [Number of Employees and Locations]

            Clients / Revenue: [Number of Clients and Estimated Revenue]

            Geographic Presence: [List of Countries/Regions]

            CSE: [Indicate if there is a CSE (Comité Social et Économique)]

            Pitch: [Company’s current business needs and areas where it might require support or partnerships. Focus on specific projects, products, or services they are interested in (e.g.,corporate gifting, e-commerce platforms, onboarding solutions).]

            Opportunities pour Atelierbox: [List of identified opportunities in terms of projects, needs, events, or ongoing recruitment for atelierbox services.]

            Events: [List of upcoming events, conferences, or trade shows they are participating in or organizing, with dates and locations.]
            Recent News: [Recent news or developments related to the company, including any new product launches, partnerships, or expansions.]

            Key Contacts: [List of contacts with names and roles at the company from linkedin]

            Score: [Provide an overall score to indicate the priority or potential of the account (from 0 to 100).]

            Ensure the output includes specific details about the company’s operations, needs, and potential business opportunities. The style should be professional, concise, and clear with proper organization. 

            Here is an example of the output:
            Company Name: **Comexposium**
            Sector:
            Événementiel / Organisation de salons et conférences professionnels
            Employees:
            Environ 800 employés répartis principalement au siège à Paris (France), avec des bureaux et collaborateurs dans plusieurs pays stratégiques.
            Clients / Revenue:
            Plus de 45 000 clients exposants chaque année, 3,5 millions de visiteurs sur leurs événements. Chiffre d'affaires estimé à plus de 350 millions d'euros annuellement.
            Geographic Presence:
            Présence internationale : France (siège), Europe (Espagne, Italie, Allemagne, Royaume-Uni), Asie (Chine, Indonesia, Singapour), Amériques (États-Unis, Canada, Brésil), Moyen-Orient.
            CSE:
            Oui – Un Comité Social et Économique actif, particulièrement pour le siège parisien.
            Pitch:
            Comexposium organise de grands salons et expositions (SIAL, Paris Games Week, SIMA...) nécessitant une gestion logistique pointue et des solutions innovantes pour renforcer l'engagement des exposants, visiteurs et partenaires. Ils recherchent régulièrement des options de cadeaux d’affaires (swag), des solutions d’accueil et d’onboarding pour les nouveaux partenaires et clients, ainsi que des plateformes de e-commerce pour le merchandising événementiel. Avec la reprise de l’activité événementielle, ils sont particulièrement attentifs à la différenciation de leur offre par le biais de cadeaux d’entreprise originaux et responsables.
            Opportunities:
            - Préparation de grands événements récurrents en 2024–2025 (SIAL, Paris Retail Week, Salon du Cheval, etc.) : opportunités de coffrets cadeaux, kits de bienvenue, cadeaux VIP.
            - Besoin croissant de cadeaux personnalisés et produits locaux ou écoresponsables pour les exposants et visiteurs premium.
            - Sensibilisation croissante au bien-être salarié, notamment via des campagnes internes pilotées par le CSE (cadeaux de fin d’année, récompenses).
            - Recrutements importants dans les équipes salons/marketing en 2024 – intégration de nouveaux collaborateurs à soutenir avec des welcome packs.
            - Intérêt possible pour des solutions digitales de gestion de cadeaux ou de boutique interne.
            Key Contacts:
            - Isabelle Charlier, Chief Human Resources Officer
            - Sandra Fournier, Head of Procurement
            - Olivier Ferraton, Directeur Général
            - Julie Rivet, Marketing & Partnerships Director
            Score: **86 / 100**
            (Potentiel élevé en raison du volume d’événements, de la récurrence des besoins cadeaux, de la présence du CSE et de l’orientation vers l’innovation et la RSE.)   
            """
    completion = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {"role": "developer", "content": "You are a company profile generators that helps my company 'atelierbox', specialist in corporate gifts, to gather information about companies, we offer mainly personnalized high quality products and textiles, giftboxes, onboarding boxes,ecommerce platforms for their clients or employees, and  everything related, you are professional, concise, and clear with proper organization."},
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