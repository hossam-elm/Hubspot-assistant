# HubSpot Assistant 🤖

> AI-powered company profiling tool that automatically researches companies and generates sales-ready lead intelligence.

Built for **Atelier Box** (corporate gifts company) to quickly understand prospects and identify sales opportunities. **Successfully enriched 1700+ leads and customer profiles and still going.**

## What it does

- 🔍 **Scrapes the web** for company info using Google CSE
- 🧠 **AI analysis** with GPT-4o-mini to generate comprehensive company profiles
- 🔍 **Semantic matching** finds relevant content using embeddings and cosine similarity
- 📊 **Scores leads** on a 100-point scale for sales prioritization
- 💰 **Identifies opportunities** like "Welcome kits for new hires" or "Corporate swag for events"
- ⚡ **Caches everything** in SQLite for speed (refreshes weekly)
- 📋 **Sends Batch results to Google Sheets** for easy HubSpot import (automatic HubSpot integration planned for future)
- 📊 **Standard HubSpot format** - A1: ID de fiche d'information, B2: Nom de l'entreprise, C3: Corp de note (just add C3 to existing exports)

## Tech Stack

- **OpenAI GPT-4o-mini** for content generation
- **SQLite** for intelligent caching
- **httpx** for fast async web scraping
- **trafilatura** for clean content extraction
- **scikit-learn** **text-embedding-ada-002** for semantic matching
- **Google Sheets API** for data export

## Quick Start

```bash
# Setup
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Mac/Linux
pip install -r requirements.txt

# Add your API keys to .env
# Required environment variables:
# - OPENAI_API_KEY: Your OpenAI API key
# - google_key: Google API key for Custom Search
# - google_cse_id: Google Custom Search Engine ID
# - serpapi_key: SerpAPI key (alternative search)
# - hubspot_key: HubSpot API key (optional)
# - sheet_key: Google Sheet ID (from sheet URL)
# - SERVICE_ACCOUNT_FILE: Path to Google service account JSON, or just make the sheet public
# - GCP_PROJECT_ID: Your Google Cloud Project ID

# Run it
python app.py
```

## Example Output

```
🏢 FICHE COMPTE – Stripe
🔹 Secteur: FinTech, paiements en ligne
• Modèle économique: API de paiement B2B, abonnement
• Date de création: 28/09/2011 (13 ans en 2025)

👥 Taille de l'entreprise
• 8000+ collaborateurs, hypercroissance
• Recrutement actif: 500+ postes ouverts

💰 Clients / Chiffre d'affaires
• 2M+ entreprises clientes
• CA estimé: $14B+ (2024)

🌍 Présence géographique
• Siège: San Francisco, bureaux dans 40+ pays
• Présence dans 46 pays

🏛️ CSE
• Présence d'un CSE: Oui (confirmé)

🔥 Actualités & signaux business
• Levée de fonds: $6.5B Series H (mars 2024)
• Expansion: Nouveaux bureaux à Singapour (2024)

🎯 Opportunités Atelier Box:
• Welcome kits onboarding (500+ nouveaux/mois)
• Hoodies corporate pour équipes tech (8000+ employés)
• Goodies événements (10+ conférences/an)
• Coffrets VIP pour direction

📅 Calendrier opportunités 2025
• 15/03/2025: Stripe Sessions (San Francisco)
• 22/04/2025: Hackathon interne (Paris)
• 10/06/2025: Summit Europe (Amsterdam)

✅ Score Atelier Box: 92/100 (Strategic key account)
• Justification: CSE confirmé, présence multi-pays, +8000 collaborateurs
```

## Architecture

```
app.py          # Main entry point
pipe.py         # Data processing pipeline  
searchfuncs/    # Web scraping (Google, SerpAPI)
utils/          # Caching, logging, API clients
```

## Cool Features

- **Token optimization** - Smart rate limiting (200K TPM)
- **Semantic search** - 30+ queries to find relevant info
- **Async scraping** - Concurrent requests with retry logic
- **Lead scoring** - Automated 100-point qualification system
- **Business intelligence** - Revenue estimates, growth indicators
- **Customizable prompts** - Modify AI prompts in `app.py` for your industry and company
- **Flexible search queries** - Edit Google search queries in `searchfuncs/searchgoogle.py`

## Performance

- **<2s response** for cached results
- **1000+ req/min** concurrent processing
- **$0.02/query** optimized token usage
- **guaranteed uptime** with robust error handling

## Security

- Rate limiting to prevent abuse
- Comprehensive error handling
- OAuth2 for Google APIs

---

**Built and shipped to help sales team target the right leads**

*Note: Automatic HubSpot integration was planned but never added because the Google Sheets import workflow was working perfectly. Will be automated in the future to automatically enrich new leads as they come in.* 