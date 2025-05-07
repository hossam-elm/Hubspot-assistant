import requests

# HubSpot API URL for fetching companies
COMPANY_URL = "https://api.hubapi.com/companies/v2/companies/paged"

def get_companies_from_hubspot(HUBSPOT_API_KEY):
    url = COMPANY_URL
    params = {
        'hapikey': HUBSPOT_API_KEY,
        'properties': 'name',
        #'limit': 10
    }

    companies = []
    while url:
        response = requests.get(url, params=params)
        data = response.json()

        if 'results' in data:
            companies.extend(data['results'])
        
        # Get the next page of results if available
        url = data.get('paging', {}).get('next', {}).get('link', None)
    
    return companies