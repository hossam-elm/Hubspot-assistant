import gspread
from google.oauth2.service_account import Credentials

def get_google_sheet(SERVICE_ACCOUNT_FILE, sheet_key):
    # Define the scope
    SCOPE = [
        "https://www.googleapis.com/auth/spreadsheets",
        "https://www.googleapis.com/auth/drive"
    ]

    # Authorize the client
    creds = Credentials.from_service_account_file(SERVICE_ACCOUNT_FILE, scopes=SCOPE)
    client = gspread.authorize(creds)
    sheet = client.open_by_key(sheet_key).sheet1
    return sheet