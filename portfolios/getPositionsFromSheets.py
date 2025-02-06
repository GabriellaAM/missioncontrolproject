from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build
from dotenv import load_dotenv
import pandas as pd
import os

load_dotenv()

# Defina o escopo
SCOPES = ['https://www.googleapis.com/auth/spreadsheets.readonly']

# Autenticação
creds = Credentials.from_service_account_file(os.getenv('SERVICE_ACCOUNT_FILE'), scopes=SCOPES)
service = build('sheets', 'v4', credentials=creds)

# ID da planilha e intervalo
SPREADSHEET_ID = '1156S3pm9vQ2JP5bE0yvpj_Y8FE55ecovHS_JvIys5dc'
RANGE_NAME = 'EXC!A:AAZ'

# Ler dados
result = service.spreadsheets().values().get(spreadsheetId=SPREADSHEET_ID, range=RANGE_NAME).execute()
values = result.get('values', [])

if not values:
    print('Nenhum dado encontrado.')
else:
    # Convert the list of values to a pandas DataFrame
    df = pd.DataFrame(values[1:], columns=values[0])  # Assuming the first row is the header
    print(df)
