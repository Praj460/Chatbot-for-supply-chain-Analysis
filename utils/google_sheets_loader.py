import gspread
import pandas as pd
from oauth2client.service_account import ServiceAccountCredentials

def load_data_from_sheets():
    # ---------------------------------------------
    # Step 1: Setup credentials and connect to sheet
    # ---------------------------------------------
    scope = ["https://spreadsheets.google.com/feeds",
             "https://www.googleapis.com/auth/drive"]
    creds = ServiceAccountCredentials.from_json_keyfile_name(
        "service_account.json", scope)  # <-- update with your correct path
    client = gspread.authorize(creds)

    # ---------------------------------------------
    # Step 2: Open the spreadsheet and worksheet
    # ---------------------------------------------
    sheet = client.open("Supply_Chain_Data").sheet1  # <-- make sure this matches your actual Sheet name
    data = sheet.get_all_records()

    # ---------------------------------------------
    # Step 3: Convert to DataFrame
    # ---------------------------------------------
    df = pd.DataFrame(data)
    df.columns = df.columns.str.strip()  # clean column names

    # ---------------------------------------------
    # Step 4: Clean & parse dates (if needed)
    # ---------------------------------------------
    date_columns = [
        "PQ First Sent to Client Date",
        "PO Sent to Vendor Date",
        "Scheduled Delivery Date",
        "Delivered to Client Date",
        "Delivery Recorded Date"
    ]
    for col in date_columns:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    # ---------------------------------------------
    # Step 5: Handle numeric conversion if needed
    # ---------------------------------------------
    num_columns = ["Weight (Kilograms)", "Freight Cost (USD)"]
    for col in num_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df, date_columns
