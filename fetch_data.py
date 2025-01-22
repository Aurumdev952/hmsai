import requests
import pandas as pd
import base64
import os
import json
from datetime import datetime, timedelta
from tqdm import tqdm

# Start a session
session = requests.Session()

# Step 1: GET Request to Login Page
login_url = "https://resensys.net:8443/login/"
response = session.get(login_url)
csrf_token = session.cookies.get('csrftoken')

# Step 2: POST Request with CSRF Token
login_data = {
    "username": "Civionics_AI_Demo",
    "password": "GIaP0A5wA1L4",
    "csrfmiddlewaretoken": csrf_token
}
headers = {"Referer": login_url}
response = session.post(login_url, data=login_data, headers=headers)
print("Login Response Status Code:", response.status_code)
print("Login Response Cookies:", response.cookies)

# Data Acquisition
data_acquire_url = "https://resensys.net:8443/api/v1/data_acquire"

def get_existing_data_files():
    """Get existing data file names and extract QuantityName and DID."""
    existing_files = [f for f in os.listdir('.') if f.startswith("data_acquire_data_") and f.endswith(".csv")]
    existing_entries = set()
    for file in existing_files:
        try:
            parts = file.replace("data_acquire_data_", "").replace(".csv", "").rsplit("_", 1)
            if len(parts) == 2:
                QuantityName = parts[0]
                DID = parts[1]
                existing_entries.add((QuantityName, DID))
        except Exception as e:
            print(f"Error processing file {file}: {e}")
    return existing_entries

def fetch_device_data(payload, session_obj):
    start_date = datetime(2024, 2, 14)
    end_date = datetime(2024, 12, 22)

    # Initialize an empty DataFrame to hold all the data
    all_data = pd.DataFrame()

    # Calculate the total number of weeks to show progress
    total_weeks = (end_date - start_date).days // 7 + 1

    with tqdm(total=total_weeks, desc=f"Fetching data for {payload['QuantityName']}") as pbar:
        while start_date < end_date:
            next_week = start_date + timedelta(weeks=1)

            # Ensure the last request doesn't go beyond the end_date
            current_end_date = min(next_week, end_date)

            # Prepare the payload for the current week
            payload_c = {
                "username": login_data["username"],
                "password": login_data["password"],
                "SID": payload["SID"],
                "DID": payload["DID"],
                "DF": payload["DataFormat"],
                "T_start": start_date.strftime("%Y-%m-%d %H:%M:%S"),
                "T_end": current_end_date.strftime("%Y-%m-%d %H:%M:%S"),
                "csrfmiddlewaretoken": session_obj.cookies.get('csrftoken')
            }
            print("payload", payload_c)
            response = session_obj.post(data_acquire_url, data=payload_c, headers={"Referer": data_acquire_url})
            print(f"Data Acquisition Response for {payload['QuantityName']} ({start_date} to {current_end_date}): Status Code: {response.status_code}")

            if response.status_code == 200:
                try:
                    data = response.json()
                    if data:
                        week_data = pd.DataFrame([y for x,y in data.items()])
                        all_data = pd.concat([all_data, week_data], ignore_index=True)
                except ValueError as e:
                    print("Error decoding JSON:", e)
            else:
                print("Request failed with status code:", response.status_code)
                with open("error_log.txt", "a") as f:
                    f.write(response.text)

            # Move to the next week
            start_date = next_week
            pbar.update(1)

    # Save all merged data to a CSV file
    if not all_data.empty:
        filename = f"data_acquire_data_{payload['QuantityName']}_{payload['DID']}.csv"
        all_data.to_csv(filename, index=False)
        print(f"Data saved to {filename}")
    else:
        print(f"No data available for {payload['QuantityName']} within the specified date range.")

# Filter registration data based on existing files
registration_url = "https://resensys.net:8443/api/v2/registration"
payload = {"username": login_data["username"], "password": login_data["password"]}
response = session.post(registration_url, data=payload)
print("Registration Response Status Code:", response.status_code)

try:
    registration_data = response.json()
    if registration_data:
        registration_data = [y for x, y in registration_data.items()]

        # Get existing entries from current directory
        existing_entries = get_existing_data_files()

        # Filter out already fetched entries
        registration_data = [p for p in registration_data if (p['QuantityName'], p['DID']) not in existing_entries]

        for p in tqdm(registration_data):
            try:
                print("Fetching", p["QuantityName"])
                fetch_device_data(p, session)
            except Exception as e:
                print(f"Error fetching data for {p['QuantityName']}: {e}")
    else:
        print("No registration data returned.")
except ValueError:
    print("Invalid JSON response received for registration.")
