from datetime import datetime

import pandas as pd
import requests

# Configuration
API_URL = "https://resensys.net:8443/api/v2/registration"
LOGIN_URL = "https://resensys.net:8443/login/"
DATA_ACQUIRE_URL = "https://resensys.net:8443/api/v1/data_acquire"
USERNAME = "Civionics_AI_Demo"
PASSWORD = "GIaP0A5wA1L4"
RESAMPLE_INTERVAL = "5min"

session = requests.Session()


def login(username, password):
    print("Attempting to log in...")
    response = session.get(LOGIN_URL)
    csrf_token = session.cookies.get("csrftoken")
    login_data = {
        "username": username,
        "password": password,
        "csrfmiddlewaretoken": csrf_token,
    }
    headers = {"Referer": LOGIN_URL}
    response = session.post(LOGIN_URL, data=login_data, headers=headers)
    if response.status_code == 200:
        print("Login successful")
    else:
        print("Login failed")
        raise Exception("Login failed")


def fetch_registered_devices(api_url, username, password):
    print("Fetching registered devices...")
    payload = {"username": username, "password": password}
    response = session.post(api_url, data=payload)
    if response.status_code == 200:
        print("Successfully fetched registered devices")
        return [device for _, device in response.json().items()]
    else:
        print("Failed to fetch registered devices")
        raise Exception("Failed to fetch registered devices")


def fetch_device_data(
    device,
    data_acquire_url,
    username,
    password,
    resample_interval,
    start_date,
    end_date,
):
    print(f"Fetching data for device SID: {device['SID']}")
    start_date = datetime.strptime(start_date, "%Y-%m-%d")
    end_date = datetime.strptime(end_date, "%Y-%m-%d")
    all_data = pd.DataFrame()

    payload = {
        "username": username,
        "password": password,
        "SID": device["SID"],
        "DID": device["DID"],
        "DF": device["DataFormat"],
        "T_start": start_date.strftime("%Y-%m-%d %H:%M:%S"),
        "T_end": end_date.strftime("%Y-%m-%d %H:%M:%S"),
        "csrfmiddlewaretoken": session.cookies.get("csrftoken"),
    }

    response = session.post(
        data_acquire_url, data=payload, headers={"Referer": data_acquire_url}
    )
    if response.status_code == 200:
        print(f"Successfully fetched data for device SID: {device['SID']}")
        try:
            data = response.json()
            if data:
                # Convert the data values to float
                processed_data = []
                for x, y in data.items():
                    # Convert all numeric values in the dictionary to float
                    processed_y = {
                        k: float(v) if k != "Time" else v for k, v in y.items()
                    }
                    processed_data.append(processed_y)

                week_data = pd.DataFrame(processed_data)
                all_data = pd.concat([all_data, week_data], ignore_index=True)
        except ValueError as e:
            print(f"Failed to parse device data: {e}")
            pass
    else:
        print(f"Failed to fetch data for device SID: {device['SID']}", response.text)

    if not all_data.empty:
        numeric_columns = all_data.columns
        all_data[numeric_columns] = all_data[numeric_columns].astype(float)
        all_data["Time"] = pd.to_datetime(all_data["Time"], unit="s")
        all_data.set_index("Time", inplace=True)
        all_data = all_data.resample(resample_interval).mean()
    else:
        raise Exception("no data available")

    return all_data


def process_devices(
    devices,
    data_acquire_url,
    username,
    password,
    resample_interval,
    start_date,
    end_date,
    quantity_names,
):
    print("Processing devices...")
    aggregated_data = {}

    for device in devices:
        try:
            quantity_name = device["QuantityName"]
            if quantity_name in quantity_names:
                device_data = fetch_device_data(
                    device,
                    data_acquire_url,
                    username,
                    password,
                    resample_interval,
                    start_date,
                    end_date,
                )
                # Rename the column to the quantity name
                device_data.columns = [
                    f"{quantity_name}_{idx}" for idx in range(len(device_data.columns))
                ]

                if quantity_name not in aggregated_data:
                    aggregated_data[quantity_name] = device_data
                else:
                    aggregated_data[quantity_name] = pd.concat(
                        [aggregated_data[quantity_name], device_data], axis=1
                    )
        except Exception as e:
            print(f"Error processing device DID: {device['DID']}", e)
            pass

    # Aggregate each quantity's data separately
    aggregated_results = {}
    for quantity_name, data in aggregated_data.items():
        # Ensure final aggregated data is float type
        mean_values = data.mean(axis=1).astype(float)
        aggregated_results[quantity_name] = pd.DataFrame(
            mean_values, columns=[quantity_name]
        )

    return aggregated_results


def get_week_data(start_date, end_date, quantity_names):
    print("Starting main process...")
    try:
        login(USERNAME, PASSWORD)
        registered_devices = fetch_registered_devices(API_URL, USERNAME, PASSWORD)
        aggregated_result = process_devices(
            registered_devices,
            DATA_ACQUIRE_URL,
            USERNAME,
            PASSWORD,
            RESAMPLE_INTERVAL,
            start_date,
            end_date,
            quantity_names,
        )

        # Save each aggregated result to a separate CSV file based on the quantity name
        # for quantity_name, data in aggregated_result.items():
        #     data.to_csv(f"aggregated_data_{quantity_name}.csv")
        #     print(f"Data for {quantity_name} saved to 'aggregated_data_{quantity_name}.csv'")
        return aggregated_result
    except Exception as e:
        print("Error:", e)
