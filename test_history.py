import requests

url = 'http://127.0.0.1:5000/history'

try:
    response = requests.get(url)
    if response.status_code == 200:
        historical_data = response.json()
        print("Successfully received historical data. First entry:")
        print(historical_data[0]) # Print first entry to avoid overwhelming output
        print(f"Total {len(historical_data)} historical entries received.")
    else:
        print(f"Failed to retrieve historical data with status code {response.status_code}:")
        print(response.json())
except requests.exceptions.ConnectionError:
    print("Could not connect to Flask app. Make sure it's running.")
except Exception as e:
    print(f"An unexpected error occurred: {e}")