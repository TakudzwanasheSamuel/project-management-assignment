import requests

url = 'http://127.0.0.1:5000/predict'
input_data = {
    "Day_of_Week": "Monday",
    "Semester": 1,
    "Starch": "Sadza",
    "Protein": "Beef",
    "Side": "Salad",
    "Academic_Event": "Regular"
}

try:
    response = requests.post(url, json=input_data)
    if response.status_code == 200:
        prediction_result = response.json()
        print("Prediction successful:", prediction_result)
    else:
        print(f"Prediction failed with status code {response.status_code}:")
        print(response.json())
except requests.exceptions.ConnectionError:
    print("Could not connect to Flask app. Make sure it's running.")
except Exception as e:
    print(f"An unexpected error occurred: {e}")