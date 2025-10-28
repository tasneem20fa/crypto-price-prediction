import requests

url = "http://127.0.0.1:5000/predict"
data = {
    "open": 27000,
    "high": 27500,
    "low": 26800,
    "volume_btc": 1200,
    "ma7": 27200,
    "ma30": 27150,
    "lag1_close": 26900,
    "day_of_week": 4
}

response = requests.post(url, json=data)
print("Status Code:", response.status_code)
print("Response:", response.json())
