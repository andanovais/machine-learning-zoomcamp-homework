import requests

url = "http://localhost:9696/predict"

sample_game = {
    "weighted_positive_ratio": 96.2,
    "price_final": 14.99,
    "rating": "Overwhelmingly Positive",
    "tags": "indie",
}

response = requests.post(url, json=sample_game).json()
print("API Response:", response)